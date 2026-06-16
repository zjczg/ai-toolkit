"""High-level provider-agnostic chat/text generation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ai_toolkit.ark import create_responses
from ai_toolkit.config import get_settings
from ai_toolkit.deepseek import create_chat_completion
from ai_toolkit.media import upload_public_url
from ai_toolkit.types import AIToolkitError, ChatCompletionResult


def complete(
    *,
    provider: str,
    messages: list[dict[str, Any]] | None = None,
    prompt: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> ChatCompletionResult:
    """Generate text with a normalized SDK interface."""
    normalized_provider = normalize_provider(provider)
    if normalized_provider == "deepseek":
        return _complete_deepseek(messages=messages, prompt=prompt, model=model, **kwargs)
    if normalized_provider == "ark":
        return _complete_ark(messages=messages, prompt=prompt, model=model, **kwargs)
    raise AIToolkitError(f"unknown chat provider: {provider}")


def complete_json(
    *,
    provider: str,
    messages: list[dict[str, Any]] | None = None,
    prompt: str | None = None,
    model: str | None = None,
    schema: dict[str, Any] | None = None,
    schema_name: str = "structured_response",
    strict: bool = True,
    disable_thinking: bool = True,
    **kwargs: Any,
) -> ChatCompletionResult:
    """Generate JSON with provider-appropriate structured-output settings.

    For ARK Responses API this uses the official `text.format` field and,
    by default, disables deep thinking for short extraction tasks.

    When ``schema`` is provided, the parsed JSON is checked against a small
    subset of JSON Schema (``type``, ``required``, top-level property types).
    On mismatch the SDK keeps ``result.text``, sets ``result.parsed_json`` to
    ``None`` and ``result.schema_error`` to a human-readable reason — callers
    can decide whether to retry, fall back, or surface the error.
    """
    normalized_provider = normalize_provider(provider)
    request_kwargs = dict(kwargs)
    if normalized_provider == "ark":
        request_kwargs.setdefault(
            "text",
            _ark_text_format(schema=schema, schema_name=schema_name, strict=strict),
        )
        if disable_thinking:
            request_kwargs.setdefault("thinking", {"type": "disabled"})
    elif normalized_provider == "deepseek":
        request_kwargs.setdefault("response_format", {"type": "json_object"})
        if disable_thinking:
            request_kwargs.setdefault("thinking", {"type": "disabled"})

    result = complete(
        provider=provider,
        messages=messages,
        prompt=prompt,
        model=model,
        **request_kwargs,
    )
    parsed = _parse_json(result.text)
    if schema is not None and parsed is not None:
        error = _validate_against_schema(parsed, schema)
        if error is not None:
            result.parsed_json = None
            result.schema_error = error
            return result
    result.parsed_json = parsed
    return result


def multimodal_user_message(
    *,
    text: str,
    images: list[str | Path] | tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """Build a Responses-compatible user message with text and image URLs.

    Local file paths are uploaded via ``media.upload_public_url`` and replaced
    with the resulting public URL, mirroring ``images.generate`` behaviour for
    ARK references. HTTP(S) URLs are passed through unchanged.
    """
    content: list[dict[str, str]] = [{"type": "input_text", "text": text}]
    content.extend(
        {"type": "input_image", "image_url": _image_reference_to_url(image)}
        for image in images
    )
    return {"role": "user", "content": content}


def _image_reference_to_url(reference: str | Path) -> str:
    value = str(reference)
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return value
    return upload_public_url(value)


def normalize_provider(provider: str) -> str:
    value = provider.strip().lower()
    if value in {"ark", "doubao", "volcengine"}:
        return "ark"
    if value == "deepseek":
        return "deepseek"
    return value


def _ark_text_format(
    *,
    schema: dict[str, Any] | None,
    schema_name: str,
    strict: bool,
) -> dict[str, Any]:
    if schema is None:
        return {"format": {"type": "json_object"}}
    return {
        "format": {
            "type": "json_schema",
            "name": schema_name,
            "strict": strict,
            "schema": schema,
        }
    }


_JSON_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "boolean": (bool,),
    "null": (type(None),),
}


def _validate_against_schema(value: Any, schema: dict[str, Any]) -> str | None:
    """Minimal JSON-Schema-style validator. Returns an error string or None.

    Only covers what high-level callers actually use: top-level ``type``,
    ``required`` keys, and the ``type`` of each declared property. Nested
    schemas and refs are out of scope — wrap a real validator if you need them.
    """
    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _matches_type(value, expected_type):
        return f"expected JSON {expected_type}, got {type(value).__name__}"

    if expected_type != "object" or not isinstance(value, dict):
        return None

    for required_key in schema.get("required", []) or []:
        if required_key not in value:
            return f"missing required key {required_key!r}"

    properties = schema.get("properties") or {}
    for key, property_schema in properties.items():
        if key not in value or not isinstance(property_schema, dict):
            continue
        property_type = property_schema.get("type")
        if isinstance(property_type, str) and not _matches_type(value[key], property_type):
            return f"property {key!r} expected {property_type}, got {type(value[key]).__name__}"

    return None


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    types = _JSON_TYPE_CHECKS.get(expected)
    if types is None:
        return True
    return isinstance(value, types)


def _parse_json(text: str) -> Any | None:
    stripped = text.strip()
    for value in (stripped, *_json_blocks(stripped)):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            continue
    return None


def _json_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    stack: list[int] = []
    for index, char in enumerate(text):
        if char == "{":
            stack.append(index)
        elif char == "}" and stack:
            start = stack.pop()
            if not stack:
                blocks.append(text[start : index + 1])
    return blocks


def _complete_deepseek(
    *,
    messages: list[dict[str, Any]] | None,
    prompt: str | None,
    model: str | None,
    **kwargs: Any,
) -> ChatCompletionResult:
    settings = get_settings()
    resolved_model = model or settings.deepseek_chat_model
    resolved_messages = _messages_or_prompt(messages, prompt)
    raw_response = create_chat_completion(resolved_model, resolved_messages, **kwargs)
    text = _extract_deepseek_text(raw_response)
    return ChatCompletionResult(
        provider="deepseek",
        model=resolved_model,
        text=text,
        raw_response=raw_response,
        request={"messages": resolved_messages, **kwargs},
        usage=raw_response.get("usage") if isinstance(raw_response.get("usage"), dict) else None,
    )


def _complete_ark(
    *,
    messages: list[dict[str, Any]] | None,
    prompt: str | None,
    model: str | None,
    **kwargs: Any,
) -> ChatCompletionResult:
    settings = get_settings()
    resolved_model = model or settings.ark_response_model
    input_payload: str | list[dict[str, Any]]
    if messages is not None:
        input_payload = messages
    elif prompt is not None:
        input_payload = prompt
    else:
        raise ValueError("messages or prompt is required")

    raw_response = create_responses(resolved_model, input_payload, **kwargs)
    text = _extract_ark_text(raw_response)
    return ChatCompletionResult(
        provider="ark",
        model=resolved_model,
        text=text,
        raw_response=raw_response,
        request={"input": input_payload, **kwargs},
        usage=raw_response.get("usage") if isinstance(raw_response.get("usage"), dict) else None,
    )


def _messages_or_prompt(
    messages: list[dict[str, Any]] | None,
    prompt: str | None,
) -> list[dict[str, Any]]:
    if messages is not None:
        return messages
    if prompt is not None:
        return [{"role": "user", "content": prompt}]
    raise ValueError("messages or prompt is required")


def _extract_deepseek_text(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message", {})
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    return ""


def _extract_ark_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text

    texts: list[str] = []
    _collect_text(response.get("output"), texts)
    return "\n".join(texts).strip()


def _collect_text(value: Any, texts: list[str]) -> None:
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            texts.append(text)
        for nested in value.values():
            _collect_text(nested, texts)
    elif isinstance(value, list):
        for item in value:
            _collect_text(item, texts)
