"""High-level ARK text and multimodal text helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ai_toolkit._structured import parse_json, validate_against_schema
from ai_toolkit.ark.responses import create_responses
from ai_toolkit.config import get_settings
from ai_toolkit.media import upload_public_url
from ai_toolkit.structured import (
    JsonOutputModel,
    output_schema,
    validate_output_model,
)
from ai_toolkit.types import ChatCompletionResult

MODEL_ALIASES = {
    "doubao-pro": "doubao-seed-2-0-pro-260215",
    "doubao-seed-pro": "doubao-seed-2-0-pro-260215",
    "doubao-seed-2-pro": "doubao-seed-2-0-pro-260215",
    "doubao-seed-2.0-pro": "doubao-seed-2-0-pro-260215",
    "doubao-seed-2-0-pro-260215": "doubao-seed-2-0-pro-260215",
}


def complete(
    *,
    messages: list[dict[str, Any]] | None = None,
    prompt: str | None = None,
    images: list[str | Path] | tuple[str | Path, ...] = (),
    model: str | None = None,
    **kwargs: Any,
) -> ChatCompletionResult:
    """Generate text with ARK Responses API.

    ``images`` is a convenience for a single multimodal user prompt. When
    ``messages`` is provided, pass already-shaped ARK Responses input.
    """
    resolved_model = resolve_model(model)
    input_payload = _input_payload(messages=messages, prompt=prompt, images=images)
    request_kwargs = _drop_none_values(kwargs)
    raw_response = create_responses(resolved_model, input_payload, **request_kwargs)
    text = _extract_text(raw_response)
    return ChatCompletionResult(
        provider="ark",
        model=resolved_model,
        text=text,
        raw_response=raw_response,
        request={"input": input_payload, **request_kwargs},
        usage=raw_response.get("usage") if isinstance(raw_response.get("usage"), dict) else None,
    )


def complete_json(
    *,
    messages: list[dict[str, Any]] | None = None,
    prompt: str | None = None,
    images: list[str | Path] | tuple[str | Path, ...] = (),
    model: str | None = None,
    schema: dict[str, Any] | None = None,
    output_type: type[JsonOutputModel] | None = None,
    schema_name: str = "structured_response",
    strict: bool = True,
    disable_thinking: bool = True,
    **kwargs: Any,
) -> ChatCompletionResult:
    """Generate structured JSON.

    JSON extraction defaults to ``thinking={"type": "disabled"}``; enabling
    thinking on ARK may mix reasoning text into ``result.text``.
    ``output_type`` enables native JSON Schema output plus strict local
    Pydantic validation. The legacy ``schema`` path remains non-raising.
    """
    resolved_schema = output_schema(
        schema=schema,
        output_type=output_type,
    )
    request_kwargs = _drop_none_values(kwargs)
    request_kwargs.setdefault(
        "text",
        _text_format(
            schema=resolved_schema,
            schema_name=schema_name,
            strict=strict,
        ),
    )
    if disable_thinking and request_kwargs.get("thinking") is None:
        request_kwargs["thinking"] = {"type": "disabled"}

    result = complete(
        messages=messages,
        prompt=prompt,
        images=images,
        model=model,
        **request_kwargs,
    )
    if output_type is not None:
        return validate_output_model(result, output_type)

    parsed = parse_json(result.text)
    if schema is not None and parsed is not None:
        error = validate_against_schema(parsed, schema)
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
    content: list[dict[str, str]] = [{"type": "input_text", "text": text}]
    content.extend(
        {"type": "input_image", "image_url": _image_reference_to_url(image)}
        for image in images
    )
    return {"role": "user", "content": content}


def resolve_model(model: str | None = None) -> str:
    settings = get_settings()
    if model is None or not str(model).strip():
        return settings.ark_response_model
    value = str(model).strip()
    return MODEL_ALIASES.get(value, MODEL_ALIASES.get(value.lower(), value))


def _input_payload(
    *,
    messages: list[dict[str, Any]] | None,
    prompt: str | None,
    images: list[str | Path] | tuple[str | Path, ...],
) -> str | list[dict[str, Any]]:
    if messages is not None:
        return messages
    if prompt is None:
        raise ValueError("messages or prompt is required")
    if images:
        return [multimodal_user_message(text=prompt, images=images)]
    return prompt


def _text_format(
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


def _extract_text(response: dict[str, Any]) -> str:
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


def _image_reference_to_url(reference: str | Path) -> str:
    value = str(reference)
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return value
    return upload_public_url(value)


def _drop_none_values(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}
