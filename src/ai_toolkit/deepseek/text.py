"""High-level DeepSeek text helpers."""

from __future__ import annotations

from typing import Any

from ai_toolkit._structured import parse_json, validate_against_schema
from ai_toolkit.config import get_settings
from ai_toolkit.deepseek.chat import create_chat_completion
from ai_toolkit.types import ChatCompletionResult

MODEL_ALIASES = {
    "v4-flash": "deepseek-v4-flash",
    "deepseek-v4-flash": "deepseek-v4-flash",
    "v4-pro": "deepseek-v4-pro",
    "deepseek-v4-pro": "deepseek-v4-pro",
}


def complete(
    *,
    messages: list[dict[str, Any]] | None = None,
    prompt: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> ChatCompletionResult:
    resolved_model = resolve_model(model)
    resolved_messages = _messages_or_prompt(messages, prompt)
    request_kwargs = _drop_none_values(kwargs)
    raw_response = create_chat_completion(resolved_model, resolved_messages, **request_kwargs)
    message = _first_message(raw_response)
    text = message.get("content") if isinstance(message.get("content"), str) else ""
    reasoning = (
        message.get("reasoning_content")
        if isinstance(message.get("reasoning_content"), str)
        else ""
    )
    return ChatCompletionResult(
        provider="deepseek",
        model=resolved_model,
        text=text,
        reasoning_text=reasoning,
        raw_response=raw_response,
        request={"messages": resolved_messages, **request_kwargs},
        usage=raw_response.get("usage") if isinstance(raw_response.get("usage"), dict) else None,
    )


def complete_json(
    *,
    messages: list[dict[str, Any]] | None = None,
    prompt: str | None = None,
    model: str | None = None,
    schema: dict[str, Any] | None = None,
    disable_thinking: bool = True,
    **kwargs: Any,
) -> ChatCompletionResult:
    request_kwargs = _drop_none_values(kwargs)
    request_kwargs.setdefault("response_format", {"type": "json_object"})
    if disable_thinking and request_kwargs.get("thinking") is None:
        request_kwargs["thinking"] = {"type": "disabled"}

    result = complete(
        messages=messages,
        prompt=prompt,
        model=model,
        **request_kwargs,
    )
    parsed = parse_json(result.text)
    if schema is not None and parsed is not None:
        error = validate_against_schema(parsed, schema)
        if error is not None:
            result.parsed_json = None
            result.schema_error = error
            return result
    result.parsed_json = parsed
    return result


def resolve_model(model: str | None = None) -> str:
    settings = get_settings()
    if model is None or not str(model).strip():
        return settings.deepseek_chat_model
    value = str(model).strip()
    return MODEL_ALIASES.get(value, MODEL_ALIASES.get(value.lower(), value))


def _messages_or_prompt(
    messages: list[dict[str, Any]] | None,
    prompt: str | None,
) -> list[dict[str, Any]]:
    if messages is not None:
        return messages
    if prompt is not None:
        return [{"role": "user", "content": prompt}]
    raise ValueError("messages or prompt is required")


def _first_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices", [])
    if not choices:
        return {}
    first = choices[0]
    if not isinstance(first, dict):
        return {}
    message = first.get("message", {})
    return message if isinstance(message, dict) else {}


def _drop_none_values(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}
