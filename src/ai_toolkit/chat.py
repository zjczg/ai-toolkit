"""High-level provider-agnostic chat/text generation helpers."""

from __future__ import annotations

from typing import Any

from ai_toolkit.ark import create_responses
from ai_toolkit.config import get_settings
from ai_toolkit.deepseek import create_chat_completion
from ai_toolkit.types import AIToolkitError
from ai_toolkit.types import ChatCompletionResult


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


def normalize_provider(provider: str) -> str:
    value = provider.strip().lower()
    if value in {"ark", "doubao", "volcengine"}:
        return "ark"
    if value == "deepseek":
        return "deepseek"
    return value


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
