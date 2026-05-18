"""DeepSeek chat — POST /chat/completions (OpenAI-compatible)."""

from __future__ import annotations

from typing import Any

from ai_toolkit._transport import post_json
from ai_toolkit.config import get_settings


class DeepSeekError(RuntimeError):
    """Raised when a DeepSeek API request fails."""


def create_chat_completion(
    model: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Submit a chat completion request and return the raw API response.

    Required:
        model:    e.g. "deepseek-v4-flash" or "deepseek-v4-pro"
        messages: [{"role": "user", "content": "hello"}]

    Optional (via kwargs):
        temperature: float ≤ 2
        max_tokens: int
        thinking: {"type": "enabled", "reasoning_effort": "high"|"max"}
        stream: bool
        tools: list of function definitions
        response_format: {"type": "json_object"}
        top_p: float ≤ 1
        frequency_penalty, presence_penalty: float ≥ -2 and ≤ 2
        stop: str | list[str] (up to 16)
    """
    settings = get_settings()
    if not settings.deepseek_api_key:
        raise DeepSeekError("DEEPSEEK_API_KEY is not configured")

    payload: dict[str, Any] = {"model": model, "messages": messages}
    payload.update(kwargs)

    api_url = f"{settings.deepseek_base_url.rstrip('/')}/chat/completions"
    return post_json(
        api_url, payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.deepseek_api_key}",
        },
        error_cls=DeepSeekError,
    )
