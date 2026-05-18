"""Gemini image generation — POST /models/{model}:generateContent."""

from __future__ import annotations

from typing import Any

from ai_toolkit._transport import post_json
from ai_toolkit.config import get_settings


class GeminiImageError(RuntimeError):
    """Raised when the Gemini image generation API request fails."""


def generate_content(
    model: str,
    contents: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Submit a generateContent request and return the raw API response.

    Required:
        model:    e.g. "gemini-3.1-flash-image-preview"
        contents: e.g. [{"parts": [{"text": "a cat"}]}]

    Optional (via kwargs):
        generationConfig: dict with temperature, topP, maxOutputTokens, etc.
        safetySettings: list of safety threshold overrides
    """
    settings = get_settings()
    if not settings.gemini_api_key:
        raise GeminiImageError("GEMINI_API_KEY is not configured")

    api_url = (
        f"{settings.gemini_base_url.rstrip('/')}/models/"
        f"{model}:generateContent"
    )
    payload: dict[str, Any] = {"contents": contents}
    payload.update(kwargs)

    return post_json(
        api_url, payload,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": settings.gemini_api_key,
        },
        error_cls=GeminiImageError,
    )
