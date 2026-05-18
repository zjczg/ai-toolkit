"""ARK (豆包/火山引擎) multimodal — POST /responses."""

from __future__ import annotations

from typing import Any

from ai_toolkit._transport import post_json
from ai_toolkit.config import get_settings


class ArkMultimodalError(RuntimeError):
    """Raised when the ARK Responses API request fails."""


def create_responses(
    model: str,
    input: str | list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Submit a Responses API request and return the raw API response.

    Required:
        model: e.g. "doubao-seed-2-0-pro-260215"
        input:  plain text string *or*
                structured list like [{"role": "user", "content": [...]}]

    Optional (via kwargs):
        stream: bool
        reasoning: dict
    """
    settings = get_settings()
    if not settings.ark_api_key:
        raise ArkMultimodalError("ARK_API_KEY is not configured")

    payload: dict[str, Any] = {"model": model, "input": input}
    payload.update(kwargs)

    api_url = f"{settings.ark_base_url.rstrip('/')}/responses"
    return post_json(
        api_url, payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.ark_api_key}",
        },
        error_cls=ArkMultimodalError,
    )
