"""ARK (豆包/火山引擎) embeddings — POST /embeddings/multimodal."""

from __future__ import annotations

from typing import Any

from ai_toolkit._transport import post_json
from ai_toolkit.config import get_settings


class ArkEmbeddingError(RuntimeError):
    """Raised when the ARK Embeddings API request fails."""


def create_multimodal_embedding(
    model: str,
    input: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Submit a multimodal embedding request and return the raw API response.

    The embedding vector is available at response["data"]["embedding"].

    Required:
        model: e.g. "doubao-embedding-vision-251215"
        input: structured list like:
            [
                {"type": "text", "text": "天很蓝，海很深"},
                {"type": "image_url", "image_url": {"url": "https://.../view.jpeg"}},
            ]

    Optional:
        Any provider-supported payload field via kwargs.
    """
    settings = get_settings()
    if not settings.ark_api_key:
        raise ArkEmbeddingError("ARK_API_KEY is not configured")

    payload: dict[str, Any] = {"model": model, "input": input}
    payload.update(kwargs)

    api_url = f"{settings.ark_base_url.rstrip('/')}/embeddings/multimodal"
    return post_json(
        api_url,
        payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.ark_api_key}",
        },
        error_cls=ArkEmbeddingError,
    )
