"""万相 / 通义 (阿里百炼 DashScope) image generation.

Synchronous multimodal-generation endpoint:
    POST /api/v1/services/aigc/multimodal-generation/generation

Covers the modern Wan image models (wan2.7-image, wan2.7-image-pro,
wan2.6-image, wan2.6-t2i) and qwen-image models, which all accept the same
``input.messages`` shape and return image URLs synchronously — no task
polling. Older async-only models (wanx2.1-t2i, wan2.2-t2i-*) use a different
endpoint and are intentionally out of scope here.
"""

from __future__ import annotations

from typing import Any

from ai_toolkit._transport import post_json
from ai_toolkit.config import get_settings

_GENERATION_PATH = "/api/v1/services/aigc/multimodal-generation/generation"


class DashScopeImageError(RuntimeError):
    """Raised when the DashScope image generation API request fails."""


def create_image_generation(
    model: str,
    prompt: str,
    *,
    image_urls: list[str] | None = None,
    **parameters: Any,
) -> dict[str, Any]:
    """Submit a synchronous Wan/Qwen image generation request.

    Required:
        model:  e.g. "wan2.7-image", "wan2.7-image-pro", "qwen-image"
        prompt: image description text

    Optional:
        image_urls: public reference image URLs. A single URL drives
            image-to-image / editing; multiple URLs drive multi-image
            reference generation.
        parameters (via kwargs): forwarded into the request ``parameters``
            block, e.g. size="2K", n=1, watermark=False, prompt_extend=True.

    Returns the raw API response. Generated image URLs live in
    ``output.choices[0].message.content[].image`` and expire after 24 hours.
    """
    settings = get_settings()
    if not settings.dashscope_api_key:
        raise DashScopeImageError("DASHSCOPE_API_KEY is not configured")

    content: list[dict[str, Any]] = [{"text": prompt}]
    for url in image_urls or []:
        content.append({"image": url})

    payload: dict[str, Any] = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": content}]},
    }
    if parameters:
        payload["parameters"] = parameters

    api_url = f"{settings.dashscope_base_url.rstrip('/')}{_GENERATION_PATH}"
    return post_json(
        api_url,
        payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.dashscope_api_key}",
        },
        error_cls=DashScopeImageError,
    )
