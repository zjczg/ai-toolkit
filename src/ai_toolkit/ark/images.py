"""ARK (豆包/火山引擎) image generation — POST /images/generations."""

from __future__ import annotations

from typing import Any

from ai_toolkit._transport import post_json
from ai_toolkit.config import get_settings


class ArkImageError(RuntimeError):
    """Raised when the ARK image generation API request fails."""


def create_image_generation(
    model: str,
    prompt: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Submit an image generation request and return the raw API response.

    Required:
        model:  e.g. "doubao-seedream-5-0-260128"
        prompt: image description text

    Optional (via kwargs):
        image: reference image URL (for image-to-image)
        size: e.g. "2K", "3K", "4K"
        response_format: "url" (default) or "b64_json"
        watermark: bool
        sequential_image_generation: "disabled" | "auto"
        stream: bool
        output_format: "png", "jpeg" (Seedream 5.0 lite)
    """
    settings = get_settings()
    if not settings.ark_api_key:
        raise ArkImageError("ARK_API_KEY is not configured")

    payload: dict[str, Any] = {"model": model, "prompt": prompt}
    payload.update(kwargs)

    api_url = f"{settings.ark_base_url.rstrip('/')}/images/generations"
    return post_json(
        api_url, payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.ark_api_key}",
        },
        error_cls=ArkImageError,
    )


def create_seedream_4_5_image_generation(
    prompt: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Submit a Seedream 4.5 image generation request.

    Defaults mirror the ARK curl-style image generation payload:
    URL response, 2K size, sequential generation disabled, no streaming,
    and watermark enabled. Pass any keyword to override these defaults.
    """
    payload_kwargs: dict[str, Any] = {
        "sequential_image_generation": "disabled",
        "response_format": "url",
        "size": "2K",
        "stream": False,
        "watermark": True,
    }
    payload_kwargs.update(kwargs)
    model = str(payload_kwargs.pop("model", "doubao-seedream-4-5-251128"))
    return create_image_generation(model, prompt, **payload_kwargs)


def create_seedream_5_0_lite_image_generation(
    prompt: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Submit a Seedream 5.0 lite image generation request.

    Defaults use the current ARK image generation API shape:
    URL response, 2K size, PNG output, sequential generation disabled,
    no streaming, and no watermark. Pass any keyword to override them.
    """
    payload_kwargs: dict[str, Any] = {
        "sequential_image_generation": "disabled",
        "response_format": "url",
        "size": "2K",
        "output_format": "png",
        "stream": False,
        "watermark": False,
    }
    payload_kwargs.update(kwargs)
    model = str(payload_kwargs.pop("model", "doubao-seedream-5-0-260128"))
    return create_image_generation(model, prompt, **payload_kwargs)
