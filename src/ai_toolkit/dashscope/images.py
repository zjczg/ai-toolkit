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

import base64
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ai_toolkit._transport import post_json
from ai_toolkit.config import get_settings
from ai_toolkit.media import upload_public_url
from ai_toolkit.types import AIToolkitError, GeneratedImage, ImageGenerationResult

_GENERATION_PATH = "/api/v1/services/aigc/multimodal-generation/generation"
MAX_REFERENCE_IMAGES = 9

MODEL_ALIASES = {
    "wan2.7": "wan2.7-image",
    "wan2.7-image": "wan2.7-image",
    "wan2.7-pro": "wan2.7-image-pro",
    "wan2.7-image-pro": "wan2.7-image-pro",
}


class DashScopeImageError(RuntimeError):
    """Raised when the DashScope image generation API request fails."""


def generate(
    *,
    prompt: str,
    references: list[str | Path] | None = None,
    output_path: str | Path | None = None,
    model: str | None = None,
    size: str | None = None,
    **kwargs: Any,
) -> ImageGenerationResult:
    """Generate an image with DashScope Wan image models."""
    settings = get_settings()
    resolved_model = resolve_model(model)
    refs = references or []
    if len(refs) > MAX_REFERENCE_IMAGES:
        raise AIToolkitError(
            f"too many reference images for {resolved_model}: got {len(refs)}, max {MAX_REFERENCE_IMAGES}"
        )
    parameters = _drop_none_values(kwargs)
    parameters.setdefault("size", size or settings.dashscope_image_size)
    parameters.setdefault("watermark", False)
    image_urls = [_reference_to_url(ref) for ref in refs]

    raw_response = create_image_generation(
        resolved_model,
        prompt,
        image_urls=image_urls or None,
        **parameters,
    )
    result = ImageGenerationResult(
        provider="dashscope",
        model=resolved_model,
        prompt=prompt,
        images=_extract_images(raw_response),
        raw_response=raw_response,
        request={"prompt": prompt, "image_urls": image_urls, "parameters": parameters},
        usage=_response_usage(raw_response),
    )
    if output_path is not None:
        result.save_first(output_path)
    return result


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


def resolve_model(model: str | None = None) -> str:
    settings = get_settings()
    if model is None or not str(model).strip():
        return settings.dashscope_image_model
    value = str(model).strip()
    return MODEL_ALIASES.get(value, MODEL_ALIASES.get(value.lower(), value))


def _extract_images(response: dict[str, Any]) -> list[GeneratedImage]:
    images: list[GeneratedImage] = []
    output = response.get("output", {}) if isinstance(response, dict) else {}
    for choice in output.get("choices", []) or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message", {}) if isinstance(choice.get("message"), dict) else {}
        for part in message.get("content", []) or []:
            if isinstance(part, dict) and isinstance(part.get("image"), str):
                images.append(GeneratedImage(url=part["image"]))
    for item in output.get("results", []) or []:
        if isinstance(item, dict) and isinstance(item.get("url"), str):
            images.append(GeneratedImage(url=item["url"]))
    if not images:
        raise AIToolkitError("DashScope image generation returned no images")
    return images


def _reference_to_url(reference: str | Path) -> str:
    value = str(reference)
    if _is_remote_reference(value) or value.startswith("data:"):
        return value
    local_path = Path(value).expanduser()
    if local_path.is_file():
        return _local_image_data_url(local_path)
    return upload_public_url(value)


def _is_remote_reference(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https", "oss"} and bool(parsed.netloc)


def _local_image_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError(f"unsupported DashScope reference image format: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _response_usage(response: dict[str, Any]) -> dict[str, Any] | None:
    usage = response.get("usage")
    return usage if isinstance(usage, dict) else None


def _drop_none_values(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}
