"""Gemini image generation — POST /models/{model}:generateContent."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ai_toolkit._transport import post_json
from ai_toolkit.config import get_settings
from ai_toolkit.types import AIToolkitError, GeneratedImage, ImageGenerationResult

DEFAULT_MODEL = "gemini-3.1-flash-image"
FLASH_MODEL = "gemini-3.1-flash-image"
LITE_MODEL = "gemini-3.1-flash-lite-image"
PRO_MODEL = "gemini-3-pro-image"
MAX_REFERENCE_IMAGES = 14

STANDARD_ASPECT_RATIOS = frozenset(
    {
        "1:1",
        "2:3",
        "3:2",
        "3:4",
        "4:3",
        "4:5",
        "5:4",
        "9:16",
        "16:9",
        "21:9",
    }
)
FLASH_ASPECT_RATIOS = STANDARD_ASPECT_RATIOS | {"1:4", "4:1", "1:8", "8:1"}
SUPPORTED_IMAGE_SIZES = frozenset({"512", "1K", "2K", "4K"})

MODEL_ALIASES = {
    "gemini-image": FLASH_MODEL,
    "nano-banana-2": FLASH_MODEL,
    FLASH_MODEL: FLASH_MODEL,
    "gemini-image-lite": LITE_MODEL,
    "nano-banana-2-lite": LITE_MODEL,
    LITE_MODEL: LITE_MODEL,
    "gemini-image-pro": PRO_MODEL,
    "nano-banana-pro": PRO_MODEL,
    PRO_MODEL: PRO_MODEL,
    # Deprecated IDs remain available only when callers request them explicitly.
    "gemini-3.1-flash-image-preview": "gemini-3.1-flash-image-preview",
}

MODEL_LIMITS = {
    FLASH_MODEL: {
        "image_sizes": SUPPORTED_IMAGE_SIZES,
        "aspect_ratios": FLASH_ASPECT_RATIOS,
    },
    LITE_MODEL: {
        "image_sizes": frozenset({"1K"}),
        "aspect_ratios": STANDARD_ASPECT_RATIOS,
    },
    PRO_MODEL: {
        "image_sizes": frozenset({"1K", "2K", "4K"}),
        "aspect_ratios": STANDARD_ASPECT_RATIOS,
    },
}


class GeminiImageError(RuntimeError):
    """Raised when the Gemini image generation API request fails."""


def generate(
    *,
    prompt: str,
    references: list[str | Path] | None = None,
    output_path: str | Path | None = None,
    model: str | None = None,
    image_size: str | None = None,
    aspect_ratio: str | None = None,
    **kwargs: Any,
) -> ImageGenerationResult:
    """Generate an image with Gemini image models.

    Local reference images are sent as inline base64 parts; no public URL
    staging is required for Gemini.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    settings = get_settings()
    resolved_model = resolve_model(model)
    reference_values = list(references or [])
    if len(reference_values) > MAX_REFERENCE_IMAGES:
        raise ValueError(
            f"Gemini image generation accepts at most {MAX_REFERENCE_IMAGES} references"
        )
    resolved_image_size = _validate_image_size(
        resolved_model,
        image_size or settings.gemini_image_size,
    )
    resolved_aspect_ratio = _validate_aspect_ratio(resolved_model, aspect_ratio)

    parts: list[dict[str, Any]] = [{"text": prompt}]
    parts.extend(_image_part(ref) for ref in reference_values)

    generation_config = dict(kwargs.pop("generationConfig", {}) or {})
    generation_config.setdefault("responseModalities", ["TEXT", "IMAGE"])
    image_config = dict(generation_config.get("imageConfig", {}) or {})
    image_config.setdefault("imageSize", resolved_image_size)
    if resolved_aspect_ratio is not None:
        image_config.setdefault("aspectRatio", resolved_aspect_ratio)
    if image_config:
        generation_config["imageConfig"] = image_config

    request_kwargs = {"generationConfig": generation_config, **_drop_none_values(kwargs)}
    contents = [{"role": "user", "parts": parts}]
    raw_response = generate_content(resolved_model, contents, **request_kwargs)
    images, text = _extract_images_and_text(raw_response)
    result = ImageGenerationResult(
        provider="gemini",
        model=resolved_model,
        prompt=prompt,
        images=images,
        text=text,
        raw_response=raw_response,
        request={"contents": contents, **request_kwargs},
        usage=_response_usage(raw_response),
    )
    if output_path is not None:
        result.save_first(output_path)
    return result


def generate_content(
    model: str,
    contents: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Submit a generateContent request and return the raw API response.

    Required:
        model:    e.g. "gemini-3.1-flash-image"
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


def resolve_model(model: str | None = None) -> str:
    settings = get_settings()
    value = settings.gemini_image_model if model is None else str(model).strip()
    if not value:
        value = DEFAULT_MODEL
    return MODEL_ALIASES.get(value, MODEL_ALIASES.get(value.lower(), value))


def _validate_image_size(model: str, image_size: object) -> str:
    value = str(image_size).strip()
    supported = MODEL_LIMITS.get(model, {}).get("image_sizes", SUPPORTED_IMAGE_SIZES)
    if value not in supported:
        choices = ", ".join(sorted(supported))
        raise ValueError(f"unsupported image_size {value!r} for {model}; choose: {choices}")
    return value


def _validate_aspect_ratio(model: str, aspect_ratio: object | None) -> str | None:
    if aspect_ratio is None:
        return None
    value = str(aspect_ratio).strip()
    supported = MODEL_LIMITS.get(model, {}).get("aspect_ratios", FLASH_ASPECT_RATIOS)
    if value not in supported:
        choices = ", ".join(sorted(supported))
        raise ValueError(f"unsupported aspect_ratio {value!r} for {model}; choose: {choices}")
    return value


def _image_part(reference: str | Path) -> dict[str, Any]:
    value = str(reference)
    if _is_http_url(value):
        return {"fileData": {"mimeType": _guess_mime_type(value), "fileUri": value}}

    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"reference image does not exist: {path}")
    return {
        "inlineData": {
            "mimeType": _guess_mime_type(str(path)),
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        }
    }


def _extract_images_and_text(response: dict[str, Any]) -> tuple[list[GeneratedImage], str]:
    images: list[GeneratedImage] = []
    texts: list[str] = []
    for candidate in response.get("candidates", []) or []:
        content = candidate.get("content", {}) if isinstance(candidate, dict) else {}
        for part in content.get("parts", []) or []:
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("text"), str):
                texts.append(part["text"])
            inline_data = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline_data, dict) and isinstance(inline_data.get("data"), str):
                images.append(
                    GeneratedImage(
                        b64_json=inline_data["data"],
                        mime_type=str(
                            inline_data.get("mimeType")
                            or inline_data.get("mime_type")
                            or "image/png"
                        ),
                    )
                )
    if not images:
        raise AIToolkitError("Gemini image generation returned no inline images")
    return images, "\n".join(texts).strip()


def _response_usage(response: dict[str, Any]) -> dict[str, Any] | None:
    usage = response.get("usageMetadata")
    return usage if isinstance(usage, dict) else None


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _guess_mime_type(value: str) -> str:
    return mimetypes.guess_type(value)[0] or "image/png"


def _drop_none_values(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}
