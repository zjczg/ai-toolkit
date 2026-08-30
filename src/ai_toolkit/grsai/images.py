"""GRS.AI Nano Banana image generation through its Gemini-compatible API."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any
from urllib import parse

from ai_toolkit._transport import post_json
from ai_toolkit.config import get_settings
from ai_toolkit.types import AIToolkitError, GeneratedImage, ImageGenerationResult

DEFAULT_MODEL = "nano-banana-2"
MAX_REFERENCE_IMAGES = 6
SUPPORTED_IMAGE_SIZES = frozenset({"1K", "2K", "4K"})
SUPPORTED_ASPECT_RATIOS = frozenset(
    {
        "auto",
        "1:1",
        "1:4",
        "4:1",
        "1:8",
        "8:1",
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
MODEL_ALIASES = {
    "grsai-image": DEFAULT_MODEL,
    "grsai-nano-banana-2": DEFAULT_MODEL,
    DEFAULT_MODEL: DEFAULT_MODEL,
}


class GRSAIImageError(RuntimeError):
    """Raised when a GRS.AI image generation request fails."""


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
    """Generate one image with GRS.AI ``nano-banana-2``.

    Local references are encoded as Gemini-compatible inline Base64 parts.
    This provider is intentionally separate from the official Gemini client,
    so credentials, provenance, and provider selection remain explicit.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    settings = get_settings()
    resolved_model = resolve_model(model)
    reference_values = list(references or [])
    if len(reference_values) > MAX_REFERENCE_IMAGES:
        raise ValueError(
            f"GRS.AI image generation accepts at most {MAX_REFERENCE_IMAGES} references"
        )
    resolved_image_size = _resolve_image_size(
        image_size or settings.grsai_image_size
    )
    resolved_aspect_ratio = _resolve_aspect_ratio(aspect_ratio)

    parts: list[dict[str, Any]] = [{"text": prompt.strip()}]
    parts.extend(_image_part(reference) for reference in reference_values)
    generation_config = dict(kwargs.pop("generationConfig", {}) or {})
    generation_config.setdefault("responseModalities", ["TEXT", "IMAGE"])
    image_config = dict(generation_config.get("imageConfig", {}) or {})
    if resolved_image_size is not None:
        image_config.setdefault("imageSize", resolved_image_size)
    if resolved_aspect_ratio is not None:
        image_config.setdefault("aspectRatio", resolved_aspect_ratio)
    if image_config:
        generation_config["imageConfig"] = image_config

    request_kwargs = {
        "generationConfig": generation_config,
        **_drop_none_values(kwargs),
    }
    contents = [{"role": "user", "parts": parts}]
    raw_response = generate_content(resolved_model, contents, **request_kwargs)
    images, text = _extract_images_and_text(raw_response)
    result = ImageGenerationResult(
        provider="grsai",
        model=resolved_model,
        prompt=prompt.strip(),
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
    """Call the GRS.AI Gemini-compatible ``generateContent`` endpoint."""

    settings = get_settings()
    if not settings.grsai_api_key:
        raise GRSAIImageError("GRSAI_API_KEY is not configured")
    encoded_model = parse.quote(model, safe="-._")
    payload = {"contents": contents, **kwargs}
    return post_json(
        (
            f"{settings.grsai_base_url.rstrip('/')}"
            f"/v1beta/models/{encoded_model}:generateContent"
        ),
        payload,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": settings.grsai_api_key,
        },
        error_cls=GRSAIImageError,
    )


def resolve_model(model: str | None = None) -> str:
    """Resolve the single supported GRS.AI Flash image route."""

    settings = get_settings()
    value = settings.grsai_image_model if model is None else str(model)
    cleaned = value.strip()
    if not cleaned:
        cleaned = settings.grsai_image_model.strip()
    resolved = MODEL_ALIASES.get(cleaned, MODEL_ALIASES.get(cleaned.lower()))
    if resolved is None:
        raise ValueError(
            f"unsupported GRS.AI image model: {model!r}; use 'grsai-image'"
        )
    return resolved


def _resolve_image_size(image_size: object) -> str | None:
    value = str(image_size).strip()
    if value.lower() == "auto":
        return None
    normalized = value.upper()
    if normalized not in SUPPORTED_IMAGE_SIZES:
        choices = ", ".join(sorted(SUPPORTED_IMAGE_SIZES))
        raise ValueError(
            f"unsupported GRS.AI image size: {image_size!r}; use auto, {choices}"
        )
    return normalized


def _resolve_aspect_ratio(aspect_ratio: object | None) -> str | None:
    if aspect_ratio is None:
        return None
    value = str(aspect_ratio).strip().lower()
    if value not in SUPPORTED_ASPECT_RATIOS:
        choices = ", ".join(sorted(SUPPORTED_ASPECT_RATIOS))
        raise ValueError(
            f"unsupported GRS.AI aspect ratio: {aspect_ratio!r}; use {choices}"
        )
    return value


def _image_part(reference: str | Path) -> dict[str, Any]:
    value = str(reference)
    if value.startswith("data:image/"):
        header, separator, data = value.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("GRS.AI data URL references must use Base64 encoding")
        mime_type = header[5:].split(";", 1)[0]
        return {"inlineData": {"mimeType": mime_type, "data": data}}

    source = Path(value).expanduser().resolve()
    if not source.is_file():
        raise ValueError(
            "GRS.AI Gemini-compatible references must be local files or Base64 data URLs"
        )
    if source.stat().st_size > 10 * 1024 * 1024:
        raise ValueError(f"GRS.AI reference image exceeds 10 MB: {source}")
    mime_type = mimetypes.guess_type(source.name)[0]
    if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError(f"unsupported GRS.AI reference image format: {source}")
    data = base64.b64encode(source.read_bytes()).decode("ascii")
    return {"inlineData": {"mimeType": mime_type, "data": data}}


def _extract_images_and_text(
    response: dict[str, Any],
) -> tuple[list[GeneratedImage], str]:
    images: list[GeneratedImage] = []
    texts: list[str] = []
    for candidate in response.get("candidates", []) or []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content", {})
        if not isinstance(content, dict):
            continue
        for part in content.get("parts", []) or []:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
            inline = part.get("inlineData") or part.get("inline_data")
            if not isinstance(inline, dict):
                continue
            data = inline.get("data")
            if isinstance(data, str):
                images.append(
                    GeneratedImage(
                        b64_json=data,
                        mime_type=str(
                            inline.get("mimeType")
                            or inline.get("mime_type")
                            or "image/png"
                        ),
                    )
                )
    if not images:
        error = response.get("error") if isinstance(response, dict) else None
        detail = f": {error}" if error else ""
        raise AIToolkitError(f"GRS.AI image generation returned no images{detail}")
    return images, "\n".join(texts).strip()


def _response_usage(response: dict[str, Any]) -> dict[str, Any] | None:
    usage = response.get("usageMetadata") or response.get("usage_metadata")
    return usage if isinstance(usage, dict) else None


def _drop_none_values(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}
