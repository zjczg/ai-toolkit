"""ARK (豆包/火山引擎) image generation — POST /images/generations."""

from __future__ import annotations

import base64
import mimetypes
import re
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ai_toolkit._transport import post_json
from ai_toolkit.config import get_settings
from ai_toolkit.types import AIToolkitError, GeneratedImage, ImageGenerationResult

MODEL_ALIASES = {
    "seedream-5": "doubao-seedream-5-0-260128",
    "seedream-5.0": "doubao-seedream-5-0-260128",
    "doubao-5.0": "doubao-seedream-5-0-260128",
    "doubao-seedream-5-0-260128": "doubao-seedream-5-0-260128",
    "seedream-5-pro": "doubao-seedream-5-0-pro-260628",
    "seedream-5.0-pro": "doubao-seedream-5-0-pro-260628",
    "doubao-5.0-pro": "doubao-seedream-5-0-pro-260628",
    "doubao-seedream-5-0-pro-260628": "doubao-seedream-5-0-pro-260628",
    "seedream-4.5": "doubao-seedream-4-5-251128",
    "doubao-4.5": "doubao-seedream-4-5-251128",
    "doubao-seedream-4-5-251128": "doubao-seedream-4-5-251128",
}

LEGACY_MODEL_ALIASES = {
    "seedream-5-lite": "doubao-seedream-5-0-260128",
    "seedream-5.0-lite": "doubao-seedream-5-0-260128",
    "doubao-5.0-lite": "doubao-seedream-5-0-260128",
}

SEEDREAM_5_MODEL = "doubao-seedream-5-0-260128"
SEEDREAM_5_PRO_MODEL = "doubao-seedream-5-0-pro-260628"
SEEDREAM_4_5_MODEL = "doubao-seedream-4-5-251128"
SEEDREAM_5_SIZE_VALUES = frozenset({"2K", "3K", "4K"})
SEEDREAM_5_PRO_SIZE_VALUES = frozenset({"1K", "2K"})


class ArkImageError(RuntimeError):
    """Raised when the ARK image generation API request fails."""


def generate(
    *,
    prompt: str,
    references: list[str | Path] | None = None,
    output_path: str | Path | None = None,
    model: str | None = None,
    size: str | None = None,
    output_format: str | None = None,
    **kwargs: Any,
) -> ImageGenerationResult:
    """Generate an image with ARK Seedream models.

    Supported model aliases: ``seedream-5``, ``seedream-5-pro``, and
    ``seedream-4.5``.
    Local reference images are sent as inline Base64 data URLs, so no public
    upload is required.
    """
    settings = get_settings()
    resolved_model = resolve_model(model)
    request_kwargs = _drop_none_values(kwargs)
    resolved_size = _normalize_size(resolved_model, size or settings.ark_image_size)
    request_kwargs.setdefault("size", resolved_size)
    request_kwargs.setdefault("response_format", "url")
    request_kwargs.setdefault("watermark", False)

    if resolved_model == SEEDREAM_5_PRO_MODEL:
        unsupported = {
            name
            for name in (
                "sequential_image_generation",
                "sequential_image_generation_options",
                "stream",
            )
            if name in request_kwargs
        }
        if unsupported:
            joined = ", ".join(sorted(unsupported))
            raise AIToolkitError(f"{resolved_model} does not support: {joined}")
    else:
        request_kwargs.setdefault("sequential_image_generation", "disabled")
        request_kwargs.setdefault("stream", False)

    refs = references or []
    if resolved_model in {SEEDREAM_5_MODEL, SEEDREAM_5_PRO_MODEL}:
        resolved_output_format = _normalize_output_format(
            output_format or _infer_output_format(output_path) or "png"
        )
        request_kwargs.setdefault("output_format", resolved_output_format)
        max_refs = 10 if resolved_model == SEEDREAM_5_PRO_MODEL else 14
        _enforce_reference_limit(refs, max_refs=max_refs, model=resolved_model)
    elif resolved_model == SEEDREAM_4_5_MODEL:
        if output_format not in {None, "", "jpeg", "jpg"}:
            raise AIToolkitError("seedream-4.5 does not support output_format override")

    image_urls = [_reference_to_url(ref) for ref in refs]
    if image_urls:
        request_kwargs["image"] = image_urls

    raw_response = create_image_generation(resolved_model, prompt, **request_kwargs)
    result = ImageGenerationResult(
        provider="ark",
        model=resolved_model,
        prompt=prompt,
        images=_extract_images(raw_response),
        raw_response=raw_response,
        request={"prompt": prompt, **request_kwargs},
        usage=_response_usage(raw_response),
    )
    if output_path is not None:
        result.save_first(output_path)
    return result


def resolve_model(model: str | None = None) -> str:
    settings = get_settings()
    value = settings.ark_image_model if model is None or not str(model).strip() else str(model).strip()
    if value.lower() in LEGACY_MODEL_ALIASES:
        warnings.warn(
            f"{value!r} is a legacy misnamed alias for Seedream 5.0; use 'seedream-5'",
            DeprecationWarning,
            stacklevel=2,
        )
        return LEGACY_MODEL_ALIASES[value.lower()]
    return MODEL_ALIASES.get(value, MODEL_ALIASES.get(value.lower(), value))


def _normalize_size(model: str, size: str) -> str:
    normalized = str(size).strip().upper()
    supported: frozenset[str] | None = None
    if model == SEEDREAM_5_MODEL:
        supported = SEEDREAM_5_SIZE_VALUES
    elif model == SEEDREAM_5_PRO_MODEL:
        supported = SEEDREAM_5_PRO_SIZE_VALUES
    if supported is None:
        return str(size).strip()
    if normalized in supported:
        return normalized
    if re.fullmatch(r"[1-9]\d*[xX][1-9]\d*", normalized):
        return normalized.replace("X", "x")
    choices = ", ".join(sorted(supported))
    raise AIToolkitError(
        f"unsupported size for {model}: {size!r}; use {choices} or '<width>x<height>'"
    )


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


def _extract_images(response: dict[str, Any]) -> list[GeneratedImage]:
    images: list[GeneratedImage] = []
    for item in response.get("data", []) or []:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        b64_json = item.get("b64_json")
        if isinstance(url, str) or isinstance(b64_json, str):
            images.append(
                GeneratedImage(
                    url=url if isinstance(url, str) else None,
                    b64_json=b64_json if isinstance(b64_json, str) else None,
                )
            )
    if not images:
        raise AIToolkitError("ARK image generation returned no images")
    return images


def _reference_to_url(reference: str | Path) -> str:
    value = str(reference)
    if _is_http_url(value):
        return value
    source = Path(value).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"reference image does not exist: {source}")
    mime_type = mimetypes.guess_type(source.name)[0] or "image/png"
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _infer_output_format(output_path: str | Path | None) -> str | None:
    if output_path is None:
        return None
    suffix = Path(str(output_path)).suffix.lower().lstrip(".")
    if suffix == "jpg":
        return "jpeg"
    if suffix in {"png", "jpeg"}:
        return suffix
    return None


def _normalize_output_format(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "jpg":
        return "jpeg"
    if normalized not in {"png", "jpeg"}:
        raise AIToolkitError("output_format must be 'png' or 'jpeg'")
    return normalized


def _enforce_reference_limit(references: list[str | Path], *, max_refs: int, model: str) -> None:
    if len(references) > max_refs:
        raise AIToolkitError(
            f"too many reference images for {model}: got {len(references)}, max {max_refs}"
        )


def _response_usage(response: dict[str, Any]) -> dict[str, Any] | None:
    usage = response.get("usage")
    return usage if isinstance(usage, dict) else None


def _drop_none_values(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


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


def create_seedream_5_0_image_generation(
    prompt: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Submit a Seedream 5.0 image generation request.

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


def create_seedream_5_0_pro_image_generation(
    prompt: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Submit one Seedream 5.0 Pro image generation request.

    Pro supports one output image at 1K or 2K. It does not support streaming
    or sequential image generation.
    """
    payload_kwargs: dict[str, Any] = {
        "response_format": "url",
        "size": "2K",
        "output_format": "png",
        "watermark": False,
    }
    payload_kwargs.update(kwargs)
    model = str(payload_kwargs.pop("model", SEEDREAM_5_PRO_MODEL))
    return create_image_generation(model, prompt, **payload_kwargs)
