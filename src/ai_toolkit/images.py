"""High-level provider-agnostic image generation helpers."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ai_toolkit.ark import create_image_generation
from ai_toolkit.config import get_settings
from ai_toolkit.gemini import generate_content
from ai_toolkit.media import upload_public_url
from ai_toolkit.types import AIToolkitError
from ai_toolkit.types import GeneratedImage
from ai_toolkit.types import ImageGenerationResult


def generate(
    *,
    provider: str,
    prompt: str,
    references: list[str | Path] | None = None,
    output_path: str | Path | None = None,
    model: str | None = None,
    size: str | None = None,
    image_size: str | None = None,
    aspect_ratio: str | None = None,
    **kwargs: Any,
) -> ImageGenerationResult:
    """Generate an image with a normalized SDK interface.

    Local reference images are handled by the SDK:
    - ARK/Doubao: upload local files first and pass public URLs.
    - Gemini: embed local files as inline base64 image parts.
    """
    normalized_provider = normalize_provider(provider)
    if normalized_provider == "ark":
        result = _generate_ark(
            prompt=prompt,
            references=references or [],
            output_path=output_path,
            model=model,
            size=size,
            **kwargs,
        )
    elif normalized_provider == "gemini":
        result = _generate_gemini(
            prompt=prompt,
            references=references or [],
            output_path=output_path,
            model=model,
            image_size=image_size or size,
            aspect_ratio=aspect_ratio,
            **kwargs,
        )
    else:
        raise AIToolkitError(f"unknown image provider: {provider}")

    return result


def normalize_provider(provider: str) -> str:
    value = provider.strip().lower()
    if value in {"ark", "doubao", "volcengine", "seedream"}:
        return "ark"
    if value in {"gemini", "google"}:
        return "gemini"
    return value


def _generate_ark(
    *,
    prompt: str,
    references: list[str | Path],
    output_path: str | Path | None,
    model: str | None,
    size: str | None,
    **kwargs: Any,
) -> ImageGenerationResult:
    settings = get_settings()
    resolved_model = model or settings.ark_image_model
    payload: dict[str, Any] = {
        "size": size or settings.ark_image_size,
        "response_format": "url",
        "sequential_image_generation": "disabled",
        "stream": False,
        "watermark": False,
    }
    payload.update(kwargs)

    image_urls = [_reference_to_url(ref) for ref in references]
    if image_urls:
        payload["image"] = image_urls

    raw_response = create_image_generation(resolved_model, prompt, **payload)
    images = _extract_ark_images(raw_response)
    result = ImageGenerationResult(
        provider="ark",
        model=resolved_model,
        prompt=prompt,
        images=images,
        raw_response=raw_response,
        request={"prompt": prompt, **payload},
    )
    if output_path is not None:
        result.save_first(output_path)
    return result


def _generate_gemini(
    *,
    prompt: str,
    references: list[str | Path],
    output_path: str | Path | None,
    model: str | None,
    image_size: str | None,
    aspect_ratio: str | None,
    **kwargs: Any,
) -> ImageGenerationResult:
    settings = get_settings()
    resolved_model = model or settings.gemini_image_model
    parts: list[dict[str, Any]] = [{"text": prompt}]
    parts.extend(_gemini_image_part(ref) for ref in references)

    generation_config = dict(kwargs.pop("generationConfig", {}) or {})
    generation_config.setdefault("responseModalities", ["TEXT", "IMAGE"])
    image_config = dict(generation_config.get("imageConfig", {}) or {})
    if image_size or settings.gemini_image_size:
        image_config.setdefault("imageSize", image_size or settings.gemini_image_size)
    if aspect_ratio:
        image_config.setdefault("aspectRatio", aspect_ratio)
    if image_config:
        generation_config["imageConfig"] = image_config

    request_kwargs = {"generationConfig": generation_config, **kwargs}
    contents = [{"role": "user", "parts": parts}]
    raw_response = generate_content(resolved_model, contents, **request_kwargs)
    images, text = _extract_gemini_images_and_text(raw_response)
    result = ImageGenerationResult(
        provider="gemini",
        model=resolved_model,
        prompt=prompt,
        images=images,
        text=text,
        raw_response=raw_response,
        request={"contents": contents, **request_kwargs},
    )
    if output_path is not None:
        result.save_first(output_path)
    return result


def _reference_to_url(reference: str | Path) -> str:
    value = str(reference)
    if _is_http_url(value):
        return value
    return upload_public_url(value)


def _gemini_image_part(reference: str | Path) -> dict[str, Any]:
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


def _extract_ark_images(response: dict[str, Any]) -> list[GeneratedImage]:
    images: list[GeneratedImage] = []
    for item in response.get("data", []) or []:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        b64_json = item.get("b64_json")
        if isinstance(url, str) or isinstance(b64_json, str):
            images.append(GeneratedImage(url=url if isinstance(url, str) else None,
                                         b64_json=b64_json if isinstance(b64_json, str) else None))
    if not images:
        raise AIToolkitError("ARK image generation returned no images")
    return images


def _extract_gemini_images_and_text(response: dict[str, Any]) -> tuple[list[GeneratedImage], str]:
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
                images.append(GeneratedImage(
                    b64_json=inline_data["data"],
                    mime_type=str(inline_data.get("mimeType") or inline_data.get("mime_type") or "image/png"),
                ))
    if not images:
        raise AIToolkitError("Gemini image generation returned no inline images")
    return images, "\n".join(texts).strip()


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _guess_mime_type(value: str) -> str:
    return mimetypes.guess_type(value)[0] or "image/png"
