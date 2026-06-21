"""High-level provider-agnostic image generation helpers."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ai_toolkit import capabilities
from ai_toolkit.ark import create_image_generation
from ai_toolkit.config import get_settings
from ai_toolkit.dashscope import create_image_generation as create_dashscope_image_generation
from ai_toolkit.gemini import generate_content
from ai_toolkit.media import upload_public_url
from ai_toolkit.types import AIToolkitError, GeneratedImage, ImageGenerationResult


def generate(
    *,
    provider: str | None = None,
    prompt: str,
    references: list[str | Path] | None = None,
    output_path: str | Path | None = None,
    model: str | None = None,
    size: str | None = None,
    image_size: str | None = None,
    aspect_ratio: str | None = None,
    path: str | None = None,
    **kwargs: Any,
) -> ImageGenerationResult:
    """Generate an image with a normalized SDK interface.

    Local reference images are handled by the SDK:
    - ARK/Doubao: upload local files first and pass public URLs.
    - Gemini: embed local files as inline base64 image parts.

    When ``path`` names a curated entry from :mod:`ai_toolkit.capabilities`
    (for example ``"doubao-5.0-lite"`` or ``"doubao-4.5"``) the SDK fills in
    ``provider``/``model``, validates the reference count, infers
    ``output_format`` from ``output_path`` for models that support it, and
    drops ``output_format`` for models that do not. Without ``path`` the call
    is forwarded with no capability awareness, preserving the previous
    hands-off behaviour.
    """
    capability = capabilities.get_image_capability(path=path, model=model)
    if capability is not None:
        provider = provider or capability.get("provider")
        model = model or capability.get("model")
        constraints = capability.get("constraints", {})
        _enforce_reference_limit(references or [], constraints, path=path)
        kwargs = _apply_capability_to_kwargs(
            kwargs,
            constraints=constraints,
            output_path=output_path,
        )

    if not provider:
        raise AIToolkitError("provider is required (or pass path=...)")

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
    elif normalized_provider == "dashscope":
        result = _generate_dashscope(
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
    if value in {"dashscope", "bailian", "wan", "wanx", "wanxiang", "万相", "qwen-image"}:
        return "dashscope"
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
        usage=_response_usage(raw_response),
    )
    if output_path is not None:
        result.save_first(output_path)
    return result


def _generate_dashscope(
    *,
    prompt: str,
    references: list[str | Path],
    output_path: str | Path | None,
    model: str | None,
    size: str | None,
    **kwargs: Any,
) -> ImageGenerationResult:
    settings = get_settings()
    resolved_model = model or settings.dashscope_image_model
    parameters: dict[str, Any] = {
        "size": size or settings.dashscope_image_size,
        "watermark": False,
    }
    parameters.update(kwargs)

    image_urls = [_reference_to_url(ref) for ref in references]
    raw_response = create_dashscope_image_generation(
        resolved_model,
        prompt,
        image_urls=image_urls or None,
        **parameters,
    )
    images = _extract_dashscope_images(raw_response)
    result = ImageGenerationResult(
        provider="dashscope",
        model=resolved_model,
        prompt=prompt,
        images=images,
        raw_response=raw_response,
        request={"prompt": prompt, "image_urls": image_urls, "parameters": parameters},
        usage=_response_usage(raw_response),
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
        usage=_response_usage(raw_response),
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


def _extract_dashscope_images(response: dict[str, Any]) -> list[GeneratedImage]:
    images: list[GeneratedImage] = []
    output = response.get("output", {}) if isinstance(response, dict) else {}

    # Synchronous multimodal-generation shape:
    # output.choices[].message.content[].image
    for choice in output.get("choices", []) or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message", {}) if isinstance(choice.get("message"), dict) else {}
        for part in message.get("content", []) or []:
            if isinstance(part, dict) and isinstance(part.get("image"), str):
                images.append(GeneratedImage(url=part["image"]))

    # Async/legacy task shape fallback: output.results[].url
    for item in output.get("results", []) or []:
        if isinstance(item, dict) and isinstance(item.get("url"), str):
            images.append(GeneratedImage(url=item["url"]))

    if not images:
        raise AIToolkitError("DashScope image generation returned no images")
    return images


def _apply_capability_to_kwargs(
    kwargs: dict[str, Any],
    *,
    constraints: dict[str, Any],
    output_path: str | Path | None,
) -> dict[str, Any]:
    result = dict(kwargs)
    output_format_param = constraints.get("output_format_param")
    if output_format_param:
        supported = [str(value).lower() for value in constraints.get("supported_output_formats", [])]
        if output_format_param not in result and output_path is not None:
            inferred = _infer_output_format(output_path, supported)
            if inferred is not None:
                result[output_format_param] = inferred
        elif output_format_param in result and supported:
            value = str(result[output_format_param]).strip().lower()
            if value in supported:
                result[output_format_param] = value
            else:
                default = constraints.get("default_output_format")
                result[output_format_param] = str(default) if default else supported[0]
    elif constraints.get("output_format_configurable") is False:
        result.pop("output_format", None)
    return result


def _infer_output_format(output_path: str | Path, supported: list[str]) -> str | None:
    if not supported:
        return None
    suffix = Path(str(output_path)).suffix.lower().lstrip(".")
    if not suffix:
        return None
    if suffix == "jpg":
        suffix = "jpeg"
    if suffix in supported:
        return suffix
    return None


def _enforce_reference_limit(
    references: list[str | Path],
    constraints: dict[str, Any],
    *,
    path: str | None,
) -> None:
    max_refs = constraints.get("max_reference_images")
    if max_refs is None:
        return
    if len(references) > max_refs:
        label = f"path {path!r}" if path else "this model"
        raise AIToolkitError(
            f"too many reference images for {label}: got {len(references)}, max {max_refs}"
        )


def _response_usage(response: dict[str, Any]) -> dict[str, Any] | None:
    usage = response.get("usage")
    if isinstance(usage, dict):
        return usage
    usage_metadata = response.get("usageMetadata")
    if isinstance(usage_metadata, dict):
        return usage_metadata
    return None


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
