"""Gemini image generation — POST /models/{model}:generateContent."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib import parse
from urllib.parse import urlparse

from ai_toolkit._transport import get_json, post_json
from ai_toolkit.config import get_settings
from ai_toolkit.types import (
    AIToolkitError,
    GeneratedImage,
    ImageGenerationBatchItem,
    ImageGenerationBatchResult,
    ImageGenerationBatchTask,
    ImageGenerationResult,
)

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
AUTO_IMAGE_SIZE = "auto"
MAX_INLINE_BATCH_BYTES = 20_000_000
BATCH_SUCCEEDED_STATUSES = frozenset({"JOB_STATE_SUCCEEDED"})
BATCH_FAILED_STATUSES = frozenset(
    {
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
)

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
    resolved_image_size = _resolve_image_size(
        resolved_model,
        image_size or settings.gemini_image_size,
    )
    resolved_aspect_ratio = _validate_aspect_ratio(resolved_model, aspect_ratio)

    parts: list[dict[str, Any]] = [{"text": prompt}]
    parts.extend(_image_part(ref) for ref in reference_values)

    generation_config = dict(kwargs.pop("generationConfig", {}) or {})
    generation_config.setdefault("responseModalities", ["TEXT", "IMAGE"])
    image_config = dict(generation_config.get("imageConfig", {}) or {})
    if resolved_image_size is not None:
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


def create_batch(
    *,
    prompts: dict[str, str],
    model: str | None = None,
    image_size: str | None = None,
    aspect_ratio: str | None = None,
    display_name: str | None = None,
) -> ImageGenerationBatchTask:
    """Submit keyed text-to-image requests as one inline Gemini batch."""

    if not isinstance(prompts, dict) or not prompts:
        raise ValueError("prompts must be a non-empty dictionary")
    normalized_prompts: dict[str, str] = {}
    for key, prompt in prompts.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("every batch request key must be a non-empty string")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"batch prompt {key!r} must be a non-empty string")
        normalized_prompts[key.strip()] = prompt.strip()
    if len(normalized_prompts) != len(prompts):
        raise ValueError("batch request keys must remain unique after trimming")

    settings = get_settings()
    resolved_model = resolve_model(model)
    resolved_image_size = _resolve_image_size(
        resolved_model,
        image_size or settings.gemini_image_size,
    )
    resolved_aspect_ratio = _validate_aspect_ratio(
        resolved_model,
        aspect_ratio,
    )
    generation_config: dict[str, Any] = {
        "responseModalities": ["TEXT", "IMAGE"],
    }
    image_config: dict[str, str] = {}
    if resolved_image_size is not None:
        image_config["imageSize"] = resolved_image_size
    if resolved_aspect_ratio is not None:
        image_config["aspectRatio"] = resolved_aspect_ratio
    if image_config:
        generation_config["imageConfig"] = image_config

    requests = [
        {
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}],
                    }
                ],
                "generationConfig": generation_config,
            },
            "metadata": {"key": key},
        }
        for key, prompt in normalized_prompts.items()
    ]
    batch: dict[str, Any] = {
        "input_config": {"requests": {"requests": requests}},
    }
    if display_name is not None:
        cleaned_display_name = str(display_name).strip()
        if not cleaned_display_name:
            raise ValueError("display_name must not be empty")
        batch["display_name"] = cleaned_display_name

    raw_response = batch_generate_content(resolved_model, batch)
    task_id = _batch_task_id(raw_response)
    if not task_id:
        raise AIToolkitError("Gemini batch response did not include a name")
    return ImageGenerationBatchTask(
        provider="gemini",
        model=resolved_model,
        task_id=task_id,
        status=_batch_status(raw_response),
        raw_response=raw_response,
        request={"batch": batch},
    )


def get_batch(
    *,
    task_id: str,
    model: str | None = None,
) -> ImageGenerationBatchTask:
    """Return the latest provider state for one Gemini image batch."""

    resolved_model = resolve_model(model)
    raw_response = get_batch_generation(task_id)
    return ImageGenerationBatchTask(
        provider="gemini",
        model=resolved_model,
        task_id=_batch_task_id(raw_response) or _normalize_batch_name(task_id),
        status=_batch_status(raw_response),
        raw_response=raw_response,
    )


def find_batch(
    *,
    display_name: str,
    model: str | None = None,
    page_size: int = 100,
) -> ImageGenerationBatchTask | None:
    """Find one batch by its unique display name for submission recovery."""

    cleaned_name = str(display_name).strip()
    if not cleaned_name:
        raise ValueError("display_name must not be empty")
    resolved_model = resolve_model(model)
    matches: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        raw_response = list_batch_generations(
            page_size=page_size,
            page_token=page_token,
        )
        batches = raw_response.get("batches", [])
        matches.extend(
            item
            for item in batches
            if isinstance(item, dict)
            and _batch_display_name(item) == cleaned_name
        )
        next_page_token = raw_response.get("nextPageToken")
        if not isinstance(next_page_token, str) or not next_page_token.strip():
            break
        page_token = next_page_token.strip()
    if len(matches) > 1:
        raise AIToolkitError(
            f"multiple Gemini batches use display name {cleaned_name!r}"
        )
    if not matches:
        return None
    item = matches[0]
    task_id = _batch_task_id(item)
    if not task_id:
        raise AIToolkitError("Gemini batch list item did not include a name")
    return ImageGenerationBatchTask(
        provider="gemini",
        model=resolved_model,
        task_id=task_id,
        status=_batch_status(item),
        raw_response=item,
        request={"display_name": cleaned_name},
    )


def batch_result(
    task: ImageGenerationBatchTask,
    *,
    prompts: dict[str, str],
) -> ImageGenerationBatchResult:
    """Decode a succeeded inline batch into independent keyed image results."""

    if task.status.upper() not in BATCH_SUCCEEDED_STATUSES:
        raise ValueError(
            f"Gemini batch {task.task_id} has not succeeded: {task.status or 'unknown'}"
        )
    responses = _inline_batch_responses(task.raw_response or {})
    keyed: dict[str, dict[str, Any]] = {}
    unkeyed: list[dict[str, Any]] = []
    for item in responses:
        key = _batch_response_key(item)
        if key is None:
            unkeyed.append(item)
            continue
        if key in keyed:
            raise AIToolkitError(f"Gemini batch returned duplicate key {key!r}")
        keyed[key] = item

    items: list[ImageGenerationBatchItem] = []
    for key, prompt in prompts.items():
        wrapped = keyed.pop(key, None)
        if wrapped is None and unkeyed:
            wrapped = unkeyed.pop(0)
        if wrapped is None:
            items.append(
                ImageGenerationBatchItem(
                    key=key,
                    error="Gemini batch returned no response for this request",
                )
            )
            continue
        error = wrapped.get("error")
        response = wrapped.get("response")
        if error is not None:
            items.append(
                ImageGenerationBatchItem(
                    key=key,
                    error=_batch_error_text(error),
                )
            )
            continue
        if not isinstance(response, dict):
            items.append(
                ImageGenerationBatchItem(
                    key=key,
                    error="Gemini batch response item has no response payload",
                )
            )
            continue
        try:
            images, text = _extract_images_and_text(response)
        except AIToolkitError as exc:
            items.append(ImageGenerationBatchItem(key=key, error=str(exc)))
            continue
        items.append(
            ImageGenerationBatchItem(
                key=key,
                result=ImageGenerationResult(
                    provider="gemini",
                    model=task.model,
                    prompt=prompt,
                    images=images,
                    text=text,
                    raw_response=response,
                    usage=_response_usage(response),
                ),
            )
        )
    return ImageGenerationBatchResult(
        provider=task.provider,
        model=task.model,
        task_id=task.task_id,
        status=task.status,
        items=items,
        raw_response=task.raw_response,
    )


def batch_generate_content(
    model: str,
    batch: dict[str, Any],
) -> dict[str, Any]:
    """Submit one non-idempotent batchGenerateContent request."""

    settings = get_settings()
    if not settings.gemini_api_key:
        raise GeminiImageError("GEMINI_API_KEY is not configured")
    api_url = (
        f"{settings.gemini_base_url.rstrip('/')}/models/"
        f"{model}:batchGenerateContent"
    )
    payload = {"batch": batch}
    payload_size = len(json.dumps(payload).encode("utf-8"))
    if payload_size >= MAX_INLINE_BATCH_BYTES:
        raise ValueError(
            "inline Gemini batch must be smaller than 20 MB; "
            "split the batch or use the JSONL file workflow"
        )
    return post_json(
        api_url,
        payload,
        headers=_gemini_headers(settings.gemini_api_key),
        error_cls=GeminiImageError,
        max_retries=0,
    )


def get_batch_generation(task_id: str) -> dict[str, Any]:
    """Query one Gemini batch by its provider name."""

    settings = get_settings()
    if not settings.gemini_api_key:
        raise GeminiImageError("GEMINI_API_KEY is not configured")
    batch_name = _normalize_batch_name(task_id)
    return get_json(
        f"{settings.gemini_base_url.rstrip('/')}/{batch_name}",
        headers=_gemini_headers(settings.gemini_api_key),
        error_cls=GeminiImageError,
    )


def list_batch_generations(
    *,
    page_size: int = 100,
    page_token: str | None = None,
) -> dict[str, Any]:
    """List recent Gemini batches for display-name recovery."""

    if not isinstance(page_size, int) or not 1 <= page_size <= 1000:
        raise ValueError("page_size must be between 1 and 1000")
    settings = get_settings()
    if not settings.gemini_api_key:
        raise GeminiImageError("GEMINI_API_KEY is not configured")
    query_parameters = {"pageSize": page_size}
    if page_token is not None:
        cleaned_page_token = str(page_token).strip()
        if not cleaned_page_token:
            raise ValueError("page_token must not be empty")
        query_parameters["pageToken"] = cleaned_page_token
    query = parse.urlencode(query_parameters)
    return get_json(
        f"{settings.gemini_base_url.rstrip('/')}/batches?{query}",
        headers=_gemini_headers(settings.gemini_api_key),
        error_cls=GeminiImageError,
    )


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


def _resolve_image_size(model: str, image_size: object) -> str | None:
    value = str(image_size).strip()
    if value.lower() == AUTO_IMAGE_SIZE:
        return None
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


def _batch_task_id(response: dict[str, Any]) -> str:
    value = response.get("name")
    return str(value).strip() if isinstance(value, str) else ""


def _batch_status(response: dict[str, Any]) -> str:
    value = response.get("state")
    if isinstance(value, str):
        return value.strip()
    metadata = response.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("state"), str):
        return str(metadata["state"]).strip()
    return ""


def _batch_display_name(response: dict[str, Any]) -> str:
    value = response.get("displayName") or response.get("display_name")
    return str(value).strip() if isinstance(value, str) else ""


def _inline_batch_responses(response: dict[str, Any]) -> list[dict[str, Any]]:
    dest = response.get("dest")
    if not isinstance(dest, dict):
        return []
    raw_items = dest.get("inlinedResponses") or dest.get("inlined_responses") or []
    if not isinstance(raw_items, list):
        return []
    return [item for item in raw_items if isinstance(item, dict)]


def _batch_response_key(response: dict[str, Any]) -> str | None:
    metadata = response.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("key"), str):
        value = str(metadata["key"]).strip()
        return value or None
    value = response.get("key")
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return None


def _batch_error_text(error: object) -> str:
    if isinstance(error, str):
        return error
    try:
        return json.dumps(error, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(error)


def _normalize_batch_name(task_id: str) -> str:
    value = str(task_id).strip().strip("/")
    parts = value.split("/")
    if len(parts) != 2 or parts[0] != "batches" or not parts[1]:
        raise ValueError(f"invalid Gemini batch task_id: {task_id!r}")
    return value


def _gemini_headers(api_key: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }


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
