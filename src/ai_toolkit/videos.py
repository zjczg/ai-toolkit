"""High-level provider-agnostic video generation helpers."""

from __future__ import annotations

import mimetypes
import time
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ai_toolkit.ark import create_video_generation_task, get_video_generation_task
from ai_toolkit.config import get_settings
from ai_toolkit.media import upload_public_url
from ai_toolkit.types import (
    AIToolkitError,
    GeneratedVideo,
    VideoGenerationResult,
    VideoGenerationTask,
)

SUCCEEDED_STATUSES = {"succeeded", "success", "completed", "complete"}
FAILED_STATUSES = {"failed", "error", "canceled", "cancelled", "expired"}


def create_task(
    *,
    provider: str,
    prompt: str | None = None,
    content: list[dict[str, Any]] | None = None,
    references: list[str | Path | dict[str, Any]] | None = None,
    reference_images: list[str | Path | dict[str, Any]] | None = None,
    reference_videos: list[str | Path | dict[str, Any]] | None = None,
    reference_audio: list[str | Path | dict[str, Any]] | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> VideoGenerationTask:
    """Submit a video generation task and return the normalized task handle.

    Local references are uploaded before provider submission when the provider
    requires public URLs.
    """
    normalized_provider = normalize_provider(provider)
    if normalized_provider != "ark":
        raise AIToolkitError(f"unknown video provider: {provider}")

    settings = get_settings()
    resolved_model = model or settings.ark_video_model
    resolved_content = build_content(
        prompt=prompt,
        content=content,
        references=references or [],
        reference_images=reference_images or [],
        reference_videos=reference_videos or [],
        reference_audio=reference_audio or [],
    )
    if not resolved_content:
        raise ValueError("prompt, content, or references must be provided")

    raw_response = create_video_generation_task(
        resolved_model,
        resolved_content,
        **kwargs,
    )
    task_id = _extract_task_id(raw_response)
    if not task_id:
        raise AIToolkitError("ARK video generation task response did not include an id")

    return VideoGenerationTask(
        provider="ark",
        model=resolved_model,
        task_id=task_id,
        status=_extract_status(raw_response),
        raw_response=raw_response,
        request={"content": resolved_content, **kwargs},
    )


def get_task(
    *,
    provider: str = "ark",
    task_id: str,
    model: str | None = None,
) -> VideoGenerationTask:
    """Fetch a video generation task by id."""
    normalized_provider = normalize_provider(provider)
    if normalized_provider != "ark":
        raise AIToolkitError(f"unknown video provider: {provider}")

    settings = get_settings()
    resolved_model = model or settings.ark_video_model
    raw_response = get_video_generation_task(task_id)
    return VideoGenerationTask(
        provider="ark",
        model=resolved_model,
        task_id=_extract_task_id(raw_response) or task_id,
        status=_extract_status(raw_response),
        raw_response=raw_response,
    )


def wait(
    *,
    provider: str = "ark",
    task_id: str,
    model: str | None = None,
    poll_interval: float = 5.0,
    timeout: float = 600.0,
) -> VideoGenerationResult:
    """Poll a video task until it succeeds, fails, or times out."""
    deadline = time.monotonic() + timeout
    last_task: VideoGenerationTask | None = None
    while True:
        last_task = get_task(provider=provider, task_id=task_id, model=model)
        status = last_task.status.lower()
        if status in SUCCEEDED_STATUSES:
            videos = _extract_videos(last_task.raw_response or {})
            if not videos:
                raise AIToolkitError("video generation succeeded but returned no video URL")
            return VideoGenerationResult(
                provider=last_task.provider,
                model=last_task.model,
                task_id=last_task.task_id,
                status=last_task.status,
                videos=videos,
                raw_response=last_task.raw_response,
            )
        if status in FAILED_STATUSES:
            raise AIToolkitError(
                f"video generation task {last_task.task_id} ended with status {last_task.status}"
            )
        if time.monotonic() >= deadline:
            latest_status = last_task.status or "unknown"
            raise TimeoutError(
                f"video generation task {last_task.task_id} timed out with status {latest_status}"
            )
        time.sleep(max(0.1, poll_interval))


def generate(
    *,
    provider: str,
    prompt: str | None = None,
    content: list[dict[str, Any]] | None = None,
    references: list[str | Path | dict[str, Any]] | None = None,
    reference_images: list[str | Path | dict[str, Any]] | None = None,
    reference_videos: list[str | Path | dict[str, Any]] | None = None,
    reference_audio: list[str | Path | dict[str, Any]] | None = None,
    output_path: str | Path | None = None,
    model: str | None = None,
    poll_interval: float = 5.0,
    timeout: float = 600.0,
    wait_for_completion: bool = True,
    **kwargs: Any,
) -> VideoGenerationResult | VideoGenerationTask:
    """Generate a video with a normalized SDK interface.

    By default this submits the async task and polls until the generated video
    URL is available. Set wait_for_completion=False to get the task handle only.
    """
    task = create_task(
        provider=provider,
        prompt=prompt,
        content=content,
        references=references,
        reference_images=reference_images,
        reference_videos=reference_videos,
        reference_audio=reference_audio,
        model=model,
        **kwargs,
    )
    if not wait_for_completion:
        return task

    result = wait(
        provider=task.provider,
        task_id=task.task_id,
        model=task.model,
        poll_interval=poll_interval,
        timeout=timeout,
    )
    result.request = task.request
    if output_path is not None:
        result.save_first(output_path)
    return result


def build_content(
    *,
    prompt: str | None = None,
    content: list[dict[str, Any]] | None = None,
    references: list[str | Path | dict[str, Any]] | None = None,
    reference_images: list[str | Path | dict[str, Any]] | None = None,
    reference_videos: list[str | Path | dict[str, Any]] | None = None,
    reference_audio: list[str | Path | dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build ARK-compatible multimodal content for video generation."""
    resolved: list[dict[str, Any]] = []
    if content:
        resolved.extend(deepcopy(content))
    if prompt:
        resolved.append({"type": "text", "text": prompt})

    for reference in references or []:
        resolved.append(_reference_part(reference, media_type=None, role=None))
    for reference in reference_images or []:
        resolved.append(_reference_part(reference, media_type="image", role="reference_image"))
    for reference in reference_videos or []:
        resolved.append(_reference_part(reference, media_type="video", role="reference_video"))
    for reference in reference_audio or []:
        resolved.append(_reference_part(reference, media_type="audio", role="reference_audio"))
    return resolved


def normalize_provider(provider: str) -> str:
    value = provider.strip().lower()
    if value in {"ark", "doubao", "volcengine", "seedance"}:
        return "ark"
    return value


def _reference_part(
    reference: str | Path | dict[str, Any],
    *,
    media_type: str | None,
    role: str | None,
) -> dict[str, Any]:
    if isinstance(reference, dict):
        return deepcopy(reference)

    url = _reference_to_url(reference)
    resolved_media_type = media_type or _infer_media_type(reference)
    payload_key = f"{resolved_media_type}_url"
    part = {
        "type": payload_key,
        payload_key: {"url": url},
    }
    resolved_role = role or f"reference_{resolved_media_type}"
    if resolved_role:
        part["role"] = resolved_role
    return part


def _reference_to_url(reference: str | Path) -> str:
    value = str(reference)
    if _is_http_url(value) or value.startswith("asset://"):
        return value
    return upload_public_url(value)


def _infer_media_type(reference: str | Path) -> str:
    mime_type = mimetypes.guess_type(str(reference))[0] or ""
    if mime_type.startswith("image/"):
        return "image"
    if mime_type.startswith("video/"):
        return "video"
    if mime_type.startswith("audio/"):
        return "audio"
    raise ValueError(f"cannot infer media type for reference: {reference}")


def _extract_task_id(response: dict[str, Any]) -> str:
    for key in ("id", "task_id", "taskId"):
        value = response.get(key)
        if isinstance(value, str):
            return value
    data = response.get("data")
    if isinstance(data, dict):
        for key in ("id", "task_id", "taskId"):
            value = data.get(key)
            if isinstance(value, str):
                return value
    return ""


def _extract_status(response: dict[str, Any]) -> str:
    for key in ("status", "state"):
        value = response.get(key)
        if isinstance(value, str):
            return value
    data = response.get("data")
    if isinstance(data, dict):
        for key in ("status", "state"):
            value = data.get(key)
            if isinstance(value, str):
                return value
    return ""


def _extract_videos(response: dict[str, Any]) -> list[GeneratedVideo]:
    urls = list(dict.fromkeys(_extract_video_urls(response)))
    return [GeneratedVideo(url=url) for url in urls]


def _extract_video_urls(value: Any, *, in_video_context: bool = False) -> list[str]:
    urls: list[str] = []
    if isinstance(value, dict):
        type_value = str(value.get("type") or "")
        object_value = str(value.get("object") or "")
        next_video_context = in_video_context or type_value == "video_url" or object_value == "video"
        for key, item in value.items():
            if key == "video_url":
                urls.extend(_extract_url_field(item))
                urls.extend(_extract_video_urls(item, in_video_context=True))
            elif key in {"file_url", "output_url", "download_url", "result_url"}:
                urls.extend(_extract_url_field(item))
                urls.extend(_extract_video_urls(item, in_video_context=next_video_context))
            elif key == "url" and next_video_context:
                urls.extend(_extract_url_field(item))
            else:
                urls.extend(_extract_video_urls(item, in_video_context=next_video_context))
    elif isinstance(value, list):
        for item in value:
            urls.extend(_extract_video_urls(item, in_video_context=in_video_context))
    return [url for url in urls if _is_http_url(url)]


def _extract_url_field(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        nested = value.get("url")
        return [nested] if isinstance(nested, str) else []
    return []


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
