"""ARK (豆包/火山引擎) video generation — POST + GET /contents/generations/tasks."""

from __future__ import annotations

import mimetypes
import time
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib import parse
from urllib.parse import urlparse

from ai_toolkit._transport import get_json, post_json
from ai_toolkit.config import get_settings
from ai_toolkit.media import upload_public_url
from ai_toolkit.types import (
    AIToolkitError,
    GeneratedVideo,
    VideoGenerationResult,
    VideoGenerationTask,
)

MODEL_ALIASES = {
    "seedance-2": "doubao-seedance-2-0-260128",
    "seedance-2.0": "doubao-seedance-2-0-260128",
    "doubao-seedance-2.0": "doubao-seedance-2-0-260128",
    "doubao-seedance-2-0-260128": "doubao-seedance-2-0-260128",
}

SUCCEEDED_STATUSES = {"succeeded", "success", "completed", "complete"}
FAILED_STATUSES = {"failed", "error", "canceled", "cancelled", "expired"}


class ArkVideoError(RuntimeError):
    """Raised when the ARK video generation API request fails."""


def generate(
    *,
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
    """Submit an ARK Seedance video task and optionally wait for the result."""
    task = create_task(
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
        task_id=task.task_id,
        model=task.model,
        poll_interval=poll_interval,
        timeout=timeout,
    )
    result.request = task.request
    if output_path is not None:
        result.save_first(output_path)
    return result


def create_task(
    *,
    prompt: str | None = None,
    content: list[dict[str, Any]] | None = None,
    references: list[str | Path | dict[str, Any]] | None = None,
    reference_images: list[str | Path | dict[str, Any]] | None = None,
    reference_videos: list[str | Path | dict[str, Any]] | None = None,
    reference_audio: list[str | Path | dict[str, Any]] | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> VideoGenerationTask:
    resolved_model = resolve_model(model)
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

    request_kwargs = _drop_none_values(kwargs)
    raw_response = create_video_generation_task(
        resolved_model,
        resolved_content,
        **request_kwargs,
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
        request={"content": resolved_content, **request_kwargs},
    )


def get_task(*, task_id: str, model: str | None = None) -> VideoGenerationTask:
    resolved_model = resolve_model(model)
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
    task_id: str,
    model: str | None = None,
    poll_interval: float = 5.0,
    timeout: float = 600.0,
) -> VideoGenerationResult:
    deadline = time.monotonic() + timeout
    last_task: VideoGenerationTask | None = None
    while True:
        last_task = get_task(task_id=task_id, model=model)
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


def build_content(
    *,
    prompt: str | None = None,
    content: list[dict[str, Any]] | None = None,
    references: list[str | Path | dict[str, Any]] | None = None,
    reference_images: list[str | Path | dict[str, Any]] | None = None,
    reference_videos: list[str | Path | dict[str, Any]] | None = None,
    reference_audio: list[str | Path | dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
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


def resolve_model(model: str | None = None) -> str:
    settings = get_settings()
    if model is None or not str(model).strip():
        return settings.ark_video_model
    value = str(model).strip()
    return MODEL_ALIASES.get(value, MODEL_ALIASES.get(value.lower(), value))


def create_video_generation_task(
    model: str,
    content: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Submit a video generation task and return the raw API response.

    Required:
        model:   e.g. "doubao-seedance-2-0-260128"
        content: e.g. [{"type": "text", "text": "..."}]
                 May include {"type": "image_url", "image_url": {"url": "..."}, "role": "reference_image"}
                 and similar for video/audio references.

    Optional (via kwargs):
        generate_audio: bool
        ratio: str
        duration: int
        watermark: bool
    """
    settings = get_settings()
    if not settings.ark_api_key:
        raise ArkVideoError("ARK_API_KEY is not configured")

    payload: dict[str, Any] = {"model": model, "content": content}
    payload.update(kwargs)

    api_url = f"{settings.ark_base_url.rstrip('/')}/contents/generations/tasks"
    return post_json(
        api_url, payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.ark_api_key}",
        },
        error_cls=ArkVideoError,
    )


def get_video_generation_task(task_id: str) -> dict[str, Any]:
    """Query a video generation task and return the raw API response.

    Required:
        task_id: the task ID returned by create_video_generation_task()
    """
    normalized_task_id = task_id.strip()
    if not normalized_task_id:
        raise ValueError("task_id must not be empty")

    settings = get_settings()
    if not settings.ark_api_key:
        raise ArkVideoError("ARK_API_KEY is not configured")

    encoded = parse.quote(normalized_task_id, safe="")
    api_url = f"{settings.ark_base_url.rstrip('/')}/contents/generations/tasks/{encoded}"
    return get_json(
        api_url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.ark_api_key}",
        },
        error_cls=ArkVideoError,
    )


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
        if isinstance(value, str) and value:
            return value
    data = response.get("data")
    if isinstance(data, dict):
        for key in ("id", "task_id", "taskId"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _extract_status(response: dict[str, Any]) -> str:
    for key in ("status", "task_status", "taskStatus"):
        value = response.get(key)
        if isinstance(value, str):
            return value
    data = response.get("data")
    if isinstance(data, dict):
        for key in ("status", "task_status", "taskStatus"):
            value = data.get(key)
            if isinstance(value, str):
                return value
    return ""


def _extract_videos(response: dict[str, Any]) -> list[GeneratedVideo]:
    videos: list[GeneratedVideo] = []
    _collect_video_urls(response, videos)
    if not videos:
        raise AIToolkitError("ARK video generation returned no videos")
    return videos


def _collect_video_urls(value: Any, videos: list[GeneratedVideo]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and key in {"url", "video_url", "videoUrl"}:
                videos.append(GeneratedVideo(url=item))
            else:
                _collect_video_urls(item, videos)
    elif isinstance(value, list):
        for item in value:
            _collect_video_urls(item, videos)


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _drop_none_values(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}
