"""HappyHorse video generation on Alibaba Cloud Model Studio (DashScope)."""

from __future__ import annotations

import base64
import mimetypes
import time
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

_VIDEO_SYNTHESIS_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"
_TASK_PATH = "/api/v1/tasks"

MODEL_ALIASES = {
    "happyhorse-t2v": "happyhorse-1.1-t2v",
    "happyhorse-i2v": "happyhorse-1.1-i2v",
    "happyhorse-r2v": "happyhorse-1.1-r2v",
    "happyhorse-1.1-t2v": "happyhorse-1.1-t2v",
    "happyhorse-1.1-i2v": "happyhorse-1.1-i2v",
    "happyhorse-1.1-r2v": "happyhorse-1.1-r2v",
}

_MODEL_MEDIA_TYPES = {
    "happyhorse-1.1-t2v": None,
    "happyhorse-1.1-i2v": "first_frame",
    "happyhorse-1.1-r2v": "reference_image",
}
_SUCCEEDED_STATUSES = {"succeeded"}
_FAILED_STATUSES = {"failed", "canceled", "cancelled", "unknown"}


class DashScopeVideoError(RuntimeError):
    """Raised when a DashScope HappyHorse video request fails."""


def generate(
    *,
    prompt: str,
    references: list[str | Path] | None = None,
    output_path: str | Path | None = None,
    model: str | None = None,
    ratio: str | None = None,
    duration: int | None = None,
    resolution: str | None = None,
    watermark: bool = False,
    seed: int | None = None,
    poll_interval: float = 15.0,
    timeout: float = 900.0,
    wait_for_completion: bool = True,
) -> VideoGenerationResult | VideoGenerationTask:
    """Create a HappyHorse task and optionally wait until its MP4 is ready.

    ``happyhorse-1.1-t2v`` accepts no references, ``i2v`` accepts exactly one
    first-frame image, and ``r2v`` accepts one to nine reference images.
    Local images are sent as DashScope-supported base64 data URLs.
    """

    task = create_task(
        prompt=prompt,
        references=references,
        model=model,
        ratio=ratio,
        duration=duration,
        resolution=resolution,
        watermark=watermark,
        seed=seed,
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
    prompt: str,
    references: list[str | Path] | None = None,
    model: str | None = None,
    ratio: str | None = None,
    duration: int | None = None,
    resolution: str | None = None,
    watermark: bool = False,
    seed: int | None = None,
) -> VideoGenerationTask:
    """Submit a HappyHorse task and return its provider-agnostic handle."""

    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    resolved_model = resolve_model(model)
    media = _build_media(resolved_model, references or [])
    parameters = _parameters_for_model(
        resolved_model,
        ratio=ratio,
        duration=duration,
        resolution=resolution,
        watermark=watermark,
        seed=seed,
    )
    raw_response = create_video_generation_task(
        resolved_model,
        prompt=prompt,
        media=media or None,
        parameters=parameters,
    )
    task_id = _extract_task_id(raw_response)
    if not task_id:
        raise AIToolkitError("DashScope video task response did not include task_id")
    return VideoGenerationTask(
        provider="dashscope",
        model=resolved_model,
        task_id=task_id,
        status=_extract_status(raw_response),
        raw_response=raw_response,
        request={
            "prompt": prompt,
            "media": media,
            "parameters": parameters,
        },
    )


def get_task(*, task_id: str, model: str | None = None) -> VideoGenerationTask:
    """Return the latest state for one HappyHorse task."""

    resolved_model = resolve_model(model)
    raw_response = get_video_generation_task(task_id)
    return VideoGenerationTask(
        provider="dashscope",
        model=resolved_model,
        task_id=_extract_task_id(raw_response) or task_id,
        status=_extract_status(raw_response),
        raw_response=raw_response,
    )


def wait(
    *,
    task_id: str,
    model: str | None = None,
    poll_interval: float = 15.0,
    timeout: float = 900.0,
) -> VideoGenerationResult:
    """Poll a HappyHorse task until it succeeds, fails, or times out."""

    deadline = time.monotonic() + timeout
    while True:
        task = get_task(task_id=task_id, model=model)
        status = task.status.lower()
        if status in _SUCCEEDED_STATUSES:
            videos = _extract_videos(task.raw_response or {})
            return VideoGenerationResult(
                provider=task.provider,
                model=task.model,
                task_id=task.task_id,
                status=task.status,
                videos=videos,
                raw_response=task.raw_response,
            )
        if status in _FAILED_STATUSES:
            message = _extract_error_message(task.raw_response or {})
            raise AIToolkitError(
                f"DashScope video task {task.task_id} ended with status {task.status}: {message}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"DashScope video task {task.task_id} timed out with status {task.status or 'unknown'}"
            )
        time.sleep(max(0.1, poll_interval))


def create_video_generation_task(
    model: str,
    *,
    prompt: str,
    media: list[dict[str, str]] | None,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Call the DashScope asynchronous HappyHorse video-synthesis endpoint."""

    settings = get_settings()
    if not settings.dashscope_api_key:
        raise DashScopeVideoError("DASHSCOPE_API_KEY is not configured")

    request_input: dict[str, Any] = {"prompt": prompt}
    if media:
        request_input["media"] = media
    payload = {"model": model, "input": request_input, "parameters": parameters}
    return post_json(
        f"{settings.dashscope_base_url.rstrip('/')}{_VIDEO_SYNTHESIS_PATH}",
        payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.dashscope_api_key}",
            "X-DashScope-Async": "enable",
        },
        error_cls=DashScopeVideoError,
    )


def get_video_generation_task(task_id: str) -> dict[str, Any]:
    """Query one asynchronous DashScope video-generation task."""

    normalized_task_id = task_id.strip()
    if not normalized_task_id:
        raise ValueError("task_id must not be empty")

    settings = get_settings()
    if not settings.dashscope_api_key:
        raise DashScopeVideoError("DASHSCOPE_API_KEY is not configured")
    encoded = parse.quote(normalized_task_id, safe="")
    return get_json(
        f"{settings.dashscope_base_url.rstrip('/')}{_TASK_PATH}/{encoded}",
        headers={"Authorization": f"Bearer {settings.dashscope_api_key}"},
        error_cls=DashScopeVideoError,
    )


def resolve_model(model: str | None = None) -> str:
    """Resolve a documented HappyHorse alias to its exact model ID."""

    settings = get_settings()
    value = settings.dashscope_video_model if model is None or not str(model).strip() else str(model)
    normalized = value.strip().lower()
    resolved = MODEL_ALIASES.get(normalized, value.strip())
    if resolved not in _MODEL_MEDIA_TYPES:
        raise ValueError(f"unsupported DashScope video model: {value}")
    return resolved


def _build_media(model: str, references: list[str | Path]) -> list[dict[str, str]]:
    media_type = _MODEL_MEDIA_TYPES[model]
    if media_type is None:
        if references:
            raise ValueError(f"{model} is text-to-video and does not accept references")
        return []
    if media_type == "first_frame" and len(references) != 1:
        raise ValueError(f"{model} requires exactly one first-frame image")
    if media_type == "reference_image" and not 1 <= len(references) <= 9:
        raise ValueError(f"{model} requires between one and nine reference images")
    return [{"type": media_type, "url": _reference_to_url(reference)} for reference in references]


def _parameters_for_model(
    model: str,
    *,
    ratio: str | None,
    duration: int | None,
    resolution: str | None,
    watermark: bool,
    seed: int | None,
) -> dict[str, Any]:
    settings = get_settings()
    parameters: dict[str, Any] = {
        "resolution": (resolution or settings.dashscope_video_resolution).upper(),
        "duration": duration or settings.dashscope_video_duration,
        "watermark": watermark,
    }
    if model != "happyhorse-1.1-i2v":
        parameters["ratio"] = ratio or settings.dashscope_video_ratio
    if seed is not None:
        parameters["seed"] = seed
    return parameters


def _reference_to_url(reference: str | Path) -> str:
    value = str(reference)
    if _is_remote_reference(value) or value.startswith("data:"):
        return value
    local_path = Path(value).expanduser()
    if local_path.is_file():
        return _local_image_data_url(local_path)
    return upload_public_url(value)


def _local_image_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError(f"unsupported DashScope video reference image format: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_task_id(response: dict[str, Any]) -> str:
    output = response.get("output", {}) if isinstance(response, dict) else {}
    value = output.get("task_id") if isinstance(output, dict) else None
    return value if isinstance(value, str) else ""


def _extract_status(response: dict[str, Any]) -> str:
    output = response.get("output", {}) if isinstance(response, dict) else {}
    value = output.get("task_status") if isinstance(output, dict) else None
    return value if isinstance(value, str) else ""


def _extract_videos(response: dict[str, Any]) -> list[GeneratedVideo]:
    output = response.get("output", {}) if isinstance(response, dict) else {}
    url = output.get("video_url") if isinstance(output, dict) else None
    if not isinstance(url, str) or not url:
        raise AIToolkitError("DashScope video task succeeded but returned no video_url")
    return [GeneratedVideo(url=url)]


def _extract_error_message(response: dict[str, Any]) -> str:
    output = response.get("output", {}) if isinstance(response, dict) else {}
    value = output.get("message") if isinstance(output, dict) else None
    return value if isinstance(value, str) else "no error message"


def _is_remote_reference(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https", "oss"} and bool(parsed.netloc)
