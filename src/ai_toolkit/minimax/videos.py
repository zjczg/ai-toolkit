"""MiniMax H3 asynchronous video generation API."""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any
from urllib import parse
from urllib.parse import urlparse

from PIL import Image

from ai_toolkit._transport import get_json, post_json
from ai_toolkit.config import get_settings
from ai_toolkit.types import (
    AIToolkitError,
    GeneratedVideo,
    VideoGenerationResult,
    VideoGenerationTask,
)

MODEL_ALIASES = {
    "h3": "MiniMax-H3",
    "minimax-h3": "MiniMax-H3",
    "MiniMax-H3": "MiniMax-H3",
}
SUPPORTED_RESOLUTIONS = frozenset({"768P", "2K"})
SUPPORTED_RATIOS = frozenset({"adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"})
SUCCEEDED_STATUSES = frozenset({"succeeded"})
FAILED_STATUSES = frozenset({"failed", "cancelled"})
MAX_IMAGE_BYTES = 30_000_000
MIN_IMAGE_EDGE = 256
MAX_IMAGE_EDGE = 5760
MIN_ASPECT_RATIO = 0.4
MAX_ASPECT_RATIO = 2.5


class MiniMaxVideoError(RuntimeError):
    """Raised when a MiniMax H3 video request fails."""


def generate(
    *,
    prompt: str,
    first_frame: str | Path | None = None,
    last_frame: str | Path | None = None,
    reference_images: list[str | Path] | None = None,
    output_path: str | Path | None = None,
    model: str | None = None,
    duration: int = 4,
    resolution: str = "768P",
    ratio: str | None = None,
    poll_interval: float = 10.0,
    timeout: float = 1800.0,
    wait_for_completion: bool = True,
) -> VideoGenerationResult | VideoGenerationTask:
    """Create a MiniMax H3 task and optionally wait for its MP4."""

    task = create_task(
        prompt=prompt,
        first_frame=first_frame,
        last_frame=last_frame,
        reference_images=reference_images,
        model=model,
        duration=duration,
        resolution=resolution,
        ratio=ratio,
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
    first_frame: str | Path | None = None,
    last_frame: str | Path | None = None,
    reference_images: list[str | Path] | None = None,
    model: str | None = None,
    duration: int = 4,
    resolution: str = "768P",
    ratio: str | None = None,
) -> VideoGenerationTask:
    """Submit one non-idempotent MiniMax H3 generation request."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not 4 <= duration <= 15:
        raise ValueError("MiniMax H3 duration must be between 4 and 15 seconds")

    resolved_model = resolve_model(model)
    resolved_resolution = str(resolution).strip().upper()
    if resolved_resolution not in SUPPORTED_RESOLUTIONS:
        raise ValueError(f"unsupported MiniMax H3 resolution: {resolution}")

    references = list(reference_images or [])
    if (first_frame is not None or last_frame is not None) and references:
        raise ValueError("first/last-frame mode cannot be mixed with reference-image mode")
    if len(references) > 9:
        raise ValueError("MiniMax H3 accepts at most nine reference images")

    resolved_ratio = _resolve_ratio(
        ratio,
        has_keyframes=first_frame is not None or last_frame is not None,
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt.strip()}]
    if first_frame is not None:
        content.append(_image_item(first_frame, role="first_frame"))
    if last_frame is not None:
        content.append(_image_item(last_frame, role="last_frame"))
    content.extend(_image_item(value, role="reference_image") for value in references)

    payload: dict[str, Any] = {
        "model": resolved_model,
        "content": content,
        "resolution": resolved_resolution,
        "duration": duration,
    }
    if resolved_ratio is not None:
        payload["ratio"] = resolved_ratio

    raw_response = create_video_generation_task(payload)
    task_id = _extract_task_id(raw_response)
    if not task_id:
        raise AIToolkitError("MiniMax video task response did not include task_id")
    return VideoGenerationTask(
        provider="minimax",
        model=resolved_model,
        task_id=task_id,
        status="queued",
        raw_response=raw_response,
        request=payload,
    )


def get_task(*, task_id: str, model: str | None = None) -> VideoGenerationTask:
    """Return the latest state for one MiniMax H3 task."""

    normalized_task_id = str(task_id).strip()
    if not normalized_task_id:
        raise ValueError("task_id must not be empty")
    resolved_model = resolve_model(model)
    raw_response = get_video_generation_task(normalized_task_id)
    task = raw_response.get("task")
    task_payload = task if isinstance(task, dict) else {}
    return VideoGenerationTask(
        provider="minimax",
        model=str(task_payload.get("model") or resolved_model),
        task_id=str(task_payload.get("id") or normalized_task_id),
        status=str(task_payload.get("status") or ""),
        raw_response=raw_response,
    )


def wait(
    *,
    task_id: str,
    model: str | None = None,
    poll_interval: float = 10.0,
    timeout: float = 1800.0,
) -> VideoGenerationResult:
    """Poll MiniMax until a task succeeds, fails, or times out."""

    deadline = time.monotonic() + timeout
    while True:
        task = get_task(task_id=task_id, model=model)
        status = task.status.lower()
        if status in SUCCEEDED_STATUSES:
            videos = _extract_videos(task.raw_response or {})
            return VideoGenerationResult(
                provider=task.provider,
                model=task.model,
                task_id=task.task_id,
                status=task.status,
                videos=videos,
                raw_response=task.raw_response,
            )
        if status in FAILED_STATUSES:
            raise AIToolkitError(
                f"MiniMax video task {task.task_id} ended with status {task.status}: "
                f"{_extract_error_message(task.raw_response or {})}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"MiniMax video task {task.task_id} timed out with status {task.status or 'unknown'}"
            )
        time.sleep(max(0.1, poll_interval))


def create_video_generation_task(payload: dict[str, Any]) -> dict[str, Any]:
    """Call POST /v2/video_generation without automatic transport retries."""

    settings = get_settings()
    if not settings.minimax_api_key:
        raise MiniMaxVideoError("MINIMAX_API_KEY is not configured")
    return post_json(
        f"{settings.minimax_base_url.rstrip('/')}/v2/video_generation",
        payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.minimax_api_key}",
        },
        error_cls=MiniMaxVideoError,
        max_retries=0,
    )


def get_video_generation_task(task_id: str) -> dict[str, Any]:
    """Call GET /v2/query/video_generation/{task_id}."""

    settings = get_settings()
    if not settings.minimax_api_key:
        raise MiniMaxVideoError("MINIMAX_API_KEY is not configured")
    encoded = parse.quote(task_id, safe="")
    return get_json(
        f"{settings.minimax_base_url.rstrip('/')}/v2/query/video_generation/{encoded}",
        headers={"Authorization": f"Bearer {settings.minimax_api_key}"},
        error_cls=MiniMaxVideoError,
    )


def resolve_model(model: str | None = None) -> str:
    """Resolve MiniMax H3 aliases and reject undocumented model IDs."""

    settings = get_settings()
    value = settings.minimax_video_model if model is None or not str(model).strip() else str(model)
    cleaned = value.strip()
    resolved = MODEL_ALIASES.get(cleaned, MODEL_ALIASES.get(cleaned.lower(), cleaned))
    if resolved != "MiniMax-H3":
        raise ValueError(f"unsupported MiniMax video model: {value}")
    return resolved


def _resolve_ratio(ratio: str | None, *, has_keyframes: bool) -> str | None:
    if has_keyframes:
        if ratio is not None and str(ratio).strip().lower() != "adaptive":
            raise ValueError("MiniMax first/last-frame mode always uses adaptive ratio")
        return "adaptive"
    if ratio is None:
        return None
    resolved = str(ratio).strip().lower()
    if resolved not in SUPPORTED_RATIOS:
        raise ValueError(f"unsupported MiniMax H3 ratio: {ratio}")
    return resolved


def _image_item(value: str | Path, *, role: str) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": _image_url(value)},
        "role": role,
    }


def _image_url(value: str | Path) -> str:
    reference = str(value)
    parsed = urlparse(reference)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return reference
    if reference.startswith("data:image/"):
        return reference

    path = Path(reference).expanduser()
    if not path.is_file():
        raise ValueError(f"MiniMax image reference is not a file or URL: {value}")
    size = path.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise ValueError(f"MiniMax image reference exceeds 30 MB: {path}")
    with Image.open(path) as image:
        width, height = image.size
        image_format = str(image.format or "").upper()
    mime_type = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }.get(image_format)
    if mime_type is None:
        raise ValueError(f"unsupported MiniMax image reference format: {path}")
    if not MIN_IMAGE_EDGE <= width <= MAX_IMAGE_EDGE or not MIN_IMAGE_EDGE <= height <= MAX_IMAGE_EDGE:
        raise ValueError(
            f"MiniMax image dimensions must be within 256..5760 pixels: {width}x{height}"
        )
    aspect_ratio = width / height
    if not MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO:
        raise ValueError(f"MiniMax image aspect ratio must be within 2:5..5:2: {width}x{height}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_task_id(response: dict[str, Any]) -> str:
    value = response.get("task_id") if isinstance(response, dict) else None
    return str(value) if value is not None else ""


def _extract_videos(response: dict[str, Any]) -> list[GeneratedVideo]:
    task = response.get("task") if isinstance(response, dict) else None
    task_payload = task if isinstance(task, dict) else {}
    content = task_payload.get("content")
    content_payload = content if isinstance(content, dict) else {}
    url = content_payload.get("url")
    if not isinstance(url, str) or not url:
        raise AIToolkitError("MiniMax video task succeeded but returned no content.url")
    return [GeneratedVideo(url=url)]


def _extract_error_message(response: dict[str, Any]) -> str:
    task = response.get("task") if isinstance(response, dict) else None
    task_payload = task if isinstance(task, dict) else {}
    error = task_payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
    return "no error message"
