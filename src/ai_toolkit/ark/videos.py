"""ARK (豆包/火山引擎) video generation — POST + GET /contents/generations/tasks."""

from __future__ import annotations

from typing import Any
from urllib import parse

from ai_toolkit._transport import get_json, post_json
from ai_toolkit.config import get_settings


class ArkVideoError(RuntimeError):
    """Raised when the ARK video generation API request fails."""


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
