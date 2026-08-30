from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from ai_toolkit import capabilities
from ai_toolkit.minimax import videos
from ai_toolkit.types import AIToolkitError, VideoGenerationTask


def _image(path: Path, size: tuple[int, int] = (640, 360)) -> Path:
    Image.new("RGB", size, (255, 0, 255)).save(path)
    return path


def test_create_first_frame_task_uses_v2_payload_without_retries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    source = _image(tmp_path / "source.png")
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")

    def fake_post_json(url, payload, headers, error_cls, **kwargs):
        captured.update(
            url=url,
            payload=payload,
            headers=headers,
            error_cls=error_cls,
            kwargs=kwargs,
        )
        return {"task_id": "task-1"}

    monkeypatch.setattr(videos, "post_json", fake_post_json)

    task = videos.create_task(prompt="walk", first_frame=source, duration=4)

    assert task.provider == "minimax"
    assert task.model == "MiniMax-H3"
    assert task.task_id == "task-1"
    assert captured["url"] == "https://api.minimax.io/v2/video_generation"
    assert captured["kwargs"] == {"max_retries": 0}
    payload = captured["payload"]
    assert payload["duration"] == 4
    assert payload["resolution"] == "768P"
    assert payload["ratio"] == "adaptive"
    assert payload["content"][1]["role"] == "first_frame"
    assert payload["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert captured["headers"]["Authorization"] == "Bearer test-key"


def test_keyframes_and_reference_images_are_mutually_exclusive(tmp_path: Path) -> None:
    source = _image(tmp_path / "source.png")

    with pytest.raises(ValueError, match="cannot be mixed"):
        videos.create_task(
            prompt="walk",
            first_frame=source,
            reference_images=[source],
        )


def test_local_image_contract_is_validated(tmp_path: Path) -> None:
    too_small = _image(tmp_path / "small.png", (128, 128))

    with pytest.raises(ValueError, match="256..5760"):
        videos.create_task(prompt="walk", first_frame=too_small)


def test_local_image_uses_detected_format_instead_of_extension(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "misnamed.png"
    Image.new("RGB", (640, 360), (255, 0, 255)).save(source, format="JPEG")
    monkeypatch.setattr(videos, "create_video_generation_task", lambda _payload: {"task_id": "1"})

    task = videos.create_task(prompt="walk", first_frame=source)

    data_url = task.request["content"][1]["image_url"]["url"]
    assert data_url.startswith("data:image/jpeg;base64,")


def test_wait_returns_completed_video(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            VideoGenerationTask("minimax", "MiniMax-H3", "task-1", "running", {}),
            VideoGenerationTask(
                "minimax",
                "MiniMax-H3",
                "task-1",
                "succeeded",
                {"task": {"content": {"url": "https://cdn.example/out.mp4"}}},
            ),
        ]
    )
    monkeypatch.setattr(videos, "get_task", lambda **_kwargs: next(responses))
    monkeypatch.setattr(videos.time, "sleep", lambda _seconds: None)

    result = videos.wait(task_id="task-1", poll_interval=0.1)

    assert result.status == "succeeded"
    assert result.first_video().url == "https://cdn.example/out.mp4"


def test_wait_surfaces_provider_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    failed = VideoGenerationTask(
        "minimax",
        "MiniMax-H3",
        "task-1",
        "failed",
        {"task": {"error": {"message": "blocked"}}},
    )
    monkeypatch.setattr(videos, "get_task", lambda **_kwargs: failed)

    with pytest.raises(AIToolkitError, match="blocked"):
        videos.wait(task_id="task-1")


def test_capability_defaults_do_not_add_unsupported_video_options() -> None:
    assert capabilities.normalize_video_options("minimax-h3") == {
        "duration": 4,
        "resolution": "768P",
    }
