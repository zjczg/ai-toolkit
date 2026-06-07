"""Shared result types for the high-level ai_toolkit SDK."""

from __future__ import annotations

import base64
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_toolkit._transport import read_binary


class AIToolkitError(RuntimeError):
    """Base error raised by high-level SDK helpers."""


@dataclass(slots=True)
class GeneratedImage:
    """A provider-agnostic generated image handle."""

    url: str | None = None
    b64_json: str | None = None
    mime_type: str = "image/png"
    local_path: str | None = None

    def save(self, output_path: str | Path) -> Path:
        """Save this image to output_path and return the resolved path."""
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        if self.local_path:
            shutil.copyfile(Path(self.local_path).expanduser(), target)
            return target

        if self.b64_json:
            target.write_bytes(base64.b64decode(self.b64_json))
            self.local_path = str(target)
            return target

        if self.url:
            data, mime_type = read_binary(
                self.url,
                headers={},
                error_cls=AIToolkitError,
                resource_name="Image",
            )
            target.write_bytes(data)
            self.mime_type = mime_type or self.mime_type
            self.local_path = str(target)
            return target

        raise AIToolkitError("generated image has no url, b64_json, or local_path")


@dataclass(slots=True)
class ImageGenerationResult:
    """Provider-agnostic image generation result."""

    provider: str
    model: str
    prompt: str
    images: list[GeneratedImage]
    text: str = ""
    raw_response: dict[str, Any] | None = None
    request: dict[str, Any] | None = None

    def first_image(self) -> GeneratedImage:
        if not self.images:
            raise AIToolkitError("image generation returned no images")
        return self.images[0]

    def save_first(self, output_path: str | Path) -> Path:
        return self.first_image().save(output_path)


@dataclass(slots=True)
class EmbeddingResult:
    """Provider-agnostic embedding result."""

    provider: str
    model: str
    embeddings: list[list[float]]
    raw_response: dict[str, Any] | None = None
    request: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None

    def first_embedding(self) -> list[float]:
        if not self.embeddings:
            raise AIToolkitError("embedding API returned no vectors")
        return self.embeddings[0]


@dataclass(slots=True)
class GeneratedVideo:
    """A provider-agnostic generated video handle."""

    url: str | None = None
    mime_type: str = "video/mp4"
    local_path: str | None = None

    def save(self, output_path: str | Path) -> Path:
        """Save this video to output_path and return the resolved path."""
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        if self.local_path:
            shutil.copyfile(Path(self.local_path).expanduser(), target)
            return target

        if self.url:
            data, mime_type = read_binary(
                self.url,
                headers={},
                error_cls=AIToolkitError,
                resource_name="Video",
            )
            target.write_bytes(data)
            self.mime_type = mime_type or self.mime_type
            self.local_path = str(target)
            return target

        raise AIToolkitError("generated video has no url or local_path")


@dataclass(slots=True)
class VideoGenerationTask:
    """Provider-agnostic video generation task state."""

    provider: str
    model: str
    task_id: str
    status: str = ""
    raw_response: dict[str, Any] | None = None
    request: dict[str, Any] | None = None


@dataclass(slots=True)
class VideoGenerationResult:
    """Provider-agnostic video generation result."""

    provider: str
    model: str
    task_id: str
    status: str
    videos: list[GeneratedVideo]
    raw_response: dict[str, Any] | None = None
    request: dict[str, Any] | None = None

    def first_video(self) -> GeneratedVideo:
        if not self.videos:
            raise AIToolkitError("video generation returned no videos")
        return self.videos[0]

    def save_first(self, output_path: str | Path) -> Path:
        return self.first_video().save(output_path)


@dataclass(slots=True)
class ChatCompletionResult:
    """Provider-agnostic text generation result."""

    provider: str
    model: str
    text: str
    parsed_json: Any | None = None
    raw_response: dict[str, Any] | None = None
    request: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
