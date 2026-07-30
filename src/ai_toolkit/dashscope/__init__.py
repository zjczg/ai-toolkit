"""DashScope platform capabilities."""

from ai_toolkit.dashscope import images, speech, videos
from ai_toolkit.dashscope.images import (
    DashScopeImageError,
    create_image_generation,
)
from ai_toolkit.dashscope.speech import (
    DashScopeSpeechError,
    create_speech_synthesis,
)
from ai_toolkit.dashscope.videos import (
    DashScopeVideoError,
    create_video_generation_task,
    get_video_generation_task,
)

__all__ = [
    "images",
    "speech",
    "videos",
    "DashScopeImageError",
    "DashScopeSpeechError",
    "DashScopeVideoError",
    "create_image_generation",
    "create_speech_synthesis",
    "create_video_generation_task",
    "get_video_generation_task",
]
