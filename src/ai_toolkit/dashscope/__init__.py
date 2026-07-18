"""DashScope platform capabilities."""

from ai_toolkit.dashscope import images, speech
from ai_toolkit.dashscope.images import (
    DashScopeImageError,
    create_image_generation,
)
from ai_toolkit.dashscope.speech import (
    DashScopeSpeechError,
    create_speech_synthesis,
)

__all__ = [
    "images",
    "speech",
    "DashScopeImageError",
    "DashScopeSpeechError",
    "create_image_generation",
    "create_speech_synthesis",
]
