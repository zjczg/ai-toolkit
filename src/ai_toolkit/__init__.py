"""ai-toolkit — platform-scoped AI model SDK."""

__version__ = "0.7.0"

from ai_toolkit import aliyun, ark, capabilities, dashscope, deepseek, gemini
from ai_toolkit.config import Settings, get_settings
from ai_toolkit.help import help, list_interfaces
from ai_toolkit.types import (
    AIToolkitError,
    ChatCompletionResult,
    EmbeddingResult,
    GeneratedImage,
    GeneratedVideo,
    ImageGenerationResult,
    ImageSegmentationResult,
    VideoGenerationResult,
    VideoGenerationTask,
)

__all__ = [
    "__version__",
    # High-level SDK modules
    "aliyun",
    "ark",
    "capabilities",
    "dashscope",
    "deepseek",
    "gemini",
    "AIToolkitError",
    "ChatCompletionResult",
    "EmbeddingResult",
    "GeneratedImage",
    "GeneratedVideo",
    "ImageGenerationResult",
    "ImageSegmentationResult",
    "VideoGenerationResult",
    "VideoGenerationTask",
    # Help
    "help",
    "list_interfaces",
    # Config
    "Settings",
    "get_settings",
]
