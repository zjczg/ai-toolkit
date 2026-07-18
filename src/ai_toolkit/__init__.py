"""ai-toolkit — platform-scoped AI model SDK."""

__version__ = "0.8.0"

from ai_toolkit import aliyun, ark, capabilities, dashscope, deepseek, gemini
from ai_toolkit.config import Settings, get_settings
from ai_toolkit.help import help, list_interfaces
from ai_toolkit.types import (
    AIToolkitError,
    ChatCompletionResult,
    EmbeddingResult,
    GeneratedAudio,
    GeneratedImage,
    GeneratedVideo,
    ImageGenerationResult,
    ImageSegmentationResult,
    SpeechSynthesisResult,
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
    "GeneratedAudio",
    "GeneratedImage",
    "GeneratedVideo",
    "ImageGenerationResult",
    "ImageSegmentationResult",
    "SpeechSynthesisResult",
    "VideoGenerationResult",
    "VideoGenerationTask",
    # Help
    "help",
    "list_interfaces",
    # Config
    "Settings",
    "get_settings",
]
