"""ai-toolkit — reusable AI model SDK.

Low-level provider wrappers are kept for compatibility. High-level modules
provide normalized SDK-style calls:

    from ai_toolkit import images, chat, capabilities
"""

__version__ = "0.4.0"

from ai_toolkit import capabilities, chat, embeddings, images, videos
from ai_toolkit.ark import (
    ArkEmbeddingError,
    ArkImageError,
    ArkMultimodalError,
    ArkVideoError,
    create_image_generation,
    create_multimodal_embedding,
    create_responses,
    create_seedream_4_5_image_generation,
    create_seedream_5_0_lite_image_generation,
    create_video_generation_task,
    get_video_generation_task,
)
from ai_toolkit.config import Settings, get_settings
from ai_toolkit.deepseek import DeepSeekError, create_chat_completion
from ai_toolkit.gemini import GeminiImageError, generate_content
from ai_toolkit.help import help, list_interfaces
from ai_toolkit.media import MediaError, upload_public_url, upload_via_ssh
from ai_toolkit.types import (
    AIToolkitError,
    ChatCompletionResult,
    EmbeddingResult,
    GeneratedImage,
    GeneratedVideo,
    ImageGenerationResult,
    VideoGenerationResult,
    VideoGenerationTask,
)

__all__ = [
    "__version__",
    # High-level SDK modules
    "capabilities",
    "chat",
    "embeddings",
    "images",
    "videos",
    "AIToolkitError",
    "ChatCompletionResult",
    "EmbeddingResult",
    "GeneratedImage",
    "GeneratedVideo",
    "ImageGenerationResult",
    "VideoGenerationResult",
    "VideoGenerationTask",
    # ARK
    "ArkEmbeddingError",
    "ArkImageError",
    "ArkMultimodalError",
    "ArkVideoError",
    "create_image_generation",
    "create_multimodal_embedding",
    "create_seedream_4_5_image_generation",
    "create_seedream_5_0_lite_image_generation",
    "create_responses",
    "create_video_generation_task",
    "get_video_generation_task",
    # Gemini
    "GeminiImageError",
    "generate_content",
    # DeepSeek
    "DeepSeekError",
    "create_chat_completion",
    # Media
    "MediaError",
    "upload_public_url",
    "upload_via_ssh",
    # Help
    "help",
    "list_interfaces",
    # Config
    "Settings",
    "get_settings",
]
