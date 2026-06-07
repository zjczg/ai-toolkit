"""ai-toolkit — reusable AI model SDK.

Low-level provider wrappers are kept for compatibility. High-level modules
provide normalized SDK-style calls:

    from ai_toolkit import images, chat, capabilities
"""

__version__ = "0.2.0"

from ai_toolkit import capabilities
from ai_toolkit import chat
from ai_toolkit import embeddings
from ai_toolkit import images
from ai_toolkit import videos
from ai_toolkit.ark import ArkEmbeddingError
from ai_toolkit.ark import ArkImageError
from ai_toolkit.ark import ArkMultimodalError
from ai_toolkit.ark import ArkVideoError
from ai_toolkit.ark import create_image_generation
from ai_toolkit.ark import create_multimodal_embedding
from ai_toolkit.ark import create_responses
from ai_toolkit.ark import create_seedream_4_5_image_generation
from ai_toolkit.ark import create_seedream_5_0_lite_image_generation
from ai_toolkit.ark import create_video_generation_task
from ai_toolkit.ark import get_video_generation_task
from ai_toolkit.config import Settings
from ai_toolkit.config import get_settings
from ai_toolkit.deepseek import DeepSeekError
from ai_toolkit.deepseek import create_chat_completion
from ai_toolkit.gemini import GeminiImageError
from ai_toolkit.gemini import generate_content
from ai_toolkit.help import help
from ai_toolkit.help import list_interfaces
from ai_toolkit.media import MediaError
from ai_toolkit.media import upload_public_url
from ai_toolkit.media import upload_via_ssh
from ai_toolkit.types import AIToolkitError
from ai_toolkit.types import ChatCompletionResult
from ai_toolkit.types import EmbeddingResult
from ai_toolkit.types import GeneratedImage
from ai_toolkit.types import GeneratedVideo
from ai_toolkit.types import ImageGenerationResult
from ai_toolkit.types import VideoGenerationResult
from ai_toolkit.types import VideoGenerationTask

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
