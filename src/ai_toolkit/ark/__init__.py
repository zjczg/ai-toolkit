"""ARK (Doubao / Volcengine) platform capabilities."""

from ai_toolkit.ark import embeddings, images, text, videos
from ai_toolkit.ark.embeddings import ArkEmbeddingError, create_multimodal_embedding
from ai_toolkit.ark.images import (
    ArkImageError,
    create_image_generation,
    create_seedream_4_5_image_generation,
    create_seedream_5_0_lite_image_generation,
)
from ai_toolkit.ark.responses import ArkMultimodalError, create_responses
from ai_toolkit.ark.videos import (
    ArkVideoError,
    create_video_generation_task,
    get_video_generation_task,
)

__all__ = [
    "embeddings",
    "images",
    "text",
    "videos",
    "ArkEmbeddingError",
    "ArkImageError",
    "ArkMultimodalError",
    "ArkVideoError",
    "create_multimodal_embedding",
    "create_image_generation",
    "create_seedream_4_5_image_generation",
    "create_seedream_5_0_lite_image_generation",
    "create_responses",
    "create_video_generation_task",
    "get_video_generation_task",
]
