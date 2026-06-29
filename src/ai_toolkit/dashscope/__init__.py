"""DashScope platform capabilities."""

from ai_toolkit.dashscope import images
from ai_toolkit.dashscope.images import (
    DashScopeImageError,
    create_image_generation,
)

__all__ = [
    "images",
    "DashScopeImageError",
    "create_image_generation",
]
