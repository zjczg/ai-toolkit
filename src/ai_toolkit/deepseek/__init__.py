"""DeepSeek API wrappers."""

from ai_toolkit.deepseek.chat import DeepSeekError
from ai_toolkit.deepseek.chat import create_chat_completion

__all__ = [
    "DeepSeekError",
    "create_chat_completion",
]
