"""DeepSeek platform capabilities."""

from ai_toolkit.deepseek import text
from ai_toolkit.deepseek.chat import DeepSeekError, create_chat_completion

__all__ = [
    "text",
    "DeepSeekError",
    "create_chat_completion",
]
