from __future__ import annotations

from ai_toolkit import __version__
from ai_toolkit import capabilities
from ai_toolkit.chat import normalize_provider as normalize_chat_provider
from ai_toolkit.images import normalize_provider as normalize_image_provider


def test_version_is_updated():
    assert __version__ == "0.2.0"


def test_provider_aliases():
    assert normalize_image_provider("doubao") == "ark"
    assert normalize_image_provider("google") == "gemini"
    assert normalize_chat_provider("doubao") == "ark"


def test_capabilities_include_current_tools():
    assert "images.generate" in capabilities.list_tools()
    assert "chat.complete" in capabilities.list_tools()
    assert "ark" in capabilities.get_tool_spec("images.generate")["providers"]
    assert "deepseek" in capabilities.get_tool_spec("chat.complete")["providers"]
