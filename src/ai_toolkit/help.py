"""Small runtime help for ai_toolkit platform capabilities."""

from __future__ import annotations

from typing import Any

from ai_toolkit import capabilities


def list_interfaces() -> list[str]:
    return capabilities.list_tools()


def help(name: str | None = None) -> dict[str, Any]:
    if name is None:
        return {
            "version": "0.9.0",
            "tools": capabilities.list_tools(),
            "providers": capabilities.list_providers(),
        }
    return capabilities.get_tool_spec(name)
