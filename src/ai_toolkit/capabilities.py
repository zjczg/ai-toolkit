"""Capability registry for ai_toolkit high-level tools."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


_PROVIDERS: dict[str, dict[str, Any]] = {
    "ark": {
        "name": "ARK / Doubao",
        "tools": ["images.generate", "chat.complete"],
        "env": ["ARK_API_KEY"],
    },
    "gemini": {
        "name": "Gemini",
        "tools": ["images.generate"],
        "env": ["GEMINI_API_KEY"],
    },
    "deepseek": {
        "name": "DeepSeek",
        "tools": ["chat.complete"],
        "env": ["DEEPSEEK_API_KEY"],
    },
    "media": {
        "name": "Media Upload",
        "tools": ["media.upload_public_url"],
        "env": ["UPLOAD_SSH_TARGET", "UPLOAD_REMOTE_DIR", "UPLOAD_PUBLIC_BASE_URL"],
    },
}

_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "images.generate": {
        "description": "Generate an image from text and optional references.",
        "providers": {
            "ark": {
                "default_model_env": "ARK_IMAGE_MODEL",
                "default_model": "doubao-seedream-4-5-251128",
                "reference_mode": "public_url_upload_for_local_files",
                "default_params": {
                    "size": "2K",
                    "response_format": "url",
                    "sequential_image_generation": "disabled",
                    "stream": False,
                    "watermark": False,
                },
            },
            "gemini": {
                "default_model_env": "GEMINI_IMAGE_MODEL",
                "default_model": "gemini-3.1-flash-image-preview",
                "reference_mode": "inline_base64_for_local_files",
                "default_params": {
                    "generationConfig": {
                        "responseModalities": ["TEXT", "IMAGE"],
                        "imageConfig": {"imageSize": "1K"},
                    }
                },
            },
        },
    },
    "chat.complete": {
        "description": "Generate text from messages or a prompt.",
        "providers": {
            "deepseek": {
                "default_model_env": "DEEPSEEK_CHAT_MODEL",
                "default_model": "deepseek-v4-flash",
                "default_params": {},
            },
            "ark": {
                "default_model_env": "ARK_RESPONSE_MODEL",
                "default_model": "doubao-seed-2-0-pro-260215",
                "default_params": {},
            },
        },
    },
}


def list_providers() -> dict[str, dict[str, Any]]:
    return deepcopy(_PROVIDERS)


def list_tools() -> list[str]:
    return sorted(_TOOL_SPECS)


def get_tool_spec(tool: str) -> dict[str, Any]:
    if tool not in _TOOL_SPECS:
        raise KeyError(f"unknown tool: {tool}")
    return deepcopy(_TOOL_SPECS[tool])


def get_default_params(provider: str, tool: str) -> dict[str, Any]:
    spec = get_tool_spec(tool)
    provider_spec = spec.get("providers", {}).get(provider)
    if provider_spec is None:
        raise KeyError(f"provider {provider!r} does not support {tool!r}")
    return deepcopy(provider_spec.get("default_params", {}))
