"""Configuration — reads from environment variables only, no .env file dependency."""

from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field


@dataclass(slots=True)
class Settings:
    ark_api_key: str = field(default_factory=lambda: os.getenv("ARK_API_KEY", ""))
    ark_base_url: str = field(
        default_factory=lambda: os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    )
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_base_url: str = field(
        default_factory=lambda: os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    )
    ark_image_model: str = field(
        default_factory=lambda: os.getenv("ARK_IMAGE_MODEL", "doubao-seedream-4-5-251128")
    )
    ark_image_size: str = field(default_factory=lambda: os.getenv("ARK_IMAGE_SIZE", "2K"))
    gemini_image_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image-preview")
    )
    gemini_image_size: str = field(default_factory=lambda: os.getenv("GEMINI_IMAGE_SIZE", "1K"))
    ark_response_model: str = field(
        default_factory=lambda: os.getenv("ARK_RESPONSE_MODEL", "doubao-seed-2-0-pro-260215")
    )
    ark_embedding_model: str = field(
        default_factory=lambda: os.getenv("ARK_EMBEDDING_MODEL", "doubao-embedding-vision-251215")
    )
    ark_video_model: str = field(
        default_factory=lambda: os.getenv("ARK_VIDEO_MODEL", "doubao-seedance-2-0-260128")
    )
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    deepseek_chat_model: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-v4-flash")
    )
    upload_ssh_target: str = field(default_factory=lambda: os.getenv("UPLOAD_SSH_TARGET", ""))
    upload_identity_file: str = field(default_factory=lambda: os.getenv("UPLOAD_IDENTITY_FILE", ""))
    upload_remote_dir: str = field(default_factory=lambda: os.getenv("UPLOAD_REMOTE_DIR", ""))
    upload_public_base_url: str = field(
        default_factory=lambda: os.getenv("UPLOAD_PUBLIC_BASE_URL", "")
    )


def get_settings() -> Settings:
    """Return settings loaded from environment variables."""
    return Settings()
