"""Configuration loaded from process, project, or user environment files."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values

EnvValues = dict[str, str]


def _read_env_file(path: Path | None) -> EnvValues:
    if path is None or not path.is_file():
        return {}
    return {name: value if value is not None else "" for name, value in dotenv_values(path).items()}


def _user_env_path() -> Path | None:
    if "AI_TOOLKIT_ENV_FILE" in os.environ:
        configured_path = os.environ["AI_TOOLKIT_ENV_FILE"].strip()
        return Path(configured_path).expanduser() if configured_path else None

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME", "").strip()
    config_home = Path(xdg_config_home).expanduser() if xdg_config_home else Path.home() / ".config"
    return config_home / "ai-toolkit" / "env"


@lru_cache(maxsize=8)
def _env_file_sources(
    project_env_path: Path,
    user_env_path: Path | None,
) -> tuple[EnvValues, EnvValues]:
    return _read_env_file(project_env_path), _read_env_file(user_env_path)


def _env(name: str, default: str = "") -> str:
    if name in os.environ:
        return os.environ[name]

    project_values, user_values = _env_file_sources(
        Path.cwd() / ".env",
        _user_env_path(),
    )
    if name in project_values:
        return project_values[name]
    if name in user_values:
        return user_values[name]
    return default


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(slots=True)
class Settings:
    ark_api_key: str = field(default_factory=lambda: _env("ARK_API_KEY"))
    ark_base_url: str = field(
        default_factory=lambda: _env("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    )
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    gemini_base_url: str = field(
        default_factory=lambda: _env(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
        )
    )
    ark_image_model: str = field(
        default_factory=lambda: _env("ARK_IMAGE_MODEL", "doubao-seedream-5-0-260128")
    )
    ark_image_size: str = field(default_factory=lambda: _env("ARK_IMAGE_SIZE", "2K"))
    gemini_image_model: str = field(
        default_factory=lambda: _env("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
    )
    gemini_image_size: str = field(default_factory=lambda: _env("GEMINI_IMAGE_SIZE", "1K"))
    grsai_api_key: str = field(default_factory=lambda: _env("GRSAI_API_KEY"))
    grsai_base_url: str = field(
        default_factory=lambda: _env("GRSAI_BASE_URL", "https://grsaiapi.com")
    )
    grsai_image_model: str = field(
        default_factory=lambda: _env("GRSAI_IMAGE_MODEL", "nano-banana-2")
    )
    grsai_image_size: str = field(
        default_factory=lambda: _env("GRSAI_IMAGE_SIZE", "1K")
    )
    minimax_api_key: str = field(default_factory=lambda: _env("MINIMAX_API_KEY"))
    minimax_base_url: str = field(
        default_factory=lambda: _env("MINIMAX_BASE_URL", "https://api.minimax.io")
    )
    minimax_video_model: str = field(
        default_factory=lambda: _env("MINIMAX_VIDEO_MODEL", "MiniMax-H3")
    )
    dashscope_api_key: str = field(default_factory=lambda: _env("DASHSCOPE_API_KEY"))
    dashscope_base_url: str = field(
        default_factory=lambda: _env("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com")
    )
    dashscope_image_model: str = field(
        default_factory=lambda: _env("DASHSCOPE_IMAGE_MODEL", "wan2.7-image")
    )
    dashscope_image_size: str = field(default_factory=lambda: _env("DASHSCOPE_IMAGE_SIZE", "2K"))
    dashscope_video_model: str = field(
        default_factory=lambda: _env("DASHSCOPE_VIDEO_MODEL", "happyhorse-1.1-r2v")
    )
    dashscope_video_ratio: str = field(
        default_factory=lambda: _env("DASHSCOPE_VIDEO_RATIO", "1:1")
    )
    dashscope_video_duration: int = field(
        default_factory=lambda: _env_int("DASHSCOPE_VIDEO_DURATION", 5)
    )
    dashscope_video_resolution: str = field(
        default_factory=lambda: _env("DASHSCOPE_VIDEO_RESOLUTION", "720P")
    )
    dashscope_speech_model: str = field(
        default_factory=lambda: _env(
            "DASHSCOPE_SPEECH_MODEL",
            "qwen-audio-3.0-tts-plus",
        )
    )
    dashscope_speech_voice: str = field(
        default_factory=lambda: _env("DASHSCOPE_SPEECH_VOICE", "longanlingxin")
    )
    ark_response_model: str = field(
        default_factory=lambda: _env("ARK_RESPONSE_MODEL", "doubao-seed-2-0-pro-260215")
    )
    ark_embedding_model: str = field(
        default_factory=lambda: _env("ARK_EMBEDDING_MODEL", "doubao-embedding-vision-251215")
    )
    ark_video_model: str = field(
        default_factory=lambda: _env("ARK_VIDEO_MODEL", "doubao-seedance-2-0-260128")
    )
    deepseek_api_key: str = field(default_factory=lambda: _env("DEEPSEEK_API_KEY"))
    deepseek_base_url: str = field(
        default_factory=lambda: _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    deepseek_chat_model: str = field(
        default_factory=lambda: _env("DEEPSEEK_CHAT_MODEL", "deepseek-v4-flash")
    )
    aliyun_access_key_id: str = field(default_factory=lambda: _env("ALIYUN_ACCESS_KEY_ID"))
    aliyun_access_key_secret: str = field(default_factory=lambda: _env("ALIYUN_ACCESS_KEY_SECRET"))
    aliyun_viapi_region: str = field(
        default_factory=lambda: _env("ALIYUN_VIAPI_REGION", "cn-shanghai")
    )
    upload_ssh_target: str = field(default_factory=lambda: _env("UPLOAD_SSH_TARGET"))
    upload_identity_file: str = field(default_factory=lambda: _env("UPLOAD_IDENTITY_FILE"))
    upload_remote_dir: str = field(default_factory=lambda: _env("UPLOAD_REMOTE_DIR"))
    upload_public_base_url: str = field(default_factory=lambda: _env("UPLOAD_PUBLIC_BASE_URL"))
    http_max_retries: int = field(
        default_factory=lambda: _env_int("AI_TOOLKIT_HTTP_MAX_RETRIES", 2)
    )
    http_retry_base_seconds: float = field(
        default_factory=lambda: _env_float("AI_TOOLKIT_HTTP_RETRY_BASE_SECONDS", 0.5)
    )
    http_retry_max_seconds: float = field(
        default_factory=lambda: _env_float("AI_TOOLKIT_HTTP_RETRY_MAX_SECONDS", 8.0)
    )
    http_timeout_seconds: float = field(
        default_factory=lambda: _env_float("AI_TOOLKIT_HTTP_TIMEOUT_SECONDS", 120.0)
    )


def get_settings() -> Settings:
    """Return settings resolved from process, project, and user configuration."""
    return Settings()
