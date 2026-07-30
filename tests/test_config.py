from __future__ import annotations

import os
from pathlib import Path

import pytest

from ai_toolkit import config
from ai_toolkit.config import get_settings

CONFIG_KEYS = (
    "AI_TOOLKIT_ENV_FILE",
    "ARK_API_KEY",
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
    "AI_TOOLKIT_HTTP_MAX_RETRIES",
    "XDG_CONFIG_HOME",
)


def _write_env(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    for name in CONFIG_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    config._env_file_sources.cache_clear()
    yield
    config._env_file_sources.cache_clear()


def test_process_environment_wins_over_project_and_user(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_env(Path.cwd() / ".env", "ARK_API_KEY=project\n")
    _write_env(
        tmp_path / "config" / "ai-toolkit" / "env",
        "ARK_API_KEY=user\n",
    )
    monkeypatch.setenv("ARK_API_KEY", "process")

    assert get_settings().ark_api_key == "process"


def test_project_values_win_and_missing_keys_fall_back_to_user(tmp_path: Path) -> None:
    _write_env(Path.cwd() / ".env", "ARK_API_KEY=project\n")
    _write_env(
        tmp_path / "config" / "ai-toolkit" / "env",
        "ARK_API_KEY=user\nDEEPSEEK_API_KEY=user-deepseek\n",
    )

    settings = get_settings()

    assert settings.ark_api_key == "project"
    assert settings.deepseek_api_key == "user-deepseek"


def test_explicit_empty_values_block_lower_priority_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_env(
        Path.cwd() / ".env",
        "ARK_API_KEY=\nDEEPSEEK_API_KEY=project\n",
    )
    _write_env(
        tmp_path / "config" / "ai-toolkit" / "env",
        "ARK_API_KEY=user\nDEEPSEEK_API_KEY=user-deepseek\n",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")

    settings = get_settings()

    assert settings.ark_api_key == ""
    assert settings.deepseek_api_key == ""


def test_user_fallback_uses_xdg_path_without_mutating_environ(tmp_path: Path) -> None:
    _write_env(
        tmp_path / "config" / "ai-toolkit" / "env",
        "GEMINI_API_KEY=user-gemini\n",
    )

    assert "GEMINI_API_KEY" not in os.environ
    assert get_settings().gemini_api_key == "user-gemini"
    assert "GEMINI_API_KEY" not in os.environ


def test_explicit_user_file_path_is_supported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    custom_env = tmp_path / "credentials" / "ai.env"
    _write_env(custom_env, "ARK_API_KEY=custom-user\n")
    monkeypatch.setenv("AI_TOOLKIT_ENV_FILE", str(custom_env))

    assert get_settings().ark_api_key == "custom-user"


def test_missing_files_do_not_search_parents_and_invalid_numbers_use_default(
    tmp_path: Path,
) -> None:
    _write_env(tmp_path / ".env", "ARK_API_KEY=parent\n")
    _write_env(
        Path.cwd() / ".env",
        "AI_TOOLKIT_HTTP_MAX_RETRIES=invalid\n",
    )
    _write_env(
        tmp_path / "config" / "ai-toolkit" / "env",
        "AI_TOOLKIT_HTTP_MAX_RETRIES=9\n",
    )

    settings = get_settings()

    assert settings.ark_api_key == ""
    assert settings.http_max_retries == 2
    assert settings.ark_image_size == "2K"
