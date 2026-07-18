"""DashScope non-real-time speech synthesis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_toolkit._transport import post_json
from ai_toolkit.config import get_settings
from ai_toolkit.types import AIToolkitError, GeneratedAudio, SpeechSynthesisResult

_SPEECH_SYNTHESIS_PATH = "/api/v1/services/audio/tts/SpeechSynthesizer"

MODEL_ALIASES = {
    "qwen-audio-tts-plus": "qwen-audio-3.0-tts-plus",
    "qwen-audio-3.0-tts-plus": "qwen-audio-3.0-tts-plus",
    "qwen-audio-tts-flash": "qwen-audio-3.0-tts-flash",
    "qwen-audio-3.0-tts-flash": "qwen-audio-3.0-tts-flash",
}

_MIME_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "pcm": "audio/pcm",
    "wav": "audio/wav",
}


class DashScopeSpeechError(RuntimeError):
    """Raised when the DashScope speech synthesis API request fails."""


def synthesize(
    *,
    text: str,
    voice: str | None = None,
    output_path: str | Path | None = None,
    model: str | None = None,
    format: str = "wav",
    sample_rate: int = 24_000,
    instruction: str | None = None,
    language_hints: list[str] | None = None,
    rate: float | None = None,
    volume: int | None = None,
    pitch: float | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> SpeechSynthesisResult:
    """Synthesize speech and optionally download the generated audio."""
    settings = get_settings()
    resolved_model = resolve_model(model)
    resolved_voice = str(voice or settings.dashscope_speech_voice).strip()
    normalized_text = str(text).strip()
    normalized_format = str(format).strip().lower()

    if not normalized_text:
        raise AIToolkitError("speech synthesis text must not be empty")
    if not resolved_voice:
        raise AIToolkitError("speech synthesis voice must not be empty")
    if normalized_format not in _MIME_TYPES:
        raise AIToolkitError(
            f"unsupported speech format: {normalized_format}; "
            f"expected one of {', '.join(sorted(_MIME_TYPES))}"
        )

    input_options = _drop_none_values(
        {
            "text": normalized_text,
            "voice": resolved_voice,
            "format": normalized_format,
            "sample_rate": sample_rate,
            "instruction": instruction,
            "language_hints": language_hints,
            "rate": rate,
            "volume": volume,
            "pitch": pitch,
            "seed": seed,
            **kwargs,
        }
    )
    raw_response = create_speech_synthesis(resolved_model, **input_options)
    result = SpeechSynthesisResult(
        provider="dashscope",
        model=resolved_model,
        text=normalized_text,
        voice=resolved_voice,
        audio=_extract_audio(raw_response, normalized_format),
        raw_response=raw_response,
        request={"input": input_options},
        usage=_response_usage(raw_response),
    )
    if output_path is not None:
        result.save(output_path)
    return result


def create_speech_synthesis(
    model: str,
    *,
    text: str,
    voice: str,
    **input_options: Any,
) -> dict[str, Any]:
    """Submit a synchronous speech synthesis request and return the raw response."""
    settings = get_settings()
    if not settings.dashscope_api_key:
        raise DashScopeSpeechError("DASHSCOPE_API_KEY is not configured")

    payload = {
        "model": model,
        "input": {
            "text": text,
            "voice": voice,
            **_drop_none_values(input_options),
        },
    }
    api_url = f"{settings.dashscope_base_url.rstrip('/')}{_SPEECH_SYNTHESIS_PATH}"
    return post_json(
        api_url,
        payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.dashscope_api_key}",
        },
        error_cls=DashScopeSpeechError,
    )


def resolve_model(model: str | None = None) -> str:
    settings = get_settings()
    if model is None or not str(model).strip():
        return settings.dashscope_speech_model
    value = str(model).strip()
    return MODEL_ALIASES.get(value, MODEL_ALIASES.get(value.lower(), value))


def _extract_audio(response: dict[str, Any], format: str) -> GeneratedAudio:
    output = response.get("output", {}) if isinstance(response, dict) else {}
    audio = output.get("audio", {}) if isinstance(output, dict) else {}
    if not isinstance(audio, dict):
        raise AIToolkitError("DashScope speech synthesis returned invalid audio metadata")

    url = audio.get("url")
    data = audio.get("data")
    if not isinstance(url, str):
        url = None
    if not isinstance(data, str):
        data = None
    if not url and not data:
        raise AIToolkitError("DashScope speech synthesis returned no audio")

    expires_at = audio.get("expires_at")
    return GeneratedAudio(
        url=url,
        b64_data=data,
        mime_type=_MIME_TYPES[format],
        audio_id=str(audio.get("id") or ""),
        expires_at=expires_at if isinstance(expires_at, int) else None,
    )


def _response_usage(response: dict[str, Any]) -> dict[str, Any] | None:
    usage = response.get("usage")
    return usage if isinstance(usage, dict) else None


def _drop_none_values(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}
