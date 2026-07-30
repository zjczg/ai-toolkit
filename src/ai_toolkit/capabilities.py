"""Capability registry for the platform-scoped ai_toolkit SDK."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_PROVIDERS: dict[str, dict[str, Any]] = {
    "ark": {
        "name": "ARK / Doubao",
        "tools": [
            "ark.images.generate",
            "ark.videos.generate",
            "ark.text.complete",
            "ark.text.complete_json",
            "ark.embeddings.generate",
        ],
        "env": ["ARK_API_KEY"],
    },
    "dashscope": {
        "name": "DashScope / Wan / Qwen",
        "tools": ["dashscope.images.generate", "dashscope.speech.synthesize"],
        "env": ["DASHSCOPE_API_KEY"],
    },
    "gemini": {
        "name": "Gemini",
        "tools": ["gemini.images.generate"],
        "env": ["GEMINI_API_KEY"],
    },
    "deepseek": {
        "name": "DeepSeek",
        "tools": ["deepseek.text.complete", "deepseek.text.complete_json"],
        "env": ["DEEPSEEK_API_KEY"],
    },
    "aliyun": {
        "name": "Aliyun ImageSeg",
        "tools": ["aliyun.images.segment_commodity", "aliyun.images.segment_hd_body"],
        "env": ["ALIYUN_ACCESS_KEY_ID", "ALIYUN_ACCESS_KEY_SECRET"],
    },
}

_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "ark.images.generate": {
        "description": "Generate images with ARK Seedream models.",
        "models": ["seedream-5-lite", "seedream-4.5"],
        "reference_mode": "public_url_upload_for_local_files",
        "default_params": {
            "size": "2K",
            "response_format": "url",
            "sequential_image_generation": "disabled",
            "stream": False,
            "watermark": False,
        },
    },
    "ark.videos.generate": {
        "description": "Submit and optionally wait for ARK Seedance video generation.",
        "models": ["seedance-2"],
        "default_params": {
            "ratio": "1:1",
            "duration": 5,
            "resolution": "720p",
            "watermark": False,
            "generate_audio": False,
        },
    },
    "ark.text.complete": {
        "description": "Generate text or multimodal text with ARK Responses API.",
        "models": ["doubao-pro"],
        "default_params": {},
    },
    "ark.text.complete_json": {
        "description": "Generate structured JSON with ARK Responses API.",
        "models": ["doubao-pro"],
        "default_params": {
            "thinking": {"type": "disabled"},
            "text": {"format": {"type": "json_object"}},
        },
    },
    "ark.embeddings.generate": {
        "description": "Generate ARK multimodal embeddings.",
        "models": ["doubao-vision"],
        "default_params": {"dimensions": None},
    },
    "dashscope.images.generate": {
        "description": "Generate images with DashScope Wan models.",
        "models": ["wan2.7", "wan2.7-pro"],
        "reference_mode": "public_url_upload_for_local_files",
        "default_params": {"size": "2K", "watermark": False},
    },
    "dashscope.speech.synthesize": {
        "description": "Synthesize speech with DashScope Qwen-Audio-TTS models.",
        "models": ["qwen-audio-tts-plus", "qwen-audio-tts-flash"],
        "default_params": {
            "format": "wav",
            "sample_rate": 24000,
            "voice": "longanlingxin",
        },
    },
    "gemini.images.generate": {
        "description": "Generate images with Gemini image models.",
        "models": ["gemini-image", "gemini-image-lite", "gemini-image-pro"],
        "reference_mode": "inline_base64_for_local_files",
        "default_params": {"image_size": "1K"},
    },
    "deepseek.text.complete": {
        "description": "Generate text with DeepSeek.",
        "models": ["v4-flash", "v4-pro"],
        "default_params": {},
    },
    "deepseek.text.complete_json": {
        "description": "Generate structured JSON with DeepSeek.",
        "models": ["v4-flash", "v4-pro"],
        "default_params": {
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
        },
    },
    "aliyun.images.segment_commodity": {
        "description": "Segment product or clothing foregrounds with Aliyun ImageSeg SegmentCommodity.",
        "models": ["viapi-segment-commodity"],
        "reference_mode": "viapi_temp_oss_upload_for_local_files",
        "default_params": {"region": "cn-shanghai", "long_side_max": 1920},
        "constraints": {
            "input": "local image path",
            "output": "transparent image saved to output_path",
        },
    },
    "aliyun.images.segment_hd_body": {
        "description": "Segment human foregrounds with Aliyun ImageSeg SegmentHDBody.",
        "models": ["viapi-segment-hd-body"],
        "reference_mode": "viapi_temp_oss_upload_for_local_files",
        "default_params": {"region": "cn-shanghai", "long_side_max": 1920},
        "constraints": {
            "input": "local image path",
            "output": "transparent image saved to output_path",
        },
    },
}

_IMAGE_MODELS: dict[str, dict[str, Any]] = {
    "seedream-5-lite": {
        "provider": "ark",
        "tool": "ark.images.generate",
        "model": "doubao-seedream-5-0-260128",
        "aliases": ["seedream-5.0-lite", "doubao-5.0-lite"],
        "default_params": {
            "size": "2K",
            "output_format": "png",
            "response_format": "url",
            "sequential_image_generation": "disabled",
            "stream": False,
            "watermark": False,
        },
        "constraints": {
            "supported_size_values": ["2K", "3K", "4K", "<width>x<height>"],
            "supported_output_formats": ["png", "jpeg"],
            "max_reference_images": 14,
            "fixed_pixel_export_requires_postprocess": True,
        },
        "last_smoke_tested": "2026-06-29",
    },
    "seedream-4.5": {
        "provider": "ark",
        "tool": "ark.images.generate",
        "model": "doubao-seedream-4-5-251128",
        "aliases": ["doubao-4.5"],
        "default_params": {
            "size": "2K",
            "response_format": "url",
            "sequential_image_generation": "disabled",
            "stream": False,
            "watermark": False,
        },
        "constraints": {
            "supported_size_values": ["2K", "4K", "<width>x<height>"],
            "output_format_configurable": False,
            "native_output_format": "jpeg",
            "fixed_pixel_export_requires_postprocess": True,
        },
        "last_smoke_tested": "2026-06-29",
    },
    "wan2.7": {
        "provider": "dashscope",
        "tool": "dashscope.images.generate",
        "model": "wan2.7-image",
        "aliases": ["wan2.7-image"],
        "default_params": {"size": "2K", "watermark": False},
        "constraints": {
            "supported_size_values": ["1K", "2K", "<width>*<height>"],
            "size_separator": "*",
            "max_reference_images": 9,
            "fixed_pixel_export_requires_postprocess": True,
        },
        "last_smoke_tested": "2026-06-29",
    },
    "wan2.7-pro": {
        "provider": "dashscope",
        "tool": "dashscope.images.generate",
        "model": "wan2.7-image-pro",
        "aliases": ["wan2.7-image-pro"],
        "default_params": {"size": "2K", "watermark": False},
        "constraints": {
            "supported_size_values": ["1K", "2K", "4K", "<width>*<height>"],
            "size_separator": "*",
            "max_reference_images": 9,
            "text_to_image_max_size": "4K",
            "image_reference_max_size": "2K",
            "fixed_pixel_export_requires_postprocess": True,
        },
        "last_smoke_tested": "2026-06-29",
    },
    "gemini-image": {
        "provider": "gemini",
        "tool": "gemini.images.generate",
        "model": "gemini-3.1-flash-image",
        "aliases": ["nano-banana-2", "gemini-3.1-flash-image"],
        "default_params": {"image_size": "1K"},
        "constraints": {
            "supported_image_size_values": ["512", "1K", "2K", "4K"],
            "supported_aspect_ratios": [
                "1:1",
                "1:4",
                "4:1",
                "1:8",
                "8:1",
                "2:3",
                "3:2",
                "3:4",
                "4:3",
                "4:5",
                "5:4",
                "9:16",
                "16:9",
                "21:9",
            ],
            "fixed_pixel_export_requires_postprocess": True,
        },
    },
    "gemini-image-lite": {
        "provider": "gemini",
        "tool": "gemini.images.generate",
        "model": "gemini-3.1-flash-lite-image",
        "aliases": ["nano-banana-2-lite", "gemini-3.1-flash-lite-image"],
        "default_params": {"image_size": "1K"},
        "constraints": {
            "supported_image_size_values": ["1K"],
            "supported_aspect_ratios": [
                "1:1",
                "2:3",
                "3:2",
                "3:4",
                "4:3",
                "4:5",
                "5:4",
                "9:16",
                "16:9",
                "21:9",
            ],
            "fixed_pixel_export_requires_postprocess": True,
        },
    },
    "gemini-image-pro": {
        "provider": "gemini",
        "tool": "gemini.images.generate",
        "model": "gemini-3-pro-image",
        "aliases": ["nano-banana-pro", "gemini-3-pro-image"],
        "default_params": {"image_size": "1K"},
        "constraints": {
            "supported_image_size_values": ["1K", "2K", "4K"],
            "supported_aspect_ratios": [
                "1:1",
                "2:3",
                "3:2",
                "3:4",
                "4:3",
                "4:5",
                "5:4",
                "9:16",
                "16:9",
                "21:9",
            ],
            "fixed_pixel_export_requires_postprocess": True,
        },
    },
}

_VIDEO_MODELS: dict[str, dict[str, Any]] = {
    "seedance-2": {
        "provider": "ark",
        "tool": "ark.videos.generate",
        "model": "doubao-seedance-2-0-260128",
        "aliases": ["seedance-2.0", "doubao-seedance-2-0-260128"],
        "default_params": {
            "ratio": "1:1",
            "duration": 5,
            "resolution": "720p",
            "watermark": False,
            "generate_audio": False,
        },
        "constraints": {
            "async_task": True,
            "supported_ratio_values": ["1:1", "4:3", "3:4", "16:9", "9:16"],
            "recommended_duration_values": [5, 8, 11],
            "recommended_resolution_values": ["720p", "1080p"],
        },
        "last_smoke_tested": "2026-06-29",
    },
}

_TEXT_MODELS: dict[str, dict[str, Any]] = {
    "doubao-pro": {
        "provider": "ark",
        "tool": "ark.text.complete",
        "model": "doubao-seed-2-0-pro-260215",
        "json_tool": "ark.text.complete_json",
        "default_json_thinking": {"type": "disabled"},
        "last_smoke_tested": "2026-06-29",
    },
    "v4-flash": {
        "provider": "deepseek",
        "tool": "deepseek.text.complete",
        "model": "deepseek-v4-flash",
        "json_tool": "deepseek.text.complete_json",
        "default_json_thinking": {"type": "disabled"},
        "last_smoke_tested": "2026-06-29",
    },
    "v4-pro": {
        "provider": "deepseek",
        "tool": "deepseek.text.complete",
        "model": "deepseek-v4-pro",
        "json_tool": "deepseek.text.complete_json",
        "supports_reasoning_text": True,
        "thinking_values": [
            {"type": "disabled"},
            {"type": "enabled", "reasoning_effort": "high"},
            {"type": "enabled", "reasoning_effort": "max"},
        ],
        "default_json_thinking": {"type": "disabled"},
        "last_smoke_tested": "2026-06-29",
    },
}

_EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    "doubao-vision": {
        "provider": "ark",
        "tool": "ark.embeddings.generate",
        "model": "doubao-embedding-vision-251215",
        "default_dimensions": 2048,
        "tested_dimensions": [1024, 2048],
        "last_smoke_tested": "2026-06-29",
    },
}

_AUDIO_MODELS: dict[str, dict[str, Any]] = {
    "qwen-audio-tts-plus": {
        "provider": "dashscope",
        "tool": "dashscope.speech.synthesize",
        "model": "qwen-audio-3.0-tts-plus",
        "aliases": ["qwen-audio-3.0-tts-plus"],
        "default_params": {
            "format": "wav",
            "sample_rate": 24000,
            "voice": "longanlingxin",
        },
        "constraints": {
            "supported_formats": ["mp3", "opus", "pcm", "wav"],
            "supported_sample_rates": [8000, 16000, 22050, 24000, 44100, 48000],
            "system_voices": ["longanlingxin", "longanlufeng"],
            "instruction_supported": True,
        },
        "last_smoke_tested": "2026-07-18",
    },
    "qwen-audio-tts-flash": {
        "provider": "dashscope",
        "tool": "dashscope.speech.synthesize",
        "model": "qwen-audio-3.0-tts-flash",
        "aliases": ["qwen-audio-3.0-tts-flash"],
        "default_params": {
            "format": "wav",
            "sample_rate": 24000,
            "voice": "longanhuan_v3.6",
        },
        "constraints": {
            "supported_formats": ["mp3", "opus", "pcm", "wav"],
            "supported_sample_rates": [8000, 16000, 22050, 24000, 44100, 48000],
            "system_voices": [
                "longanhuan_v3.6",
                "longjielidou_v3.6",
                "loongeva_v3.6",
                "loongjohn",
            ],
            "instruction_supported": True,
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


def get_default_params(tool: str) -> dict[str, Any]:
    return deepcopy(get_tool_spec(tool).get("default_params", {}))


def list_image_models() -> dict[str, dict[str, Any]]:
    return deepcopy(_IMAGE_MODELS)


def get_image_model(model: str) -> dict[str, Any]:
    return _get_model(model, _IMAGE_MODELS, "image")


def list_video_models() -> dict[str, dict[str, Any]]:
    return deepcopy(_VIDEO_MODELS)


def get_video_model(model: str) -> dict[str, Any]:
    return _get_model(model, _VIDEO_MODELS, "video")


def list_text_models() -> dict[str, dict[str, Any]]:
    return deepcopy(_TEXT_MODELS)


def get_text_model(model: str) -> dict[str, Any]:
    return _get_model(model, _TEXT_MODELS, "text")


def list_embedding_models() -> dict[str, dict[str, Any]]:
    return deepcopy(_EMBEDDING_MODELS)


def get_embedding_model(model: str) -> dict[str, Any]:
    return _get_model(model, _EMBEDDING_MODELS, "embedding")


def list_audio_models() -> dict[str, dict[str, Any]]:
    return deepcopy(_AUDIO_MODELS)


def get_audio_model(model: str) -> dict[str, Any]:
    return _get_model(model, _AUDIO_MODELS, "audio")


def normalize_image_options(model: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    config = get_image_model(model)
    params = deepcopy(config.get("default_params", {}))
    params.update(_non_null_options(options))
    constraints = config.get("constraints", {})

    if "output_format" in params and "supported_output_formats" in constraints:
        supported = constraints["supported_output_formats"]
        value = str(params.get("output_format") or "").lower()
        if value == "jpg":
            value = "jpeg"
        if value not in supported:
            value = supported[0]
        params["output_format"] = value

    if constraints.get("output_format_configurable") is False:
        params.pop("output_format", None)

    return params


def normalize_video_options(model: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    config = get_video_model(model)
    params = deepcopy(config.get("default_params", {}))
    params.update(_non_null_options(options))
    constraints = config.get("constraints", {})

    ratios = constraints.get("supported_ratio_values") or []
    if ratios and params.get("ratio") not in ratios:
        params["ratio"] = config["default_params"]["ratio"]

    durations = constraints.get("recommended_duration_values") or []
    if durations:
        params["duration"] = _coerce_int(params.get("duration"), config["default_params"]["duration"])

    resolutions = constraints.get("recommended_resolution_values") or []
    if resolutions and params.get("resolution") not in resolutions:
        params["resolution"] = config["default_params"]["resolution"]

    params["watermark"] = _coerce_bool(params.get("watermark"), False)
    params["generate_audio"] = _coerce_bool(params.get("generate_audio"), False)
    return params


def _get_model(model: str, registry: dict[str, dict[str, Any]], kind: str) -> dict[str, Any]:
    if model in registry:
        return deepcopy(registry[model])
    for key, config in registry.items():
        aliases = set(config.get("aliases", []))
        aliases.add(config.get("model", ""))
        if model in aliases:
            resolved = deepcopy(config)
            resolved["id"] = key
            return resolved
    raise KeyError(f"unknown {kind} model: {model}")


def _non_null_options(options: dict[str, Any] | None) -> dict[str, Any]:
    return {key: value for key, value in (options or {}).items() if value is not None}


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)
