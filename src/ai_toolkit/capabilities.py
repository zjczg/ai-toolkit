"""Capability registry for ai_toolkit high-level tools."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_PROVIDERS: dict[str, dict[str, Any]] = {
    "ark": {
        "name": "ARK / Doubao",
        "tools": [
            "images.generate",
            "videos.generate",
            "chat.complete",
            "chat.complete_json",
            "embeddings.generate",
        ],
        "env": ["ARK_API_KEY"],
    },
    "gemini": {
        "name": "Gemini",
        "tools": ["images.generate"],
        "env": ["GEMINI_API_KEY"],
    },
    "dashscope": {
        "name": "万相 / 通义 (阿里百炼 DashScope)",
        "tools": ["images.generate"],
        "env": ["DASHSCOPE_API_KEY"],
    },
    "deepseek": {
        "name": "DeepSeek",
        "tools": ["chat.complete", "chat.complete_json"],
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
                "default_model": "doubao-seedream-5-0-260128",
                "reference_mode": "public_url_upload_for_local_files",
                "default_params": {
                    "size": "2K",
                    "output_format": "png",
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
            "dashscope": {
                "default_model_env": "DASHSCOPE_IMAGE_MODEL",
                "default_model": "wan2.7-image",
                "endpoint": "POST /api/v1/services/aigc/multimodal-generation/generation",
                "reference_mode": "public_url_upload_for_local_files",
                "default_params": {
                    "size": "2K",
                    "watermark": False,
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
    "chat.complete_json": {
        "description": "Generate structured JSON from messages or a prompt.",
        "providers": {
            "deepseek": {
                "default_model_env": "DEEPSEEK_CHAT_MODEL",
                "default_model": "deepseek-v4-flash",
                "default_params": {
                    "response_format": {"type": "json_object"},
                    "thinking": {"type": "disabled"},
                },
            },
            "ark": {
                "default_model_env": "ARK_RESPONSE_MODEL",
                "default_model": "doubao-seed-2-0-pro-260215",
                "endpoint": "POST /responses",
                "default_params": {
                    "thinking": {"type": "disabled"},
                    "text": {"format": {"type": "json_object"}},
                },
            },
        },
    },
    "embeddings.generate": {
        "description": "Generate multimodal embeddings from text and/or image references.",
        "providers": {
            "ark": {
                "default_model_env": "ARK_EMBEDDING_MODEL",
                "default_model": "doubao-embedding-vision-251215",
                "reference_mode": "public_url_upload_for_local_files",
                "default_params": {},
            },
        },
    },
    "videos.generate": {
        "description": "Submit and optionally poll an asynchronous video generation task.",
        "providers": {
            "ark": {
                "default_model_env": "ARK_VIDEO_MODEL",
                "default_model": "doubao-seedance-2-0-260128",
                "endpoint": "POST /contents/generations/tasks",
                "query_endpoint": "GET /contents/generations/tasks/{task_id}",
                "reference_mode": "public_url_upload_for_local_files",
                "default_params": {
                    "watermark": False,
                    "generate_audio": False,
                },
            },
        },
    },
}

_IMAGE_GENERATION_PATHS: dict[str, dict[str, Any]] = {
    "doubao-5.0-lite": {
        "label": "doubao-5.0-lite",
        "provider": "ark",
        "model": "doubao-seedream-5-0-260128",
        "default_params": {
            "size": "2K",
            "output_format": "png",
            "response_format": "url",
            "sequential_image_generation": "disabled",
            "stream": False,
            "watermark": False,
        },
        "constraints": {
            "generation_size_param": "size",
            "supported_size_values": ["2K", "3K", "4K", "<width>x<height>"],
            "default_size": "2K",
            "supports_arbitrary_small_size": False,
            "min_total_pixels": 3686400,
            "min_total_pixels_hint": "2560x1440",
            "max_total_pixels": 16777216,
            "max_total_pixels_hint": "4096x4096",
            "aspect_ratio_range": "[1/16, 16]",
            "fixed_pixel_export_requires_postprocess": True,
            "output_format_param": "output_format",
            "supported_output_formats": ["png", "jpeg"],
            "default_output_format": "png",
            "max_reference_images": 14,
            "supported_input_formats": [
                "jpeg",
                "png",
                "webp",
                "bmp",
                "tiff",
                "gif",
                "heic",
                "heif",
            ],
            "supports_web_search_tool": True,
            "sequential_max_images": 15,
        },
        "notes": [
            "Seedream 5.0 lite uses the same ARK /images/generations endpoint as 4.5 and supports text, single-image, and multi-image generation.",
            "Use output_format=png or jpeg when the caller needs the generated file format to match a local suffix.",
            "The web_search tool is available but should be enabled only for time-sensitive visual content.",
            "For fixed small final assets, generate within documented pixel limits and post-process deterministically.",
        ],
        "source_urls": [
            "https://www.volcengine.com/docs/6791/1541523",
            "https://www.volcengine.com/docs/82379/1824692",
        ],
        "last_verified": "2026-06-08",
    },
    "doubao-4.5": {
        "label": "doubao-4.5",
        "provider": "ark",
        "model": "doubao-seedream-4-5-251128",
        "default_params": {
            "size": "2048x2048",
            "response_format": "url",
            "sequential_image_generation": "disabled",
            "stream": False,
            "watermark": False,
        },
        "constraints": {
            "generation_size_param": "size",
            "supported_size_values": ["2K", "4K", "<width>x<height>"],
            "default_size": "2048x2048",
            "supports_arbitrary_small_size": False,
            "min_total_pixels": 3686400,
            "min_total_pixels_hint": "2560x1440",
            "max_total_pixels_hint": "4096x4096",
            "aspect_ratio_range": "[1/16, 16]",
            "known_invalid_size_examples": ["1024x256", "2048x512"],
            "fixed_pixel_export_requires_postprocess": True,
            "output_format": "jpeg",
            "output_format_configurable": False,
        },
        "notes": [
            "Use the size parameter for generation resolution or aspect-ratio-derived dimensions.",
            "Custom WxH must meet Seedream 4.5 pixel limits; small sprite strips such as 1024x256 should be generated larger and post-processed.",
            "Small final assets such as 200x200 should be produced by deterministic post-processing.",
            "Seedream 4.5 returns jpeg images by default and does not support output_format override.",
        ],
        "source_urls": [
            "https://www.volcengine.com/docs/6492/2172373?lang=zh",
            "https://www.volcengine.com/docs/6492/2221472?lang=zh",
        ],
        "last_verified": "2026-05-20",
    },
    "wan2.7-image": {
        "label": "wan2.7-image (万相)",
        "provider": "dashscope",
        "model": "wan2.7-image",
        "default_params": {
            "size": "2K",
            "watermark": False,
        },
        "constraints": {
            "generation_size_param": "size",
            "supported_size_values": ["1K", "2K", "<width>*<height>"],
            "default_size": "2K",
            "supports_arbitrary_small_size": False,
            "max_total_pixels_hint": "2048x2048",
            "aspect_ratio_range": "[1/16, 16]",
            "size_separator": "*",
            "fixed_pixel_export_requires_postprocess": True,
            "output_format_configurable": False,
            "max_reference_images": 5,
            "supported_input_formats": ["jpeg", "png", "webp", "bmp", "tiff", "heic"],
            "sync_call": True,
        },
        "notes": [
            "万相 wan2.7-image runs on Alibaba Bailian (DashScope), a different platform/API key than ARK/Doubao — set DASHSCOPE_API_KEY.",
            "Uses the synchronous multimodal-generation endpoint; no task polling required.",
            "DashScope size uses '*' as the separator (e.g. 1024*1024), not 'x'.",
            "Local reference images are uploaded to public URLs first and sent as image content parts.",
            "Output image URLs returned by DashScope expire after 24 hours; save them promptly.",
        ],
        "source_urls": [
            "https://help.aliyun.com/zh/model-studio/wan-image-generation-and-editing-api-reference",
            "https://help.aliyun.com/zh/model-studio/text-to-image",
        ],
        "last_verified": "2026-06-22",
    },
    "wan2.7-image-pro": {
        "label": "wan2.7-image-pro (万相)",
        "provider": "dashscope",
        "model": "wan2.7-image-pro",
        "default_params": {
            "size": "2K",
            "watermark": False,
        },
        "constraints": {
            "generation_size_param": "size",
            "supported_size_values": ["1K", "2K", "4K", "<width>*<height>"],
            "default_size": "2K",
            "supports_arbitrary_small_size": False,
            "max_total_pixels_hint": "4096x4096",
            "aspect_ratio_range": "[1/16, 16]",
            "size_separator": "*",
            "fixed_pixel_export_requires_postprocess": True,
            "output_format_configurable": False,
            "max_reference_images": 5,
            "supported_input_formats": ["jpeg", "png", "webp", "bmp", "tiff", "heic"],
            "sync_call": True,
            "sequential_image_generation": True,
        },
        "notes": [
            "万相 wan2.7-image-pro is the most capable Wan image model: text-to-image up to 4096x4096, sequential composition, brand-color and character-consistency controls.",
            "Runs on Alibaba Bailian (DashScope) — set DASHSCOPE_API_KEY, not ARK_API_KEY.",
            "Uses the synchronous multimodal-generation endpoint; no task polling required.",
            "DashScope size uses '*' as the separator (e.g. 2048*2048), not 'x'.",
        ],
        "source_urls": [
            "https://help.aliyun.com/zh/model-studio/wan-image-generation-and-editing-api-reference",
        ],
        "last_verified": "2026-06-22",
    },
    "gemini-banano2": {
        "label": "gemini-banano2",
        "provider": "gemini",
        "model": "gemini-3.1-flash-image-preview",
        "default_params": {
            "image_size": "2K",
            "aspect_ratio": "1:1",
        },
        "constraints": {
            "generation_size_param": "image_size",
            "aspect_ratio_param": "aspect_ratio",
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
            "default_image_size": "2K",
            "default_aspect_ratio": "1:1",
            "supports_arbitrary_pixel_size": False,
            "fixed_pixel_export_requires_postprocess": True,
        },
        "notes": [
            "Use image_size plus aspect_ratio for generation control.",
            "Small final assets such as 200x200 should be produced by deterministic post-processing.",
        ],
        "source_urls": [
            "https://ai.google.dev/gemini-api/docs/image-generation",
            "https://ai.google.dev/api/generate-content",
        ],
        "last_verified": "2026-05-20",
    },
}

_VIDEO_GENERATION_PATHS: dict[str, dict[str, Any]] = {
    "doubao-seedance-2-0-260128": {
        "label": "doubao-seedance-2.0",
        "provider": "ark",
        "model": "doubao-seedance-2-0-260128",
        "tool": "videos.generate",
        "default_params": {
            "watermark": False,
            "generate_audio": False,
        },
        "constraints": {
            "endpoint": "POST /contents/generations/tasks",
            "query_endpoint": "GET /contents/generations/tasks/{task_id}",
            "async_task": True,
            "supported_content_types": ["text", "image_url", "video_url", "audio_url"],
            "supported_reference_roles": [
                "reference_image",
                "reference_video",
                "reference_audio",
            ],
            "ratio_param": "ratio",
            "supported_ratio_values": ["1:1", "4:3", "3:4", "16:9", "9:16"],
            "default_ratio": "1:1",
            "duration_param": "duration",
            "recommended_duration_values": [5, 8, 11],
            "default_duration": 5,
            "resolution_param": "resolution",
            "recommended_resolution_values": ["720p", "1080p"],
            "default_resolution": "720p",
            "watermark_param": "watermark",
            "generate_audio_param": "generate_audio",
        },
        "notes": [
            "Use create_task for non-blocking submission, get_task for polling, or generate with wait_for_completion=True for a blocking call.",
            "Local image/video/audio references are uploaded first and sent as public URLs.",
            "Do not mix this with images.generate; Seedance video generation uses the asynchronous contents/generations/tasks endpoint.",
        ],
        "source_urls": [
            "https://www.volcengine.com/docs/82379/1520757?lang=zh",
            "https://www.volcengine.com/docs/82379/1521309?lang=zh",
        ],
        "last_verified": "2026-05-22",
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


def list_image_generation_paths() -> dict[str, dict[str, Any]]:
    """Return curated image generation paths with model constraints."""
    return deepcopy(_IMAGE_GENERATION_PATHS)


def get_image_generation_path(path: str) -> dict[str, Any]:
    """Return one curated image generation path by id."""
    if path not in _IMAGE_GENERATION_PATHS:
        raise KeyError(f"unknown image generation path: {path}")
    return deepcopy(_IMAGE_GENERATION_PATHS[path])


def find_image_path_by_model(model_id: str | None) -> str | None:
    """Return the curated path id whose ``model`` equals ``model_id``, or None."""
    if not model_id:
        return None
    for path, config in _IMAGE_GENERATION_PATHS.items():
        if config.get("model") == model_id:
            return path
    return None


def get_image_capability(
    *,
    path: str | None = None,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Return the curated capability for an image model.

    Prefers ``path`` when given. Falls back to a reverse lookup by ``model``.
    Returns None when neither is resolvable so callers can opt out of the
    capability-aware path silently.
    """
    if path:
        try:
            return get_image_generation_path(path)
        except KeyError:
            return None
    if model:
        resolved = find_image_path_by_model(model)
        if resolved is not None:
            return get_image_generation_path(resolved)
    return None


def normalize_image_generation_options(
    path: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return validated image generation params for a curated path.

    The result is ready to pass into ``images.generate`` as keyword arguments.
    Unsupported enum-like image size or aspect-ratio values fall back to the
    documented default for that path.
    """
    config = get_image_generation_path(path)
    constraints = config.get("constraints", {})
    params = deepcopy(config.get("default_params", {}))
    params.update(_non_null_options(options))

    image_size_values = constraints.get("supported_image_size_values")
    if image_size_values and params.get("image_size") not in image_size_values:
        params["image_size"] = constraints.get("default_image_size")

    aspect_ratio_param = constraints.get("aspect_ratio_param")
    aspect_ratio_values = constraints.get("supported_aspect_ratios")
    if aspect_ratio_param and aspect_ratio_values:
        value = params.get(aspect_ratio_param)
        if value not in aspect_ratio_values:
            params[aspect_ratio_param] = constraints.get("default_aspect_ratio")

    size_param = constraints.get("generation_size_param")
    if size_param and not params.get(size_param):
        default_size = constraints.get("default_size") or constraints.get("default_image_size")
        if default_size:
            params[size_param] = default_size

    output_format_param = constraints.get("output_format_param")
    output_format_values = constraints.get("supported_output_formats")
    if output_format_param and output_format_values:
        value = str(params.get(output_format_param) or "").strip().lower()
        if value not in output_format_values:
            value = str(constraints.get("default_output_format") or output_format_values[0])
        params[output_format_param] = value

    return params


def list_video_generation_paths() -> dict[str, dict[str, Any]]:
    """Return curated video generation paths with model constraints."""
    return deepcopy(_VIDEO_GENERATION_PATHS)


def get_video_generation_path(path: str) -> dict[str, Any]:
    """Return one curated video generation path by id."""
    if path not in _VIDEO_GENERATION_PATHS:
        raise KeyError(f"unknown video generation path: {path}")
    return deepcopy(_VIDEO_GENERATION_PATHS[path])


def normalize_video_generation_options(
    path: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return validated video generation params for a curated path.

    The result is ready to pass into ``videos.generate`` as keyword arguments.
    """
    config = get_video_generation_path(path)
    constraints = config.get("constraints", {})
    params = deepcopy(config.get("default_params", {}))
    params.update(_non_null_options(options))

    ratio_param = constraints.get("ratio_param", "ratio")
    supported_ratios = constraints.get("supported_ratio_values", [])
    if supported_ratios:
        ratio = str(params.get(ratio_param) or constraints.get("default_ratio") or "").strip()
        if ratio not in supported_ratios:
            ratio = str(constraints.get("default_ratio") or supported_ratios[0])
        params[ratio_param] = ratio

    duration_param = constraints.get("duration_param", "duration")
    duration = _coerce_int(
        params.get(duration_param),
        int(constraints.get("default_duration") or 5),
    )
    params[duration_param] = min(30, max(1, duration))

    resolution_param = constraints.get("resolution_param", "resolution")
    supported_resolutions = (
        constraints.get("supported_resolution_values")
        or constraints.get("recommended_resolution_values")
        or []
    )
    if supported_resolutions:
        resolution = str(
            params.get(resolution_param)
            or constraints.get("default_resolution")
            or supported_resolutions[0]
        ).strip().lower()
        if resolution not in supported_resolutions:
            resolution = str(constraints.get("default_resolution") or supported_resolutions[0])
        params[resolution_param] = resolution
    elif resolution_param in params or constraints.get("default_resolution"):
        params[resolution_param] = str(
            params.get(resolution_param)
            or constraints.get("default_resolution")
        ).strip().lower()

    watermark_param = constraints.get("watermark_param")
    if watermark_param:
        params[watermark_param] = _coerce_bool(
            params.get(watermark_param),
            bool(config.get("default_params", {}).get(watermark_param, False)),
        )

    audio_param = constraints.get("generate_audio_param")
    if audio_param:
        params[audio_param] = _coerce_bool(
            params.get(audio_param),
            bool(config.get("default_params", {}).get(audio_param, False)),
        )

    return params


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
