from __future__ import annotations

from pathlib import Path

from ai_toolkit import __version__, _transport, capabilities, chat, images
from ai_toolkit.chat import normalize_provider as normalize_chat_provider
from ai_toolkit.images import normalize_provider as normalize_image_provider
from ai_toolkit.videos import _extract_videos, build_content
from ai_toolkit.videos import normalize_provider as normalize_video_provider


def test_version_is_updated():
    assert __version__ == "0.5.0"


def test_provider_aliases():
    assert normalize_image_provider("doubao") == "ark"
    assert normalize_image_provider("google") == "gemini"
    assert normalize_image_provider("wan") == "dashscope"
    assert normalize_image_provider("wanx") == "dashscope"
    assert normalize_image_provider("万相") == "dashscope"
    assert normalize_image_provider("qwen-image") == "dashscope"
    assert normalize_chat_provider("doubao") == "ark"
    assert normalize_video_provider("seedance") == "ark"


def test_capabilities_include_current_tools():
    assert "images.generate" in capabilities.list_tools()
    assert "videos.generate" in capabilities.list_tools()
    assert "chat.complete" in capabilities.list_tools()
    assert "chat.complete_json" in capabilities.list_tools()
    assert "ark" in capabilities.get_tool_spec("images.generate")["providers"]
    assert "dashscope" in capabilities.get_tool_spec("images.generate")["providers"]
    assert "ark" in capabilities.get_tool_spec("videos.generate")["providers"]
    assert "deepseek" in capabilities.get_tool_spec("chat.complete")["providers"]
    assert "ark" in capabilities.get_tool_spec("chat.complete_json")["providers"]


def test_multimodal_user_message_builds_responses_input():
    message = chat.multimodal_user_message(
        text="只输出名称",
        images=["https://example.com/item.png"],
    )
    assert message == {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "只输出名称"},
            {"type": "input_image", "image_url": "https://example.com/item.png"},
        ],
    }


def test_multimodal_user_message_uploads_local_files(monkeypatch, tmp_path):
    local = tmp_path / "garment.png"
    local.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(
        chat,
        "upload_public_url",
        lambda value: f"https://uploads.example.com/{Path(value).name}",
    )

    message = chat.multimodal_user_message(
        text="describe",
        images=[local, "https://cdn.example.com/already.png"],
    )

    assert message["content"][1]["image_url"] == f"https://uploads.example.com/{local.name}"
    assert message["content"][2]["image_url"] == "https://cdn.example.com/already.png"


def test_ark_complete_json_uses_responses_structured_output(monkeypatch):
    captured = {}

    def fake_create_responses(model, input, **kwargs):
        captured["model"] = model
        captured["input"] = input
        captured["kwargs"] = kwargs
        return {"output_text": '{"name":"深蓝色双肩包"}', "usage": {"total_tokens": 12}}

    monkeypatch.setattr(chat, "create_responses", fake_create_responses)
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }

    result = chat.complete_json(
        provider="ark",
        model="doubao-seed-2-0-pro-260215",
        messages=[chat.multimodal_user_message(text="命名", images=["https://example.com/item.png"])],
        schema=schema,
        schema_name="garment_name",
        max_output_tokens=64,
    )

    assert result.parsed_json == {"name": "深蓝色双肩包"}
    assert captured["model"] == "doubao-seed-2-0-pro-260215"
    assert captured["kwargs"]["thinking"] == {"type": "disabled"}
    assert captured["kwargs"]["text"] == {
        "format": {
            "type": "json_schema",
            "name": "garment_name",
            "strict": True,
            "schema": schema,
        }
    }
    assert captured["kwargs"]["max_output_tokens"] == 64


def test_complete_json_rejects_payload_that_misses_required_key(monkeypatch):
    monkeypatch.setattr(
        chat,
        "create_responses",
        lambda model, input, **kwargs: {"output_text": '{"unrelated":"value"}'},
    )
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    result = chat.complete_json(
        provider="ark",
        model="doubao-seed-2-0-pro-260215",
        messages=[{"role": "user", "content": "x"}],
        schema=schema,
    )

    assert result.parsed_json is None
    assert result.schema_error is not None
    assert "name" in result.schema_error


def test_complete_json_rejects_payload_with_wrong_property_type(monkeypatch):
    monkeypatch.setattr(
        chat,
        "create_responses",
        lambda model, input, **kwargs: {"output_text": '{"name":123}'},
    )
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    result = chat.complete_json(
        provider="ark",
        model="doubao-seed-2-0-pro-260215",
        messages=[{"role": "user", "content": "x"}],
        schema=schema,
    )

    assert result.parsed_json is None
    assert result.schema_error is not None
    assert "name" in result.schema_error


def test_complete_json_accepts_payload_that_matches_schema(monkeypatch):
    monkeypatch.setattr(
        chat,
        "create_responses",
        lambda model, input, **kwargs: {
            "output_text": '{"name":"深蓝色双肩包","extra":42}'
        },
    )
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    result = chat.complete_json(
        provider="ark",
        model="doubao-seed-2-0-pro-260215",
        messages=[{"role": "user", "content": "x"}],
        schema=schema,
    )

    assert result.parsed_json == {"name": "深蓝色双肩包", "extra": 42}
    assert result.schema_error is None


def test_image_generation_paths_include_model_constraints():
    paths = capabilities.list_image_generation_paths()
    assert set(paths) >= {"doubao-5.0-lite", "doubao-4.5", "gemini-banano2"}

    seedream5 = capabilities.get_image_generation_path("doubao-5.0-lite")
    assert seedream5["provider"] == "ark"
    assert seedream5["model"] == "doubao-seedream-5-0-260128"
    assert seedream5["default_params"]["output_format"] == "png"
    assert "3K" in seedream5["constraints"]["supported_size_values"]
    assert "jpeg" in seedream5["constraints"]["supported_output_formats"]
    assert seedream5["constraints"]["max_reference_images"] == 14

    doubao = capabilities.get_image_generation_path("doubao-4.5")
    assert doubao["provider"] == "ark"
    assert doubao["model"] == "doubao-seedream-4-5-251128"
    assert doubao["constraints"]["supports_arbitrary_small_size"] is False
    assert doubao["constraints"]["min_total_pixels"] == 3686400
    assert "1024x256" in doubao["constraints"]["known_invalid_size_examples"]
    assert doubao["constraints"]["fixed_pixel_export_requires_postprocess"] is True
    assert doubao["default_params"]["size"] == "2048x2048"

    gemini = capabilities.get_image_generation_path("gemini-banano2")
    assert gemini["provider"] == "gemini"
    assert gemini["model"] == "gemini-3.1-flash-image-preview"
    assert "1:1" in gemini["constraints"]["supported_aspect_ratios"]
    assert "2K" in gemini["constraints"]["supported_image_size_values"]
    assert gemini["constraints"]["supports_arbitrary_pixel_size"] is False


def test_ark_image_generation_only_sends_explicit_output_format(monkeypatch):
    captured = {}

    def fake_create_image_generation(model, prompt, **kwargs):
        captured["model"] = model
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return {
            "data": [{"url": "https://example.com/out.png"}],
            "usage": {"output_tokens": 12, "total_tokens": 12},
        }

    monkeypatch.setattr(images, "create_image_generation", fake_create_image_generation)

    result = images.generate(
        provider="ark",
        model="doubao-seedream-5-0-260128",
        prompt="生成一张图",
    )

    assert result.images[0].url == "https://example.com/out.png"
    assert result.usage == {"output_tokens": 12, "total_tokens": 12}
    assert "output_format" not in captured["kwargs"]

    images.generate(
        provider="ark",
        model="doubao-seedream-5-0-260128",
        prompt="生成一张图",
        output_format="png",
    )

    assert captured["kwargs"]["output_format"] == "png"


def test_find_image_path_by_model_resolves_known_ids():
    assert (
        capabilities.find_image_path_by_model("doubao-seedream-5-0-260128") == "doubao-5.0-lite"
    )
    assert capabilities.find_image_path_by_model("doubao-seedream-4-5-251128") == "doubao-4.5"
    assert capabilities.find_image_path_by_model("unknown-model") is None
    assert capabilities.find_image_path_by_model(None) is None


def test_get_image_capability_prefers_path_then_falls_back_to_model():
    by_path = capabilities.get_image_capability(path="doubao-4.5")
    assert by_path is not None and by_path["model"] == "doubao-seedream-4-5-251128"

    by_model = capabilities.get_image_capability(model="doubao-seedream-5-0-260128")
    assert by_model is not None and by_model["model"] == "doubao-seedream-5-0-260128"

    assert capabilities.get_image_capability(path="bogus") is None
    assert capabilities.get_image_capability(model="bogus") is None
    assert capabilities.get_image_capability() is None


_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _fake_b64_response(captured):
    def fake_create_image_generation(model, prompt, **kwargs):
        captured["model"] = model
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return {"data": [{"b64_json": _TINY_PNG_B64}]}

    return fake_create_image_generation


def test_generate_with_path_resolves_provider_and_model(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(images, "create_image_generation", _fake_b64_response(captured))

    images.generate(
        path="doubao-5.0-lite",
        prompt="生成",
        output_path=tmp_path / "result.png",
    )

    assert captured["model"] == "doubao-seedream-5-0-260128"
    # 5.0 lite supports output_format; SDK infers from output_path suffix
    assert captured["kwargs"]["output_format"] == "png"


def test_generate_with_path_drops_output_format_for_4_5(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(images, "create_image_generation", _fake_b64_response(captured))

    images.generate(
        path="doubao-4.5",
        prompt="生成",
        output_path=tmp_path / "result.jpg",
        output_format="jpeg",  # caller mistakenly provides it
    )

    # 4.5 does not support output_format; SDK drops it silently
    assert "output_format" not in captured["kwargs"]


def test_generate_with_path_infers_jpeg_from_jpg_suffix(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(images, "create_image_generation", _fake_b64_response(captured))

    images.generate(
        path="doubao-5.0-lite",
        prompt="生成",
        output_path=tmp_path / "result.jpg",
    )

    assert captured["kwargs"]["output_format"] == "jpeg"


def test_generate_with_path_enforces_reference_limit(monkeypatch):
    monkeypatch.setattr(images, "_reference_to_url", lambda value: str(value))

    too_many = [f"https://example.com/{index}.png" for index in range(15)]
    try:
        images.generate(
            path="doubao-5.0-lite",
            prompt="生成",
            references=too_many,
        )
    except images.AIToolkitError as exc:
        assert "max 14" in str(exc)
    else:
        raise AssertionError("expected AIToolkitError for reference overflow")


def test_generate_falls_back_to_model_capability_when_path_absent(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(images, "create_image_generation", _fake_b64_response(captured))

    # No path — but model is a known capability, so SDK still applies its rules.
    images.generate(
        provider="ark",
        model="doubao-seedream-5-0-260128",
        prompt="生成",
        output_path=tmp_path / "out.png",
    )

    assert captured["kwargs"]["output_format"] == "png"


def test_video_generation_paths_include_seedance_task_api():
    paths = capabilities.list_video_generation_paths()
    assert set(paths) >= {"doubao-seedance-2-0-260128"}

    seedance = capabilities.get_video_generation_path("doubao-seedance-2-0-260128")
    assert seedance["provider"] == "ark"
    assert seedance["model"] == "doubao-seedance-2-0-260128"
    assert seedance["tool"] == "videos.generate"
    assert seedance["constraints"]["async_task"] is True
    assert "video_url" in seedance["constraints"]["supported_content_types"]
    assert seedance["constraints"]["endpoint"] == "POST /contents/generations/tasks"


def test_capability_option_normalizers_validate_common_params():
    image_options = capabilities.normalize_image_generation_options(
        "doubao-5.0-lite",
        {"output_format": "bad"},
    )
    assert image_options["output_format"] == "png"

    image_options = capabilities.normalize_image_generation_options(
        "gemini-banano2",
        {"image_size": "bad", "aspect_ratio": "4:1"},
    )
    assert image_options["image_size"] == "2K"
    assert image_options["aspect_ratio"] == "4:1"

    video_options = capabilities.normalize_video_generation_options(
        "doubao-seedance-2-0-260128",
        {
            "ratio": "bad",
            "duration": "8",
            "resolution": "4k",
            "watermark": "false",
            "generate_audio": "true",
        },
    )
    assert video_options["ratio"] == "1:1"
    assert video_options["duration"] == 8
    assert video_options["resolution"] == "720p"
    assert video_options["watermark"] is False
    assert video_options["generate_audio"] is True


def test_sleep_for_retry_only_sleeps(monkeypatch):
    delays = []
    monkeypatch.setattr(_transport.time, "sleep", delays.append)
    monkeypatch.setattr(_transport.random, "uniform", lambda *_args: 0.0)

    _transport._sleep_for_retry(attempt=2, base_seconds=0.5, max_seconds=8.0)

    assert delays == [1.0]


def test_video_content_builder_keeps_raw_content_and_reference_roles():
    content = build_content(
        content=[{"type": "text", "text": "make a short pixel animation"}],
        reference_images=["https://example.com/base.png"],
        reference_videos=["https://example.com/motion.mp4"],
        reference_audio=["https://example.com/bgm.mp3"],
    )
    assert content[0] == {"type": "text", "text": "make a short pixel animation"}
    assert content[1]["type"] == "image_url"
    assert content[1]["role"] == "reference_image"
    assert content[2]["type"] == "video_url"
    assert content[2]["role"] == "reference_video"
    assert content[3]["type"] == "audio_url"
    assert content[3]["role"] == "reference_audio"


def test_dashscope_image_generation_extracts_choice_image_and_uploads_refs(monkeypatch):
    captured = {}

    def fake_create_dashscope(model, prompt, *, image_urls=None, **parameters):
        captured["model"] = model
        captured["prompt"] = prompt
        captured["image_urls"] = image_urls
        captured["parameters"] = parameters
        return {
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "image", "image": "https://dashscope.example/out.png"}
                            ]
                        }
                    }
                ]
            },
            "usage": {"image_count": 1, "size": "1488*704"},
        }

    monkeypatch.setattr(images, "create_dashscope_image_generation", fake_create_dashscope)
    monkeypatch.setattr(
        images, "_reference_to_url", lambda value: f"https://uploads.example/{value}"
    )

    result = images.generate(
        provider="wan",
        prompt="生成一张图",
        references=["local.png"],
    )

    assert result.provider == "dashscope"
    assert result.model == "wan2.7-image"
    assert result.images[0].url == "https://dashscope.example/out.png"
    assert result.usage == {"image_count": 1, "size": "1488*704"}
    assert captured["image_urls"] == ["https://uploads.example/local.png"]
    assert captured["parameters"]["size"] == "2K"
    assert captured["parameters"]["watermark"] is False


def test_dashscope_extractor_supports_legacy_results_shape():
    response = {"output": {"results": [{"url": "https://dashscope.example/legacy.png"}]}}
    extracted = images._extract_dashscope_images(response)
    assert extracted[0].url == "https://dashscope.example/legacy.png"


def test_dashscope_image_paths_resolve_provider_and_constraints():
    paths = capabilities.list_image_generation_paths()
    assert {"wan2.7-image", "wan2.7-image-pro"} <= set(paths)

    wan = capabilities.get_image_generation_path("wan2.7-image-pro")
    assert wan["provider"] == "dashscope"
    assert wan["model"] == "wan2.7-image-pro"
    assert wan["constraints"]["output_format_configurable"] is False
    assert wan["constraints"]["max_reference_images"] == 5
    assert capabilities.find_image_path_by_model("wan2.7-image") == "wan2.7-image"


def test_video_result_extraction_supports_common_task_shapes():
    upstream = {
        "id": "cgt-1",
        "model": "doubao-seedance-2-0-260128",
        "status": "succeeded",
        "content": {"video_url": "https://example.com/out.mp4"},
    }
    compatible = {
        "id": "cgt-2",
        "object": "video",
        "status": "completed",
        "metadata": {"url": "https://example.com/metadata.mp4"},
    }
    assert _extract_videos(upstream)[0].url == "https://example.com/out.mp4"
    assert _extract_videos(compatible)[0].url == "https://example.com/metadata.mp4"
