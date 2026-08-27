from __future__ import annotations

from pathlib import Path

import pytest

from ai_toolkit import (
    GeneratedAudio,
    ImageGenerationBatchTask,
    __version__,
    capabilities,
    help,
)
from ai_toolkit.aliyun import images as aliyun_images
from ai_toolkit.ark import embeddings as ark_embeddings
from ai_toolkit.ark import images as ark_images
from ai_toolkit.ark import text as ark_text
from ai_toolkit.ark import videos as ark_videos
from ai_toolkit.dashscope import images as dashscope_images
from ai_toolkit.dashscope import speech as dashscope_speech
from ai_toolkit.dashscope import videos as dashscope_videos
from ai_toolkit.deepseek import text as deepseek_text
from ai_toolkit.gemini import images as gemini_images
from ai_toolkit.grsai import images as grsai_images
from ai_toolkit.minimax import videos as minimax_videos


def test_version_is_updated():
    assert __version__ == "0.10.0"
    assert help()["version"] == __version__


def test_capabilities_use_platform_scoped_tool_names():
    tools = capabilities.list_tools()
    assert "ark.images.generate" in tools
    assert "ark.videos.generate" in tools
    assert "ark.text.complete_json" in tools
    assert "deepseek.text.complete" in tools
    assert "dashscope.speech.synthesize" in tools
    assert "gemini.images.create_batch" in tools
    assert "gemini.images.get_batch" in tools
    assert "grsai.images.generate" in tools
    assert "minimax.videos.generate" in tools
    assert "aliyun.images.segment_commodity" in tools
    assert "aliyun.images.segment_hd_body" in tools
    assert "images.generate" not in tools
    assert "chat.complete" not in tools


def test_model_aliases_resolve_to_raw_provider_ids():
    assert ark_images.resolve_model("seedream-5") == "doubao-seedream-5-0-260128"
    assert (
        ark_images.resolve_model("seedream-5-pro")
        == "doubao-seedream-5-0-pro-260628"
    )
    assert ark_images.resolve_model("seedream-4.5") == "doubao-seedream-4-5-251128"
    assert ark_videos.resolve_model("seedance-2") == "doubao-seedance-2-0-260128"
    assert ark_text.resolve_model("doubao-pro") == "doubao-seed-2-0-pro-260215"
    assert deepseek_text.resolve_model("v4-pro") == "deepseek-v4-pro"
    assert dashscope_images.resolve_model("wan2.7-pro") == "wan2.7-image-pro"
    assert dashscope_videos.resolve_model("happyhorse-r2v") == "happyhorse-1.1-r2v"
    assert (
        dashscope_speech.resolve_model("qwen-audio-tts-plus")
        == "qwen-audio-3.0-tts-plus"
    )
    assert gemini_images.resolve_model("gemini-image") == "gemini-3.1-flash-image"
    assert gemini_images.resolve_model("nano-banana-2") == "gemini-3.1-flash-image"
    assert gemini_images.resolve_model("nano-banana-2-lite") == "gemini-3.1-flash-lite-image"
    assert gemini_images.resolve_model("nano-banana-pro") == "gemini-3-pro-image"
    assert grsai_images.resolve_model("grsai-image") == "nano-banana-2"
    assert minimax_videos.resolve_model("minimax-h3") == "MiniMax-H3"
    assert ark_embeddings.resolve_model("doubao-vision") == "doubao-embedding-vision-251215"


def test_ark_multimodal_user_message_uploads_local_files(monkeypatch, tmp_path):
    local = tmp_path / "item.png"
    local.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(
        ark_text,
        "upload_public_url",
        lambda value: f"https://uploads.example.com/{Path(value).name}",
    )

    message = ark_text.multimodal_user_message(
        text="name it",
        images=[local, "https://cdn.example.com/already.png"],
    )

    assert message["content"][1]["image_url"] == f"https://uploads.example.com/{local.name}"
    assert message["content"][2]["image_url"] == "https://cdn.example.com/already.png"


def test_ark_complete_json_defaults_to_disabled_thinking(monkeypatch):
    captured = {}

    def fake_create_responses(model, input, **kwargs):
        captured["model"] = model
        captured["input"] = input
        captured["kwargs"] = kwargs
        return {"output_text": '{"ok":true}', "usage": {"total_tokens": 3}}

    monkeypatch.setattr(ark_text, "create_responses", fake_create_responses)
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }

    result = ark_text.complete_json(
        model="doubao-pro",
        prompt="Return JSON only",
        schema=schema,
        thinking=None,
    )

    assert result.parsed_json == {"ok": True}
    assert captured["model"] == "doubao-seed-2-0-pro-260215"
    assert captured["kwargs"]["thinking"] == {"type": "disabled"}


def test_deepseek_complete_exposes_reasoning_text(monkeypatch):
    def fake_create_chat_completion(model, messages, **kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "think",
                        "content": "answer",
                    }
                }
            ],
            "usage": {"total_tokens": 4},
        }

    monkeypatch.setattr(deepseek_text, "create_chat_completion", fake_create_chat_completion)

    result = deepseek_text.complete(
        model="v4-pro",
        prompt="hello",
        thinking={"type": "enabled", "reasoning_effort": "high"},
    )

    assert result.text == "answer"
    assert result.reasoning_text == "think"


def test_ark_image_generate_infers_seedream_5_output_format(monkeypatch, tmp_path):
    captured = {}

    def fake_create_image_generation(model, prompt, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return {"data": [{"b64_json": _TINY_PNG_B64}]}

    monkeypatch.setattr(ark_images, "create_image_generation", fake_create_image_generation)

    ark_images.generate(
        model="seedream-5",
        prompt="red square",
        output_path=tmp_path / "out.jpg",
    )

    assert captured["model"] == "doubao-seedream-5-0-260128"
    assert captured["kwargs"]["output_format"] == "jpeg"


@pytest.mark.parametrize(
    "model",
    ["seedream-5-lite", "seedream-5.0-lite", "doubao-5.0-lite"],
)
def test_ark_image_warns_for_incorrect_seedream_5_lite_aliases(model):
    with pytest.warns(DeprecationWarning, match="use 'seedream-5'"):
        resolved = ark_images.resolve_model(model)

    assert resolved == "doubao-seedream-5-0-260128"


def test_ark_image_seedream_5_pro_uses_pro_contract(monkeypatch):
    captured = {}

    def fake_create_image_generation(model, prompt, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return {"data": [{"b64_json": _TINY_PNG_B64}]}

    monkeypatch.setattr(ark_images, "create_image_generation", fake_create_image_generation)

    ark_images.generate(
        model="seedream-5-pro",
        prompt="red square",
        size="1k",
    )

    assert captured["model"] == "doubao-seedream-5-0-pro-260628"
    assert captured["kwargs"]["size"] == "1K"
    assert captured["kwargs"]["output_format"] == "png"
    assert "sequential_image_generation" not in captured["kwargs"]
    assert "stream" not in captured["kwargs"]


def test_ark_image_seedream_5_pro_rejects_4k_before_network(monkeypatch):
    monkeypatch.setattr(
        ark_images,
        "create_image_generation",
        lambda *args, **kwargs: pytest.fail("network request should not be sent"),
    )

    with pytest.raises(ark_images.AIToolkitError, match="unsupported size"):
        ark_images.generate(
            model="seedream-5-pro",
            prompt="red square",
            size="4K",
        )


def test_ark_image_seedream_5_pro_rejects_more_than_ten_references(monkeypatch):
    monkeypatch.setattr(
        ark_images,
        "create_image_generation",
        lambda *args, **kwargs: pytest.fail("network request should not be sent"),
    )

    with pytest.raises(ark_images.AIToolkitError, match="max 10"):
        ark_images.generate(
            model="seedream-5-pro",
            prompt="compose",
            references=[f"https://example.test/{index}.png" for index in range(11)],
        )


def test_seedream_5_pro_capabilities_publish_resolution_contract():
    model = capabilities.get_image_model("seedream-5-pro")

    assert model["model"] == "doubao-seedream-5-0-pro-260628"
    assert model["constraints"]["supported_size_values"] == [
        "1K",
        "2K",
        "<width>x<height>",
    ]
    assert model["constraints"]["sequential_image_generation"] is False
    assert model["constraints"]["stream"] is False


def test_ark_image_seedream_4_rejects_output_format_override():
    try:
        ark_images.generate(
            model="seedream-4.5",
            prompt="red square",
            output_format="png",
        )
    except ark_images.AIToolkitError as exc:
        assert "does not support output_format" in str(exc)
    else:
        raise AssertionError("expected AIToolkitError")


def test_ark_image_generation_encodes_local_reference(monkeypatch, tmp_path):
    reference = tmp_path / "character.png"
    reference.write_bytes(b"png-data")
    captured = {}

    def fake_create_image_generation(model, prompt, **kwargs):
        captured["model"] = model
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return {"data": [{"b64_json": _TINY_PNG_B64}]}

    monkeypatch.setattr(
        ark_images,
        "create_image_generation",
        fake_create_image_generation,
    )

    ark_images.generate(
        model="seedream-4.5",
        prompt="paper doll",
        references=[reference],
    )

    assert captured["kwargs"]["image"] == [
        "data:image/png;base64,cG5nLWRhdGE="
    ]


def test_dashscope_image_generation_supports_nine_references(monkeypatch):
    captured = {}

    def fake_create_image_generation(model, prompt, *, image_urls=None, **parameters):
        captured["model"] = model
        captured["image_urls"] = image_urls
        captured["parameters"] = parameters
        return {
            "output": {
                "choices": [
                    {"message": {"content": [{"image": "https://dashscope.example/out.png"}]}}
                ]
            }
        }

    monkeypatch.setattr(dashscope_images, "create_image_generation", fake_create_image_generation)
    monkeypatch.setattr(
        dashscope_images,
        "upload_public_url",
        lambda value: f"https://uploads.example.com/{value}",
    )

    result = dashscope_images.generate(
        model="wan2.7-pro",
        prompt="compose",
        references=[f"ref-{index}.png" for index in range(9)],
    )

    assert result.model == "wan2.7-image-pro"
    assert result.images[0].url == "https://dashscope.example/out.png"
    assert len(captured["image_urls"]) == 9
    assert captured["parameters"]["size"] == "2K"


def test_dashscope_image_generation_encodes_local_reference(monkeypatch, tmp_path):
    reference = tmp_path / "cat.png"
    reference.write_bytes(b"png-data")
    captured = {}

    def fake_create_image_generation(model, prompt, *, image_urls=None, **parameters):
        captured["image_urls"] = image_urls
        return {
            "output": {
                "choices": [
                    {"message": {"content": [{"image": "https://dashscope.example/out.png"}]}}
                ]
            }
        }

    monkeypatch.setattr(dashscope_images, "create_image_generation", fake_create_image_generation)
    result = dashscope_images.generate(
        model="wan2.7-pro",
        prompt="compose",
        references=[reference],
    )

    assert result.images[0].url == "https://dashscope.example/out.png"
    assert captured["image_urls"] == ["data:image/png;base64,cG5nLWRhdGE="]


def test_dashscope_happyhorse_r2v_uses_reference_images_and_polls(monkeypatch, tmp_path):
    reference = tmp_path / "cat.png"
    reference.write_bytes(b"png-data")
    captured = {}

    def fake_create_video_generation_task(model, *, prompt, media, parameters):
        captured.update(model=model, prompt=prompt, media=media, parameters=parameters)
        return {"output": {"task_id": "task-1", "task_status": "PENDING"}}

    def fake_get_video_generation_task(task_id):
        assert task_id == "task-1"
        return {
            "output": {
                "task_id": task_id,
                "task_status": "SUCCEEDED",
                "video_url": "https://dashscope.example/out.mp4",
            }
        }

    monkeypatch.setattr(
        dashscope_videos,
        "create_video_generation_task",
        fake_create_video_generation_task,
    )
    monkeypatch.setattr(
        dashscope_videos,
        "get_video_generation_task",
        fake_get_video_generation_task,
    )

    result = dashscope_videos.generate(
        model="happyhorse-r2v",
        prompt="The kitten in [Image 1] walks.",
        references=[reference],
        poll_interval=0.1,
    )

    assert result.model == "happyhorse-1.1-r2v"
    assert result.first_video().url == "https://dashscope.example/out.mp4"
    assert captured["media"] == [
        {"type": "reference_image", "url": "data:image/png;base64,cG5nLWRhdGE="}
    ]
    assert captured["parameters"] == {
        "resolution": "720P",
        "duration": 5,
        "watermark": False,
        "ratio": "1:1",
    }


def test_dashscope_speech_synthesis_returns_downloadable_audio(monkeypatch):
    captured = {}

    def fake_create_speech_synthesis(model, *, text, voice, **input_options):
        captured["model"] = model
        captured["text"] = text
        captured["voice"] = voice
        captured["input_options"] = input_options
        return {
            "output": {
                "finish_reason": "stop",
                "audio": {
                    "url": "https://dashscope.example/out.wav",
                    "id": "audio-1",
                    "expires_at": 1784462605,
                },
            },
            "usage": {"characters": 12},
        }

    monkeypatch.setattr(
        dashscope_speech,
        "create_speech_synthesis",
        fake_create_speech_synthesis,
    )

    result = dashscope_speech.synthesize(
        model="qwen-audio-tts-plus",
        text="自然地介绍这个项目。",
        voice="longanlingxin",
        instruction="使用自然、沉稳的普通话。",
        language_hints=["zh"],
    )

    assert result.model == "qwen-audio-3.0-tts-plus"
    assert result.audio.url == "https://dashscope.example/out.wav"
    assert result.audio.audio_id == "audio-1"
    assert result.audio.mime_type == "audio/wav"
    assert result.usage == {"characters": 12}
    assert captured == {
        "model": "qwen-audio-3.0-tts-plus",
        "text": "自然地介绍这个项目。",
        "voice": "longanlingxin",
        "input_options": {
            "format": "wav",
            "sample_rate": 24000,
            "instruction": "使用自然、沉稳的普通话。",
            "language_hints": ["zh"],
        },
    }


def test_generated_audio_saves_base64_data(tmp_path):
    output = tmp_path / "sample.wav"
    audio = GeneratedAudio(b64_data="UklGRg==")

    assert audio.save(output) == output.resolve()
    assert output.read_bytes() == b"RIFF"


def test_gemini_generate_embeds_local_reference(monkeypatch, tmp_path):
    local = tmp_path / "ref.png"
    local.write_bytes(b"\x89PNG\r\n\x1a\n")
    captured = {}

    def fake_generate_content(model, contents, **kwargs):
        captured["model"] = model
        captured["contents"] = contents
        captured["kwargs"] = kwargs
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"inlineData": {"data": _TINY_PNG_B64, "mimeType": "image/png"}}
                        ]
                    }
                }
            ]
        }

    monkeypatch.setattr(gemini_images, "generate_content", fake_generate_content)

    result = gemini_images.generate(
        model="gemini-image",
        prompt="red square",
        references=[local],
        image_size="512",
        aspect_ratio="1:1",
    )

    assert result.model == "gemini-3.1-flash-image"
    assert "inlineData" in captured["contents"][0]["parts"][1]
    assert captured["kwargs"]["generationConfig"]["imageConfig"]["imageSize"] == "512"
    assert captured["kwargs"]["generationConfig"]["imageConfig"]["aspectRatio"] == "1:1"


def test_gemini_omits_aspect_ratio_when_not_requested(monkeypatch):
    captured = {}

    def fake_generate_content(model, contents, **kwargs):
        captured["image_config"] = kwargs["generationConfig"]["imageConfig"]
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"inlineData": {"data": _TINY_PNG_B64, "mimeType": "image/png"}}
                        ]
                    }
                }
            ]
        }

    monkeypatch.setattr(gemini_images, "generate_content", fake_generate_content)
    gemini_images.generate(prompt="match the input ratio", image_size="1K")

    assert captured["image_config"] == {"imageSize": "1K"}


def test_gemini_auto_image_size_omits_resolution(monkeypatch):
    captured = {}

    def fake_generate_content(model, contents, **kwargs):
        captured["image_config"] = kwargs["generationConfig"]["imageConfig"]
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "data": _TINY_PNG_B64,
                                    "mimeType": "image/png",
                                }
                            }
                        ]
                    }
                }
            ]
        }

    monkeypatch.setattr(gemini_images, "generate_content", fake_generate_content)
    gemini_images.generate(
        model="gemini-image",
        prompt="model decides the resolution",
        image_size="auto",
        aspect_ratio="16:9",
    )

    assert captured["image_config"] == {"aspectRatio": "16:9"}


def test_gemini_image_batch_builds_keyed_inline_requests(monkeypatch):
    captured = {}

    def fake_batch_generate_content(model, batch):
        captured["model"] = model
        captured["batch"] = batch
        return {
            "name": "batches/batch-123",
            "state": "JOB_STATE_PENDING",
        }

    monkeypatch.setattr(
        gemini_images,
        "batch_generate_content",
        fake_batch_generate_content,
    )

    task = gemini_images.create_batch(
        model="gemini-image-lite",
        prompts={"coop": "wooden coop", "trough": "wooden trough"},
        image_size="auto",
        aspect_ratio=None,
        display_name="pasture-batch-1",
    )

    assert task.task_id == "batches/batch-123"
    assert task.model == "gemini-3.1-flash-lite-image"
    assert captured["model"] == "gemini-3.1-flash-lite-image"
    requests = captured["batch"]["input_config"]["requests"]["requests"]
    assert [item["metadata"]["key"] for item in requests] == ["coop", "trough"]
    assert requests[0]["request"]["generationConfig"] == {
        "responseModalities": ["TEXT", "IMAGE"]
    }


def test_gemini_image_batch_result_keeps_per_request_errors():
    task = ImageGenerationBatchTask(
        provider="gemini",
        model="gemini-3.1-flash-lite-image",
        task_id="batches/batch-123",
        status="JOB_STATE_SUCCEEDED",
        raw_response={
            "name": "batches/batch-123",
            "state": "JOB_STATE_SUCCEEDED",
            "dest": {
                "inlinedResponses": [
                    {
                        "metadata": {"key": "coop"},
                        "response": {
                            "candidates": [
                                {
                                    "content": {
                                        "parts": [
                                            {
                                                "inlineData": {
                                                    "data": _TINY_PNG_B64,
                                                    "mimeType": "image/png",
                                                }
                                            }
                                        ]
                                    }
                                }
                            ]
                        },
                    },
                    {
                        "metadata": {"key": "trough"},
                        "error": {"code": 429, "message": "quota"},
                    },
                ]
            },
        },
    )

    result = gemini_images.batch_result(
        task,
        prompts={"coop": "wooden coop", "trough": "wooden trough"},
    )

    assert result.items[0].key == "coop"
    assert result.items[0].result is not None
    assert result.items[0].result.first_image().b64_json == _TINY_PNG_B64
    assert result.items[1].key == "trough"
    assert result.items[1].error == '{"code": 429, "message": "quota"}'


def test_gemini_batch_create_disables_transport_retries(monkeypatch):
    captured = {}

    class Settings:
        gemini_api_key = "test-key"
        gemini_base_url = "https://example.test/v1beta"

    def fake_post_json(url, payload, **kwargs):
        captured["url"] = url
        captured["payload"] = payload
        captured["kwargs"] = kwargs
        return {"name": "batches/batch-123"}

    monkeypatch.setattr(gemini_images, "get_settings", lambda: Settings())
    monkeypatch.setattr(gemini_images, "post_json", fake_post_json)

    gemini_images.batch_generate_content(
        "gemini-3.1-flash-lite-image",
        {"input_config": {}},
    )

    assert captured["url"].endswith(
        "/models/gemini-3.1-flash-lite-image:batchGenerateContent"
    )
    assert captured["kwargs"]["max_retries"] == 0


def test_gemini_batch_rejects_oversize_inline_payload(monkeypatch):
    class Settings:
        gemini_api_key = "test-key"
        gemini_base_url = "https://example.test/v1beta"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("network request should not be sent")

    monkeypatch.setattr(gemini_images, "get_settings", lambda: Settings())
    monkeypatch.setattr(gemini_images, "post_json", fail_if_called)
    monkeypatch.setattr(gemini_images, "MAX_INLINE_BATCH_BYTES", 20)

    with pytest.raises(ValueError, match="smaller than 20 MB"):
        gemini_images.batch_generate_content(
            "gemini-3.1-flash-lite-image",
            {"input_config": {"requests": {"requests": []}}},
        )


def test_gemini_find_batch_follows_pagination(monkeypatch):
    calls = []

    def fake_list_batch_generations(*, page_size, page_token=None):
        calls.append((page_size, page_token))
        if page_token is None:
            return {
                "batches": [],
                "nextPageToken": "page-2",
            }
        return {
            "batches": [
                {
                    "name": "batches/batch-123",
                    "displayName": "pasture-batch-1",
                    "state": "JOB_STATE_PENDING",
                }
            ]
        }

    monkeypatch.setattr(
        gemini_images,
        "list_batch_generations",
        fake_list_batch_generations,
    )

    task = gemini_images.find_batch(
        display_name="pasture-batch-1",
        model="gemini-image-lite",
    )

    assert calls == [(100, None), (100, "page-2")]
    assert task is not None
    assert task.task_id == "batches/batch-123"


@pytest.mark.parametrize(
    ("model", "image_size", "aspect_ratio"),
    [
        ("gemini-image", "1k", "1:1"),
        ("gemini-image", "1K", "7:1"),
        ("gemini-image-lite", "2K", "1:1"),
        ("gemini-image-pro", "1K", "8:1"),
    ],
)
def test_gemini_rejects_unsupported_options_before_request(
    monkeypatch,
    model,
    image_size,
    aspect_ratio,
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("network request should not be sent")

    monkeypatch.setattr(gemini_images, "generate_content", fail_if_called)
    with pytest.raises(ValueError):
        gemini_images.generate(
            model=model,
            prompt="sprite sheet",
            image_size=image_size,
            aspect_ratio=aspect_ratio,
        )


def test_gemini_rejects_more_than_fourteen_references(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("network request should not be sent")

    monkeypatch.setattr(gemini_images, "generate_content", fail_if_called)
    with pytest.raises(ValueError, match="at most 14 references"):
        gemini_images.generate(
            prompt="compose",
            references=[f"ref-{index}.png" for index in range(15)],
        )


def test_ark_video_content_builder_and_extractor():
    content = ark_videos.build_content(
        content=[{"type": "text", "text": "move"}],
        reference_images=["https://example.com/base.png"],
        reference_videos=["https://example.com/motion.mp4"],
    )
    assert content[0] == {"type": "text", "text": "move"}
    assert content[1]["type"] == "image_url"
    assert content[1]["role"] == "reference_image"
    assert content[2]["type"] == "video_url"
    assert content[2]["role"] == "reference_video"

    response = {
        "id": "cgt-1",
        "status": "succeeded",
        "content": {"video_url": "https://example.com/out.mp4"},
    }
    assert ark_videos._extract_videos(response)[0].url == "https://example.com/out.mp4"


def test_ark_embeddings_dimensions_are_forwarded(monkeypatch):
    captured = {}

    def fake_create_multimodal_embedding(model, input, **kwargs):
        captured["model"] = model
        captured["input"] = input
        captured["kwargs"] = kwargs
        return {"data": {"embedding": [0.1, 0.2]}}

    monkeypatch.setattr(
        ark_embeddings,
        "create_multimodal_embedding",
        fake_create_multimodal_embedding,
    )

    result = ark_embeddings.generate(model="doubao-vision", text="hello", dimensions=1024)

    assert result.first_embedding() == [0.1, 0.2]
    assert captured["model"] == "doubao-embedding-vision-251215"
    assert captured["kwargs"]["dimensions"] == 1024


def test_aliyun_segment_commodity_stages_and_saves(monkeypatch, tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\n")
    output = tmp_path / "out.png"
    captured = {}

    class FakeRequest:
        def set_ImageURL(self, value):
            self.image_url = value

    monkeypatch.setattr(
        aliyun_images,
        "_import_request_class",
        lambda module_name, class_name: FakeRequest,
    )
    monkeypatch.setattr(
        aliyun_images,
        "stage_image",
        lambda image_path, *, long_side_max: "https://viapi.example/source.png",
    )

    def fake_invoke(request, *, output_path, api_name, region=None):
        captured["request_url"] = request.image_url
        captured["api_name"] = api_name
        captured["region"] = region
        Path(output_path).write_bytes(b"png")
        return {"Data": {"ImageURL": "https://viapi.example/out.png"}}

    monkeypatch.setattr(aliyun_images, "invoke_segmentation_request", fake_invoke)

    result = aliyun_images.segment_commodity(
        image_path=source,
        output_path=output,
        region="cn-shanghai",
    )

    assert result.provider == "aliyun"
    assert result.model == "viapi-segment-commodity"
    assert result.path == output.resolve()
    assert output.read_bytes() == b"png"
    assert captured == {
        "request_url": "https://viapi.example/source.png",
        "api_name": "SegmentCommodity",
        "region": "cn-shanghai",
    }
    assert result.request["staged_url"] == "https://viapi.example/source.png"


def test_aliyun_segment_hd_body_uses_hd_body_contract(monkeypatch, tmp_path):
    source = tmp_path / "person.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\n")
    output = tmp_path / "person-out.png"
    captured = {}

    class FakeRequest:
        def set_ImageURL(self, value):
            self.image_url = value

    monkeypatch.setattr(
        aliyun_images,
        "_import_request_class",
        lambda module_name, class_name: FakeRequest,
    )
    monkeypatch.setattr(
        aliyun_images,
        "stage_image",
        lambda image_path, *, long_side_max: "https://viapi.example/person.png",
    )

    def fake_invoke(request, *, output_path, api_name, region=None):
        captured["request_url"] = request.image_url
        captured["api_name"] = api_name
        Path(output_path).write_bytes(b"png")
        return {"Data": {"ImageURL": "https://viapi.example/person-out.png"}}

    monkeypatch.setattr(aliyun_images, "invoke_segmentation_request", fake_invoke)

    result = aliyun_images.segment_hd_body(image_path=source, output_path=output)

    assert result.model == "viapi-segment-hd-body"
    assert result.path == output.resolve()
    assert captured == {
        "request_url": "https://viapi.example/person.png",
        "api_name": "SegmentHDBody",
    }


def test_aliyun_shrink_to_limit_resizes_large_image(tmp_path):
    source = tmp_path / "large.png"
    from PIL import Image

    Image.new("RGB", (100, 50), "white").save(source)
    resized = aliyun_images.shrink_to_limit(source, long_side_max=20)

    try:
        assert resized != source
        with Image.open(resized) as image:
            assert max(image.size) <= 20
    finally:
        if resized != source:
            resized.unlink(missing_ok=True)


_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
