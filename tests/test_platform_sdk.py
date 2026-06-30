from __future__ import annotations

from pathlib import Path

from ai_toolkit import __version__, capabilities
from ai_toolkit.aliyun import images as aliyun_images
from ai_toolkit.ark import embeddings as ark_embeddings
from ai_toolkit.ark import images as ark_images
from ai_toolkit.ark import text as ark_text
from ai_toolkit.ark import videos as ark_videos
from ai_toolkit.dashscope import images as dashscope_images
from ai_toolkit.deepseek import text as deepseek_text
from ai_toolkit.gemini import images as gemini_images


def test_version_is_updated():
    assert __version__ == "0.7.0"


def test_capabilities_use_platform_scoped_tool_names():
    tools = capabilities.list_tools()
    assert "ark.images.generate" in tools
    assert "ark.videos.generate" in tools
    assert "ark.text.complete_json" in tools
    assert "deepseek.text.complete" in tools
    assert "aliyun.images.segment_commodity" in tools
    assert "aliyun.images.segment_hd_body" in tools
    assert "images.generate" not in tools
    assert "chat.complete" not in tools


def test_model_aliases_resolve_to_raw_provider_ids():
    assert ark_images.resolve_model("seedream-5-lite") == "doubao-seedream-5-0-260128"
    assert ark_images.resolve_model("seedream-4.5") == "doubao-seedream-4-5-251128"
    assert ark_videos.resolve_model("seedance-2") == "doubao-seedance-2-0-260128"
    assert ark_text.resolve_model("doubao-pro") == "doubao-seed-2-0-pro-260215"
    assert deepseek_text.resolve_model("v4-pro") == "deepseek-v4-pro"
    assert dashscope_images.resolve_model("wan2.7-pro") == "wan2.7-image-pro"
    assert gemini_images.resolve_model("gemini-image") == "gemini-3.1-flash-image-preview"
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
        model="seedream-5-lite",
        prompt="red square",
        output_path=tmp_path / "out.jpg",
    )

    assert captured["model"] == "doubao-seedream-5-0-260128"
    assert captured["kwargs"]["output_format"] == "jpeg"


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

    assert result.model == "gemini-3.1-flash-image-preview"
    assert "inlineData" in captured["contents"][0]["parts"][1]
    assert captured["kwargs"]["generationConfig"]["imageConfig"]["imageSize"] == "512"


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
