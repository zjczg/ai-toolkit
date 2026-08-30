from __future__ import annotations

from pathlib import Path

import pytest

from ai_toolkit.grsai import images


def test_generate_uses_independent_grsai_provider_and_gemini_protocol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"\x89PNG\r\n\x1a\n")

    def fake_generate_content(model, contents, **kwargs):
        captured.update(model=model, contents=contents, kwargs=kwargs)
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "done"},
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": "aW1hZ2U=",
                                }
                            },
                        ]
                    }
                }
            ],
            "usageMetadata": {"totalTokenCount": 7},
        }

    monkeypatch.setattr(images, "generate_content", fake_generate_content)
    output = tmp_path / "output.png"

    result = images.generate(
        model="grsai-image",
        prompt="pixel farmer",
        references=[reference],
        output_path=output,
        image_size="2k",
        aspect_ratio="4:3",
    )

    assert result.provider == "grsai"
    assert result.model == "nano-banana-2"
    assert result.text == "done"
    assert result.usage == {"totalTokenCount": 7}
    assert output.read_bytes() == b"image"
    assert captured["model"] == "nano-banana-2"
    parts = captured["contents"][0]["parts"]
    assert parts[1]["inlineData"]["mimeType"] == "image/png"
    assert captured["kwargs"]["generationConfig"]["imageConfig"] == {
        "imageSize": "2K",
        "aspectRatio": "4:3",
    }


def test_generate_content_uses_grsai_key_and_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("GRSAI_API_KEY", "test-key")
    monkeypatch.setenv("GRSAI_BASE_URL", "https://grs.example.test/")

    def fake_post_json(url, payload, headers, error_cls, **kwargs):
        captured.update(
            url=url,
            payload=payload,
            headers=headers,
            error_cls=error_cls,
            kwargs=kwargs,
        )
        return {"ok": True}

    monkeypatch.setattr(images, "post_json", fake_post_json)

    response = images.generate_content(
        "nano-banana-2",
        [{"role": "user", "parts": [{"text": "hello"}]}],
    )

    assert response == {"ok": True}
    assert captured["url"] == (
        "https://grs.example.test/v1beta/models/nano-banana-2:generateContent"
    )
    assert captured["headers"] == {
        "Content-Type": "application/json",
        "x-goog-api-key": "test-key",
    }
    assert captured["error_cls"] is images.GRSAIImageError


@pytest.mark.parametrize("model", ["nano-banana-fast", "nano-banana-2-lite"])
def test_only_nano_banana_2_is_supported(model: str) -> None:
    with pytest.raises(ValueError, match="unsupported GRS.AI image model"):
        images.resolve_model(model)


def test_invalid_size_and_reference_count_fail_before_network(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="image size"):
        images.generate(prompt="test", image_size="512")

    references = [tmp_path / f"{index}.png" for index in range(7)]
    with pytest.raises(ValueError, match="at most 6"):
        images.generate(prompt="test", references=references)
