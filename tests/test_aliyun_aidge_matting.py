from types import SimpleNamespace

import pytest

from ai_toolkit import capabilities
from ai_toolkit.aliyun import images as aliyun_images


def test_image_matting_stages_invokes_and_saves(monkeypatch, tmp_path):
    source = tmp_path / "pet.png"
    source.write_bytes(b"source")
    output = tmp_path / "pet-cutout.png"
    captured = {}

    class FakeRequest:
        def __init__(self, **kwargs):
            captured["request"] = kwargs

    body = SimpleNamespace(
        success=True,
        code="success",
        message="Success",
        data=SimpleNamespace(
            image_url="https://aidge.example/pet-cutout.png"
        ),
        to_map=lambda: {
            "Success": True,
            "Code": "success",
            "Data": {
                "ImageUrl": "https://aidge.example/pet-cutout.png"
            },
        },
    )

    class FakeClient:
        def image_matting(self, request):
            captured["client_request"] = request
            return SimpleNamespace(body=body)

    monkeypatch.setattr(
        aliyun_images,
        "stage_aidge_image",
        lambda image_path, *, long_side_max: (
            "https://oss.example/pet.png"
        ),
    )
    monkeypatch.setattr(
        aliyun_images,
        "_build_aidge_client",
        lambda: (FakeClient(), FakeRequest),
    )

    def fake_download(url, output_path):
        captured["download"] = (url, output_path)
        output_path.write_bytes(b"transparent-png")

    monkeypatch.setattr(aliyun_images, "_download", fake_download)

    result = aliyun_images.image_matting(
        image_path=source,
        output_path=output,
    )

    assert result.provider == "aliyun"
    assert result.model == "aidge-image-matting"
    assert result.path == output.resolve()
    assert output.read_bytes() == b"transparent-png"
    assert captured["request"] == {
        "image_url": "https://oss.example/pet.png",
        "back_ground_type": "TRANSPARENT",
    }
    assert captured["download"] == (
        "https://aidge.example/pet-cutout.png",
        output.resolve(),
    )
    assert "aliyun.images.image_matting" in capabilities.list_tools()


def test_image_matting_does_not_download_failed_response(
    monkeypatch,
    tmp_path,
):
    class FakeRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    body = SimpleNamespace(
        success=False,
        code="InvalidParameter",
        message="bad image",
        data=None,
        to_map=lambda: {"Success": False},
    )
    client = SimpleNamespace(
        image_matting=lambda request: SimpleNamespace(body=body)
    )
    monkeypatch.setattr(
        aliyun_images,
        "stage_aidge_image",
        lambda image_path, *, long_side_max: "https://oss.example/bad.png",
    )
    monkeypatch.setattr(
        aliyun_images,
        "_build_aidge_client",
        lambda: (client, FakeRequest),
    )
    monkeypatch.setattr(
        aliyun_images,
        "_download",
        lambda *args, **kwargs: pytest.fail("download must not run"),
    )
    source = tmp_path / "bad.png"
    source.write_bytes(b"source")

    with pytest.raises(
        aliyun_images.AliyunImageError,
        match="InvalidParameter bad image",
    ):
        aliyun_images.image_matting(
            image_path=source,
            output_path=tmp_path / "result.png",
        )
