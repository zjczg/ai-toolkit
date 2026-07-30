from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from ai_toolkit import capabilities
from ai_toolkit.aliyun import images as aliyun_images


class SubmitRequest:
    def __init__(self) -> None:
        self.image_url: str | None = None
        self.accept_format: str | None = None

    def set_ImageUrl(self, value: str) -> None:
        self.image_url = value

    def set_accept_format(self, value: str) -> None:
        self.accept_format = value


class PollRequest:
    def __init__(self) -> None:
        self.job_id: str | None = None
        self.accept_format: str | None = None

    def set_JobId(self, value: str) -> None:
        self.job_id = value

    def set_accept_format(self, value: str) -> None:
        self.accept_format = value


class HttpStatusError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status


def _request_class(_module_name: str, class_name: str) -> type[Any]:
    if class_name == "SegmentHDCommonImageRequest":
        return SubmitRequest
    if class_name == "GetAsyncJobResultRequest":
        return PollRequest
    raise AssertionError(class_name)


def test_hd_common_segmentation_submits_polls_and_downloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"source")
    output = tmp_path / "cutout.png"
    requests: list[Any] = []
    responses = iter(
        [
            {"RequestId": "job-1"},
            {"Data": {"Status": "QUEUING", "JobId": "job-1"}},
            {
                "Data": {
                    "Status": "PROCESS_SUCCESS",
                    "JobId": "job-1",
                    "Result": json.dumps({"imageUrl": "https://example.com/cutout.png"}),
                }
            },
        ]
    )

    class FakeClient:
        def do_action_with_exception(self, request: Any) -> bytes:
            requests.append(request)
            return json.dumps(next(responses)).encode()

    def fake_download(url: str, path: Path) -> None:
        assert url == "https://example.com/cutout.png"
        path.write_bytes(b"transparent")

    monkeypatch.setattr(aliyun_images, "_import_request_class", _request_class)
    monkeypatch.setattr(
        aliyun_images,
        "stage_hd_common_image",
        lambda _source: "https://example.com/source.jpg",
    )
    monkeypatch.setattr(aliyun_images, "_build_client", lambda **_kwargs: FakeClient())
    monkeypatch.setattr(aliyun_images, "_download", fake_download)
    monkeypatch.setattr(aliyun_images.time, "sleep", lambda _seconds: None)

    result = aliyun_images.segment_hd_common_image(
        image_path=source,
        output_path=output,
    )

    assert result.model == "viapi-segment-hd-common-image"
    assert result.path == output
    assert output.read_bytes() == b"transparent"
    assert isinstance(requests[0], SubmitRequest)
    assert requests[0].image_url == "https://example.com/source.jpg"
    assert requests[0].accept_format == "JSON"
    assert [request.job_id for request in requests[1:]] == ["job-1", "job-1"]
    assert all(request.accept_format == "JSON" for request in requests[1:])
    assert result.raw_response["submission"]["RequestId"] == "job-1"
    assert "aliyun.images.segment_hd_common_image" in capabilities.list_tools()


def test_hd_common_segmentation_retries_transient_staging_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"source")
    output = tmp_path / "cutout.png"
    staging_attempts = 0
    sleep_calls: list[float] = []

    def fake_stage(_source: Path) -> str:
        nonlocal staging_attempts
        staging_attempts += 1
        if staging_attempts < 3:
            raise HttpStatusError(502)
        return "https://example.com/source.jpg"

    def fake_invoke(_request: Any, **kwargs: Any) -> dict[str, Any]:
        Path(kwargs["output_path"]).write_bytes(b"transparent")
        return {"submission": {"RequestId": "job-retry"}}

    monkeypatch.setattr(aliyun_images, "_import_request_class", _request_class)
    monkeypatch.setattr(aliyun_images, "stage_hd_common_image", fake_stage)
    monkeypatch.setattr(aliyun_images, "invoke_async_segmentation_request", fake_invoke)
    monkeypatch.setattr(aliyun_images.time, "sleep", sleep_calls.append)

    result = aliyun_images.segment_hd_common_image(
        image_path=source,
        output_path=output,
    )

    assert result.path == output
    assert output.read_bytes() == b"transparent"
    assert staging_attempts == 3
    assert sleep_calls == [2.0, 4.0]


def test_hd_common_segmentation_does_not_retry_permanent_staging_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"source")
    staging_attempts = 0
    sleep_calls: list[float] = []

    def fake_stage(_source: Path) -> str:
        nonlocal staging_attempts
        staging_attempts += 1
        raise HttpStatusError(403)

    monkeypatch.setattr(aliyun_images, "_import_request_class", _request_class)
    monkeypatch.setattr(aliyun_images, "stage_hd_common_image", fake_stage)
    monkeypatch.setattr(aliyun_images.time, "sleep", sleep_calls.append)

    with pytest.raises(HttpStatusError):
        aliyun_images.segment_hd_common_image(
            image_path=source,
            output_path=tmp_path / "cutout.png",
        )

    assert staging_attempts == 1
    assert sleep_calls == []


def test_hd_common_segmentation_retries_result_download_without_resubmitting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = SubmitRequest()
    requests: list[Any] = []
    download_attempts = 0
    sleep_calls: list[float] = []
    responses = iter(
        [
            {"RequestId": "job-download"},
            {
                "Data": {
                    "Status": "PROCESS_SUCCESS",
                    "Result": json.dumps({"imageUrl": "https://example.com/cutout.png"}),
                }
            },
        ]
    )

    class FakeClient:
        def do_action_with_exception(self, current_request: Any) -> bytes:
            requests.append(current_request)
            return json.dumps(next(responses)).encode()

    def fake_download(_url: str, path: Path) -> None:
        nonlocal download_attempts
        download_attempts += 1
        if download_attempts == 1:
            raise HttpStatusError(502)
        path.write_bytes(b"transparent")

    monkeypatch.setattr(aliyun_images, "_import_request_class", _request_class)
    monkeypatch.setattr(aliyun_images, "_build_client", lambda **_kwargs: FakeClient())
    monkeypatch.setattr(aliyun_images, "_download", fake_download)
    monkeypatch.setattr(aliyun_images.time, "sleep", sleep_calls.append)

    result = aliyun_images.invoke_async_segmentation_request(
        request,
        output_path=tmp_path / "cutout.png",
        api_name="SegmentHDCommonImage",
    )

    assert result["submission"]["RequestId"] == "job-download"
    assert sum(isinstance(item, SubmitRequest) for item in requests) == 1
    assert download_attempts == 2
    assert sleep_calls == [2.0]


def test_hd_common_segmentation_retries_poll_without_resubmitting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = SubmitRequest()
    submit_count = 0
    poll_job_ids: list[str | None] = []
    sleep_calls: list[float] = []

    class FakeClient:
        def do_action_with_exception(self, current_request: Any) -> bytes:
            nonlocal submit_count
            if isinstance(current_request, SubmitRequest):
                submit_count += 1
                return json.dumps({"RequestId": "job-poll"}).encode()
            poll_job_ids.append(current_request.job_id)
            if len(poll_job_ids) == 1:
                raise HttpStatusError(502)
            return json.dumps(
                {
                    "Data": {
                        "Status": "PROCESS_SUCCESS",
                        "Result": json.dumps(
                            {"imageUrl": "https://example.com/cutout.png"}
                        ),
                    }
                }
            ).encode()

    def fake_download(_url: str, path: Path) -> None:
        path.write_bytes(b"transparent")

    monkeypatch.setattr(aliyun_images, "_import_request_class", _request_class)
    monkeypatch.setattr(aliyun_images, "_build_client", lambda **_kwargs: FakeClient())
    monkeypatch.setattr(aliyun_images, "_download", fake_download)
    monkeypatch.setattr(aliyun_images.time, "sleep", sleep_calls.append)

    result = aliyun_images.invoke_async_segmentation_request(
        request,
        output_path=tmp_path / "cutout.png",
        api_name="SegmentHDCommonImage",
    )

    assert result["submission"]["RequestId"] == "job-poll"
    assert submit_count == 1
    assert poll_job_ids == ["job-poll", "job-poll"]
    assert sleep_calls == [2.0]


@pytest.mark.parametrize(
    "status",
    ["PROCESS_FAILED", "TIMEOUT_FAILED", "LIMIT_RETRY_FAILED"],
)
def test_hd_common_segmentation_stops_on_failed_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: str,
) -> None:
    request = SubmitRequest()
    responses = iter(
        [
            {"RequestId": "job-2"},
            {
                "Data": {
                    "Status": status,
                    "ErrorMessage": "segmentation failed",
                }
            },
        ]
    )

    class FakeClient:
        def do_action_with_exception(self, _request: Any) -> bytes:
            return json.dumps(next(responses)).encode()

    monkeypatch.setattr(aliyun_images, "_import_request_class", _request_class)
    monkeypatch.setattr(aliyun_images, "_build_client", lambda **_kwargs: FakeClient())
    monkeypatch.setattr(
        aliyun_images,
        "_download",
        lambda *_args: pytest.fail("failed jobs must not download"),
    )

    with pytest.raises(aliyun_images.AliyunImageError, match=status):
        aliyun_images.invoke_async_segmentation_request(
            request,
            output_path=tmp_path / "cutout.png",
            api_name="SegmentHDCommonImage",
        )


def test_hd_common_segmentation_times_out_while_processing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = SubmitRequest()
    responses = iter(
        [
            {"RequestId": "job-3"},
            {"Data": {"Status": "PROCESSING"}},
        ]
    )

    class FakeClient:
        def do_action_with_exception(self, _request: Any) -> bytes:
            return json.dumps(next(responses)).encode()

    times = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(aliyun_images, "_import_request_class", _request_class)
    monkeypatch.setattr(aliyun_images, "_build_client", lambda **_kwargs: FakeClient())
    monkeypatch.setattr(aliyun_images.time, "monotonic", lambda: next(times))

    with pytest.raises(aliyun_images.AliyunImageError, match="timed out"):
        aliyun_images.invoke_async_segmentation_request(
            request,
            output_path=tmp_path / "cutout.png",
            api_name="SegmentHDCommonImage",
            timeout_seconds=1.0,
        )


def test_hd_common_staging_keeps_a_2048_image_at_original_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (2048, 2048), "white").save(source)
    staged_paths: list[Path] = []

    class FakeFileUtils:
        def get_oss_url(self, path: str, suffix: str, is_local: bool) -> str:
            staged_paths.append(Path(path))
            assert suffix == "png"
            assert is_local is True
            return "https://example.com/source.png"

    monkeypatch.setattr(aliyun_images, "_build_file_utils", FakeFileUtils)

    url = aliyun_images.stage_hd_common_image(source)

    assert url == "https://example.com/source.png"
    assert staged_paths == [source]
    with Image.open(source) as image:
        assert image.size == (2048, 2048)
