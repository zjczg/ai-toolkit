"""Aliyun image foreground extraction helpers.

Aliyun's vision APIs accept a public ``ImageURL``. For local files this module
stages the image with ``viapi-utils`` to a managed temporary OSS URL, invokes
the selected API, then immediately downloads the temporary result URL.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from PIL import Image

from ai_toolkit.config import get_settings
from ai_toolkit.types import AIToolkitError, ImageSegmentationResult

PROVIDER = "aliyun"
MODEL_IMAGE_MATTING = "aidge-image-matting"
MODEL_SEGMENT_COMMODITY = "viapi-segment-commodity"
MODEL_SEGMENT_HD_BODY = "viapi-segment-hd-body"
MODEL_SEGMENT_HD_COMMON_IMAGE = "viapi-segment-hd-common-image"
DEFAULT_REGION = "cn-shanghai"
AIDGE_REGION = "cn-beijing"
AIDGE_ENDPOINT = "aidge.cn-beijing.aliyuncs.com"
AIDGE_INPUT_LONG_SIDE_MAX = 3000
AIDGE_INPUT_MIN_SIDE = 256
AIDGE_INPUT_MAX_BYTES = 10_000_000
VIAPI_INPUT_LONG_SIDE_MAX = 1920
VIAPI_HD_INPUT_LONG_SIDE_MAX = 9999
VIAPI_HD_INPUT_MIN_SIDE = 33
VIAPI_HD_INPUT_MAX_BYTES = 40_000_000
VIAPI_ASYNC_POLL_INTERVAL_SECONDS = 1.0
VIAPI_ASYNC_TIMEOUT_SECONDS = 120.0
_VIAPI_ASYNC_PENDING_STATUSES = {"QUEUING", "PROCESSING"}
_VIAPI_ASYNC_FAILURE_STATUSES = {
    "PROCESS_FAILED",
    "TIMEOUT_FAILED",
    "LIMIT_RETRY_FAILED",
}
_RESULT_DOWNLOAD_TIMEOUT_SECONDS = 60
_HD_RETRYABLE_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}
_HD_MAX_RETRIES = 3
_HD_RETRY_BASE_SECONDS = 2.0
_HD_RETRY_MAX_SECONDS = 8.0

RetryResultT = TypeVar("RetryResultT")


class AliyunImageError(AIToolkitError):
    """Raised when Aliyun foreground extraction fails."""


def image_matting(
    *,
    image_path: str | Path,
    output_path: str | Path,
    long_side_max: int = AIDGE_INPUT_LONG_SIDE_MAX,
) -> ImageSegmentationResult:
    """Extract the salient subject with Aidge ImageMatting as transparent PNG."""
    source = Path(image_path).expanduser().resolve()
    if not source.exists():
        raise AliyunImageError(f"input image does not exist: {source}")
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    staged_url = stage_aidge_image(source, long_side_max=long_side_max)
    payload = invoke_image_matting(staged_url, output_path=target)
    return ImageSegmentationResult(
        provider=PROVIDER,
        model=MODEL_IMAGE_MATTING,
        path=target,
        raw_response=payload,
        request={
            "api_name": "ImageMatting",
            "image_path": str(source),
            "staged_url": staged_url,
            "endpoint": AIDGE_ENDPOINT,
            "background_type": "TRANSPARENT",
            "long_side_max": long_side_max,
        },
    )


def invoke_image_matting(
    image_url: str,
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Invoke synchronous Aidge ImageMatting and download its result."""
    client, request_cls = _build_aidge_client()
    request = request_cls(
        image_url=image_url,
        back_ground_type="TRANSPARENT",
    )
    response = client.image_matting(request)
    body = getattr(response, "body", None)
    payload = body.to_map() if body is not None else {}
    if body is None or getattr(body, "success", None) is not True:
        raise AliyunImageError(
            f"ImageMatting failed: {getattr(body, 'code', None)} {getattr(body, 'message', None)}"
        )
    data = getattr(body, "data", None)
    result_url = getattr(data, "image_url", None)
    if not isinstance(result_url, str) or not result_url:
        raise AliyunImageError(f"ImageMatting: missing Data.ImageUrl in response: {payload!r}")
    _download(result_url, Path(output_path).expanduser().resolve())
    return payload


def segment_commodity(
    *,
    image_path: str | Path,
    output_path: str | Path,
    region: str | None = None,
    long_side_max: int = VIAPI_INPUT_LONG_SIDE_MAX,
) -> ImageSegmentationResult:
    """Segment a commodity/product image and save the transparent result."""
    return _segment(
        image_path=image_path,
        output_path=output_path,
        model=MODEL_SEGMENT_COMMODITY,
        api_name="SegmentCommodity",
        request_import=(
            "aliyunsdkimageseg.request.v20191230.SegmentCommodityRequest",
            "SegmentCommodityRequest",
        ),
        region=region,
        long_side_max=long_side_max,
    )


def segment_hd_body(
    *,
    image_path: str | Path,
    output_path: str | Path,
    region: str | None = None,
    long_side_max: int = VIAPI_INPUT_LONG_SIDE_MAX,
) -> ImageSegmentationResult:
    """Segment a human body image and save the transparent result."""
    return _segment(
        image_path=image_path,
        output_path=output_path,
        model=MODEL_SEGMENT_HD_BODY,
        api_name="SegmentHDBody",
        request_import=(
            "aliyunsdkimageseg.request.v20191230.SegmentHDBodyRequest",
            "SegmentHDBodyRequest",
        ),
        region=region,
        long_side_max=long_side_max,
    )


def segment_hd_common_image(
    *,
    image_path: str | Path,
    output_path: str | Path,
    region: str | None = None,
    poll_interval_seconds: float = VIAPI_ASYNC_POLL_INTERVAL_SECONDS,
    timeout_seconds: float = VIAPI_ASYNC_TIMEOUT_SECONDS,
    max_retries: int = _HD_MAX_RETRIES,
    retry_base_seconds: float = _HD_RETRY_BASE_SECONDS,
    retry_max_seconds: float = _HD_RETRY_MAX_SECONDS,
) -> ImageSegmentationResult:
    """Segment a general foreground with the asynchronous HD ImageSeg API."""
    source = Path(image_path).expanduser().resolve()
    if not source.exists():
        raise AliyunImageError(f"input image does not exist: {source}")
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    request_cls = _import_request_class(
        "aliyunsdkimageseg.request.v20191230.SegmentHDCommonImageRequest",
        "SegmentHDCommonImageRequest",
    )
    request = request_cls()
    staged_url = _retry_transient_call(
        lambda: stage_hd_common_image(source),
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
    )
    request.set_ImageUrl(staged_url)
    payload = invoke_async_segmentation_request(
        request,
        output_path=target,
        api_name="SegmentHDCommonImage",
        region=region,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
    )
    return ImageSegmentationResult(
        provider=PROVIDER,
        model=MODEL_SEGMENT_HD_COMMON_IMAGE,
        path=target,
        raw_response=payload,
        request={
            "api_name": "SegmentHDCommonImage",
            "image_path": str(source),
            "staged_url": staged_url,
            "region": _resolved_region(region),
            "async": True,
            "max_retries": max_retries,
        },
    )


def _segment(
    *,
    image_path: str | Path,
    output_path: str | Path,
    model: str,
    api_name: str,
    request_import: tuple[str, str],
    region: str | None,
    long_side_max: int,
) -> ImageSegmentationResult:
    source = Path(image_path).expanduser().resolve()
    if not source.exists():
        raise AliyunImageError(f"input image does not exist: {source}")
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    request_cls = _import_request_class(*request_import)
    request = request_cls()
    staged_url = stage_image(source, long_side_max=long_side_max)
    request.set_ImageURL(staged_url)
    payload = invoke_segmentation_request(
        request, output_path=target, api_name=api_name, region=region
    )
    return ImageSegmentationResult(
        provider=PROVIDER,
        model=model,
        path=target,
        raw_response=payload,
        request={
            "api_name": api_name,
            "image_path": str(source),
            "staged_url": staged_url,
            "region": _resolved_region(region),
            "long_side_max": long_side_max,
        },
    )


def invoke_segmentation_request(
    request: Any,
    *,
    output_path: str | Path,
    api_name: str,
    region: str | None = None,
) -> dict[str, Any]:
    """Invoke an Aliyun ImageSeg request and download Data.ImageURL."""
    request.set_accept_format("JSON")
    client = _build_client(region=region)
    raw = client.do_action_with_exception(request)
    payload = json.loads(raw)
    result_url = _extract_result_url(payload, api_name=api_name)
    _download(result_url, Path(output_path).expanduser().resolve())
    return payload


def invoke_async_segmentation_request(
    request: Any,
    *,
    output_path: str | Path,
    api_name: str,
    region: str | None = None,
    poll_interval_seconds: float = VIAPI_ASYNC_POLL_INTERVAL_SECONDS,
    timeout_seconds: float = VIAPI_ASYNC_TIMEOUT_SECONDS,
    max_retries: int = _HD_MAX_RETRIES,
    retry_base_seconds: float = _HD_RETRY_BASE_SECONDS,
    retry_max_seconds: float = _HD_RETRY_MAX_SECONDS,
) -> dict[str, Any]:
    """Submit an asynchronous ImageSeg request, poll it, and save its result."""
    request.set_accept_format("JSON")
    client = _build_client(region=region)
    submission = json.loads(
        _retry_transient_call(
            lambda: client.do_action_with_exception(request),
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
        )
    )
    job_id = submission.get("RequestId")
    if not isinstance(job_id, str) or not job_id:
        raise AliyunImageError(f"{api_name}: missing RequestId in submission")

    poll_cls = _import_request_class(
        "aliyunsdkimageseg.request.v20191230.GetAsyncJobResultRequest",
        "GetAsyncJobResultRequest",
    )
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        poll_request = poll_cls()
        poll_request.set_JobId(job_id)
        poll_request.set_accept_format("JSON")
        result = json.loads(
            _retry_transient_call(
                lambda poll_request=poll_request: client.do_action_with_exception(
                    poll_request
                ),
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
            )
        )
        data = result.get("Data")
        if not isinstance(data, dict):
            raise AliyunImageError(f"{api_name}: missing Data in job result")

        status = data.get("Status")
        if status == "PROCESS_SUCCESS":
            result_data = _decode_async_result(data.get("Result"), api_name=api_name)
            result_url = _extract_image_url(result_data, api_name=api_name)
            _retry_transient_call(
                lambda result_url=result_url: _download(
                    result_url,
                    Path(output_path).expanduser().resolve(),
                ),
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
            )
            return {"submission": submission, "result": result}
        if status in _VIAPI_ASYNC_FAILURE_STATUSES:
            message = data.get("ErrorMessage") or data.get("ErrorCode") or status
            raise AliyunImageError(f"{api_name} failed ({status}): {message}")
        if status not in _VIAPI_ASYNC_PENDING_STATUSES:
            raise AliyunImageError(f"{api_name}: unknown job status: {status!r}")
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            break
        time.sleep(min(poll_interval_seconds, remaining_seconds))
    raise AliyunImageError(f"{api_name}: timed out waiting for job {job_id}")


def stage_image(image_path: str | Path, *, long_side_max: int = VIAPI_INPUT_LONG_SIDE_MAX) -> str:
    """Stage a local image to the viapi temporary OSS bucket and return its URL."""
    source = Path(image_path).expanduser().resolve()
    suffix = source.suffix.lstrip(".").lower() or "png"
    staged_path = shrink_to_limit(source, long_side_max=long_side_max)
    try:
        return _build_file_utils().get_oss_url(str(staged_path), suffix, True)
    finally:
        if staged_path != source:
            staged_path.unlink(missing_ok=True)


def stage_aidge_image(
    image_path: str | Path,
    *,
    long_side_max: int = AIDGE_INPUT_LONG_SIDE_MAX,
) -> str:
    """Validate and stage a local image for Aidge ImageMatting."""
    source = Path(image_path).expanduser().resolve()
    with Image.open(source) as image:
        if min(image.size) < AIDGE_INPUT_MIN_SIDE:
            raise AliyunImageError(
                "ImageMatting input dimensions must be at least "
                f"{AIDGE_INPUT_MIN_SIDE}x{AIDGE_INPUT_MIN_SIDE}: {image.size}"
            )

    suffix = source.suffix.lstrip(".").lower() or "png"
    staged_path = shrink_to_limit(source, long_side_max=long_side_max)
    try:
        if staged_path.stat().st_size > AIDGE_INPUT_MAX_BYTES:
            raise AliyunImageError("ImageMatting input must not exceed 10 MB after resizing")
        return _build_file_utils().get_oss_url(str(staged_path), suffix, True)
    finally:
        if staged_path != source:
            staged_path.unlink(missing_ok=True)


def stage_hd_common_image(image_path: str | Path) -> str:
    """Validate and stage a local image without reducing ordinary HD inputs."""
    source = Path(image_path).expanduser().resolve()
    with Image.open(source) as image:
        if min(image.size) < VIAPI_HD_INPUT_MIN_SIDE:
            raise AliyunImageError(
                f"SegmentHDCommonImage input dimensions must be greater than 32x32: {image.size}"
            )

    suffix = source.suffix.lstrip(".").lower() or "png"
    staged_path = shrink_to_limit(
        source,
        long_side_max=VIAPI_HD_INPUT_LONG_SIDE_MAX,
    )
    try:
        if staged_path.stat().st_size > VIAPI_HD_INPUT_MAX_BYTES:
            raise AliyunImageError("SegmentHDCommonImage input must not exceed 40 MB")
        return _build_file_utils().get_oss_url(str(staged_path), suffix, True)
    finally:
        if staged_path != source:
            staged_path.unlink(missing_ok=True)


def shrink_to_limit(
    image_path: str | Path, *, long_side_max: int = VIAPI_INPUT_LONG_SIDE_MAX
) -> Path:
    """Return image_path or a temporary resized copy whose long side is within limit."""
    source = Path(image_path).expanduser().resolve()
    with Image.open(source) as image:
        long_side = max(image.size)
        if long_side <= long_side_max:
            return source
        scale = long_side_max / long_side
        new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        resized = image.resize(new_size, Image.Resampling.LANCZOS)
    fd, tmp_name = tempfile.mkstemp(suffix=source.suffix or ".png", prefix="viapi-")
    os.close(fd)
    target = Path(tmp_name)
    resized.save(target)
    return target


def _build_aidge_client() -> tuple[Any, type[Any]]:
    try:
        from alibabacloud_aidge20260428 import models
        from alibabacloud_aidge20260428.client import Client
        from alibabacloud_tea_openapi import models as open_api_models
    except ImportError as exc:
        raise AliyunImageError(
            "alibabacloud-aidge20260428 is not installed. Install ai-toolkit[aliyun]."
        ) from exc

    settings = get_settings()
    _require_credentials(
        settings.aliyun_access_key_id,
        settings.aliyun_access_key_secret,
    )
    config = open_api_models.Config(
        access_key_id=settings.aliyun_access_key_id,
        access_key_secret=settings.aliyun_access_key_secret,
        region_id=AIDGE_REGION,
        endpoint=AIDGE_ENDPOINT,
    )
    return Client(config), models.ImageMattingRequest


def _build_client(*, region: str | None = None) -> Any:
    try:
        from aliyunsdkcore.client import AcsClient
    except ImportError as exc:
        raise AliyunImageError(
            "aliyun-python-sdk-core is not installed. Install ai-toolkit[aliyun] or the required Aliyun SDK packages."
        ) from exc
    settings = get_settings()
    _require_credentials(settings.aliyun_access_key_id, settings.aliyun_access_key_secret)
    return AcsClient(
        settings.aliyun_access_key_id,
        settings.aliyun_access_key_secret,
        _resolved_region(region),
    )


def _build_file_utils() -> Any:
    try:
        from viapi.fileutils import FileUtils
    except ImportError as exc:
        raise AliyunImageError(
            "viapi-utils or oss2 is not installed. Install ai-toolkit[aliyun]."
        ) from exc
    settings = get_settings()
    _require_credentials(settings.aliyun_access_key_id, settings.aliyun_access_key_secret)
    return FileUtils(settings.aliyun_access_key_id, settings.aliyun_access_key_secret)


def _import_request_class(module_name: str, class_name: str) -> type[Any]:
    try:
        module = __import__(module_name, fromlist=[class_name])
    except ImportError as exc:
        raise AliyunImageError(
            "aliyun-python-sdk-imageseg is not installed. Install ai-toolkit[aliyun] or aliyun-python-sdk-imageseg."
        ) from exc
    return getattr(module, class_name)


def _resolved_region(region: str | None) -> str:
    return (
        region or get_settings().aliyun_viapi_region or DEFAULT_REGION
    ).strip() or DEFAULT_REGION


def _require_credentials(access_key_id: str, access_key_secret: str) -> None:
    if not access_key_id or not access_key_secret:
        raise AliyunImageError(
            "Aliyun credentials are not configured: ALIYUN_ACCESS_KEY_ID / ALIYUN_ACCESS_KEY_SECRET"
        )


def _extract_result_url(payload: dict[str, Any], *, api_name: str) -> str:
    data = payload.get("Data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        raise AliyunImageError(f"{api_name}: missing Data in response: {payload!r}")
    url = data.get("ImageURL") or data.get("ImageUrl")
    if not isinstance(url, str) or not url:
        raise AliyunImageError(f"{api_name}: missing Data.ImageURL in response: {payload!r}")
    return url


def _decode_async_result(value: Any, *, api_name: str) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        raise AliyunImageError(f"{api_name}: missing Data.Result")
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AliyunImageError(f"{api_name}: invalid Data.Result JSON") from exc
    if not isinstance(result, dict):
        raise AliyunImageError(f"{api_name}: Data.Result must be an object")
    return result


def _extract_image_url(payload: dict[str, Any], *, api_name: str) -> str:
    url = payload.get("imageUrl") or payload.get("ImageUrl") or payload.get("ImageURL")
    if not isinstance(url, str) or not url:
        raise AliyunImageError(f"{api_name}: missing ImageUrl in job result")
    return url


def _retry_transient_call(
    operation: Callable[[], RetryResultT],
    *,
    max_retries: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
) -> RetryResultT:
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    if retry_base_seconds <= 0 or retry_max_seconds <= 0:
        raise ValueError("retry delays must be positive")

    retry_count = 0
    while True:
        try:
            return operation()
        except Exception as exc:
            if retry_count >= max_retries or not _is_retryable_network_error(exc):
                raise
            delay = min(
                retry_base_seconds * (2**retry_count),
                retry_max_seconds,
            )
            retry_count += 1
            time.sleep(delay)


def _is_retryable_network_error(exc: Exception) -> bool:
    status = _error_http_status(exc)
    if status is not None:
        return status in _HD_RETRYABLE_HTTP_STATUSES
    if isinstance(
        exc,
        (
            urllib.error.URLError,
            http.client.RemoteDisconnected,
            ConnectionError,
            TimeoutError,
            socket.timeout,
        ),
    ):
        return True
    try:
        from oss2.exceptions import RequestError
    except ImportError:
        return False
    return isinstance(exc, RequestError)


def _error_http_status(exc: Exception) -> int | None:
    if isinstance(exc, urllib.error.HTTPError):
        return int(exc.code)
    status = getattr(exc, "status", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status
    get_http_status = getattr(exc, "get_http_status", None)
    if callable(get_http_status):
        value = get_http_status()
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _download(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=_RESULT_DOWNLOAD_TIMEOUT_SECONDS) as response:
        output_path.write_bytes(response.read())
