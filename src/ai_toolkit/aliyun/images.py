"""Aliyun image segmentation helpers.

The Aliyun ImageSeg viapi APIs only accept an ``ImageURL`` in a supported OSS
region. For local files this module mirrors Aliyun's recommended helper flow:
stage the local image with ``viapi-utils`` to a managed temporary cn-shanghai
OSS URL, call the segmentation API, then download the transparent PNG result.
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from ai_toolkit.config import get_settings
from ai_toolkit.types import AIToolkitError, ImageSegmentationResult

PROVIDER = "aliyun"
MODEL_SEGMENT_COMMODITY = "viapi-segment-commodity"
MODEL_SEGMENT_HD_BODY = "viapi-segment-hd-body"
DEFAULT_REGION = "cn-shanghai"
VIAPI_INPUT_LONG_SIDE_MAX = 1920
_RESULT_DOWNLOAD_TIMEOUT_SECONDS = 60


class AliyunImageError(AIToolkitError):
    """Raised when Aliyun image segmentation fails."""


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
    payload = invoke_segmentation_request(request, output_path=target, api_name=api_name, region=region)
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


def shrink_to_limit(image_path: str | Path, *, long_side_max: int = VIAPI_INPUT_LONG_SIDE_MAX) -> Path:
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
    return (region or get_settings().aliyun_viapi_region or DEFAULT_REGION).strip() or DEFAULT_REGION


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


def _download(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=_RESULT_DOWNLOAD_TIMEOUT_SECONDS) as response:
        output_path.write_bytes(response.read())
