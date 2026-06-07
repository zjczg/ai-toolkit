"""Internal HTTP helpers shared by API clients."""

from __future__ import annotations

import json
import http.client
import random
import socket
import time
from typing import Any
from urllib import error
from urllib import request

from ai_toolkit.config import get_settings

_RETRYABLE_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}
_RETRYABLE_NETWORK_ERRORS = (
    http.client.RemoteDisconnected,
    ConnectionResetError,
    TimeoutError,
    socket.timeout,
)


def get_json(
    url: str,
    headers: dict[str, str],
    error_cls: type[RuntimeError],
    max_retries: int | None = None,
    retry_base_seconds: float | None = None,
    retry_max_seconds: float | None = None,
    retriable_statuses: set[int] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    req = request.Request(url, headers=headers, method="GET")
    return _request_json(
        req,
        error_cls=error_cls,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
        retriable_statuses=retriable_statuses,
        timeout=timeout,
    )


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    error_cls: type[RuntimeError],
    max_retries: int | None = None,
    retry_base_seconds: float | None = None,
    retry_max_seconds: float | None = None,
    retriable_statuses: set[int] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    return _request_json(
        req,
        error_cls=error_cls,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
        retriable_statuses=retriable_statuses,
        timeout=timeout,
    )


def _request_json(
    req: request.Request,
    error_cls: type[RuntimeError],
    max_retries: int | None = None,
    retry_base_seconds: float | None = None,
    retry_max_seconds: float | None = None,
    retriable_statuses: set[int] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    resolved_max_retries = (
        settings.http_max_retries if max_retries is None else max_retries
    )
    resolved_retry_base = (
        settings.http_retry_base_seconds
        if retry_base_seconds is None
        else retry_base_seconds
    )
    resolved_retry_max = (
        settings.http_retry_max_seconds if retry_max_seconds is None else retry_max_seconds
    )
    resolved_timeout = settings.http_timeout_seconds if timeout is None else timeout
    resolved_retriable_statuses = (
        set(_RETRYABLE_HTTP_STATUSES)
        if retriable_statuses is None
        else set(retriable_statuses)
    )

    attempt = 0
    while True:
        try:
            with request.urlopen(req, timeout=resolved_timeout) as response:
                raw_body = response.read().decode("utf-8")
                break
        except error.HTTPError as exc:
            status = exc.code or 0
            detail = exc.read().decode("utf-8", errors="replace")
            if status in resolved_retriable_statuses and attempt < resolved_max_retries:
                attempt += 1
                retry_after = _get_retry_after_seconds(exc)
                _sleep_for_retry(
                    attempt=attempt,
                    base_seconds=resolved_retry_base,
                    max_seconds=resolved_retry_max,
                    retry_after_seconds=retry_after,
                )
                continue
            raise error_cls(f"API request failed with status {status}: {detail}") from exc
        except error.URLError as exc:
            if attempt < resolved_max_retries:
                attempt += 1
                _sleep_for_retry(
                    attempt=attempt,
                    base_seconds=resolved_retry_base,
                    max_seconds=resolved_retry_max,
                    retry_after_seconds=None,
                )
                continue
            raise error_cls(f"API request failed: {exc.reason}") from exc
        except _RETRYABLE_NETWORK_ERRORS as exc:
            if attempt < resolved_max_retries:
                attempt += 1
                _sleep_for_retry(
                    attempt=attempt,
                    base_seconds=resolved_retry_base,
                    max_seconds=resolved_retry_max,
                    retry_after_seconds=None,
                )
                continue
            raise error_cls(f"API request failed: {exc}") from exc

    try:
        return json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise error_cls("API returned invalid JSON") from exc


def read_binary(
    url: str,
    headers: dict[str, str],
    error_cls: type[RuntimeError],
    resource_name: str = "Resource",
    max_retries: int | None = None,
    retry_base_seconds: float | None = None,
    retry_max_seconds: float | None = None,
    retriable_statuses: set[int] | None = None,
    timeout: float | None = None,
) -> tuple[bytes, str]:
    req = request.Request(url, headers=headers)
    settings = get_settings()
    resolved_max_retries = (
        settings.http_max_retries if max_retries is None else max_retries
    )
    resolved_retry_base = (
        settings.http_retry_base_seconds
        if retry_base_seconds is None
        else retry_base_seconds
    )
    resolved_retry_max = (
        settings.http_retry_max_seconds if retry_max_seconds is None else retry_max_seconds
    )
    resolved_timeout = settings.http_timeout_seconds if timeout is None else timeout
    resolved_retriable_statuses = (
        set(_RETRYABLE_HTTP_STATUSES)
        if retriable_statuses is None
        else set(retriable_statuses)
    )

    attempt = 0
    while True:
        try:
            with request.urlopen(req, timeout=resolved_timeout) as response:
                return response.read(), response.headers.get("Content-Type", "")
        except error.HTTPError as exc:
            status = exc.code or 0
            detail = exc.read().decode("utf-8", errors="replace")
            if status in resolved_retriable_statuses and attempt < resolved_max_retries:
                attempt += 1
                retry_after = _get_retry_after_seconds(exc)
                _sleep_for_retry(
                    attempt=attempt,
                    base_seconds=resolved_retry_base,
                    max_seconds=resolved_retry_max,
                    retry_after_seconds=retry_after,
                )
                continue
            raise error_cls(f"{resource_name} download failed with status {status}: {detail}") from exc
        except error.URLError as exc:
            if attempt < resolved_max_retries:
                attempt += 1
                _sleep_for_retry(
                    attempt=attempt,
                    base_seconds=resolved_retry_base,
                    max_seconds=resolved_retry_max,
                )
                continue
            raise error_cls(f"{resource_name} download failed: {exc.reason}") from exc
        except _RETRYABLE_NETWORK_ERRORS as exc:
            if attempt < resolved_max_retries:
                attempt += 1
                _sleep_for_retry(
                    attempt=attempt,
                    base_seconds=resolved_retry_base,
                    max_seconds=resolved_retry_max,
                    retry_after_seconds=None,
                )
                continue
            raise error_cls(f"{resource_name} download failed: {exc}") from exc


def _get_retry_after_seconds(exc: error.HTTPError) -> float | None:
    header_value = exc.headers.get("Retry-After") if exc.headers else None
    if not header_value:
        return None
    value = header_value.strip()
    if not value:
        return None
    if value.isdigit():
        return float(value)
    return None


def _sleep_for_retry(
    attempt: int,
    base_seconds: float,
    max_seconds: float,
    retry_after_seconds: float | None = None,
) -> None:
    # attempt starts at 1 on first retry.
    delay = min(base_seconds * (2 ** (attempt - 1)), max_seconds)
    if retry_after_seconds is not None:
        delay = max(delay, retry_after_seconds)
    # add jitter for herd prevention
    delay += random.uniform(0.0, min(0.5, delay * 0.3))
    time.sleep(delay)
