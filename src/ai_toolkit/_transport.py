"""Internal HTTP helpers shared by API clients."""

from __future__ import annotations

import json
from typing import Any
from urllib import error
from urllib import request


def get_json(
    url: str,
    headers: dict[str, str],
    error_cls: type[RuntimeError],
) -> dict[str, Any]:
    req = request.Request(url, headers=headers, method="GET")
    return _request_json(req, error_cls=error_cls)


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    error_cls: type[RuntimeError],
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    return _request_json(req, error_cls=error_cls)


def _request_json(
    req: request.Request,
    error_cls: type[RuntimeError],
) -> dict[str, Any]:
    try:
        with request.urlopen(req, timeout=120) as response:
            raw_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise error_cls(f"API request failed with status {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise error_cls(f"API request failed: {exc.reason}") from exc

    try:
        return json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise error_cls("API returned invalid JSON") from exc


def read_binary(
    url: str,
    headers: dict[str, str],
    error_cls: type[RuntimeError],
) -> tuple[bytes, str]:
    req = request.Request(url, headers=headers)
    try:
        with request.urlopen(req, timeout=120) as response:
            return response.read(), response.headers.get("Content-Type", "")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise error_cls(f"Image download failed with status {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise error_cls(f"Image download failed: {exc.reason}") from exc
