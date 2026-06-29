"""ARK embeddings: high-level helpers and raw API wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ai_toolkit._transport import post_json
from ai_toolkit.config import get_settings
from ai_toolkit.media import upload_public_url
from ai_toolkit.types import AIToolkitError, EmbeddingResult

MODEL_ALIASES = {
    "doubao-vision": "doubao-embedding-vision-251215",
    "doubao-embedding-vision": "doubao-embedding-vision-251215",
    "doubao-embedding-vision-251215": "doubao-embedding-vision-251215",
}


class ArkEmbeddingError(RuntimeError):
    """Raised when the ARK Embeddings API request fails."""


def generate(
    *,
    input: list[dict[str, Any]] | None = None,
    text: str | list[str] | None = None,
    images: list[str | Path] | None = None,
    model: str | None = None,
    dimensions: int | None = None,
    **kwargs: Any,
) -> EmbeddingResult:
    """Generate ARK multimodal embeddings."""
    resolved_model = resolve_model(model)
    payload_input = _build_payload_input(input=input, text=text, images=images)
    if not payload_input:
        raise ValueError("text, images, or input is required")

    request_kwargs = _drop_none_values(kwargs)
    if dimensions is not None:
        request_kwargs["dimensions"] = dimensions

    raw_response = create_multimodal_embedding(
        resolved_model,
        payload_input,
        **request_kwargs,
    )
    return EmbeddingResult(
        provider="ark",
        model=resolved_model,
        embeddings=_extract_embeddings(raw_response),
        raw_response=raw_response,
        request={"input": payload_input, **request_kwargs},
        usage=raw_response.get("usage") if isinstance(raw_response.get("usage"), dict) else None,
    )


def create_multimodal_embedding(
    model: str,
    input: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    settings = get_settings()
    if not settings.ark_api_key:
        raise ArkEmbeddingError("ARK_API_KEY is not configured")

    payload: dict[str, Any] = {"model": model, "input": input}
    payload.update(kwargs)

    api_url = f"{settings.ark_base_url.rstrip('/')}/embeddings/multimodal"
    return post_json(
        api_url,
        payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.ark_api_key}",
        },
        error_cls=ArkEmbeddingError,
    )


def resolve_model(model: str | None = None) -> str:
    settings = get_settings()
    if model is None or not str(model).strip():
        return settings.ark_embedding_model
    value = str(model).strip()
    return MODEL_ALIASES.get(value, MODEL_ALIASES.get(value.lower(), value))


def _build_payload_input(
    *,
    input: list[dict[str, Any]] | None,
    text: str | list[str] | None,
    images: list[str | Path] | None,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    if input:
        payload.extend(_normalize_input_item(item) for item in input)
    if isinstance(text, list):
        payload.extend({"type": "text", "text": item} for item in text)
    elif text is not None:
        payload.append({"type": "text", "text": text})
    if images:
        payload.extend(_image_reference_to_part(reference) for reference in images)
    return payload


def _normalize_input_item(item: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    if isinstance(item, (str, Path)):
        return {"type": "text", "text": str(item)}
    raise TypeError("input entries must be dict, str, or pathlib.Path")


def _image_reference_to_part(reference: str | Path) -> dict[str, Any]:
    url = _reference_to_url(reference)
    return {"type": "image_url", "image_url": {"url": url}}


def _reference_to_url(reference: str | Path) -> str:
    value = str(reference)
    if _is_http_url(value):
        return value
    return upload_public_url(value)


def _extract_embeddings(response: dict[str, Any]) -> list[list[float]]:
    data = response.get("data")
    vectors: list[list[float]] = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            vector = _coerce_vector(item.get("embedding"))
            if vector is not None:
                vectors.append(vector)
    elif isinstance(data, dict):
        vector = _coerce_vector(data.get("embedding"))
        if vector is not None:
            vectors.append(vector)

    if not vectors:
        single = _coerce_vector(response.get("embedding"))
        if single is not None:
            vectors.append(single)
    if not vectors:
        raise AIToolkitError("ARK embedding API returned no embedding vectors")
    return vectors


def _coerce_vector(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    result: list[float] = []
    for item in value:
        if not isinstance(item, int | float):
            return None
        result.append(float(item))
    return result


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _drop_none_values(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}
