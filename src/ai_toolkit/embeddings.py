"""High-level provider-agnostic multimodal embedding helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ai_toolkit.ark import create_multimodal_embedding
from ai_toolkit.config import get_settings
from ai_toolkit.media import upload_public_url
from ai_toolkit.types import AIToolkitError
from ai_toolkit.types import EmbeddingResult


def generate(
    *,
    provider: str,
    input: list[dict[str, Any]] | None = None,
    text: str | list[str] | None = None,
    images: list[str | Path] | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> EmbeddingResult:
    """Generate multimodal embeddings with a normalized SDK interface."""
    normalized_provider = normalize_provider(provider)
    if normalized_provider == "ark":
        return _generate_ark(
            input=input,
            text=text,
            images=images,
            model=model,
            **kwargs,
        )
    raise AIToolkitError(f"unknown embedding provider: {provider}")


def normalize_provider(provider: str) -> str:
    value = provider.strip().lower()
    if value in {"ark", "doubao", "volcengine", "seedream"}:
        return "ark"
    return value


def _generate_ark(
    *,
    input: list[dict[str, Any]] | None,
    text: str | list[str] | None,
    images: list[str | Path] | None,
    model: str | None,
    **kwargs: Any,
) -> EmbeddingResult:
    settings = get_settings()
    resolved_model = model or settings.ark_embedding_model
    payload_input = _build_payload_input(
        input=input,
        text=text,
        images=images,
    )
    if not payload_input:
        raise ValueError("text, images, or input is required")

    raw_response = create_multimodal_embedding(
        resolved_model,
        payload_input,
        **kwargs,
    )
    return EmbeddingResult(
        provider="ark",
        model=resolved_model,
        embeddings=_extract_embeddings(raw_response),
        raw_response=raw_response,
        request={"input": payload_input, **kwargs},
        usage=raw_response.get("usage") if isinstance(raw_response.get("usage"), dict) else None,
    )


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
        # Keep string shorthand for backward compatibility: treat as text by default.
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


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _extract_embeddings(response: dict[str, Any]) -> list[list[float]]:
    data = response.get("data")
    vectors: list[list[float]] = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            embedding = item.get("embedding")
            vector = _coerce_vector(embedding)
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
