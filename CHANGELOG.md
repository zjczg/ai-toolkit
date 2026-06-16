# Changelog

## 0.4.0

- `chat.multimodal_user_message` now auto-uploads local file paths via `media.upload_public_url`, mirroring `images.generate` reference handling. HTTP(S) URLs still pass through unchanged.
- `chat.complete_json` validates `parsed_json` against the provided `schema` (top-level `type`, `required`, declared property types). On mismatch it clears `parsed_json` and surfaces a human-readable `schema_error` on `ChatCompletionResult`.
- `ChatCompletionResult` gained a new `schema_error: str | None` field for the above. Existing callers that only read `parsed_json` and `text` are unaffected.

## 0.3.0

- `images.generate` now accepts a curated `path` (e.g. `"doubao-5.0-lite"`, `"doubao-4.5"`) that resolves provider, model, and applies model-specific constraints.
- When a known model is passed without a `path`, the SDK still reverse-looks-up the capability and applies the same rules; callers can opt out by passing an unknown model.
- For ARK models with `output_format_param`, the SDK infers `output_format` from the `output_path` suffix when the caller does not specify it.
- For ARK models with `output_format_configurable=False` (Seedream 4.5), the SDK silently drops a caller-supplied `output_format` so the request remains valid.
- Reference image count is validated against `max_reference_images` from the capability and overflow raises `AIToolkitError` before the HTTP call.
- New `capabilities.find_image_path_by_model` and `capabilities.get_image_capability` helpers for reverse lookups.
- Fixed video option normalizer reading `supported_resolution_values` from the wrong field; it now also accepts `recommended_resolution_values`.
- `ImageGenerationResult` now carries `usage` from the upstream response.

## 0.2.0

- Added high-level `ai_toolkit.images.generate` for ARK/Doubao and Gemini image generation.
- Added high-level `ai_toolkit.chat.complete` for DeepSeek and ARK text generation.
- Added `ai_toolkit.capabilities` provider/tool registry.
- Added normalized result dataclasses in `ai_toolkit.types`.
- Added `media.upload_public_url` and `UPLOAD_PUBLIC_BASE_URL` support.
- Kept low-level provider wrappers backward compatible.

## 0.1.0

- Initial low-level API wrappers for ARK, Gemini, DeepSeek, and media upload.
