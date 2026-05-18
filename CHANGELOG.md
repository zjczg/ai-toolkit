# Changelog

## 0.2.0

- Added high-level `ai_toolkit.images.generate` for ARK/Doubao and Gemini image generation.
- Added high-level `ai_toolkit.chat.complete` for DeepSeek and ARK text generation.
- Added `ai_toolkit.capabilities` provider/tool registry.
- Added normalized result dataclasses in `ai_toolkit.types`.
- Added `media.upload_public_url` and `UPLOAD_PUBLIC_BASE_URL` support.
- Kept low-level provider wrappers backward compatible.

## 0.1.0

- Initial low-level API wrappers for ARK, Gemini, DeepSeek, and media upload.
