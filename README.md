# ai-toolkit

Platform-scoped Python SDK for reusable AI model calls.

The package does not store API keys. Runtime configuration is read from
environment variables only.

## Environment

Required as needed:

```bash
export ARK_API_KEY="..."
export DASHSCOPE_API_KEY="..."
export GEMINI_API_KEY="..."
export DEEPSEEK_API_KEY="..."

# Used internally when ARK / DashScope need public reference URLs.
export UPLOAD_SSH_TARGET="root@example.com"
export UPLOAD_IDENTITY_FILE="~/.ssh/id_ed25519"
export UPLOAD_REMOTE_DIR="/var/www/images"
export UPLOAD_PUBLIC_BASE_URL="https://example.com/images"
```

Optional defaults:

```bash
export ARK_IMAGE_MODEL="doubao-seedream-5-0-260128"
export ARK_IMAGE_SIZE="2K"
export ARK_VIDEO_MODEL="doubao-seedance-2-0-260128"
export ARK_RESPONSE_MODEL="doubao-seed-2-0-pro-260215"
export ARK_EMBEDDING_MODEL="doubao-embedding-vision-251215"

export DASHSCOPE_IMAGE_MODEL="wan2.7-image"
export DASHSCOPE_IMAGE_SIZE="2K"

export GEMINI_IMAGE_MODEL="gemini-3.1-flash-image-preview"
export GEMINI_IMAGE_SIZE="2K"

export DEEPSEEK_CHAT_MODEL="deepseek-v4-flash"
```

## Platform Entry Points

Use platform modules directly:

```python
from ai_toolkit import ark, dashscope, deepseek, gemini
```

### ARK / Doubao

Image generation:

```python
result = ark.images.generate(
    model="seedream-5-lite",  # or "seedream-4.5"
    prompt="A flat red square icon on a white background.",
    output_path="./out.png",
    size="2K",
)
print(result.model, result.first_image().local_path)
```

Video generation:

```python
result = ark.videos.generate(
    model="seedance-2",
    prompt="A red square gently moving on a white background.",
    output_path="./out.mp4",
    ratio="1:1",
    duration=5,
    resolution="720p",
    watermark=False,
    generate_audio=False,
)
```

Text and structured JSON:

```python
result = ark.text.complete_json(
    model="doubao-pro",
    prompt="Return JSON only: {\"ok\": true}",
    schema={
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    },
)
print(result.parsed_json)
```

`ark.text.complete_json` defaults to `thinking={"type": "disabled"}`. Do not
enable thinking for JSON extraction unless you explicitly handle reasoning text
mixed into the model text.

Embeddings:

```python
result = ark.embeddings.generate(
    model="doubao-vision",
    text="red square icon",
    dimensions=1024,
)
print(len(result.first_embedding()))
```

### DashScope / Wan

```python
result = dashscope.images.generate(
    model="wan2.7-pro",  # or "wan2.7"
    prompt="A clean product render of a red square icon.",
    output_path="./wan.png",
    size="1K",
)
```

Local references are uploaded to public URLs internally. `wan2.7` and
`wan2.7-pro` accept up to 9 reference images.

### Gemini

```python
result = gemini.images.generate(
    model="gemini-image",
    prompt="A clean red square icon.",
    output_path="./gemini.png",
    image_size="512",
    aspect_ratio="1:1",
)
```

Local references are sent as inline base64 parts; no public URL upload is
required for Gemini.

### DeepSeek

```python
result = deepseek.text.complete(
    model="v4-pro",  # or "v4-flash"
    prompt="2+2 equals?",
    thinking={"type": "enabled", "reasoning_effort": "high"},
)
print(result.reasoning_text, result.text)
```

JSON extraction:

```python
result = deepseek.text.complete_json(
    model="v4-flash",
    prompt="Return JSON only: {\"ok\": true}",
    schema={
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    },
)
```

`deepseek.text.complete_json` defaults to `thinking={"type": "disabled"}` and
`response_format={"type": "json_object"}`.

## Supported Model Aliases

Smoke-tested on 2026-06-29:

| Platform | Ability | Aliases | Raw model |
|---|---|---|---|
| ARK | image | `seedream-5-lite` | `doubao-seedream-5-0-260128` |
| ARK | image | `seedream-4.5` | `doubao-seedream-4-5-251128` |
| ARK | video | `seedance-2` | `doubao-seedance-2-0-260128` |
| ARK | text | `doubao-pro` | `doubao-seed-2-0-pro-260215` |
| ARK | embedding | `doubao-vision` | `doubao-embedding-vision-251215` |
| DashScope | image | `wan2.7` | `wan2.7-image` |
| DashScope | image | `wan2.7-pro` | `wan2.7-image-pro` |
| Gemini | image | `gemini-image` | `gemini-3.1-flash-image-preview` |
| DeepSeek | text | `v4-flash` | `deepseek-v4-flash` |
| DeepSeek | text | `v4-pro` | `deepseek-v4-pro` |

Raw model IDs are still accepted by platform modules, but aliases are the
documented SDK contract.

## Capabilities

```python
from ai_toolkit import capabilities

print(capabilities.list_tools())
print(capabilities.list_image_models())
print(capabilities.list_text_models())
print(capabilities.list_embedding_models())
```
