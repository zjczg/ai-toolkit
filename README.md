# ai-toolkit

Reusable Python SDK for common model calls across local projects.

It keeps low-level provider wrappers available, and adds high-level normalized tools for project code.

## Environment

API keys and deployment-specific values are read from environment variables only.

Required as needed:

```bash
export ARK_API_KEY="..."
export DASHSCOPE_API_KEY="..."   # 万相 / 通义 (阿里百炼)
export GEMINI_API_KEY="..."
export DEEPSEEK_API_KEY="..."

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
export DASHSCOPE_IMAGE_MODEL="wan2.7-image"
export DASHSCOPE_IMAGE_SIZE="2K"
export GEMINI_IMAGE_MODEL="gemini-3.1-flash-image-preview"
export GEMINI_IMAGE_SIZE="1K"
export DEEPSEEK_CHAT_MODEL="deepseek-v4-flash"
export ARK_RESPONSE_MODEL="doubao-seed-2-0-pro-260215"

# Transport retry/timeout defaults (optional)
export AI_TOOLKIT_HTTP_MAX_RETRIES="2"
export AI_TOOLKIT_HTTP_RETRY_BASE_SECONDS="0.5"
export AI_TOOLKIT_HTTP_RETRY_MAX_SECONDS="8.0"
export AI_TOOLKIT_HTTP_TIMEOUT_SECONDS="120"
```

## High-level image generation

Prefer passing a curated `path` so the SDK resolves provider, model, and model-specific quirks (output_format inference, reference-count limits, dropping unsupported params):

```python
from ai_toolkit import images

result = images.generate(
    path="doubao-5.0-lite",                # or "doubao-4.5", "gemini-banano2"
    prompt="pixel art orange tabby cat",
    references=["./base.png"],
    size="2K",
    output_path="./out.png",
)

print(result.provider, result.model, result.first_image().local_path)
```

When `path` is omitted, `provider`+`model` still works. If the `model` matches a curated path the SDK reverse-looks-up the capability and applies the same rules; an unknown `model` is forwarded as-is:

```python
images.generate(
    provider="ark",
    model="doubao-seedream-4-5-251128",
    prompt="...",
    output_path="./out.jpg",
)
```

Per-model behaviour applied automatically:

| path | output_format | reference cap |
|---|---|---|
| `doubao-5.0-lite` | inferred from `output_path` suffix (png/jpeg) | 14 |
| `doubao-4.5` | not supported — silently dropped | per docs |
| `wan2.7-image` / `wan2.7-image-pro` | not configurable | 5 |

Local references are handled internally:

- `ark` / Doubao uploads local files and passes public URLs.
- `dashscope` / 万相 uploads local files and passes public URLs.
- `gemini` embeds local files as inline base64 parts.

### 万相 / 通义 (阿里百炼 DashScope)

万相 runs on a **different platform** from ARK/Doubao — it needs its own `DASHSCOPE_API_KEY`. The toolkit uses the synchronous `multimodal-generation` endpoint, so calls return image URLs directly (no task polling). Provider aliases: `wan`, `wanx`, `万相`, `dashscope`, `bailian`, `qwen-image`.

```python
from ai_toolkit import images

result = images.generate(
    path="wan2.7-image-pro",               # or "wan2.7-image"
    prompt="一只橙色虎斑猫的工笔画",
    size="2K",                              # "1K"/"2K"/"4K" or "1024*1024" (separator is *, not x)
    output_path="./out.png",
)
print(result.provider, result.model, result.first_image().local_path)
```

> Note: only the modern sync Wan/Qwen models (`wan2.7-image`, `wan2.7-image-pro`, `wan2.6-image`, `qwen-image`) are wired here. The legacy async-only series (`wanx2.1-t2i`, `wan2.2-t2i-*`) uses a different task-polling endpoint and is out of scope. DashScope output URLs expire after 24 hours.

## High-level video generation

```python
from ai_toolkit import videos

task = videos.generate(
    provider="ark",
    prompt="A pixel art orange tabby cat running in a smooth loop.",
    reference_images=["./base.png"],
    ratio="1:1",
    duration=5,
    watermark=False,
    wait_for_completion=False,
)

print(task.task_id)
```

Seedance video generation uses ARK's asynchronous task API:

```text
POST /contents/generations/tasks
GET  /contents/generations/tasks/{task_id}
```

Use `videos.generate(..., wait_for_completion=True)` for a blocking call that polls until the video URL is ready, or use `videos.create_task()` and `videos.get_task()` when the application needs its own task lifecycle.

## High-level text generation

```python
from ai_toolkit import chat

result = chat.complete(
    provider="deepseek",
    messages=[{"role": "user", "content": "Return JSON for a cat spec"}],
    response_format={"type": "json_object"},
)

print(result.text)
```

For ARK Responses API structured output, prefer `complete_json`.
It uses `text.format` for JSON output and disables thinking by default for short extraction tasks:

```python
from ai_toolkit import chat

schema = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "additionalProperties": False,
}

result = chat.complete_json(
    provider="ark",
    model="doubao-seed-2-0-pro-260215",
    messages=[
        chat.multimodal_user_message(
            text="Name the main garment. Return JSON only.",
            images=["https://example.com/garment.png"],
        )
    ],
    schema=schema,
    schema_name="garment_name",
)

print(result.parsed_json)
```

## High-level multimodal embeddings

```python
from ai_toolkit import embeddings

result = embeddings.generate(
    provider="ark",
    text="A red pixel-art cat sitting on a table",
    images=["./cat_reference.png"],
)

print(result.provider, result.model, len(result.first_embedding()))
```

## Capabilities

```python
from ai_toolkit import capabilities

print(capabilities.list_providers())
print(capabilities.get_tool_spec("images.generate"))
print(capabilities.get_tool_spec("videos.generate"))
print(capabilities.get_default_params("ark", "images.generate"))
print(capabilities.list_image_generation_paths())
print(capabilities.get_image_generation_path("doubao-5.0-lite"))
print(capabilities.list_video_generation_paths())
print(capabilities.get_video_generation_path("doubao-seedance-2-0-260128"))
```

Curated image generation paths include provider/model mapping, supported size controls, source URLs, and whether exact small exports such as `200x200` require post-processing. Current paths:

```text
doubao-5.0-lite -> provider=ark,    model=doubao-seedream-5-0-260128
doubao-4.5      -> provider=ark,    model=doubao-seedream-4-5-251128
gemini-banano2  -> provider=gemini, model=gemini-3.1-flash-image-preview
```

Curated video generation paths are separate from image generation paths. Current paths:

```text
doubao-seedance-2.0 -> provider=ark, model=doubao-seedance-2-0-260128
```

## Low-level wrappers

Existing raw API wrappers remain available:

```python
from ai_toolkit.ark import create_image_generation
from ai_toolkit.ark import create_video_generation_task
from ai_toolkit.ark import get_video_generation_task
from ai_toolkit.gemini import generate_content
from ai_toolkit.deepseek import create_chat_completion
```
