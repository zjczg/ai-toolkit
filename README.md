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
export ALIYUN_ACCESS_KEY_ID="..."
export ALIYUN_ACCESS_KEY_SECRET="..."

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
export DASHSCOPE_SPEECH_MODEL="qwen-audio-3.0-tts-plus"
export DASHSCOPE_SPEECH_VOICE="longanlingxin"

export GEMINI_IMAGE_MODEL="gemini-3.1-flash-image"
export GEMINI_IMAGE_SIZE="1K"

export DEEPSEEK_CHAT_MODEL="deepseek-v4-flash"

export ALIYUN_VIAPI_REGION="cn-shanghai"
```

## Platform Entry Points

Use platform modules directly:

```python
from ai_toolkit import aliyun, ark, dashscope, deepseek, gemini
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

Speech synthesis:

```python
result = dashscope.speech.synthesize(
    model="qwen-audio-tts-plus",
    text="请介绍一下你负责的项目。",
    voice="longanlingxin",
    instruction="使用自然、沉稳、有交流感的普通话，避免播音腔。",
    language_hints=["zh"],
    output_path="./question.wav",
)
print(result.usage)
```

The non-streaming API returns a temporary audio URL. When `output_path` is
provided, `ai-toolkit` downloads it immediately.

### Gemini

```python
result = gemini.images.generate(
    model="gemini-image",
    prompt="Four evenly spaced pixel-art walking frames in one horizontal row.",
    output_path="./gemini.png",
    image_size="1K",
    aspect_ratio="4:1",
)
```

Local references are sent as inline base64 parts; no public URL upload is
required for Gemini. Omit `aspect_ratio` when an input image should determine
the output ratio. `gemini-image` maps to stable Nano Banana 2 and supports
`512`, `1K`, `2K`, `4K`, plus the long `4:1` and `8:1` ratios. Lite supports
only `1K`; Pro supports `1K`, `2K`, and `4K`. Invalid options fail before the
network request.

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

### Aliyun ImageSeg

Install the optional Aliyun SDK dependencies when using this platform:

```bash
pip install "ai-toolkit[aliyun]"
```

Product or clothing foreground segmentation:

```python
result = aliyun.images.segment_commodity(
    image_path="./source.jpg",
    output_path="./cutout.png",
)
print(result.model, result.path)
```

Human foreground segmentation:

```python
result = aliyun.images.segment_hd_body(
    image_path="./person.jpg",
    output_path="./person-cutout.png",
)
```

Local inputs are staged through Aliyun's viapi temporary OSS helper internally;
callers do not need to provide a public image URL. The saved output is the
transparent PNG returned by ImageSeg.

## Supported Model Aliases and API Names

Documented SDK contract. Generation/text/embedding aliases were smoke-tested on 2026-06-29.
Aliyun `SegmentCommodity` was live smoke-tested on 2026-06-30 with a synthetic image;
`SegmentHDBody` is covered by the same SDK flow and unit contract tests.

| Platform | Ability | Aliases | Raw model |
|---|---|---|---|
| ARK | image | `seedream-5-lite` | `doubao-seedream-5-0-260128` |
| ARK | image | `seedream-4.5` | `doubao-seedream-4-5-251128` |
| ARK | video | `seedance-2` | `doubao-seedance-2-0-260128` |
| ARK | text | `doubao-pro` | `doubao-seed-2-0-pro-260215` |
| ARK | embedding | `doubao-vision` | `doubao-embedding-vision-251215` |
| DashScope | image | `wan2.7` | `wan2.7-image` |
| DashScope | image | `wan2.7-pro` | `wan2.7-image-pro` |
| DashScope | speech | `qwen-audio-tts-plus` | `qwen-audio-3.0-tts-plus` |
| DashScope | speech | `qwen-audio-tts-flash` | `qwen-audio-3.0-tts-flash` |
| Gemini | image | `gemini-image`, `nano-banana-2` | `gemini-3.1-flash-image` |
| Gemini | image | `gemini-image-lite`, `nano-banana-2-lite` | `gemini-3.1-flash-lite-image` |
| Gemini | image | `gemini-image-pro`, `nano-banana-pro` | `gemini-3-pro-image` |
| DeepSeek | text | `v4-flash` | `deepseek-v4-flash` |
| DeepSeek | text | `v4-pro` | `deepseek-v4-pro` |
| Aliyun | image segmentation | `viapi-segment-commodity` | `SegmentCommodity` |
| Aliyun | image segmentation | `viapi-segment-hd-body` | `SegmentHDBody` |

Raw model IDs are still accepted by platform modules, but aliases are the
documented SDK contract.

## Capabilities

```python
from ai_toolkit import capabilities

print(capabilities.list_tools())
print(capabilities.list_image_models())
print(capabilities.list_text_models())
print(capabilities.list_embedding_models())
print(capabilities.list_audio_models())
```
