# ai-toolkit

Platform-scoped Python SDK for reusable AI model calls.

The package does not store API keys. Runtime configuration is resolved, in
order, from process environment variables, `.env` in the current working
directory, `${XDG_CONFIG_HOME:-~/.config}/ai-toolkit/env`, and built-in
defaults. Set `AI_TOOLKIT_ENV_FILE` to select a different user file.

Resolution is based on key presence, so an explicitly empty value blocks
lower-priority fallbacks. Dotenv files are read without modifying
`os.environ`; restart the process after editing one because file contents are
cached. Keep the user file private (for example, `chmod 600`) and run commands
directly when you want project `.env` values to override user defaults. An
`ai-run` wrapper that exports the user file instead promotes those values to
the highest-priority process environment.

## Environment

Required as needed:

```bash
export ARK_API_KEY="..."
export DASHSCOPE_API_KEY="..."
export GEMINI_API_KEY="..."
export GRSAI_API_KEY="..."
export MINIMAX_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export ALIYUN_ACCESS_KEY_ID="..."
export ALIYUN_ACCESS_KEY_SECRET="..."

# Used internally when a provider requires public reference URLs.
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
export DASHSCOPE_VIDEO_MODEL="happyhorse-1.1-r2v"
export DASHSCOPE_VIDEO_RATIO="1:1"
export DASHSCOPE_VIDEO_DURATION="5"
export DASHSCOPE_VIDEO_RESOLUTION="720P"
export DASHSCOPE_SPEECH_MODEL="qwen-audio-3.0-tts-plus"
export DASHSCOPE_SPEECH_VOICE="longanlingxin"

export GEMINI_IMAGE_MODEL="gemini-3.1-flash-image"
export GEMINI_IMAGE_SIZE="1K"

export GRSAI_BASE_URL="https://grsaiapi.com"
export GRSAI_IMAGE_MODEL="nano-banana-2"
export GRSAI_IMAGE_SIZE="1K"

export MINIMAX_BASE_URL="https://api.minimax.io"
export MINIMAX_VIDEO_MODEL="MiniMax-H3"

export DEEPSEEK_CHAT_MODEL="deepseek-v4-flash"

export ALIYUN_VIAPI_REGION="cn-shanghai"
```

## Platform Entry Points

Use platform modules directly:

```python
from ai_toolkit import aliyun, ark, dashscope, deepseek, gemini, grsai, minimax
```

### ARK / Doubao

Image generation:

```python
result = ark.images.generate(
    model="seedream-5",  # or "seedream-5-pro" / "seedream-4.5"
    prompt="A flat red square icon on a white background.",
    output_path="./out.png",
    size="2K",
)
print(result.model, result.first_image().local_path)
```

Local image references for Seedream are sent as inline Base64 data URLs; no
public upload is required.

`seedream-5` accepts `2K`, `3K`, `4K`, or an explicit `<width>x<height>`
value. `seedream-5-pro` accepts `1K`, `2K`, or explicit dimensions; it does
not support `3K`/`4K`, streaming, or sequential image generation. The
provider may align explicit dimensions, so callers must inspect the returned
image size instead of treating the request as a fixed-pixel export contract.
The old `seedream-5-lite` / `doubao-5.0-lite` names remain compatibility-only
aliases for standard Seedream 5.0 and emit `DeprecationWarning`; new code must
not use them.

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

Local image references are sent as DashScope-supported inline base64 data;
`wan2.7` and `wan2.7-pro` accept up to 9 reference images.

HappyHorse video generation:

```python
result = dashscope.videos.generate(
    model="happyhorse-1.1-r2v",
    prompt="The orange kitten in [Image 1] takes two gentle steps forward.",
    references=["./kitten.png"],
    output_path="./kitten-walk.mp4",
    ratio="1:1",
    duration=5,
    resolution="720P",
    watermark=False,
)
```

`happyhorse-1.1-t2v` accepts only text, `happyhorse-1.1-i2v` accepts exactly
one first-frame image, and `happyhorse-1.1-r2v` accepts 1–9 reference images.
Tasks are asynchronous and are polled every 15 seconds by default. Local image
references are sent as inline base64 data, so no public upload is required.

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

Small non-urgent text-to-image sets can use Gemini's inline Batch API:

```python
task = gemini.images.create_batch(
    model="gemini-image-lite",
    prompts={
        "coop": "A wooden chicken coop on a flat magenta background.",
        "trough": "A wooden chicken trough on a flat magenta background.",
    },
    image_size="auto",
    display_name="pasture-props-2026-08-04",
)

# Query this again later; Batch completion may take up to 24 hours.
task = gemini.images.get_batch(task_id=task.task_id, model=task.model)
if task.status == "JOB_STATE_SUCCEEDED":
    result = gemini.images.batch_result(
        task,
        prompts={
            "coop": "A wooden chicken coop on a flat magenta background.",
            "trough": "A wooden chicken trough on a flat magenta background.",
        },
    )
```

`create_batch` accepts keyed prompts only: reference-image and JSONL batches
are intentionally outside this initial interface. Inline payloads at or above
20 MB are rejected locally. Batch creation disables
transport retries because Google's create operation is not idempotent. Use
`find_batch(display_name=...)` to recover a task when submission completed but
the client did not receive its response.

### GRS.AI

GRS.AI is an independent provider entry and never reuses Gemini credentials:

```python
result = grsai.images.generate(
    model="grsai-image",
    prompt="A clean pixel-art farmer on a flat magenta background.",
    references=["./farmer-reference.png"],
    output_path="./grsai-farmer.png",
    image_size="1K",
    aspect_ratio="1:1",
)
print(result.provider, result.model)
```

`grsai-image` resolves only to GRS.AI `nano-banana-2`. The client uses the
provider's Gemini-compatible endpoint, supports 1K/2K/4K and the long 4:1/8:1
ratios, and sends local references as inline Base64. Set `GRSAI_BASE_URL` to
`https://grsai.dakka.com.cn` when the documented China-direct host is needed.
The SDK does not automatically fall back between GRS.AI and official Gemini.

### MiniMax

```python
result = minimax.videos.generate(
    model="minimax-h3",
    prompt="The character walks in place while facing right.",
    first_frame="./character.png",
    output_path="./walk.mp4",
    duration=4,
    resolution="768P",
)
```

MiniMax H3 tasks are asynchronous. Local keyframes and reference images are
validated and sent as inline Base64 data URLs.

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

### Typed JSON outputs

Define a reusable output model once, then use it in both the prompt and the
model call:

```python
from pydantic import Field

from ai_toolkit import JsonOutputModel, deepseek


class PetSceneOutput(JsonOutputModel):
    asset_prompt: str = Field(
        min_length=1,
        description="完整的图片或视频生成提示词",
    )
    event_summary: str = Field(
        min_length=1,
        description="会影响下一轮互动的简洁事件摘要",
    )


result = deepseek.text.complete_json(
    model="v4-flash",
    prompt=(
        "描述宠物的自然反应。\n\n"
        f"{PetSceneOutput.prompt_fragment()}"
    ),
    output_type=PetSceneOutput,
)
scene = result.output
print(scene.asset_prompt, scene.event_summary)
```

`prompt_fragment()` renders a compact JSON example for explicit prompt
composition. It resolves local `$ref` / `$defs` definitions and renders nested
objects plus required array items; optional nested objects with a `null`
default stay `null` unless the field supplies an explicit example. ARK sends
`output_type.model_json_schema()` through its native JSON Schema format;
DeepSeek continues to use JSON-object mode. Both providers validate the parsed
response with Pydantic and raise `StructuredOutputError` when decoding or
validation fails.

The existing `schema=` form remains available for dictionary results and keeps
its non-raising `parsed_json` / `schema_error` behavior. Do not pass `schema`
and `output_type` together.

### Aliyun ImageSeg

Install the optional Aliyun SDK dependencies when using this platform:

```bash
pip install "ai-toolkit[aliyun]"
```

General foreground extraction with Aidge ImageMatting:

```python
result = aliyun.images.image_matting(
    image_path="./pet.png",
    output_path="./pet-cutout.png",
)
print(result.model, result.path)
```

`image_matting` preserves the canvas size and downloads a transparent PNG. It
accepts local images from 256×256 through 3000×3000 and stages them to a
temporary OSS URL internally.

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

General high-definition foreground segmentation:

```python
result = aliyun.images.segment_hd_common_image(
    image_path="./pet.jpg",
    output_path="./pet-cutout.png",
)
```

`segment_hd_common_image` accepts images smaller than 10000×10000 and no
larger than 40 MB. The SDK wrapper submits the asynchronous task, polls it,
and downloads the transparent PNG before returning. Temporary network errors
during OSS staging, API calls, polling, or result download are retried up to
three times with 2, 4, and 8 second delays. Authentication, permission,
parameter, response-format, and terminal job errors fail immediately.

Local inputs are staged through Aliyun's viapi temporary OSS helper internally;
callers do not need to provide a public image URL. The saved output is the
transparent PNG returned by ImageSeg.

## Supported Model Aliases and API Names

Documented SDK contract. Generation/text/embedding aliases were smoke-tested on 2026-06-29.
Aliyun `SegmentCommodity` was live smoke-tested on 2026-06-30 with a synthetic image;
`SegmentHDBody` is covered by the same SDK flow and unit contract tests.

| Platform | Ability | Aliases | Raw model |
|---|---|---|---|
| ARK | image | `seedream-5` | `doubao-seedream-5-0-260128` |
| ARK | image | `seedream-5-pro` | `doubao-seedream-5-0-pro-260628` |
| ARK | image | `seedream-4.5` | `doubao-seedream-4-5-251128` |
| ARK | video | `seedance-2` | `doubao-seedance-2-0-260128` |
| ARK | text | `doubao-pro` | `doubao-seed-2-0-pro-260215` |
| ARK | embedding | `doubao-vision` | `doubao-embedding-vision-251215` |
| DashScope | image | `wan2.7` | `wan2.7-image` |
| DashScope | image | `wan2.7-pro` | `wan2.7-image-pro` |
| DashScope | video | `happyhorse-1.1-t2v` | `happyhorse-1.1-t2v` |
| DashScope | video | `happyhorse-1.1-i2v` | `happyhorse-1.1-i2v` |
| DashScope | video | `happyhorse-1.1-r2v` | `happyhorse-1.1-r2v` |
| DashScope | speech | `qwen-audio-tts-plus` | `qwen-audio-3.0-tts-plus` |
| DashScope | speech | `qwen-audio-tts-flash` | `qwen-audio-3.0-tts-flash` |
| Gemini | image | `gemini-image`, `nano-banana-2` | `gemini-3.1-flash-image` |
| Gemini | image | `gemini-image-lite`, `nano-banana-2-lite` | `gemini-3.1-flash-lite-image` |
| Gemini | image | `gemini-image-pro`, `nano-banana-pro` | `gemini-3-pro-image` |
| GRS.AI | image | `grsai-image`, `grsai-nano-banana-2` | `nano-banana-2` |
| MiniMax | video | `minimax-h3`, `h3` | `MiniMax-H3` |
| DeepSeek | text | `v4-flash` | `deepseek-v4-flash` |
| DeepSeek | text | `v4-pro` | `deepseek-v4-pro` |
| Aliyun | image matting | `aidge-image-matting` | `ImageMatting` |
| Aliyun | image segmentation | `viapi-segment-commodity` | `SegmentCommodity` |
| Aliyun | image segmentation | `viapi-segment-hd-body` | `SegmentHDBody` |
| Aliyun | image segmentation | `viapi-segment-hd-common-image` | `SegmentHDCommonImage` |

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
