# ai-toolkit

Reusable Python SDK for common model calls across local projects.

It keeps low-level provider wrappers available, and adds high-level normalized tools for project code.

## Environment

API keys and deployment-specific values are read from environment variables only.

Required as needed:

```bash
export ARK_API_KEY="..."
export GEMINI_API_KEY="..."
export DEEPSEEK_API_KEY="..."

export UPLOAD_SSH_TARGET="root@example.com"
export UPLOAD_IDENTITY_FILE="~/.ssh/id_ed25519"
export UPLOAD_REMOTE_DIR="/var/www/images"
export UPLOAD_PUBLIC_BASE_URL="https://example.com/images"
```

Optional defaults:

```bash
export ARK_IMAGE_MODEL="doubao-seedream-4-5-251128"
export ARK_IMAGE_SIZE="2K"
export GEMINI_IMAGE_MODEL="gemini-3.1-flash-image-preview"
export GEMINI_IMAGE_SIZE="1K"
export DEEPSEEK_CHAT_MODEL="deepseek-v4-flash"
export ARK_RESPONSE_MODEL="doubao-seed-2-0-pro-260215"
```

## High-level image generation

```python
from ai_toolkit import images

result = images.generate(
    provider="ark",
    prompt="pixel art orange tabby cat, side view",
    references=["./base.png"],
    size="2K",
    output_path="./out.png",
)

print(result.provider, result.model, result.first_image().local_path)
```

Local references are handled internally:

- `ark` / Doubao uploads local files and passes public URLs.
- `gemini` embeds local files as inline base64 parts.

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

## Capabilities

```python
from ai_toolkit import capabilities

print(capabilities.list_providers())
print(capabilities.get_tool_spec("images.generate"))
print(capabilities.get_default_params("ark", "images.generate"))
```

## Low-level wrappers

Existing raw API wrappers remain available:

```python
from ai_toolkit.ark import create_image_generation
from ai_toolkit.gemini import generate_content
from ai_toolkit.deepseek import create_chat_completion
```
