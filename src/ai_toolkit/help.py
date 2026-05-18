"""Interface registry — query available AI API endpoints and their parameters.

Usage:
    from ai_toolkit import help
    help()                        # list all
    help("ark")                   # filter by provider
    help("create_image")          # filter by function name
"""

from __future__ import annotations

from typing import Any

InterfaceSpec = dict[str, Any]

_INTERFACES: list[InterfaceSpec] = [
    # -- ARK Image Generation -------------------------------------------------
    {
        "provider": "ARK (豆包/火山引擎)",
        "endpoint": "POST /images/generations",
        "function": "create_image_generation",
        "description": "图片生成 — 返回原始 JSON，data[0].url 即生成图片链接",
        "parameters": [
            ("model", "str", "必填 — 模型名，如 doubao-seedream-4-5-251128"),
            ("prompt", "str", "必填 — 图片描述文本"),
            ("image", "list[str]", '可选 (kwargs) — 参考图片 URL 数组，单张为图生图，多张为多图融合'),
            ("size", "str", '可选 (kwargs) — "2k"(默认)≈2048²、"3k"(仅5.0)、"4k"，或自定义 "1920x1080"。1K 仅 seedream-4-0 支持'),
            ("response_format", "str", '可选 (kwargs) — "url"（默认）或 "b64_json"'),
            ("watermark", "bool", "可选 (kwargs) — 是否添加水印，各版本默认值不同建议显式设置"),
            ("stream", "bool", "可选 (kwargs) — 是否开启 SSE 流式输出"),
            ("sequential_image_generation", "str", '可选 (kwargs) — "disabled"（编辑/融合模式）或 "auto"（批量序列模式）'),
            ("sequential_image_generation_options", "dict", "可选 (kwargs) — {\"max_images\": int}，输入+输出总数 ≤ 15"),
            ("n", "int", "可选 (kwargs) — 生成图片数量"),
        ],
    },
    {
        "provider": "ARK (豆包/火山引擎)",
        "endpoint": "POST /images/generations",
        "function": "create_seedream_4_5_image_generation",
        "description": "Seedream 4.5 图片生成便捷方法，默认 model=doubao-seedream-4-5-251128、size=2K、response_format=url、sequential_image_generation=disabled、stream=false、watermark=true",
        "parameters": [
            ("prompt", "str", "必填 — 图片描述文本"),
            ("**kwargs", "Any", "可选 — 覆盖默认 payload 字段，如 watermark=False 或 image=[...]"),
        ],
    },
    # -- ARK Multimodal -------------------------------------------------------
    {
        "provider": "ARK (豆包/火山引擎)",
        "endpoint": "POST /responses",
        "function": "create_responses",
        "description": "多模态对话 — 返回原始 JSON，output 中提取回复文本。支持文本生成/深度思考/多模态理解/工具调用/结构化输出等功能",
        "parameters": [
            ("model", "str", "必填 — 模型名，如 doubao-seed-2-0-pro-260215"),
            ("input", "str | list[dict]", "必填 — 纯文本字符串 或 结构化 [{\"role\": \"user\", \"content\": [...]}]"),
            ("stream", "bool", "可选 (kwargs) — 是否 SSE 流式返回"),
            ("reasoning", "dict", "可选 (kwargs) — 深度思考配置，如 {\"effort\": \"high\"}"),
            ("tools", "list[dict]", "可选 (kwargs) — function calling 工具定义"),
            ("temperature", "float", "可选 (kwargs) — 温度参数"),
            ("max_output_tokens", "int", "可选 (kwargs) — 最大输出 token 数"),
        ],
    },
    {
        "provider": "ARK (豆包/火山引擎)",
        "endpoint": "POST /embeddings/multimodal",
        "function": "create_multimodal_embedding",
        "description": "多模态向量 — 返回原始 JSON，data.embedding 为向量，可对文本、图片 URL 等输入生成 embedding",
        "parameters": [
            ("model", "str", "必填 — 模型名，如 doubao-embedding-vision-251215"),
            ("input", "list[dict]", "必填 — [{\"type\":\"text\",\"text\":\"...\"}] 或包含 image_url 的多模态数组"),
            ("**kwargs", "Any", "可选 — 透传 provider 支持的其他 payload 字段"),
        ],
    },
    # -- ARK Video Generation -------------------------------------------------
    {
        "provider": "ARK (豆包/火山引擎)",
        "endpoint": "POST /contents/generations/tasks",
        "function": "create_video_generation_task",
        "description": "提交视频生成任务 — 返回原始 JSON，id 为 task_id",
        "parameters": [
            ("model", "str", "必填 — 模型名，如 doubao-seedance-2-0-260128"),
            ("content", "list[dict]", "必填 — [{\"type\": \"text\", \"text\": \"...\"}]，可含 image_url/video_url/audio_url 参考素材"),
            ("ratio", "str", '可选 (kwargs) — 画面比例，如 "16:9"'),
            ("resolution", "str", '可选 (kwargs) — 分辨率，如 "720p"'),
            ("duration", "int", "可选 (kwargs) — 视频时长（秒）"),
            ("generate_audio", "bool", "可选 (kwargs) — 是否生成音频"),
            ("return_last_frame", "bool", "可选 (kwargs) — 是否返回最后一帧图片"),
            ("callback_url", "str", "可选 (kwargs) — 异步回调地址"),
            ("watermark", "bool", "可选 (kwargs) — 是否添加水印"),
        ],
    },
    {
        "provider": "ARK (豆包/火山引擎)",
        "endpoint": "GET /contents/generations/tasks/{task_id}",
        "function": "get_video_generation_task",
        "description": "查询视频任务状态 — 返回原始 JSON，status 为任务状态（queued/running/succeeded/failed/canceled）",
        "parameters": [
            ("task_id", "str", "必填 — create_video_generation_task 返回的任务 ID"),
        ],
    },
    # -- Gemini Image Generation ----------------------------------------------
    {
        "provider": "Gemini (Google)",
        "endpoint": "POST /models/{model}:generateContent",
        "function": "generate_content",
        "description": (
            "内容生成（含图片）— 返回原始 JSON。"
            "图片在 candidates[].content.parts[].inlineData 中 (base64)。"
            "⚠️ 图片生成时必须设置 generationConfig.responseModalities=[\"IMAGE\"] 或 [\"TEXT\",\"IMAGE\"]"
        ),
        "parameters": [
            ("model", "str", "必填 — 如 gemini-3.1-flash-image-preview"),
            ("contents", "list[dict]", "必填 — [{\"parts\": [{\"text\": \"a cat\"}]}]"),
            ("generationConfig", "dict", (
                "可选 (kwargs) — 生成配置字典，关键子字段：\n"
                "        responseModalities: [\"IMAGE\"] 或 [\"TEXT\",\"IMAGE\"] — 图片生成必设！\n"
                "        imageConfig: {\"aspectRatio\": \"16:9\", \"imageSize\": \"2K\"}\n"
                "          aspectRatio: \"1:1\"|\"4:3\"|\"3:4\"|\"16:9\"|\"9:16\"|\"1:4\"|\"4:1\"|\"1:8\"|\"8:1\"\n"
                "          imageSize: \"0.5K\"|\"1K\"(默认)|\"2K\"|\"4K\"\n"
                "        temperature: float\n"
                "        topP: float\n"
                "        topK: int\n"
                "        maxOutputTokens: int\n"
                "        candidateCount: int (1-8)\n"
                "        stopSequences: list[str]\n"
                "        responseMimeType: \"application/json\" (JSON 模式)"
            )),
            ("systemInstruction", "dict", "可选 (kwargs) — 系统指令，格式同 contents"),
            ("tools", "list[dict]", "可选 (kwargs) — function calling 工具定义"),
            ("safetySettings", "list[dict]", "可选 (kwargs) — 安全过滤设置"),
        ],
    },
    # -- DeepSeek Chat --------------------------------------------------------
    {
        "provider": "DeepSeek",
        "endpoint": "POST /chat/completions",
        "function": "create_chat_completion",
        "description": "对话补全 (OpenAI 兼容) — 返回原始 JSON，choices[0].message.content 为回复文本",
        "parameters": [
            ("model", "str", "必填 — deepseek-v4-flash 或 deepseek-v4-pro"),
            ("messages", "list[dict]", "必填 — [{\"role\": \"user\", \"content\": \"hello\"}]，支持 system/user/assistant/tool 角色"),
            ("temperature", "float", "可选 (kwargs) — 温度，≤ 2，默认 1"),
            ("max_tokens", "int", "可选 (kwargs) — 最大输出 token 数"),
            ("thinking", "dict", "可选 (kwargs) — {\"type\": \"enabled\"|\"disabled\", \"reasoning_effort\": \"high\"|\"max\"}"),
            ("stream", "bool", "可选 (kwargs) — 是否 SSE 流式返回"),
            ("stream_options", "dict", '可选 (kwargs) — {\"include_usage\": true}，流式结束时返回 token 用量'),
            ("top_p", "float", "可选 (kwargs) — nucleus 采样，≤ 1，默认 1"),
            ("frequency_penalty", "float", "可选 (kwargs) — 频率惩罚，-2 ~ 2，默认 0"),
            ("presence_penalty", "float", "可选 (kwargs) — 存在惩罚，-2 ~ 2，默认 0"),
            ("response_format", "dict", '可选 (kwargs) — {\"type\": \"json_object\"} 启用 JSON 模式'),
            ("stop", "str | list[str]", "可选 (kwargs) — 停止序列，最多 16 个"),
            ("tools", "list[dict]", "可选 (kwargs) — function calling 工具定义，最多 128 个"),
            ("tool_choice", "str | dict", '可选 (kwargs) — \"none\"|\"auto\"|\"required\"|{\"type\":\"function\",\"function\":{\"name\":\"xxx\"}}'),
            ("logprobs", "bool", "可选 (kwargs) — 是否返回 log 概率"),
            ("top_logprobs", "int", "可选 (kwargs) — 每个位置返回最可能的 token 数，≤ 20，需 logprobs=true"),
        ],
    },
    # -- Media Utilities ------------------------------------------------------
    {
        "provider": "Media Tools",
        "endpoint": "(local)",
        "function": "upload_via_ssh",
        "description": "通过 SCP 上传本地文件到远程服务器。需在 ~/.zshrc 中配置 UPLOAD_SSH_TARGET / UPLOAD_IDENTITY_FILE / UPLOAD_REMOTE_DIR",
        "parameters": [
            ("local_path", "str", "必填 — 本地文件路径"),
        ],
    },
]


def help(query: str | None = None) -> None:
    """Print available API endpoints and their parameters.

    Args:
        query: Filter by provider name, function name, or keyword.
               Omit to list all.
    """
    matched = _INTERFACES
    if query:
        q = query.lower()
        matched = [
            iface
            for iface in _INTERFACES
            if q in iface["function"].lower()
            or q in iface["provider"].lower()
            or q in iface.get("endpoint", "").lower()
        ]

    if not matched:
        print(f"未找到匹配 '{query}' 的接口。")
        print(f"可用的 provider: ARK, Gemini, DeepSeek, Media Tools")
        return

    current_provider = None
    for iface in matched:
        if iface["provider"] != current_provider:
            current_provider = iface["provider"]
            print(f"\n{'=' * 60}")
            print(f"  {current_provider}")
            print(f"{'=' * 60}")

        print(f"\n  {iface['function']}()  —  {iface['endpoint']}")
        print(f"    {iface['description']}")
        print(f"    参数:")
        for name, ptype, desc in iface["parameters"]:
            print(f"      {name}: {ptype}")
            print(f"        {desc}")

    print()


def list_interfaces() -> list[InterfaceSpec]:
    """Return the raw interface spec list for programmatic use."""
    return _INTERFACES
