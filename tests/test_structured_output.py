from __future__ import annotations

import json

import pytest
from pydantic import Field

from ai_toolkit import JsonOutputModel, StructuredOutputError
from ai_toolkit.ark import text as ark_text
from ai_toolkit.deepseek import text as deepseek_text


class SceneOutput(JsonOutputModel):
    asset_prompt: str = Field(
        min_length=1,
        description="完整媒体生成提示词",
    )
    event_summary: str = Field(
        min_length=1,
        description="会影响下一轮互动的事件摘要",
    )


def test_prompt_fragment_renders_one_compact_json_example():
    fragment = SceneOutput.prompt_fragment()
    instruction, example_text = fragment.split("\n\n", 1)

    assert instruction == "只返回以下结构的 JSON 对象，不要输出解释或 Markdown："
    assert json.loads(example_text) == {
        "asset_prompt": "<完整媒体生成提示词>",
        "event_summary": "<会影响下一轮互动的事件摘要>",
    }
    assert fragment.count("完整媒体生成提示词") == 1
    assert fragment.count("会影响下一轮互动的事件摘要") == 1


def test_deepseek_output_type_returns_validated_model(monkeypatch):
    captured = {}

    def fake_create_chat_completion(model, messages, **kwargs):
        captured["kwargs"] = kwargs
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"asset_prompt":"  prompt  ",'
                            '"event_summary":"summary"}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(
        deepseek_text,
        "create_chat_completion",
        fake_create_chat_completion,
    )

    result = deepseek_text.complete_json(
        model="v4-flash",
        prompt=SceneOutput.prompt_fragment(),
        output_type=SceneOutput,
    )

    assert isinstance(result.output, SceneOutput)
    assert result.output.asset_prompt == "prompt"
    assert result.parsed_json == {
        "asset_prompt": "  prompt  ",
        "event_summary": "summary",
    }
    assert captured["kwargs"]["response_format"] == {"type": "json_object"}


def test_ark_output_type_sends_native_schema(monkeypatch):
    captured = {}

    def fake_create_responses(model, input, **kwargs):
        captured["kwargs"] = kwargs
        return {
            "output_text": (
                '{"asset_prompt":"prompt","event_summary":"summary"}'
            )
        }

    monkeypatch.setattr(ark_text, "create_responses", fake_create_responses)

    result = ark_text.complete_json(
        model="doubao-pro",
        prompt="generate",
        output_type=SceneOutput,
    )

    output_format = captured["kwargs"]["text"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["strict"] is True
    assert output_format["schema"] == SceneOutput.model_json_schema()
    assert isinstance(result.output, SceneOutput)


@pytest.mark.parametrize(
    ("content", "stage"),
    [
        ("not json", "json_decode"),
        (
            '{"asset_prompt":" ","event_summary":"summary"}',
            "validation",
        ),
        (
            '{"asset_prompt":"prompt"}',
            "validation",
        ),
        (
            (
                '{"asset_prompt":"prompt","event_summary":"summary",'
                '"unexpected":true}'
            ),
            "validation",
        ),
    ],
)
def test_output_type_raises_structured_error(monkeypatch, content, stage):
    def fake_create_chat_completion(model, messages, **kwargs):
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(
        deepseek_text,
        "create_chat_completion",
        fake_create_chat_completion,
    )

    with pytest.raises(StructuredOutputError) as raised:
        deepseek_text.complete_json(
            model="v4-flash",
            prompt="generate",
            output_type=SceneOutput,
        )

    assert raised.value.stage == stage
    assert raised.value.output_type is SceneOutput
    assert raised.value.completion.text == content


def test_schema_and_output_type_are_mutually_exclusive():
    with pytest.raises(ValueError, match="schema and output_type"):
        deepseek_text.complete_json(
            prompt="generate",
            schema={"type": "object"},
            output_type=SceneOutput,
        )


def test_legacy_schema_path_remains_soft_on_validation_failure(monkeypatch):
    def fake_create_chat_completion(model, messages, **kwargs):
        return {"choices": [{"message": {"content": '{"ok":"yes"}'}}]}

    monkeypatch.setattr(
        deepseek_text,
        "create_chat_completion",
        fake_create_chat_completion,
    )

    result = deepseek_text.complete_json(
        model="v4-flash",
        prompt="generate",
        schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        },
    )

    assert result.output is None
    assert result.parsed_json is None
    assert result.schema_error is not None
