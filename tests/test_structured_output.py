from __future__ import annotations

import json
from typing import Literal

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


class ConstrainedOutput(JsonOutputModel):
    commitment_deadline_minutes: int | None = Field(
        default=None,
        ge=1,
        le=10080,
    )
    target_id: Literal["chen_mo", "lin_xiao"] | None
    retry_count: int = Field(ge=1, le=3)
    negative_score: float = Field(ge=-1, le=-0.25)


class ActivityChangeOutput(JsonOutputModel):
    transition: Literal["start", "continue", "none"]
    referenced_activity_id: str | None = None
    new_activity_description: str | None = Field(
        default=None,
        description="开始新活动时填写的活动描述",
    )


class NestedDecisionOutput(JsonOutputModel):
    activity_change: ActivityChangeOutput


class PlanStepOutput(JsonOutputModel):
    action: str = Field(description="待执行动作")
    completed: bool = False


class NestedArrayOutput(JsonOutputModel):
    steps: list[PlanStepOutput] = Field(min_length=1)


class CommunicationOutput(JsonOutputModel):
    mode: Literal["face_to_face", "message"]
    content: str


class OptionalCommunicationOutput(JsonOutputModel):
    communication: CommunicationOutput | None = None


class ExplicitCommunicationOutput(JsonOutputModel):
    communication: CommunicationOutput | None = Field(
        default=None,
        examples=[{"mode": "message", "content": "稍后联系。"}],
    )


class SpokenActionOutput(JsonOutputModel):
    kind: Literal["speak"]
    content: str = Field(description="说出的内容")


class SilentActionOutput(JsonOutputModel):
    kind: Literal["wait"]
    duration_minutes: int = Field(ge=1)


class DiscriminatedActionOutput(JsonOutputModel):
    action: SpokenActionOutput | SilentActionOutput = Field(discriminator="kind")


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


def test_prompt_fragment_example_satisfies_constrained_schema():
    _, example_text = ConstrainedOutput.prompt_fragment().split("\n\n", 1)
    example = json.loads(example_text)

    assert example == {
        "commitment_deadline_minutes": None,
        "target_id": "chen_mo",
        "retry_count": 1,
        "negative_score": -0.25,
    }
    assert ConstrainedOutput.model_validate(example).model_dump() == example


def test_prompt_fragment_resolves_required_nested_object_refs():
    _, example_text = NestedDecisionOutput.prompt_fragment().split("\n\n", 1)
    example = json.loads(example_text)

    assert example == {
        "activity_change": {
            "transition": "start",
            "referenced_activity_id": None,
            "new_activity_description": None,
        }
    }
    assert NestedDecisionOutput.model_validate(example).model_dump() == example


def test_prompt_fragment_resolves_nested_array_item_refs():
    _, example_text = NestedArrayOutput.prompt_fragment().split("\n\n", 1)
    example = json.loads(example_text)

    assert example == {
        "steps": [
            {
                "action": "<待执行动作>",
                "completed": False,
            }
        ]
    }
    assert NestedArrayOutput.model_validate(example).model_dump() == example


def test_prompt_fragment_keeps_optional_nested_object_null_by_default():
    _, example_text = OptionalCommunicationOutput.prompt_fragment().split("\n\n", 1)

    assert json.loads(example_text) == {"communication": None}


def test_prompt_fragment_prefers_explicit_nested_object_example():
    _, example_text = ExplicitCommunicationOutput.prompt_fragment().split("\n\n", 1)
    example = json.loads(example_text)

    assert example == {
        "communication": {
            "mode": "message",
            "content": "稍后联系。",
        }
    }
    assert ExplicitCommunicationOutput.model_validate(example).model_dump() == example


def test_prompt_fragment_resolves_one_of_nested_object_ref():
    _, example_text = DiscriminatedActionOutput.prompt_fragment().split("\n\n", 1)
    example = json.loads(example_text)

    assert example == {
        "action": {
            "kind": "speak",
            "content": "<说出的内容>",
        }
    }
    assert DiscriminatedActionOutput.model_validate(example).model_dump() == example


def test_deepseek_output_type_returns_validated_model(monkeypatch):
    captured = {}

    def fake_create_chat_completion(model, messages, **kwargs):
        captured["kwargs"] = kwargs
        return {
            "choices": [
                {
                    "message": {
                        "content": ('{"asset_prompt":"  prompt  ","event_summary":"summary"}')
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
        return {"output_text": ('{"asset_prompt":"prompt","event_summary":"summary"}')}

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
            ('{"asset_prompt":"prompt","event_summary":"summary","unexpected":true}'),
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
