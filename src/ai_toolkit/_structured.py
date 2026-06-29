"""Structured-output helpers shared by text providers."""

from __future__ import annotations

import json
from typing import Any

JSON_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "boolean": (bool,),
    "null": (type(None),),
}


def parse_json(text: str) -> Any | None:
    stripped = text.strip()
    for value in (stripped, *json_blocks(stripped)):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            continue
    return None


def json_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    stack: list[int] = []
    for index, char in enumerate(text):
        if char == "{":
            stack.append(index)
        elif char == "}" and stack:
            start = stack.pop()
            if not stack:
                blocks.append(text[start : index + 1])
    return blocks


def validate_against_schema(value: Any, schema: dict[str, Any]) -> str | None:
    """Small JSON-Schema subset validator for SDK-level extraction results."""
    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not matches_type(value, expected_type):
        return f"expected JSON {expected_type}, got {type(value).__name__}"

    if expected_type != "object" or not isinstance(value, dict):
        return None

    for required_key in schema.get("required", []) or []:
        if required_key not in value:
            return f"missing required key {required_key!r}"

    properties = schema.get("properties") or {}
    for key, property_schema in properties.items():
        if key not in value or not isinstance(property_schema, dict):
            continue
        property_type = property_schema.get("type")
        if isinstance(property_type, str) and not matches_type(value[key], property_type):
            return f"property {key!r} expected {property_type}, got {type(value[key]).__name__}"

    return None


def matches_type(value: Any, expected: str) -> bool:
    if expected == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    types = JSON_TYPE_CHECKS.get(expected)
    if types is None:
        return True
    return isinstance(value, types)
