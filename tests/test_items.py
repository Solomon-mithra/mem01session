from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pytest

from mem01session.items import latest_user_text, messages_from_items, text_from_item


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        ({"role": "user", "content": " hello "}, "hello"),
        (
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "first"},
                    {"type": "input_image", "image_url": "data:image/png,..."},
                    {"type": "input_text", "text": "second"},
                ],
            },
            "first\nsecond",
        ),
        (
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "answer", "annotations": []}
                ],
            },
            "answer",
        ),
        ({"type": "reasoning", "summary": []}, None),
        ({"type": "function_call", "arguments": '{"secret": true}'}, None),
        ({"role": "user", "content": "   \n "}, None),
    ],
)
def test_text_from_item_supports_responses_text_shapes(
    item: dict[str, Any], expected: str | None
) -> None:
    assert text_from_item(item) == expected


@dataclass
class ContentPart:
    type: str
    text: str


@dataclass
class ObjectItem:
    role: str
    content: list[ContentPart]


def test_text_from_item_supports_attribute_based_sdk_objects() -> None:
    item = ObjectItem(role="assistant", content=[ContentPart("output_text", "Hi")])

    assert text_from_item(item) == "Hi"


def test_messages_normalize_only_user_and_assistant_without_mutation() -> None:
    items: list[Any] = [
        {"role": "system", "content": "do not persist"},
        {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
        {"type": "function_call", "name": "lookup", "arguments": "{}"},
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello"}],
        },
        {"role": "tool", "content": "tool result"},
        {"role": "user", "content": "  "},
    ]
    before = deepcopy(items)

    messages = messages_from_items(items)

    assert messages == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    assert items == before


def test_latest_user_text_selects_newest_textual_user_item() -> None:
    items: list[Any] = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": [{"type": "input_image", "image_url": "x"}]},
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "latest question"}],
        },
    ]

    assert latest_user_text(items) == "latest question"


def test_latest_user_text_returns_none_for_string_input_without_role() -> None:
    assert latest_user_text(["What do you remember?"]) is None
