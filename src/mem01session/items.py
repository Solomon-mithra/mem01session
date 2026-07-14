"""Safe conversion helpers for OpenAI Responses input and output items."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_TEXT_PART_TYPES = {"input_text", "output_text", "text"}


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def text_from_item(item: Any) -> str | None:
    """Extract only visible text from a string or Responses-style item."""
    if isinstance(item, str):
        return _clean_text(item)

    content = _field(item, "content")
    if isinstance(content, str):
        return _clean_text(content)
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return None

    parts: list[str] = []
    for part in content:
        if _field(part, "type") not in _TEXT_PART_TYPES:
            continue
        text = _clean_text(_field(part, "text"))
        if text is not None:
            parts.append(text)
    return "\n".join(parts) or None


def messages_from_items(items: Sequence[Any]) -> list[dict[str, str]]:
    """Normalize textual user/assistant items for mem01 extraction."""
    messages: list[dict[str, str]] = []
    for item in items:
        role = _field(item, "role")
        if role not in {"user", "assistant"}:
            continue
        content = text_from_item(item)
        if content is not None:
            messages.append({"role": role, "content": content})
    return messages


def latest_user_text(items: Sequence[Any]) -> str | None:
    """Return the newest textual user message without changing the input sequence."""
    for item in reversed(items):
        if _field(item, "role") != "user":
            continue
        text = text_from_item(item)
        if text is not None:
            return text
    return None
