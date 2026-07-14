"""Prompt-shape estimates for demos; these are not provider usage metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .items import text_from_item
from .memory_block import estimate_tokens

APPROXIMATE_CONTEXT_LABEL = (
    "Approximate prompt context (local estimate; not exact billed tokens)"
)
OFFLINE_PREPARED_INPUT_MEASUREMENT = "offline_prepared_model_input"
TOKEN_ESTIMATOR_UTF8_BYTES_UPPER_BOUND = "utf8_bytes_upper_bound"


@dataclass(frozen=True, slots=True)
class ContextEstimate:
    """A clearly labeled local estimate of prepared prompt shape."""

    item_count: int
    text_characters: int
    estimated_tokens: int
    label: str = APPROXIMATE_CONTEXT_LABEL


def estimate_prompt_context(items: Sequence[Any]) -> ContextEstimate:
    """Estimate visible text size without representing provider-billed usage."""
    texts = [text for item in items if (text := text_from_item(item)) is not None]
    combined = "\n".join(texts)
    return ContextEstimate(
        item_count=len(items),
        text_characters=sum(len(text) for text in texts),
        estimated_tokens=estimate_tokens(combined),
    )
