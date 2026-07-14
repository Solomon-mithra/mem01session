from __future__ import annotations

from mem01session.metrics import (
    APPROXIMATE_CONTEXT_LABEL,
    OFFLINE_PREPARED_INPUT_MEASUREMENT,
    TOKEN_ESTIMATOR_UTF8_BYTES_UPPER_BOUND,
    estimate_prompt_context,
)


def test_context_estimate_is_transparently_approximate() -> None:
    items = [
        {"role": "system", "content": "memory"},
        {"role": "user", "content": "hello"},
        {"type": "reasoning", "summary": []},
    ]

    estimate = estimate_prompt_context(items)

    assert estimate.label == APPROXIMATE_CONTEXT_LABEL
    assert "Approximate" in estimate.label
    assert "not exact billed tokens" in estimate.label
    assert estimate.item_count == 3
    assert estimate.text_characters == 11
    assert estimate.estimated_tokens > 0


def test_measurement_names_the_offline_upper_bound_explicitly() -> None:
    assert OFFLINE_PREPARED_INPUT_MEASUREMENT == "offline_prepared_model_input"
    assert TOKEN_ESTIMATOR_UTF8_BYTES_UPPER_BOUND == "utf8_bytes_upper_bound"
