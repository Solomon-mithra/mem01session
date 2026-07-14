from __future__ import annotations

from typing import Any, cast

import pytest
from agents import Agent, RunConfig, Runner, function_tool
from agents.items import ModelResponse
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall

from mem01session.memory_block import is_memory_item
from mem01session.session import Mem01Session
from tests.fakes import EchoModel, FakeInnerSession, FakeMemoryClient
from tests.test_session import active_belief


@pytest.mark.asyncio
async def test_runner_without_config_does_not_recall_or_inject_memory() -> None:
    inner = FakeInnerSession()
    memory = FakeMemoryClient([active_belief()])
    session = Mem01Session(
        user_id="runner-user",
        session_id="runner-session",
        inner=inner,
        runtime=memory,
    )
    model = EchoModel("You live in SF.")
    agent = Agent(name="memory-test", instructions="Answer briefly.", model=model)

    result = await Runner.run(
        agent,
        "Where do I live?",
        session=session,
        run_config=RunConfig(tracing_disabled=True),
    )

    assert result.final_output == "You live in SF."
    prepared = cast(list[Any], model.prepared_inputs[0])
    assert not any(is_memory_item(item) for item in prepared)
    assert prepared[-1]["role"] == "user"
    assert prepared[-1]["content"] == "Where do I live?"
    assert len(inner.items) == 2
    assert inner.items[0]["role"] == "user"
    assert inner.items[1]["role"] == "assistant"
    assert not any(is_memory_item(item) for item in inner.items)
    assert memory.recall_calls == []
    assert memory.remember_calls == [
        {
            "messages": [
                {"role": "user", "content": "Where do I live?"},
                {"role": "assistant", "content": "You live in SF."},
            ],
            "user_id": "runner-user",
        }
    ]


@pytest.mark.asyncio
async def test_real_runner_query_aware_config_calls_recall_with_current_input() -> None:
    inner = FakeInnerSession()
    memory = FakeMemoryClient([active_belief()])
    session = Mem01Session(
        user_id="runner-user",
        session_id="query-aware-session",
        inner=inner,
        runtime=memory,
        max_memory_tokens=800,
    )
    model = EchoModel("You live in SF.")
    agent = Agent(name="memory-test", model=model)
    config = session.run_config()
    config.tracing_disabled = True

    await Runner.run(
        agent,
        "What is my current city?",
        session=session,
        run_config=config,
    )

    assert memory.recall_calls == [
        {
            "query": "What is my current city?",
            "user_id": "runner-user",
            "max_memory_tokens": 800,
            "include_history": False,
        }
    ]
    prepared = cast(list[Any], model.prepared_inputs[0])
    assert len([item for item in prepared if is_memory_item(item)]) == 1
    assert any(
        item.get("role") == "user" and item.get("content") == "What is my current city?"
        for item in prepared
    )
    assert len(inner.items) == 2
    assert not any(is_memory_item(item) for item in inner.items)
    assert memory.remember_calls == [
        {
            "messages": [
                {"role": "user", "content": "What is my current city?"},
                {"role": "assistant", "content": "You live in SF."},
            ],
            "user_id": "runner-user",
        }
    ]


@pytest.mark.asyncio
async def test_real_runner_tool_loop_reuses_cached_memory_across_model_calls() -> None:
    class ToolThenAnswerModel(EchoModel):
        async def get_response(
            self,
            system_instructions: str | None,
            input: Any,
            *args: Any,
            **kwargs: Any,
        ) -> ModelResponse:
            if not self.prepared_inputs:
                self.prepared_inputs.append(input)
                return ModelResponse(
                    output=[
                        ResponseFunctionToolCall(
                            arguments='{"city":"SF"}',
                            call_id="call-1",
                            name="lookup_weather",
                            type="function_call",
                            id="function-1",
                            status="completed",
                        )
                    ],
                    usage=Usage(
                        requests=1,
                        input_tokens=10,
                        output_tokens=3,
                        total_tokens=13,
                    ),
                    response_id="response-1",
                )
            return await super().get_response(
                system_instructions, input, *args, **kwargs
            )

    @function_tool
    def lookup_weather(city: str) -> str:
        """Return deterministic local weather for a city."""
        return f"sunny in {city}"

    memory = FakeMemoryClient([active_belief()])
    inner = FakeInnerSession()
    session = Mem01Session("multi-call", "runner-user", inner=inner, runtime=memory)
    model = ToolThenAnswerModel("Final answer")
    agent = Agent(name="memory-test", model=model, tools=[lookup_weather])
    config = session.run_config()
    config.tracing_disabled = True

    result = await Runner.run(
        agent,
        "Where do I live?",
        session=session,
        run_config=config,
    )

    assert result.final_output == "Final answer"
    assert len(model.prepared_inputs) == 2
    assert len(memory.recall_calls) == 1
    first, second = cast(list[list[Any]], model.prepared_inputs)
    assert len([item for item in first if is_memory_item(item)]) == 1
    assert len([item for item in second if is_memory_item(item)]) == 1
    assert first[0] == second[0]
    assert first[-1] == {"role": "user", "content": "Where do I live?"}
    assert any(
        item.get("role") == "user" and item.get("content") == "Where do I live?"
        for item in second
    )
    assert any(item.get("type") == "function_call_output" for item in second)

    assert not any(is_memory_item(item) for item in inner.items)
    assert not any(item.get("role") == "system" for item in inner.items)
    assert [item.get("type", item.get("role")) for item in inner.items] == [
        "user",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert memory.remember_calls == [
        {
            "messages": [
                {"role": "user", "content": "Where do I live?"},
                {"role": "assistant", "content": "Final answer"},
            ],
            "user_id": "runner-user",
        },
    ]
