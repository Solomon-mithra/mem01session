from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from agents import Model
from agents.items import ModelResponse
from agents.usage import Usage
from mem01 import ApplyResult
from mem01.types import Belief, PackedMemory
from openai.types.responses import ResponseOutputMessage, ResponseOutputText


class FakeInnerSession:
    def __init__(
        self, items: Sequence[Any] = (), *, events: list[str] | None = None
    ) -> None:
        self.items = list(items)
        self.events = events
        self.get_limits: list[int | None] = []
        self.clear_calls = 0
        self.close_calls = 0
        self.session_settings = None

    async def get_items(self, limit: int | None = None) -> list[Any]:
        self.get_limits.append(limit)
        items = self.items if limit is None else self.items[-limit:]
        return list(items)

    async def add_items(self, items: list[Any]) -> None:
        if self.events is not None:
            self.events.append("inner.add")
        self.items.extend(items)

    async def pop_item(self) -> Any | None:
        return self.items.pop() if self.items else None

    async def clear_session(self) -> None:
        self.clear_calls += 1
        self.items.clear()

    def close(self) -> None:
        self.close_calls += 1


class FakeMemoryClient:
    def __init__(
        self,
        beliefs: Sequence[Belief] = (),
        *,
        events: list[str] | None = None,
    ) -> None:
        self.beliefs = list(beliefs)
        self.events = events
        self.remember_calls: list[dict[str, Any]] = []
        self.history_calls: list[dict[str, Any]] = []
        self.recall_calls: list[dict[str, Any]] = []
        self.correct_calls: list[dict[str, Any]] = []
        self.forget_calls: list[dict[str, Any]] = []
        self.clear_user_calls: list[dict[str, str]] = []
        self.remember_error: Exception | None = None
        self.history_error: Exception | None = None
        self.recall_error: Exception | None = None
        self.correct_error: Exception | None = None
        self.forget_error: Exception | None = None
        self.clear_user_error: Exception | None = None
        self.remember_result = object()
        self.correct_result = ApplyResult()
        self.forget_result = ApplyResult()
        self.clear_user_result = 0
        self.close_calls = 0

    async def remember(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        session_id: str | None = None,
    ) -> object:
        if self.events is not None:
            self.events.append("memory.remember")
        call: dict[str, Any] = {"messages": messages, "user_id": user_id}
        if session_id is not None:
            call["session_id"] = session_id
        self.remember_calls.append(call)
        if self.remember_error is not None:
            raise self.remember_error
        return self.remember_result

    async def enqueue_remember(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
    ) -> object:
        return await self.remember(messages, user_id=user_id)

    async def flush_pending(self, *, user_id: str) -> None:
        return None

    async def history(
        self,
        *,
        user_id: str,
        include_invalidated: bool = False,
        limit: int = 100,
    ) -> list[Belief]:
        self.history_calls.append(
            {
                "user_id": user_id,
                "include_invalidated": include_invalidated,
                "limit": limit,
            }
        )
        if self.history_error is not None:
            raise self.history_error
        return list(self.beliefs)

    async def recall(
        self,
        query: str,
        *,
        user_id: str,
        max_memory_tokens: int,
        include_history: bool = False,
    ) -> PackedMemory:
        self.recall_calls.append(
            {
                "query": query,
                "user_id": user_id,
                "max_memory_tokens": max_memory_tokens,
                "include_history": include_history,
            }
        )
        if self.recall_error is not None:
            raise self.recall_error
        return PackedMemory(
            text="\n".join(belief.content for belief in self.beliefs),
            tokens_used=0,
            max_memory_tokens=max_memory_tokens,
            candidate_count=len(self.beliefs),
            latency_ms=0,
            beliefs=self.beliefs,
        )

    async def correct(self, memory_id: str, new_value: str, *, user_id: str) -> object:
        self.correct_calls.append(
            {"memory_id": memory_id, "new_value": new_value, "user_id": user_id}
        )
        if self.correct_error is not None:
            raise self.correct_error
        return self.correct_result

    async def forget(
        self,
        memory_id: str,
        *,
        user_id: str,
        reason: str | None = None,
    ) -> object:
        self.forget_calls.append(
            {"memory_id": memory_id, "user_id": user_id, "reason": reason}
        )
        if self.forget_error is not None:
            raise self.forget_error
        return self.forget_result

    async def clear_user(self, *, user_id: str) -> int:
        self.clear_user_calls.append({"user_id": user_id})
        if self.clear_user_error is not None:
            raise self.clear_user_error
        return self.clear_user_result

    async def aclose(self) -> None:
        self.close_calls += 1


class EchoModel(Model):
    """Deterministic SDK Model that performs no I/O and records prepared input."""

    def __init__(self, answer: str = "deterministic answer") -> None:
        self.answer = answer
        self.prepared_inputs: list[Any] = []

    async def get_response(
        self,
        system_instructions: str | None,
        input: Any,
        *args: Any,
        **kwargs: Any,
    ) -> ModelResponse:
        self.prepared_inputs.append(input)
        output = ResponseOutputMessage(
            id=f"message-{len(self.prepared_inputs)}",
            content=[
                ResponseOutputText(
                    annotations=[],
                    text=self.answer,
                    type="output_text",
                    logprobs=[],
                )
            ],
            role="assistant",
            status="completed",
            type="message",
        )
        return ModelResponse(
            output=[output],
            usage=Usage(requests=1, input_tokens=10, output_tokens=3, total_tokens=13),
            response_id=f"response-{len(self.prepared_inputs)}",
        )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: Any,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        if False:
            yield None
        raise AssertionError("EchoModel streaming was not expected")
