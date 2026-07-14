from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from agents.memory import Session
from agents.run import RunConfig
from mem01 import Belief
from mem01.types import PackedMemory

from tests.fakes import FakeInnerSession, FakeMemoryClient


class InnerSession:
    async def get_items(self, limit: int | None = None) -> list[Any]:
        return []

    async def add_items(self, items: list[Any]) -> None:
        return None

    async def pop_item(self) -> Any | None:
        return None

    async def clear_session(self) -> None:
        return None


class MemoryClient:
    pass


def test_mem01_session_satisfies_runtime_protocol() -> None:
    from mem01session.session import Mem01Session

    session = Mem01Session(
        user_id="user-1",
        session_id="session-1",
        inner=InnerSession(),
        runtime=MemoryClient(),
    )

    assert isinstance(session, Session)
    assert session.session_id == "session-1"
    assert session.session_settings is None


def test_canonical_mem_session_symbol_aliases_product_class() -> None:
    from mem01session import Mem01Session, memSession

    assert memSession is Mem01Session


def active_belief(content: str = "User lives in SF.") -> Belief:
    timestamp = datetime(2026, 7, 15, tzinfo=UTC)
    return Belief(
        id="belief-1",
        content=content,
        status="active",
        source="extraction",
        created_at=timestamp,
        updated_at=timestamp,
    )


class UserScopedMemory:
    def __init__(self) -> None:
        self.remember_users: list[str] = []
        self.history_users: list[str] = []
        self.recall_users: list[str] = []

    async def remember(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        session_id: str | None = None,
    ) -> object:
        assert session_id is None
        self.remember_users.append(user_id)
        return object()

    async def history(
        self,
        *,
        user_id: str,
        include_invalidated: bool = False,
        limit: int = 100,
    ) -> list[Belief]:
        self.history_users.append(user_id)
        return []

    async def recall(
        self,
        query: str,
        *,
        user_id: str,
        max_memory_tokens: int,
        include_history: bool = False,
    ) -> PackedMemory:
        self.recall_users.append(user_id)
        return PackedMemory(
            text="",
            tokens_used=0,
            max_memory_tokens=max_memory_tokens,
            candidate_count=0,
            latency_ms=0,
            beliefs=[],
        )


def test_locked_constructor_accepts_positional_session_and_user_ids() -> None:
    from mem01session.session import Mem01Session

    keyword_user = Mem01Session(
        "monday",
        user_id="shared-user",
        inner=FakeInnerSession(),
        runtime=UserScopedMemory(),
    )
    positional_user = Mem01Session(
        "wednesday",
        "shared-user",
        inner=FakeInnerSession(),
        runtime=UserScopedMemory(),
    )

    assert keyword_user.session_id == "monday"
    assert positional_user.user_id == "shared-user"


@pytest.mark.asyncio
async def test_falsey_injected_inner_remains_caller_owned_and_receives_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mem01session.session as session_module

    class FalseyInner(FakeInnerSession):
        def __bool__(self) -> bool:
            return False

    inner = FalseyInner([{"role": "user", "content": "preserved"}])
    monkeypatch.setattr(
        session_module,
        "SQLiteSession",
        lambda session_id: (_ for _ in ()).throw(
            AssertionError("injected falsey Session must be used")
        ),
    )
    session = session_module.Mem01Session(
        "falsey-session",
        "shared-user",
        inner=inner,
        runtime=FakeMemoryClient(),
    )

    result = await session.get_items(limit=1)
    await session.close()

    assert result[-1] is inner.items[0]
    assert inner.get_limits == [1]
    assert inner.close_calls == 0


@pytest.mark.asyncio
async def test_add_items_stores_raw_chain_before_remembering_normalized_text() -> None:
    from mem01session.session import Mem01Session

    events: list[str] = []
    inner = FakeInnerSession(events=events)
    memory = FakeMemoryClient(events=events)
    session = Mem01Session(
        user_id="user-1",
        session_id="monday",
        inner=inner,
        runtime=memory,
    )
    items: list[Any] = [
        {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
        {"type": "reasoning", "summary": []},
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello"}],
        },
    ]

    await session.add_items(cast(Any, items))

    assert events == ["inner.add", "memory.remember"]
    assert inner.items == items
    assert memory.remember_calls == [
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            "user_id": "user-1",
        }
    ]


@pytest.mark.asyncio
async def test_add_items_buffers_user_until_assistant_then_extracts_turn_once() -> None:
    from mem01session.session import Mem01Session

    inner = FakeInnerSession()
    memory = FakeMemoryClient()
    session = Mem01Session("turn", "user-1", inner=inner, runtime=memory)
    user_item = cast(Any, {"role": "user", "content": "Remember blue."})
    assistant_item = cast(Any, {"role": "assistant", "content": "Noted."})

    await session.add_items([user_item])

    assert inner.items == [user_item]
    assert memory.remember_calls == []

    await session.add_items([assistant_item])

    assert inner.items == [user_item, assistant_item]
    assert memory.remember_calls == [
        {
            "messages": [
                {"role": "user", "content": "Remember blue."},
                {"role": "assistant", "content": "Noted."},
            ],
            "user_id": "user-1",
        }
    ]


@pytest.mark.asyncio
async def test_add_items_enqueues_each_completed_turn_in_one_batch() -> None:
    from mem01session.session import Mem01Session

    memory = FakeMemoryClient()
    session = Mem01Session(
        "two-turns", "user-1", inner=FakeInnerSession(), runtime=memory
    )

    await session.add_items(
        cast(
            Any,
            [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second question"},
                {"role": "assistant", "content": "second answer"},
            ],
        )
    )

    assert memory.remember_calls == [
        {
            "messages": [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
            ],
            "user_id": "user-1",
        },
        {
            "messages": [
                {"role": "user", "content": "second question"},
                {"role": "assistant", "content": "second answer"},
            ],
            "user_id": "user-1",
        },
    ]


@pytest.mark.asyncio
async def test_consecutive_user_items_in_same_batch_are_preserved_as_one_turn() -> None:
    from mem01session.session import Mem01Session

    memory = FakeMemoryClient()
    session = Mem01Session(
        "multi-user-items", "user-1", inner=FakeInnerSession(), runtime=memory
    )

    await session.add_items(
        cast(
            Any,
            [
                {"role": "user", "content": "first user item"},
                {"role": "user", "content": "second user item"},
                {"role": "assistant", "content": "combined answer"},
            ],
        )
    )

    assert memory.remember_calls == [
        {
            "messages": [
                {"role": "user", "content": "first user item"},
                {"role": "user", "content": "second user item"},
                {"role": "assistant", "content": "combined answer"},
            ],
            "user_id": "user-1",
        }
    ]


@pytest.mark.asyncio
async def test_consecutive_assistant_items_complete_the_same_turn() -> None:
    from mem01session.session import Mem01Session

    memory = FakeMemoryClient()
    session = Mem01Session(
        "multi-assistant-items",
        "user-1",
        inner=FakeInnerSession(),
        runtime=memory,
    )

    await session.add_items(
        cast(
            Any,
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "first answer part"},
                {"role": "assistant", "content": "second answer part"},
            ],
        )
    )

    assert memory.remember_calls == [
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "first answer part"},
                {"role": "assistant", "content": "second answer part"},
            ],
            "user_id": "user-1",
        }
    ]


@pytest.mark.asyncio
async def test_add_items_does_not_extract_assistant_without_pending_user() -> None:
    from mem01session.session import Mem01Session

    inner = FakeInnerSession()
    memory = FakeMemoryClient()
    session = Mem01Session("assistant-only", "user-1", inner=inner, runtime=memory)
    item = cast(Any, {"role": "assistant", "content": "orphan reply"})

    await session.add_items([item])

    assert inner.items == [item]
    assert memory.remember_calls == []


@pytest.mark.asyncio
async def test_new_user_batch_replaces_abandoned_pending_user() -> None:
    from mem01session.session import Mem01Session

    inner = FakeInnerSession()
    memory = FakeMemoryClient()
    session = Mem01Session("replace-pending", "user-1", inner=inner, runtime=memory)

    await session.add_items(
        cast(Any, [{"role": "user", "content": "abandoned question"}])
    )
    await session.add_items(cast(Any, [{"role": "user", "content": "fresh question"}]))
    await session.add_items(
        cast(Any, [{"role": "assistant", "content": "fresh answer"}])
    )

    assert memory.remember_calls == [
        {
            "messages": [
                {"role": "user", "content": "fresh question"},
                {"role": "assistant", "content": "fresh answer"},
            ],
            "user_id": "user-1",
        }
    ]


@pytest.mark.asyncio
async def test_clear_session_discards_pending_user_extraction() -> None:
    from mem01session.session import Mem01Session

    inner = FakeInnerSession()
    memory = FakeMemoryClient()
    session = Mem01Session("clear-pending", "user-1", inner=inner, runtime=memory)

    await session.add_items(
        cast(Any, [{"role": "user", "content": "stale before clear"}])
    )
    await session.clear_session()
    await session.add_items(
        cast(
            Any,
            [
                {"role": "user", "content": "fresh after clear"},
                {"role": "assistant", "content": "fresh reply"},
            ],
        )
    )

    assert inner.items == [
        {"role": "user", "content": "fresh after clear"},
        {"role": "assistant", "content": "fresh reply"},
    ]
    assert memory.remember_calls == [
        {
            "messages": [
                {"role": "user", "content": "fresh after clear"},
                {"role": "assistant", "content": "fresh reply"},
            ],
            "user_id": "user-1",
        }
    ]


@pytest.mark.asyncio
async def test_pop_item_conservatively_discards_pending_user_extraction() -> None:
    from mem01session.session import Mem01Session

    inner = FakeInnerSession()
    memory = FakeMemoryClient()
    session = Mem01Session("pop-pending", "user-1", inner=inner, runtime=memory)
    stale = cast(Any, {"role": "user", "content": "stale before pop"})

    await session.add_items([stale])
    assert await session.pop_item() == stale
    await session.add_items(
        cast(
            Any,
            [
                {"role": "user", "content": "fresh after pop"},
                {"role": "assistant", "content": "fresh reply"},
            ],
        )
    )

    assert memory.remember_calls == [
        {
            "messages": [
                {"role": "user", "content": "fresh after pop"},
                {"role": "assistant", "content": "fresh reply"},
            ],
            "user_id": "user-1",
        }
    ]


@pytest.mark.asyncio
async def test_add_items_skips_memory_for_non_text_batch() -> None:
    from mem01session.session import Mem01Session

    inner = FakeInnerSession()
    memory = FakeMemoryClient()
    session = Mem01Session(
        user_id="user-1",
        session_id="monday",
        inner=inner,
        runtime=memory,
    )
    items = [{"type": "reasoning", "summary": []}]

    await session.add_items(cast(Any, items))

    assert inner.items == items
    assert memory.remember_calls == []


@pytest.mark.asyncio
async def test_pop_and_clear_delegate_without_forgetting_long_term_beliefs() -> None:
    from mem01session.session import Mem01Session

    item = {"role": "user", "content": "Hi"}
    inner = FakeInnerSession([item])
    memory = FakeMemoryClient([active_belief()])
    session = Mem01Session(
        user_id="user-1",
        session_id="monday",
        inner=inner,
        runtime=memory,
    )

    assert await session.pop_item() is item
    await inner.add_items([item])
    await session.clear_session()

    assert inner.items == []
    assert inner.clear_calls == 1
    assert memory.beliefs == [active_belief()]
    assert memory.remember_calls == []


@pytest.mark.asyncio
async def test_close_does_not_close_injected_resources() -> None:
    from mem01session.session import Mem01Session

    inner = FakeInnerSession()
    memory = FakeMemoryClient()
    session = Mem01Session(
        user_id="user-1",
        session_id="monday",
        inner=inner,
        runtime=memory,
    )

    await session.close()

    assert inner.close_calls == 0
    assert memory.close_calls == 0


@pytest.mark.asyncio
async def test_close_closes_owned_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mem01session.session as session_module

    inner = FakeInnerSession()
    memory = FakeMemoryClient()
    release_calls = 0

    class Lease:
        runtime = memory

        async def release(self) -> None:
            nonlocal release_calls
            release_calls += 1

    async def acquire(settings: object) -> Lease:
        return Lease()

    monkeypatch.setattr(
        session_module, "SQLiteSession", lambda session_id, **kwargs: inner
    )
    monkeypatch.setattr(session_module, "acquire_shared_runtime", acquire)

    session = session_module.Mem01Session(
        user_id="user-1",
        session_id="monday",
        runtime_settings=cast(Any, object()),
    )
    assert release_calls == 0
    await session.add_items(
        cast(
            Any,
            [
                {"role": "user", "content": "acquire"},
                {"role": "assistant", "content": "acquired"},
            ],
        )
    )
    await session.close()

    assert inner.close_calls == 1
    assert release_calls == 1
    assert memory.close_calls == 0


@pytest.mark.asyncio
async def test_close_waits_for_lazy_acquisition_and_releases_its_lease_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mem01session.session as session_module

    acquisition_started = asyncio.Event()
    allow_acquisition = asyncio.Event()
    release_calls = 0
    runtime = FakeMemoryClient()

    class Lease:
        def __init__(self, acquired_runtime: FakeMemoryClient) -> None:
            self.runtime = acquired_runtime

        async def release(self) -> None:
            nonlocal release_calls
            release_calls += 1

    async def gated_acquire(settings: object) -> Lease:
        acquisition_started.set()
        await allow_acquisition.wait()
        return Lease(runtime)

    monkeypatch.setattr(session_module, "acquire_shared_runtime", gated_acquire)
    session = session_module.Mem01Session(
        "race-session",
        "shared-user",
        inner=FakeInnerSession(),
        runtime_settings=cast(Any, object()),
        strict=True,
    )

    memory_task = asyncio.create_task(
        session.add_items(
            cast(
                Any,
                [
                    {"role": "user", "content": "acquire"},
                    {"role": "assistant", "content": "acquired"},
                ],
            )
        )
    )
    await asyncio.wait_for(acquisition_started.wait(), timeout=0.5)
    close_task = asyncio.create_task(session.close())
    await asyncio.sleep(0)

    assert close_task.done() is False
    allow_acquisition.set()
    await asyncio.wait_for(asyncio.gather(memory_task, close_task), timeout=0.5)
    await session.close()

    assert release_calls == 1


@pytest.mark.asyncio
async def test_concurrent_session_closers_wait_for_shared_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mem01session.session as session_module

    release_started = asyncio.Event()
    allow_release = asyncio.Event()
    release_calls = 0
    runtime = FakeMemoryClient()
    runtime_ref = runtime
    inner = FakeInnerSession()

    class Lease:
        runtime = runtime_ref

        async def release(self) -> None:
            nonlocal release_calls
            release_calls += 1
            release_started.set()
            await allow_release.wait()

    async def acquire(settings: object) -> Lease:
        return Lease()

    monkeypatch.setattr(
        session_module, "SQLiteSession", lambda session_id, **kwargs: inner
    )
    monkeypatch.setattr(session_module, "acquire_shared_runtime", acquire)
    session = session_module.Mem01Session(
        "close-barrier",
        "shared-user",
        runtime_settings=cast(Any, object()),
    )
    await session.add_items(
        cast(
            Any,
            [
                {"role": "user", "content": "acquire"},
                {"role": "assistant", "content": "acquired"},
            ],
        )
    )
    first = asyncio.create_task(session.close())
    await asyncio.wait_for(release_started.wait(), timeout=0.5)
    second = asyncio.create_task(session.close())
    await asyncio.sleep(0)

    second_returned_early = second.done()
    allow_release.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=0.5)

    assert second_returned_early is False
    assert release_calls == 1
    assert inner.close_calls == 1


@pytest.mark.asyncio
async def test_cancelled_session_close_drains_cleanup_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mem01session.session as session_module

    release_started = asyncio.Event()
    allow_release = asyncio.Event()
    release_finished = asyncio.Event()
    runtime = FakeMemoryClient()
    runtime_ref = runtime
    inner = FakeInnerSession()

    class Lease:
        runtime = runtime_ref

        async def release(self) -> None:
            release_started.set()
            await allow_release.wait()
            release_finished.set()

    async def acquire(settings: object) -> Lease:
        return Lease()

    monkeypatch.setattr(
        session_module, "SQLiteSession", lambda session_id, **kwargs: inner
    )
    monkeypatch.setattr(session_module, "acquire_shared_runtime", acquire)
    session = session_module.Mem01Session(
        "cancelled-close",
        "shared-user",
        runtime_settings=cast(Any, object()),
    )
    await session.add_items(
        cast(
            Any,
            [
                {"role": "user", "content": "acquire"},
                {"role": "assistant", "content": "acquired"},
            ],
        )
    )
    cancelled_close = asyncio.create_task(session.close())
    await asyncio.wait_for(release_started.wait(), timeout=0.5)

    cancelled_close.cancel()
    await asyncio.sleep(0)
    cancellation_propagated_early = cancelled_close.done()
    allow_release.set()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_close

    assert cancellation_propagated_early is False
    assert release_finished.is_set()
    assert inner.close_calls == 1
    await asyncio.wait_for(session.close(), timeout=0.5)


@pytest.mark.asyncio
async def test_session_cleanup_attempts_both_resources_and_replays_safe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mem01session.session as session_module

    lease_secret = "sk-private-lease-error"
    inner_secret = "postgresql://private:password@secret.example/mem01"
    runtime = FakeMemoryClient()
    runtime_ref = runtime
    release_calls = 0

    class Lease:
        runtime = runtime_ref

        async def release(self) -> None:
            nonlocal release_calls
            release_calls += 1
            raise RuntimeError(lease_secret)

    class FailingAsyncInner(FakeInnerSession):
        async def close(self) -> None:
            self.close_calls += 1
            await asyncio.sleep(0)
            raise RuntimeError(inner_secret)

    async def acquire(settings: object) -> Lease:
        return Lease()

    inner = FailingAsyncInner()
    monkeypatch.setattr(
        session_module, "SQLiteSession", lambda session_id, **kwargs: inner
    )
    monkeypatch.setattr(session_module, "acquire_shared_runtime", acquire)
    session = session_module.Mem01Session(
        "failing-close",
        "shared-user",
        runtime_settings=cast(Any, object()),
    )
    await session.add_items(
        cast(
            Any,
            [
                {"role": "user", "content": "acquire"},
                {"role": "assistant", "content": "acquired"},
            ],
        )
    )

    with pytest.raises(RuntimeError) as first_error:
        await session.close()
    with pytest.raises(RuntimeError) as second_error:
        await session.close()

    assert release_calls == 1
    assert inner.close_calls == 1
    assert str(first_error.value) == str(second_error.value)
    assert lease_secret not in str(first_error.value)
    assert inner_secret not in str(first_error.value)
    assert "private" not in str(first_error.value)


@pytest.mark.asyncio
async def test_owned_sync_inner_close_runs_off_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mem01session.session as session_module

    close_threads: list[int] = []

    class ThreadRecordingInner(FakeInnerSession):
        def close(self) -> None:
            close_threads.append(threading.get_ident())
            super().close()

    inner = ThreadRecordingInner()
    monkeypatch.setattr(
        session_module, "SQLiteSession", lambda session_id, **kwargs: inner
    )
    session = session_module.Mem01Session("threaded-close", "shared-user")
    event_loop_thread = threading.get_ident()

    await session.close()

    assert close_threads
    assert close_threads[0] != event_loop_thread
    assert inner.close_calls == 1


def test_run_config_binds_query_aware_callback() -> None:
    from mem01session.session import Mem01Session

    session = Mem01Session(
        user_id="user-1",
        session_id="friday",
        inner=FakeInnerSession(),
        runtime=FakeMemoryClient(),
    )

    config = session.run_config()

    assert isinstance(config, RunConfig)
    assert config.session_input_callback is not None
    assert config.call_model_input_filter is not None
