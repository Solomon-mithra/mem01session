from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, fields
from typing import Any

import pytest
from mem01.runtime import OpenAIRuntimeSettings

import mem01session.runtime as runtime_module
from mem01session.runtime import (
    EmbeddedMem01Runtime,
    acquire_shared_runtime,
    close_shared_runtimes,
)


@dataclass
class RecordingStore:
    close_calls: int = 0

    def close(self) -> None:
        self.close_calls += 1


class RecordingClient:
    def __init__(self) -> None:
        self.store = RecordingStore()
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any], int]] = []
        self.results = {
            name: object()
            for name in (
                "remember",
                "recall",
                "history",
                "correct",
                "forget",
                "clear_user",
            )
        }
        self.results["clear_user"] = 0

    def _record(self, name: str, *args: Any, **kwargs: Any) -> object:
        self.calls.append((name, args, kwargs, threading.get_ident()))
        return self.results[name]

    def remember(self, *args: Any, **kwargs: Any) -> object:
        return self._record("remember", *args, **kwargs)

    def recall(self, *args: Any, **kwargs: Any) -> object:
        return self._record("recall", *args, **kwargs)

    def history(self, *args: Any, **kwargs: Any) -> object:
        return self._record("history", *args, **kwargs)

    def correct(self, *args: Any, **kwargs: Any) -> object:
        return self._record("correct", *args, **kwargs)

    def forget(self, *args: Any, **kwargs: Any) -> object:
        return self._record("forget", *args, **kwargs)

    def clear_user(self, *args: Any, **kwargs: Any) -> object:
        return self._record("clear_user", *args, **kwargs)


def settings(
    *,
    api_key: str = "sk-runtime-secret",
    database_url: str = "postgresql://private:password@db.example/mem01",
    llm_model: str = "gpt-5.6-sol",
) -> OpenAIRuntimeSettings:
    return OpenAIRuntimeSettings(
        api_key=api_key,
        database_url=database_url,
        llm_model=llm_model,
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        base_url="https://api.openai.com/v1",
    )


def wait_until_closed(runtime: EmbeddedMem01Runtime, *, timeout: float = 0.5) -> None:
    deadline = time.monotonic() + timeout
    while not runtime.is_closed:
        if time.monotonic() >= deadline:
            raise AssertionError("runtime did not begin closing")
        time.sleep(0.001)


@pytest.fixture(autouse=True)
async def empty_runtime_registry() -> Any:
    await close_shared_runtimes()
    yield
    await close_shared_runtimes()


@pytest.mark.asyncio
async def test_all_sync_operations_run_in_threads_with_arguments_and_results() -> None:
    client = RecordingClient()
    runtime = EmbeddedMem01Runtime(client=client)
    loop_thread = threading.get_ident()
    messages = [{"role": "user", "content": "I moved."}]

    results = [
        await runtime.remember(messages, user_id="alex", project_id="work"),
        await runtime.recall("where now?", user_id="alex", max_memory_tokens=800, k=12),
        await runtime.history(user_id="alex", include_invalidated=False, limit=25),
        await runtime.correct("belief-1", "Lives in SF", confidence=0.98),
        await runtime.forget("belief-2", reason="user requested deletion"),
    ]

    assert results == [
        client.results["remember"],
        client.results["recall"],
        client.results["history"],
        client.results["correct"],
        client.results["forget"],
    ]
    assert client.calls == [
        (
            "remember",
            (messages,),
            {"user_id": "alex", "project_id": "work"},
            client.calls[0][3],
        ),
        (
            "recall",
            ("where now?",),
            {"user_id": "alex", "max_memory_tokens": 800, "k": 12},
            client.calls[1][3],
        ),
        (
            "history",
            (),
            {"user_id": "alex", "include_invalidated": False, "limit": 25},
            client.calls[2][3],
        ),
        (
            "correct",
            ("belief-1", "Lives in SF"),
            {"confidence": 0.98},
            client.calls[3][3],
        ),
        (
            "forget",
            ("belief-2",),
            {"reason": "user requested deletion"},
            client.calls[4][3],
        ),
    ]
    assert all(call_thread != loop_thread for *_, call_thread in client.calls)


@pytest.mark.asyncio
async def test_blocking_engine_call_does_not_block_event_loop() -> None:
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()

    class BlockingClient(RecordingClient):
        def recall(self, *args: Any, **kwargs: Any) -> object:
            loop.call_soon_threadsafe(entered.set)
            release.wait(timeout=1)
            return super().recall(*args, **kwargs)

    runtime = EmbeddedMem01Runtime(client=BlockingClient())
    recall_task = asyncio.create_task(runtime.recall("query", user_id="alex"))

    await asyncio.wait_for(entered.wait(), timeout=0.5)
    assert not recall_task.done()
    release.set()

    await recall_task


@pytest.mark.asyncio
async def test_clear_user_delegates_scope_in_worker_thread_and_returns_count() -> None:
    client = RecordingClient()
    client.results["clear_user"] = 3
    runtime = EmbeddedMem01Runtime(client=client)
    loop_thread = threading.get_ident()

    deleted = await runtime.clear_user(user_id="scoped-user")

    assert deleted == 3
    assert client.calls == [
        ("clear_user", (), {"user_id": "scoped-user"}, client.calls[0][3])
    ]
    assert client.calls[0][3] != loop_thread


@pytest.mark.parametrize("user_id", [None, "", "   ", 1])
@pytest.mark.asyncio
async def test_clear_user_rejects_invalid_user_id(user_id: Any) -> None:
    runtime = EmbeddedMem01Runtime(client=RecordingClient())
    with pytest.raises((TypeError, ValueError), match="user_id"):
        await runtime.clear_user(user_id=user_id)
    await runtime.aclose()


@pytest.mark.asyncio
async def test_identical_concurrent_acquisitions_share_until_last_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = RecordingClient()
    build_calls = 0

    def build(*, settings: OpenAIRuntimeSettings) -> RecordingClient:
        nonlocal build_calls
        build_calls += 1
        return client

    monkeypatch.setattr(runtime_module, "build_openai_memory_client", build)

    first, second = await asyncio.gather(
        acquire_shared_runtime(settings()),
        acquire_shared_runtime(settings()),
    )

    assert first.runtime is second.runtime
    assert first.runtime.client is client
    assert build_calls == 1
    await first.release()
    assert client.store.close_calls == 0
    await second.release()
    assert client.store.close_calls == 1


@pytest.mark.asyncio
async def test_different_configurations_do_not_share(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[RecordingClient] = []

    def build(*, settings: OpenAIRuntimeSettings) -> RecordingClient:
        client = RecordingClient()
        clients.append(client)
        return client

    monkeypatch.setattr(runtime_module, "build_openai_memory_client", build)

    first = await acquire_shared_runtime(settings())
    second = await acquire_shared_runtime(settings(llm_model="gpt-other"))

    assert first.runtime is not second.runtime
    assert first.runtime.client is not second.runtime.client
    await first.release()
    await second.release()
    assert [client.store.close_calls for client in clients] == [1, 1]


@pytest.mark.asyncio
async def test_cancelled_shared_acquisition_releases_completed_worker_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_started = threading.Event()
    allow_worker = threading.Event()
    worker_finished = threading.Event()
    client = RecordingClient()
    monkeypatch.setattr(
        runtime_module,
        "build_openai_memory_client",
        lambda *, settings: client,
    )
    original_acquire = runtime_module._acquire_shared_runtime_sync

    def gated_acquire(settings: OpenAIRuntimeSettings) -> Any:
        worker_started.set()
        allow_worker.wait(timeout=1)
        try:
            return original_acquire(settings)
        finally:
            worker_finished.set()

    monkeypatch.setattr(runtime_module, "_acquire_shared_runtime_sync", gated_acquire)
    acquire_task = asyncio.create_task(acquire_shared_runtime(settings()))
    assert await asyncio.to_thread(worker_started.wait, 0.5)

    acquire_task.cancel()
    await asyncio.sleep(0)
    cancellation_propagated_early = acquire_task.done()
    allow_worker.set()
    with pytest.raises(asyncio.CancelledError):
        await acquire_task
    assert await asyncio.to_thread(worker_finished.wait, 0.5)

    assert cancellation_propagated_early is False
    assert runtime_module._shared_runtimes == {}
    assert client.store.close_calls == 1


@pytest.mark.asyncio
async def test_cancelled_final_release_finishes_close_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()
    close_started = asyncio.Event()
    allow_close = threading.Event()
    close_finished = threading.Event()

    class BlockingStore:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            loop.call_soon_threadsafe(close_started.set)
            allow_close.wait(timeout=1)
            close_finished.set()

    client = RecordingClient()
    client.store = BlockingStore()
    monkeypatch.setattr(
        runtime_module,
        "build_openai_memory_client",
        lambda *, settings: client,
    )
    lease = await acquire_shared_runtime(settings())
    release_task = asyncio.create_task(lease.release())
    await asyncio.wait_for(close_started.wait(), timeout=0.5)

    release_task.cancel()
    await asyncio.sleep(0)
    cancellation_propagated_early = release_task.done()
    allow_close.set()
    with pytest.raises(asyncio.CancelledError):
        await release_task
    assert await asyncio.to_thread(close_finished.wait, 0.5)

    assert cancellation_propagated_early is False
    assert runtime_module._shared_runtimes == {}
    assert client.store.close_calls == 1


@pytest.mark.asyncio
async def test_concurrent_release_callers_wait_for_final_store_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()
    close_started = asyncio.Event()
    allow_close = threading.Event()

    class BlockingStore:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            loop.call_soon_threadsafe(close_started.set)
            allow_close.wait(timeout=1)

    client = RecordingClient()
    client.store = BlockingStore()
    monkeypatch.setattr(
        runtime_module,
        "build_openai_memory_client",
        lambda *, settings: client,
    )
    lease = await acquire_shared_runtime(settings())
    first = asyncio.create_task(lease.release())
    await asyncio.wait_for(close_started.wait(), timeout=0.5)
    others = [asyncio.create_task(lease.release()) for _ in range(2)]

    completed, _ = await asyncio.wait(
        others,
        timeout=0.05,
        return_when=asyncio.FIRST_COMPLETED,
    )
    allow_close.set()
    await asyncio.wait_for(asyncio.gather(first, *others), timeout=0.5)

    assert completed == set()
    assert client.store.close_calls == 1


@pytest.mark.asyncio
async def test_release_failure_is_shared_and_replayed_secret_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()
    close_started = asyncio.Event()
    allow_close = threading.Event()
    secret = "postgresql://private:password@secret.example/mem01"

    class FailingStore:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            loop.call_soon_threadsafe(close_started.set)
            allow_close.wait(timeout=1)
            raise RuntimeError(f"could not close {secret}")

    client = RecordingClient()
    client.store = FailingStore()
    monkeypatch.setattr(
        runtime_module,
        "build_openai_memory_client",
        lambda *, settings: client,
    )
    lease = await acquire_shared_runtime(settings())
    first = asyncio.create_task(lease.release())
    await asyncio.wait_for(close_started.wait(), timeout=0.5)
    others = [asyncio.create_task(lease.release()) for _ in range(2)]

    completed, _ = await asyncio.wait(
        others,
        timeout=0.05,
        return_when=asyncio.FIRST_COMPLETED,
    )
    allow_close.set()
    results = await asyncio.wait_for(
        asyncio.gather(first, *others, return_exceptions=True),
        timeout=0.5,
    )
    with pytest.raises(RuntimeError) as later_error:
        await lease.release()

    errors = [*results, later_error.value]
    assert completed == set()
    assert all(isinstance(error, RuntimeError) for error in errors)
    assert len({str(error) for error in errors}) == 1
    assert all(secret not in str(error) for error in errors)
    assert all("private" not in str(error) for error in errors)
    assert client.store.close_calls == 1


@pytest.mark.asyncio
async def test_registry_reprs_and_construction_errors_do_not_expose_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_key = "sk-never-render-this"
    database_url = "postgresql://private:password@secret.example/mem01"
    configured = settings(api_key=api_key, database_url=database_url)
    client = RecordingClient()
    monkeypatch.setattr(
        runtime_module,
        "build_openai_memory_client",
        lambda *, settings: client,
    )

    lease = await acquire_shared_runtime(configured)
    rendered = repr(runtime_module._shared_runtimes)
    rendered += repr(lease)
    rendered += repr(lease.runtime)

    assert api_key not in rendered
    assert database_url not in rendered
    assert "private" not in rendered
    assert "password" not in rendered
    assert "secret.example" not in rendered
    assert runtime_module._fingerprint(api_key) in rendered
    assert runtime_module._fingerprint(database_url) in rendered
    await lease.release()

    def fail(*, settings: OpenAIRuntimeSettings) -> RecordingClient:
        raise RuntimeError(
            f"could not connect with {settings.database_url} {settings.api_key}"
        )

    monkeypatch.setattr(runtime_module, "build_openai_memory_client", fail)
    with pytest.raises(RuntimeError) as exc_info:
        await acquire_shared_runtime(configured)

    assert api_key not in str(exc_info.value)
    assert database_url not in str(exc_info.value)
    assert "private" not in str(exc_info.value)
    assert "password" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_runtime_close_closes_store_once_and_operations_then_fail() -> None:
    client = RecordingClient()
    runtime = EmbeddedMem01Runtime(client=client)

    await asyncio.gather(runtime.aclose(), runtime.aclose())
    await runtime.aclose()

    assert client.store.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        await runtime.history(user_id="alex")


@pytest.mark.asyncio
async def test_runtime_closed_state_is_prompt_during_blocking_store_close() -> None:
    loop = asyncio.get_running_loop()
    close_entered = asyncio.Event()
    release_close = threading.Event()

    class BlockingStore:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            loop.call_soon_threadsafe(close_entered.set)
            release_close.wait(timeout=1)

    client = RecordingClient()
    client.store = BlockingStore()
    runtime = EmbeddedMem01Runtime(client=client)
    close_task = asyncio.create_task(runtime.aclose())

    await asyncio.wait_for(close_entered.wait(), timeout=0.5)
    watchdog = threading.Timer(0.5, release_close.set)
    watchdog.start()
    started = loop.time()
    try:
        assert runtime.is_closed is True
        assert loop.time() - started < 0.1
        with pytest.raises(RuntimeError, match="closed"):
            await runtime.history(user_id="alex")
    finally:
        release_close.set()
        watchdog.cancel()
        await close_task

    assert client.store.close_calls == 1


@pytest.mark.asyncio
async def test_cancelled_runtime_close_drains_store_before_propagating() -> None:
    loop = asyncio.get_running_loop()
    close_started = asyncio.Event()
    allow_close = threading.Event()
    close_finished = threading.Event()

    class BlockingStore:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            loop.call_soon_threadsafe(close_started.set)
            allow_close.wait(timeout=1)
            close_finished.set()

    client = RecordingClient()
    client.store = BlockingStore()
    runtime = EmbeddedMem01Runtime(client=client)
    close_task = asyncio.create_task(runtime.aclose())
    await asyncio.wait_for(close_started.wait(), timeout=0.5)

    close_task.cancel()
    await asyncio.sleep(0)
    cancellation_propagated_early = close_task.done()
    allow_close.set()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert cancellation_propagated_early is False
    assert close_finished.is_set()
    assert client.store.close_calls == 1


@pytest.mark.asyncio
async def test_cancelled_recall_remains_in_flight_until_worker_finishes() -> None:
    loop = asyncio.get_running_loop()
    recall_entered = asyncio.Event()
    release_recall = threading.Event()
    client = RecordingClient()

    def blocking_recall(*args: Any, **kwargs: Any) -> object:
        loop.call_soon_threadsafe(recall_entered.set)
        release_recall.wait(timeout=1)
        return client._record("recall", *args, **kwargs)

    client.recall = blocking_recall  # type: ignore[method-assign]
    runtime = EmbeddedMem01Runtime(client=client)
    recall_task = asyncio.create_task(runtime.recall("query", user_id="alex"))
    await asyncio.wait_for(recall_entered.wait(), timeout=0.5)

    recall_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await recall_task
    close_task = asyncio.create_task(runtime.aclose())
    await asyncio.to_thread(wait_until_closed, runtime)

    assert close_task.done() is False
    assert client.store.close_calls == 0
    with pytest.raises(RuntimeError, match="closing|closed"):
        await asyncio.wait_for(runtime.history(user_id="alex"), timeout=0.2)

    release_recall.set()
    await asyncio.wait_for(close_task, timeout=0.5)
    assert client.store.close_calls == 1


@pytest.mark.asyncio
async def test_all_concurrent_closers_wait_for_one_store_close() -> None:
    loop = asyncio.get_running_loop()
    close_entered = asyncio.Event()
    release_close = threading.Event()

    class BlockingStore:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            loop.call_soon_threadsafe(close_entered.set)
            release_close.wait(timeout=1)

    client = RecordingClient()
    client.store = BlockingStore()
    runtime = EmbeddedMem01Runtime(client=client)
    first = asyncio.create_task(runtime.aclose())
    await asyncio.wait_for(close_entered.wait(), timeout=0.5)
    others = [asyncio.create_task(runtime.aclose()) for _ in range(2)]

    completed, _ = await asyncio.wait(
        others,
        timeout=0.05,
        return_when=asyncio.FIRST_COMPLETED,
    )
    assert completed == set()
    assert first.done() is False

    release_close.set()
    await asyncio.wait_for(asyncio.gather(first, *others), timeout=0.5)
    assert client.store.close_calls == 1


@pytest.mark.asyncio
async def test_store_close_failure_is_shared_and_secret_safe() -> None:
    secret = "postgresql://private:password@secret.example/mem01"

    class FailingStore:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError(f"could not close {secret}")

    client = RecordingClient()
    client.store = FailingStore()
    runtime = EmbeddedMem01Runtime(client=client)

    results = await asyncio.gather(
        runtime.aclose(),
        runtime.aclose(),
        runtime.aclose(),
        return_exceptions=True,
    )
    with pytest.raises(RuntimeError) as later_error:
        await runtime.aclose()

    errors = [*results, later_error.value]
    assert all(isinstance(error, RuntimeError) for error in errors)
    assert len({str(error) for error in errors}) == 1
    assert all(secret not in str(error) for error in errors)
    assert all("private" not in str(error) for error in errors)
    assert client.store.close_calls == 1


@pytest.mark.asyncio
async def test_injected_runtime_remains_caller_owned_by_session() -> None:
    from mem01session.session import Mem01Session

    client = RecordingClient()
    runtime = EmbeddedMem01Runtime(client=client)
    session = Mem01Session(
        "conversation-1",
        "alex",
        inner=object(),
        runtime=runtime,
    )

    await session.close()

    assert client.store.close_calls == 0
    await runtime.aclose()
    assert client.store.close_calls == 1


@pytest.mark.asyncio
async def test_close_shared_runtimes_closes_each_runtime_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[RecordingClient] = []

    def build(*, settings: OpenAIRuntimeSettings) -> RecordingClient:
        client = RecordingClient()
        clients.append(client)
        return client

    monkeypatch.setattr(runtime_module, "build_openai_memory_client", build)
    leases = [
        await acquire_shared_runtime(settings()),
        await acquire_shared_runtime(settings(llm_model="gpt-other")),
    ]

    await close_shared_runtimes()
    await close_shared_runtimes()
    await asyncio.gather(*(lease.release() for lease in leases))

    assert [client.store.close_calls for client in clients] == [1, 1]


@pytest.mark.asyncio
async def test_shutdown_attempts_every_runtime_and_reports_safe_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "postgresql://private:password@secret.example/mem01"

    class FailingStore:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError(f"failed with {secret}")

    first = RecordingClient()
    first.store = FailingStore()
    second = RecordingClient()
    clients = iter((first, second))
    monkeypatch.setattr(
        runtime_module,
        "build_openai_memory_client",
        lambda *, settings: next(clients),
    )
    await acquire_shared_runtime(settings())
    await acquire_shared_runtime(settings(llm_model="gpt-other"))

    with pytest.raises(RuntimeError) as exc_info:
        await close_shared_runtimes()
    await close_shared_runtimes()

    assert first.store.close_calls == 1
    assert second.store.close_calls == 1
    assert runtime_module._shared_runtimes == {}
    assert secret not in str(exc_info.value)
    assert "private" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_concurrent_shutdown_and_acquisition_share_one_shutdown_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()
    close_started = asyncio.Event()
    allow_close = threading.Event()
    close_finished = threading.Event()

    class BlockingStore:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            loop.call_soon_threadsafe(close_started.set)
            allow_close.wait(timeout=1)
            close_finished.set()

    first = RecordingClient()
    first.store = BlockingStore()
    second = RecordingClient()
    build_calls = 0
    second_built_during_shutdown = False

    def build(*, settings: OpenAIRuntimeSettings) -> RecordingClient:
        nonlocal build_calls, second_built_during_shutdown
        build_calls += 1
        if build_calls == 1:
            return first
        second_built_during_shutdown = not close_finished.is_set()
        return second

    monkeypatch.setattr(runtime_module, "build_openai_memory_client", build)
    first_lease = await acquire_shared_runtime(settings())
    first_shutdown = asyncio.create_task(close_shared_runtimes())
    await asyncio.wait_for(close_started.wait(), timeout=0.5)
    second_shutdown = asyncio.create_task(close_shared_runtimes())
    acquire_task = asyncio.create_task(
        acquire_shared_runtime(settings(llm_model="gpt-other"))
    )

    completed, _ = await asyncio.wait(
        (second_shutdown, acquire_task),
        timeout=0.05,
        return_when=asyncio.FIRST_COMPLETED,
    )
    allow_close.set()
    await asyncio.wait_for(asyncio.gather(first_shutdown, second_shutdown), timeout=0.5)
    second_lease = await asyncio.wait_for(acquire_task, timeout=0.5)
    await first_lease.release()
    await second_lease.release()

    assert completed == set()
    assert second_built_during_shutdown is False
    assert first.store.close_calls == 1
    assert second.store.close_calls == 1
    assert runtime_module._shared_runtimes == {}


@pytest.mark.asyncio
async def test_queued_remember_is_fifo_per_user_and_independent_between_users() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    calls: list[tuple[str, str]] = []
    calls_lock = threading.Lock()

    class QueuedClient(RecordingClient):
        def remember(self, messages: list[dict[str, str]], **kwargs: Any) -> object:
            label = messages[0]["content"]
            with calls_lock:
                calls.append((kwargs["user_id"], label))
            if label == "alex-1":
                first_started.set()
                release_first.wait(timeout=1)
            return super().remember(messages, **kwargs)

    runtime = EmbeddedMem01Runtime(client=QueuedClient())
    await runtime.enqueue_remember(
        [{"role": "user", "content": "alex-1"}], user_id="alex"
    )
    assert await asyncio.to_thread(first_started.wait, 0.5)
    await runtime.enqueue_remember(
        [{"role": "user", "content": "alex-2"}], user_id="alex"
    )
    await runtime.enqueue_remember(
        [{"role": "user", "content": "blair-1"}], user_id="blair"
    )

    await asyncio.wait_for(runtime.history(user_id="blair"), timeout=0.5)
    with calls_lock:
        before_release = list(calls)
    release_first.set()
    await asyncio.wait_for(runtime.history(user_id="alex"), timeout=0.5)
    await runtime.aclose()

    assert ("blair", "blair-1") in before_release
    assert ("alex", "alex-2") not in before_release
    alex_calls = [label for user, label in calls if user == "alex"]
    assert alex_calls == ["alex-1", "alex-2"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation", ["recall", "history", "correct", "forget", "clear_user"]
)
async def test_user_operations_wait_for_that_users_queued_writes(
    operation: str,
) -> None:
    remember_started = threading.Event()
    release_remember = threading.Event()
    events: list[str] = []

    class BlockingClient(RecordingClient):
        def remember(self, *args: Any, **kwargs: Any) -> object:
            events.append("remember.start")
            remember_started.set()
            release_remember.wait(timeout=1)
            events.append("remember.done")
            return super().remember(*args, **kwargs)

        def _record(self, name: str, *args: Any, **kwargs: Any) -> object:
            events.append(name)
            return super()._record(name, *args, **kwargs)

    runtime = EmbeddedMem01Runtime(client=BlockingClient())
    await runtime.enqueue_remember(
        [{"role": "user", "content": "queued"}], user_id="alex"
    )
    assert await asyncio.to_thread(remember_started.wait, 0.5)
    if operation == "recall":
        barrier = asyncio.create_task(runtime.recall("where?", user_id="alex"))
    elif operation == "history":
        barrier = asyncio.create_task(runtime.history(user_id="alex"))
    elif operation == "correct":
        barrier = asyncio.create_task(
            runtime.correct("belief-1", "new value", user_id="alex")
        )
    elif operation == "forget":
        barrier = asyncio.create_task(runtime.forget("belief-1", user_id="alex"))
    else:
        barrier = asyncio.create_task(runtime.clear_user(user_id="alex"))

    await asyncio.sleep(0.02)
    returned_early = barrier.done()
    release_remember.set()
    await asyncio.wait_for(barrier, timeout=0.5)
    await runtime.aclose()

    assert returned_early is False
    assert events.index("remember.done") < events.index(operation)


@pytest.mark.asyncio
async def test_clear_user_waits_only_for_its_scoped_users_queue() -> None:
    alex_started = threading.Event()
    release_alex = threading.Event()

    class BlockingClient(RecordingClient):
        def remember(self, messages: list[dict[str, str]], **kwargs: Any) -> object:
            if kwargs["user_id"] == "alex":
                alex_started.set()
                release_alex.wait(timeout=1)
            return super().remember(messages, **kwargs)

    client = BlockingClient()
    client.results["clear_user"] = 1
    runtime = EmbeddedMem01Runtime(client=client)
    await runtime.enqueue_remember(
        [{"role": "user", "content": "queued"}], user_id="alex"
    )
    assert await asyncio.to_thread(alex_started.wait, 0.5)

    deleted = await asyncio.wait_for(runtime.clear_user(user_id="blair"), timeout=0.5)
    release_alex.set()
    await runtime.flush_pending(user_id="alex")
    await runtime.aclose()

    assert deleted == 1
    assert any(
        name == "clear_user" and kwargs == {"user_id": "blair"}
        for name, _args, kwargs, _thread in client.calls
    )


@pytest.mark.asyncio
async def test_cancelled_clear_user_waits_for_delete_before_propagating() -> None:
    clear_started = threading.Event()
    release_clear = threading.Event()
    clear_finished = threading.Event()

    class BlockingClient(RecordingClient):
        def clear_user(self, *args: Any, **kwargs: Any) -> object:
            clear_started.set()
            release_clear.wait(timeout=1)
            result = super().clear_user(*args, **kwargs)
            clear_finished.set()
            return result

    runtime = EmbeddedMem01Runtime(client=BlockingClient())
    clear_task = asyncio.create_task(runtime.clear_user(user_id="alex"))
    assert await asyncio.to_thread(clear_started.wait, 0.5)

    clear_task.cancel()
    await asyncio.sleep(0)
    cancellation_visible_before_delete = clear_task.done()
    close_task = asyncio.create_task(runtime.aclose())
    await asyncio.to_thread(wait_until_closed, runtime)
    close_finished_before_delete = close_task.done()

    release_clear.set()
    with pytest.raises(asyncio.CancelledError):
        await clear_task
    await asyncio.wait_for(close_task, timeout=0.5)

    assert cancellation_visible_before_delete is False
    assert close_finished_before_delete is False
    assert clear_finished.is_set()
    assert runtime.client.store.close_calls == 1


@pytest.mark.asyncio
async def test_queued_write_error_is_caught_and_surfaced_secret_safe_at_barrier() -> (
    None
):
    secret = "postgresql://private:password@secret.example/mem01"

    class FailingClient(RecordingClient):
        def remember(self, *args: Any, **kwargs: Any) -> object:
            raise RuntimeError(secret)

    runtime = EmbeddedMem01Runtime(client=FailingClient())
    await runtime.enqueue_remember(
        [{"role": "user", "content": "queued"}], user_id="alex"
    )

    with pytest.raises(RuntimeError, match="queued memory write failed") as error:
        await runtime.history(user_id="alex")
    with pytest.raises(RuntimeError, match="queued memory write failed"):
        await runtime.history(user_id="alex")
    await runtime.aclose()

    assert secret not in repr(error.value)
    assert "private" not in repr(error.value)


@pytest.mark.asyncio
async def test_concurrent_same_target_barriers_all_observe_queued_failure() -> None:
    remember_started = threading.Event()
    release_remember = threading.Event()
    both_barriers_waiting = threading.Event()
    waiters = 0
    waiters_lock = threading.Lock()

    class GatedFailingClient(RecordingClient):
        def remember(self, *args: Any, **kwargs: Any) -> object:
            remember_started.set()
            release_remember.wait(timeout=1)
            raise RuntimeError("queued failure")

    runtime = EmbeddedMem01Runtime(client=GatedFailingClient())
    original_wait = runtime._wait_for_user_writes

    def observed_wait(state: Any, target: int) -> None:
        nonlocal waiters
        with waiters_lock:
            waiters += 1
            if waiters == 2:
                both_barriers_waiting.set()
        original_wait(state, target)

    runtime._wait_for_user_writes = observed_wait
    await runtime.enqueue_remember(
        [{"role": "user", "content": "queued"}], user_id="alex"
    )
    assert await asyncio.to_thread(remember_started.wait, 0.5)
    history = asyncio.create_task(runtime.history(user_id="alex"))
    recall = asyncio.create_task(runtime.recall("query", user_id="alex"))
    assert await asyncio.to_thread(both_barriers_waiting.wait, 0.5)

    release_remember.set()
    results = await asyncio.gather(history, recall, return_exceptions=True)
    await runtime.aclose()

    assert all(
        isinstance(result, runtime_module.PendingMemoryWriteError) for result in results
    )


@pytest.mark.asyncio
async def test_failed_target_repeats_until_later_success_recovers_state() -> None:
    attempts = 0

    class FailThenSucceedClient(RecordingClient):
        def remember(self, *args: Any, **kwargs: Any) -> object:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("first write failed")
            return object()

    runtime = EmbeddedMem01Runtime(client=FailThenSucceedClient())
    await runtime.enqueue_remember(
        [{"role": "user", "content": "fails"}], user_id="alex"
    )

    with pytest.raises(runtime_module.PendingMemoryWriteError):
        await runtime.flush_pending(user_id="alex")
    with pytest.raises(runtime_module.PendingMemoryWriteError):
        await runtime.flush_pending(user_id="alex")

    await runtime.enqueue_remember(
        [{"role": "user", "content": "succeeds"}], user_id="alex"
    )
    await runtime.flush_pending(user_id="alex")

    assert runtime._user_writes == {}
    await runtime.aclose()


@pytest.mark.asyncio
async def test_unseen_failure_is_not_repaired_by_later_success() -> None:
    attempts = 0

    class FailThenSucceedClient(RecordingClient):
        def remember(self, *args: Any, **kwargs: Any) -> object:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("unseen first write failure")
            return object()

    runtime = EmbeddedMem01Runtime(client=FailThenSucceedClient())
    await runtime.enqueue_remember(
        [{"role": "user", "content": "lost turn"}], user_id="alex"
    )
    await runtime.enqueue_remember(
        [{"role": "user", "content": "later success"}], user_id="alex"
    )

    with pytest.raises(runtime_module.PendingMemoryWriteError):
        await runtime.history(user_id="alex")
    await runtime.recall("recovered?", user_id="alex")

    assert runtime._user_writes == {}
    await runtime.aclose()


@pytest.mark.asyncio
async def test_unseen_failure_cohort_all_fails_before_recorded_recovery() -> None:
    attempts = 0
    both_barriers_waiting = threading.Event()
    release_barriers = threading.Event()
    writes_finished = threading.Event()
    waiters = 0
    waiters_lock = threading.Lock()

    class FailThenSucceedClient(RecordingClient):
        def remember(self, *args: Any, **kwargs: Any) -> object:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("unseen failure")
            writes_finished.set()
            return object()

    runtime = EmbeddedMem01Runtime(client=FailThenSucceedClient())
    await runtime.enqueue_remember(
        [{"role": "user", "content": "fails"}], user_id="alex"
    )
    await runtime.enqueue_remember(
        [{"role": "user", "content": "recovery candidate"}], user_id="alex"
    )
    assert await asyncio.to_thread(writes_finished.wait, 0.5)

    def wait_until_idle() -> None:
        with runtime._condition:
            while runtime._in_flight:
                runtime._condition.wait()

    await asyncio.to_thread(wait_until_idle)
    original_wait = runtime._wait_for_user_writes

    def cohort_wait(state: Any, target: int) -> None:
        nonlocal waiters
        with waiters_lock:
            waiters += 1
            if waiters == 2:
                both_barriers_waiting.set()
        release_barriers.wait(timeout=1)
        original_wait(state, target)

    runtime._wait_for_user_writes = cohort_wait
    history = asyncio.create_task(runtime.history(user_id="alex"))
    recall = asyncio.create_task(runtime.recall("query", user_id="alex"))
    assert await asyncio.to_thread(both_barriers_waiting.wait, 0.5)
    release_barriers.set()

    cohort = await asyncio.gather(history, recall, return_exceptions=True)
    assert all(
        isinstance(result, runtime_module.PendingMemoryWriteError) for result in cohort
    )

    await runtime.flush_pending(user_id="alex")
    assert runtime._user_writes == {}
    await runtime.aclose()


@pytest.mark.asyncio
async def test_user_write_state_never_retains_raw_exception_or_secret() -> None:
    secret = "postgresql://private:password@secret.example/mem01"

    class SecretFailingClient(RecordingClient):
        def remember(self, *args: Any, **kwargs: Any) -> object:
            raise RuntimeError(secret)

    runtime = EmbeddedMem01Runtime(client=SecretFailingClient())
    await runtime.enqueue_remember(
        [{"role": "user", "content": "fails"}], user_id="alex"
    )
    with pytest.raises(runtime_module.PendingMemoryWriteError):
        await runtime.flush_pending(user_id="alex")
    state = runtime._user_writes["alex"]
    values = [getattr(state, field.name) for field in fields(state)]

    assert state.last_failed_sequence == 1
    assert not any(isinstance(value, BaseException) for value in values)
    assert secret not in repr(state)
    assert "private" not in repr(state)

    await runtime.aclose()


@pytest.mark.asyncio
async def test_failure_metadata_stays_constant_after_many_failures() -> None:
    class AlwaysFailingClient(RecordingClient):
        def remember(self, *args: Any, **kwargs: Any) -> object:
            raise RuntimeError("automatic write failed")

    runtime = EmbeddedMem01Runtime(client=AlwaysFailingClient())
    for sequence in range(250):
        await runtime.enqueue_remember(
            [{"role": "user", "content": f"failure-{sequence}"}],
            user_id="alex",
        )

    def wait_until_idle() -> None:
        with runtime._condition:
            while runtime._in_flight:
                runtime._condition.wait()

    await asyncio.to_thread(wait_until_idle)
    state = runtime._user_writes["alex"]

    assert state.last_failed_sequence == 250
    assert state.active_failure_sequence == 1
    assert not hasattr(state, "failed_sequences")
    assert len(fields(state)) == 8

    with pytest.raises(runtime_module.PendingMemoryWriteError):
        await runtime.flush_pending(user_id="alex")
    await runtime.aclose()


@pytest.mark.asyncio
async def test_executor_submit_failure_is_accounted_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "submit-failure-secret"
    runtime = EmbeddedMem01Runtime(client=RecordingClient())
    original_submit = runtime._write_executor.submit
    submit_calls = 0

    def fail_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal submit_calls
        submit_calls += 1
        if submit_calls == 1:
            raise RuntimeError(secret)
        return original_submit(*args, **kwargs)

    monkeypatch.setattr(runtime._write_executor, "submit", fail_once)

    await runtime.enqueue_remember(
        [{"role": "user", "content": "unscheduled"}], user_id="alex"
    )
    with pytest.raises(runtime_module.PendingMemoryWriteError):
        await runtime.flush_pending(user_id="alex")
    assert runtime._in_flight == 0
    assert secret not in repr(runtime._user_writes["alex"])

    await runtime.enqueue_remember(
        [{"role": "user", "content": "recovery"}], user_id="alex"
    )
    await runtime.flush_pending(user_id="alex")

    assert runtime._user_writes == {}
    await runtime.aclose()


@pytest.mark.asyncio
async def test_runtime_close_drains_queued_writes_before_store_close() -> None:
    remember_started = threading.Event()
    release_remember = threading.Event()
    events: list[str] = []

    class Store(RecordingStore):
        def close(self) -> None:
            events.append("store.close")
            super().close()

    class BlockingClient(RecordingClient):
        def __init__(self) -> None:
            super().__init__()
            self.store = Store()

        def remember(self, *args: Any, **kwargs: Any) -> object:
            remember_started.set()
            release_remember.wait(timeout=1)
            events.append("remember.done")
            return super().remember(*args, **kwargs)

    runtime = EmbeddedMem01Runtime(client=BlockingClient())
    await runtime.enqueue_remember(
        [{"role": "user", "content": "queued"}], user_id="alex"
    )
    assert await asyncio.to_thread(remember_started.wait, 0.5)
    close_task = asyncio.create_task(runtime.aclose())
    await asyncio.sleep(0.02)
    closed_early = close_task.done()
    release_remember.set()
    await asyncio.wait_for(close_task, timeout=0.5)

    assert closed_early is False
    assert events == ["remember.done", "store.close"]


@pytest.mark.asyncio
async def test_direct_remember_uses_same_user_fifo_and_awaits_its_result() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    calls: list[str] = []

    class OrderedClient(RecordingClient):
        def remember(self, messages: list[dict[str, str]], **kwargs: Any) -> object:
            label = messages[0]["content"]
            calls.append(label)
            if label == "first":
                first_started.set()
                release_first.wait(timeout=1)
            return label

    runtime = EmbeddedMem01Runtime(client=OrderedClient())
    await runtime.enqueue_remember(
        [{"role": "user", "content": "first"}], user_id="alex"
    )
    assert await asyncio.to_thread(first_started.wait, 0.5)
    direct = asyncio.create_task(
        runtime.remember([{"role": "user", "content": "direct"}], user_id="alex")
    )
    await asyncio.sleep(0)
    await runtime.enqueue_remember(
        [{"role": "user", "content": "later"}], user_id="alex"
    )

    release_first.set()
    direct_result = await asyncio.wait_for(direct, timeout=0.5)
    await runtime.flush_pending(user_id="alex")
    await runtime.aclose()

    assert direct_result == "direct"
    assert calls == ["first", "direct", "later"]


@pytest.mark.asyncio
async def test_direct_remember_preserves_raw_exception_and_lane_continues() -> None:
    direct_error = RuntimeError("direct engine failure")
    calls: list[str] = []

    class FailingDirectClient(RecordingClient):
        def remember(self, messages: list[dict[str, str]], **kwargs: Any) -> object:
            label = messages[0]["content"]
            calls.append(label)
            if label == "direct":
                raise direct_error
            return label

    runtime = EmbeddedMem01Runtime(client=FailingDirectClient())

    with pytest.raises(RuntimeError) as error:
        await runtime.remember([{"role": "user", "content": "direct"}], user_id="alex")
    await runtime.enqueue_remember(
        [{"role": "user", "content": "later"}], user_id="alex"
    )
    await runtime.flush_pending(user_id="alex")
    await runtime.aclose()

    assert error.value is direct_error
    assert calls == ["direct", "later"]


def test_runtime_queue_can_be_reused_across_sequential_event_loops() -> None:
    client = RecordingClient()
    runtime = EmbeddedMem01Runtime(client=client)

    async def write_and_flush(label: str) -> None:
        await runtime.enqueue_remember(
            [{"role": "user", "content": label}], user_id="alex"
        )
        await runtime.flush_pending(user_id="alex")

    asyncio.run(write_and_flush("first-loop"))
    asyncio.run(write_and_flush("second-loop"))
    asyncio.run(runtime.aclose())

    assert [
        call[1][0][0]["content"] for call in client.calls if call[0] == "remember"
    ] == ["first-loop", "second-loop"]


@pytest.mark.asyncio
async def test_enqueue_copies_payload_before_returning() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    observed: list[str] = []

    class CopyingClient(RecordingClient):
        def remember(self, messages: list[dict[str, str]], **kwargs: Any) -> object:
            label = messages[0]["content"]
            if label == "blocker":
                first_started.set()
                release_first.wait(timeout=1)
            observed.append(label)
            return object()

    runtime = EmbeddedMem01Runtime(client=CopyingClient())
    await runtime.enqueue_remember(
        [{"role": "user", "content": "blocker"}], user_id="alex"
    )
    assert await asyncio.to_thread(first_started.wait, 0.5)
    payload = [{"role": "user", "content": "original"}]
    await runtime.enqueue_remember(payload, user_id="alex")
    payload[0]["content"] = "mutated"

    release_first.set()
    await runtime.flush_pending(user_id="alex")
    await runtime.aclose()

    assert observed == ["blocker", "original"]
