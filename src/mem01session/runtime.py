"""Async ownership boundary for the embedded synchronous mem01 engine."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, cast

from mem01 import ApplyResult, MemoryClient
from mem01.runtime import OpenAIRuntimeSettings, build_openai_memory_client


class PendingMemoryWriteError(RuntimeError):
    """Secret-safe signal that an accepted automatic memory write failed."""


@dataclass(slots=True)
class _QueuedRemember:
    sequence: int
    messages: list[dict[str, str]]
    kwargs: dict[str, Any]
    completion: Future[Any] | None = None


@dataclass(slots=True)
class _UserWriteState:
    pending: deque[_QueuedRemember] = field(default_factory=deque)
    next_sequence: int = 0
    completed_sequence: int = 0
    worker_active: bool = False
    last_failed_sequence: int | None = None
    active_failure_sequence: int | None = None
    active_failure_observed: bool = False
    recovery_sequence: int | None = None


def _fingerprint(value: str) -> str:
    """Return a stable one-way identity suitable for secret-bearing settings."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class _RuntimeKey:
    api_key_fingerprint: str
    database_url_fingerprint: str
    llm_model: str
    embedding_model: str
    embedding_dimensions: int
    base_url: str

    @classmethod
    def from_settings(cls, settings: OpenAIRuntimeSettings) -> _RuntimeKey:
        return cls(
            api_key_fingerprint=_fingerprint(settings.api_key),
            database_url_fingerprint=_fingerprint(settings.database_url),
            llm_model=settings.llm_model,
            embedding_model=settings.embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
            base_url=settings.base_url,
        )


class EmbeddedMem01Runtime:
    """Run a synchronous mem01 client without blocking an asyncio event loop."""

    def __init__(self, *, client: MemoryClient) -> None:
        self._client = client
        self._condition = threading.Condition()
        self._state = "open"
        self._in_flight = 0
        self._close_error: RuntimeError | None = None
        self._write_executor = ThreadPoolExecutor(
            thread_name_prefix="mem01session-write"
        )
        self._user_writes: dict[str, _UserWriteState] = {}

    @property
    def client(self) -> MemoryClient:
        """The wrapped client, exposed for diagnostics and explicit integration."""
        return self._client

    @property
    def is_closed(self) -> bool:
        with self._condition:
            return self._state != "open"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(closed={self.is_closed})"

    def _run_operation(
        self,
        operation: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        with self._condition:
            if self._state != "open":
                raise RuntimeError("Embedded mem01 runtime is closing or closed")
            self._in_flight += 1
        try:
            return operation(*args, **kwargs)
        finally:
            with self._condition:
                self._in_flight -= 1
                self._condition.notify_all()

    async def remember(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        """Extract and store beliefs through the embedded engine."""
        user_id = kwargs.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            return await asyncio.to_thread(
                self._run_operation,
                self._client.remember,
                messages,
                **kwargs,
            )
        completion: Future[Any] = Future()
        self._accept_remember(
            messages,
            kwargs,
            user_id=user_id,
            completion=completion,
        )
        wrapped = asyncio.wrap_future(completion)
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            try:
                await wrapped
            except Exception:
                pass
            raise

    def _schedule_user_head_locked(
        self,
        user_id: str,
        state: _UserWriteState,
    ) -> None:
        while state.pending:
            job = state.pending.popleft()
            state.worker_active = True
            try:
                self._write_executor.submit(
                    self._run_queued_remember,
                    user_id,
                    state,
                    job,
                )
            except Exception as error:
                state.completed_sequence = job.sequence
                if job.completion is not None:
                    job.completion.set_exception(error)
                else:
                    state.last_failed_sequence = job.sequence
                    if state.active_failure_sequence is None:
                        state.active_failure_sequence = job.sequence
                    state.active_failure_observed = False
                    state.recovery_sequence = None
                self._in_flight -= 1
                state.worker_active = False
                if (
                    job.completion is not None
                    and state.active_failure_sequence is None
                    and not state.pending
                ):
                    self._user_writes.pop(user_id, None)
                self._condition.notify_all()
                continue
            return
        state.worker_active = False

    def _run_queued_remember(
        self,
        user_id: str,
        state: _UserWriteState,
        job: _QueuedRemember,
    ) -> None:
        operation_error: Exception | None = None
        job_failed = False
        result: Any = None
        try:
            result = self._client.remember(job.messages, **job.kwargs)
            apply_result = getattr(result, "apply", None)
            if isinstance(apply_result, ApplyResult) and not apply_result.ok:
                job_failed = True
        except Exception as error:
            operation_error = error
            job_failed = True
        finally:
            with self._condition:
                state.completed_sequence = job.sequence
                if job_failed and job.completion is None:
                    state.last_failed_sequence = job.sequence
                    if state.active_failure_sequence is None:
                        state.active_failure_sequence = job.sequence
                    state.active_failure_observed = False
                    state.recovery_sequence = None
                elif not job_failed and state.active_failure_sequence is not None:
                    if state.active_failure_observed:
                        state.active_failure_sequence = None
                        state.recovery_sequence = None
                    elif state.recovery_sequence is None:
                        state.recovery_sequence = job.sequence
                if job.completion is not None:
                    if operation_error is not None:
                        job.completion.set_exception(operation_error)
                    else:
                        job.completion.set_result(result)
                self._in_flight -= 1
                if state.pending:
                    self._schedule_user_head_locked(user_id, state)
                else:
                    state.worker_active = False
                    if not job_failed and (state.active_failure_sequence is None):
                        self._user_writes.pop(user_id, None)
                    elif job.completion is not None and (
                        state.active_failure_sequence is None
                    ):
                        self._user_writes.pop(user_id, None)
                self._condition.notify_all()

    def _accept_remember(
        self,
        messages: list[dict[str, str]],
        kwargs: dict[str, Any],
        *,
        user_id: str,
        completion: Future[Any] | None = None,
    ) -> None:
        payload = [dict(message) for message in messages]
        copied_kwargs = dict(kwargs)
        with self._condition:
            if self._state != "open":
                raise RuntimeError("Embedded mem01 runtime is closing or closed")
            state = self._user_writes.get(user_id)
            if state is None:
                state = _UserWriteState()
                self._user_writes[user_id] = state
            state.next_sequence += 1
            state.pending.append(
                _QueuedRemember(
                    sequence=state.next_sequence,
                    messages=payload,
                    kwargs=copied_kwargs,
                    completion=completion,
                )
            )
            self._in_flight += 1
            if not state.worker_active:
                self._schedule_user_head_locked(user_id, state)

    async def enqueue_remember(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> None:
        """Accept an automatic write and return before engine processing finishes."""
        user_id = kwargs.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("user_id is required for queued memory writes")
        self._accept_remember(messages, kwargs, user_id=user_id)

    def _snapshot_user_writes(
        self, user_id: str
    ) -> tuple[_UserWriteState | None, int, bool, int]:
        with self._condition:
            state = self._user_writes.get(user_id)
            if state is None:
                return None, 0, False, 0
            active = state.active_failure_sequence
            recovery = state.recovery_sequence
            if (
                active is not None
                and state.active_failure_observed
                and recovery is not None
                and recovery > active
            ):
                state.active_failure_sequence = None
                state.recovery_sequence = None
                active = None
                if not state.worker_active and not state.pending:
                    self._user_writes.pop(user_id, None)
                    return None, 0, False, state.completed_sequence
            target = state.next_sequence
            must_fail = active is not None and (
                not state.active_failure_observed or target <= active
            )
            return state, target, must_fail, state.completed_sequence

    def _wait_for_user_writes(self, state: _UserWriteState, target: int) -> None:
        with self._condition:
            while state.completed_sequence < target:
                self._condition.wait()

    def _barrier_failed(
        self,
        state: _UserWriteState,
        target: int,
        *,
        must_fail: bool,
        completed_before: int,
    ) -> bool:
        with self._condition:
            if must_fail:
                return True
            last_failed = state.last_failed_sequence
            if last_failed is not None and completed_before < last_failed <= target:
                return True
            active = state.active_failure_sequence
            return active is not None and active <= target

    def _mark_failure_observed(
        self,
        state: _UserWriteState,
        target: int,
    ) -> None:
        with self._condition:
            active = state.active_failure_sequence
            last_failed = state.last_failed_sequence
            if (
                active is not None
                and active <= target
                and (last_failed is None or last_failed <= target)
            ):
                state.active_failure_observed = True

    async def flush_pending(self, *, user_id: str) -> None:
        """Wait for writes accepted for one user before this barrier."""
        state, target, must_fail, completed_before = self._snapshot_user_writes(user_id)
        if state is None or target == 0:
            return
        waiter = asyncio.create_task(
            asyncio.to_thread(self._wait_for_user_writes, state, target)
        )
        try:
            await asyncio.shield(waiter)
        except asyncio.CancelledError:
            try:
                await waiter
            except Exception:
                pass
            raise
        if self._barrier_failed(
            state,
            target,
            must_fail=must_fail,
            completed_before=completed_before,
        ):
            self._mark_failure_observed(state, target)
            raise PendingMemoryWriteError("queued memory write failed") from None

    async def recall(self, query: str, **kwargs: Any) -> Any:
        """Retrieve a budgeted belief block through the embedded engine."""
        user_id = kwargs.get("user_id")
        if isinstance(user_id, str):
            await self.flush_pending(user_id=user_id)
        return await asyncio.to_thread(
            self._run_operation,
            self._client.recall,
            query,
            **kwargs,
        )

    async def history(self, **kwargs: Any) -> Any:
        """Read belief history through the embedded engine."""
        user_id = kwargs.get("user_id")
        if isinstance(user_id, str):
            await self.flush_pending(user_id=user_id)
        return await asyncio.to_thread(
            self._run_operation,
            self._client.history,
            **kwargs,
        )

    async def correct(
        self,
        memory_id: str,
        new_value: str,
        **kwargs: Any,
    ) -> ApplyResult:
        """Correct a belief through the embedded engine."""
        user_id = kwargs.get("user_id")
        if isinstance(user_id, str):
            await self.flush_pending(user_id=user_id)
        result = await asyncio.to_thread(
            self._run_operation,
            self._client.correct,
            memory_id,
            new_value,
            **kwargs,
        )
        return cast(ApplyResult, result)

    async def forget(self, memory_id: str, **kwargs: Any) -> ApplyResult:
        """Invalidate a belief through the embedded engine."""
        user_id = kwargs.get("user_id")
        if isinstance(user_id, str):
            await self.flush_pending(user_id=user_id)
        result = await asyncio.to_thread(
            self._run_operation,
            self._client.forget,
            memory_id,
            **kwargs,
        )
        return cast(ApplyResult, result)

    async def clear_user(self, *, user_id: str) -> int:
        """Delete all durable memory for one user after pending writes finish."""
        if not isinstance(user_id, str):
            raise TypeError("user_id must be a non-empty string")
        if not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        await self.flush_pending(user_id=user_id)
        worker = asyncio.create_task(
            asyncio.to_thread(
                self._run_operation,
                self._client.clear_user,
                user_id=user_id,
            )
        )
        try:
            result = await asyncio.shield(worker)
        except asyncio.CancelledError:
            try:
                await worker
            except Exception:
                pass
            raise
        return int(result)

    def _close_sync(self) -> None:
        with self._condition:
            if self._state == "closed":
                if self._close_error is not None:
                    raise self._close_error
                return
            if self._state == "closing":
                while self._state == "closing":
                    self._condition.wait()
                if self._close_error is not None:
                    raise self._close_error
                return
            self._state = "closing"
            self._condition.notify_all()
            while self._in_flight:
                self._condition.wait()

        self._write_executor.shutdown(wait=True)

        store = getattr(self._client, "store", None)
        close = getattr(store, "close", None)
        close_error: RuntimeError | None = None
        try:
            if callable(close):
                close()
        except Exception:
            close_error = RuntimeError("Embedded mem01 runtime close failed")

        with self._condition:
            self._close_error = close_error
            self._state = "closed"
            self._condition.notify_all()

        if close_error is not None:
            raise close_error from None

    async def aclose(self) -> None:
        """Close the owned mem01 store once, without blocking the event loop."""
        worker = asyncio.create_task(asyncio.to_thread(self._close_sync))
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            try:
                await worker
            except Exception:
                pass
            raise


@dataclass(slots=True)
class _SharedRuntime:
    runtime: EmbeddedMem01Runtime
    references: int


_registry_lock = threading.RLock()
_registry_condition = threading.Condition(_registry_lock)
_shared_runtimes: dict[_RuntimeKey, _SharedRuntime] = {}
_shutdown_in_progress = False
_last_shutdown_error: RuntimeError | None = None


class SharedRuntimeLease:
    """One reference to a process-shared embedded runtime."""

    def __init__(self, *, key: _RuntimeKey, runtime: EmbeddedMem01Runtime) -> None:
        self._key = key
        self.runtime = runtime
        self._release_condition = threading.Condition()
        self._release_state = "unreleased"
        self._release_error: RuntimeError | None = None

    def __repr__(self) -> str:
        with self._release_condition:
            state = self._release_state
        return (
            f"{type(self).__name__}(key={self._key!r}, "
            f"state={state!r}, runtime={self.runtime!r})"
        )

    def _release_sync(self) -> None:
        with self._release_condition:
            if self._release_state == "released":
                if self._release_error is not None:
                    raise self._release_error
                return
            if self._release_state == "releasing":
                while self._release_state == "releasing":
                    self._release_condition.wait()
                if self._release_error is not None:
                    raise self._release_error
                return
            self._release_state = "releasing"

        release_error: RuntimeError | None = None
        try:
            runtime_to_close: EmbeddedMem01Runtime | None = None
            with _registry_lock:
                shared = _shared_runtimes.get(self._key)
                if shared is not None and shared.runtime is self.runtime:
                    shared.references -= 1
                    if shared.references == 0:
                        del _shared_runtimes[self._key]
                        runtime_to_close = shared.runtime

            if runtime_to_close is not None:
                runtime_to_close._close_sync()
        except Exception:
            release_error = RuntimeError("Shared mem01 runtime release failed")

        with self._release_condition:
            self._release_error = release_error
            self._release_state = "released"
            self._release_condition.notify_all()

        if release_error is not None:
            raise release_error from None

    async def release(self) -> None:
        """Release this reference and close the runtime after the final lease."""
        worker = asyncio.create_task(asyncio.to_thread(self._release_sync))
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            try:
                await worker
            except Exception:
                pass
            raise


def _acquire_shared_runtime_sync(
    settings: OpenAIRuntimeSettings,
) -> SharedRuntimeLease:
    key = _RuntimeKey.from_settings(settings)
    with _registry_condition:
        while _shutdown_in_progress:
            _registry_condition.wait()
        shared = _shared_runtimes.get(key)
        if shared is None:
            try:
                client = build_openai_memory_client(settings=settings)
            except Exception:
                raise RuntimeError(
                    f"Failed to construct embedded mem01 runtime for {key!r}"
                ) from None
            shared = _SharedRuntime(
                runtime=EmbeddedMem01Runtime(client=client),
                references=0,
            )
            _shared_runtimes[key] = shared
        shared.references += 1
        return SharedRuntimeLease(key=key, runtime=shared.runtime)


async def acquire_shared_runtime(
    settings: OpenAIRuntimeSettings,
) -> SharedRuntimeLease:
    """Acquire one process-shared runtime for an exact settings identity."""
    worker = asyncio.create_task(
        asyncio.to_thread(_acquire_shared_runtime_sync, settings)
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        lease: SharedRuntimeLease | None = None
        try:
            lease = await worker
        except Exception:
            pass
        if lease is not None:
            try:
                await lease.release()
            except Exception:
                pass
        raise


def _close_shared_runtimes_sync() -> None:
    global _last_shutdown_error, _shutdown_in_progress

    with _registry_condition:
        if _shutdown_in_progress:
            while _shutdown_in_progress:
                _registry_condition.wait()
            if _last_shutdown_error is not None:
                raise _last_shutdown_error
            return
        _shutdown_in_progress = True
        _last_shutdown_error = None
        runtimes = [shared.runtime for shared in _shared_runtimes.values()]
        _shared_runtimes.clear()

    failures = 0
    for runtime in runtimes:
        try:
            runtime._close_sync()
        except Exception:
            failures += 1

    shutdown_error = (
        RuntimeError(f"Failed to close {failures} shared mem01 runtime(s)")
        if failures
        else None
    )
    with _registry_condition:
        _last_shutdown_error = shutdown_error
        _shutdown_in_progress = False
        _registry_condition.notify_all()

    if shutdown_error is not None:
        raise shutdown_error from None


async def close_shared_runtimes() -> None:
    """Close and unregister all process-shared runtimes."""
    worker = asyncio.create_task(asyncio.to_thread(_close_shared_runtimes_sync))
    try:
        await asyncio.shield(worker)
    except asyncio.CancelledError:
        try:
            await worker
        except Exception:
            pass
        raise


__all__ = [
    "EmbeddedMem01Runtime",
    "SharedRuntimeLease",
    "acquire_shared_runtime",
    "close_shared_runtimes",
]
