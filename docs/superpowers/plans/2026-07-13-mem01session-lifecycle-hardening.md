# Mem01Session Lifecycle Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make runtime operations, shared registry shutdown, and Session cleanup durable under concurrency, cancellation, and close failures.

**Architecture:** The embedded runtime owns a condition-protected `open -> closing -> closed` state machine and admits operations inside worker threads so cancellation cannot lose in-flight accounting. Registry acquisition/release and Session cleanup use shielded task handoffs: caller cancellation waits for irreversible cleanup before propagating, while global shutdown gates new acquisition and attempts every captured runtime.

**Tech Stack:** Python 3.11+, asyncio, threading conditions, pytest/pytest-asyncio, Ruff, mypy, Hatch/build.

**Repository rule:** Do not commit, push, deploy, publish, or touch video assets. Review status and artifacts only.

---

### Task 1: Runtime operation and close state machine

**Files:**
- Modify: `tests/test_runtime.py`
- Modify: `src/mem01session/runtime.py`

- [ ] Write deterministic blocking-client tests proving a cancelled recall remains in-flight, close waits for it, new work is rejected after closing begins, concurrent closers wait for the same store close, and one secret-safe close failure is replayed to every closer.
- [ ] Run the focused tests and capture failures against the boolean-close implementation.
- [ ] Replace `_closed` with condition-protected state, in-flight count, and shared close completion. Execute operation admission/call/decrement entirely inside `asyncio.to_thread` workers; perform store close outside the condition lock.
- [ ] Re-run focused runtime tests to GREEN.

Required operation-worker shape:

```python
def _run_operation(self, operation, *args, **kwargs):
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
```

### Task 2: Cancellation-safe shared acquisition and release

**Files:**
- Modify: `tests/test_runtime.py`
- Modify: `src/mem01session/runtime.py`

- [ ] Add gated worker tests for cancellation during acquisition and final release.
- [ ] Verify RED: acquisition leaves an unreachable registry reference and release cancellation propagates before close completes.
- [ ] Wrap worker futures in tasks and shield them. On acquisition cancellation, wait for the worker, release its resulting lease, then re-raise; on release cancellation, wait for decrement/close before re-raising.
- [ ] Re-run focused cancellation tests to GREEN.

Required cancellation handoff:

```python
worker = asyncio.create_task(asyncio.to_thread(sync_operation))
try:
    return await asyncio.shield(worker)
except asyncio.CancelledError:
    result = await worker
    await cleanup(result)
    raise
```

### Task 3: Serialize process-wide shutdown

**Files:**
- Modify: `tests/test_runtime.py`
- Modify: `src/mem01session/runtime.py`

- [ ] Add tests with two runtimes, one failing close, concurrent shutdown callers, and a gated acquisition started during shutdown.
- [ ] Verify RED: current shutdown clears early, aborts after one failure, and permits a new entry before shutdown completes.
- [ ] Add a registry condition and shutdown flag. One shutdown captures and clears entries, attempts every close, resets/notifies in `finally`, and raises a secret-safe aggregate error. Other shutdown calls and acquisition wait for the active shutdown.
- [ ] Re-run registry lifecycle tests to GREEN.

### Task 4: Durable Session cleanup barrier

**Files:**
- Modify: `tests/test_session.py`
- Modify: `src/mem01session/session.py`

- [ ] Add tests proving concurrent/cancelled close callers share cleanup, release and inner close are both attempted on failure, sync close is off-loop, async close is awaited, and acquisition racing close cannot leak.
- [ ] Verify RED against the early-return close implementation.
- [ ] Atomically mark closed and detach the lease under `_runtime_lock`, create one cleanup task, then shield it for all callers. Run synchronous inner close with `asyncio.to_thread`, await async close directly, attempt both resources, and replay one secret-safe aggregate result.
- [ ] Re-run focused Session lifecycle tests to GREEN.

Required close shape:

```python
async with self._runtime_lock:
    if self._cleanup_task is None:
        self._closed = True
        lease, self._runtime_lease = self._runtime_lease, None
        self._cleanup_task = asyncio.create_task(self._cleanup(lease))
task = self._cleanup_task
await asyncio.shield(task)
```

### Task 5: Full verification and artifact review

**Files:**
- Verify: all package files and `dist/mem01session-0.1.0-py3-none-any.whl`

- [x] Run `.venv/bin/pytest -q`.
- [x] Run `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .`.
- [x] Run `.venv/bin/mypy src` and `.venv/bin/python -m build`.
- [x] Run legacy and loopback scans excluding `.venv` and `dist`.
- [x] Inspect wheel members and metadata for canonical package identity and direct dependencies.
- [x] Self-review every cancellation path and confirm no commit/push/deploy/publication occurred.

### Task 6: Completion barriers for release and cancellation

**Files:**
- Modify: `tests/test_runtime.py`
- Modify: `tests/test_session.py`
- Modify: `src/mem01session/runtime.py`
- Modify: `src/mem01session/session.py`

- [x] Add focused regressions proving concurrent and later lease releases wait for and replay the final close outcome, direct runtime close cancellation drains its worker, and Session close cancellation drains shared cleanup before returning cancellation.
- [x] Run the focused tests and capture RED against the current claimed-release flag and non-draining close awaits.
- [x] Replace the lease flag with a condition-protected unreleased/releasing/released completion state and persisted sanitized error.
- [x] Apply the shield-and-drain worker pattern to direct runtime close and the shield-and-drain shared-task pattern to Session close.
- [x] Re-run focused tests to GREEN, then repeat every Task 5 verification and artifact check.
