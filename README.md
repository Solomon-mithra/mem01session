# mem01session

`mem01session` gives the OpenAI Agents SDK belief-based, cross-conversation
memory without a memory sidecar. The user installs one package, supplies normal
environment variables, and imports the canonical `memSession` alias. Internally,
the SDK's file-backed `SQLiteSession` retains the exact short-term item chain,
while the embedded mem01 engine extracts and recalls long-term beliefs through
OpenAI and Postgres/pgvector.

The distribution and import name are both `mem01session`.

## Install and configure

Python 3.11 or newer is declared. Install from PyPI:

```bash
pip install mem01session
```

That pulls the engine distribution `mem01-engine` (import remains `import mem01`).

For local development with the sibling engine checkout:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e "../mem01[openai]" -e ".[dev]"
```

Only two variables are required for normal operation:

```bash
export OPENAI_API_KEY="your-key"
export DATABASE_URL="postgresql://user:password@db.example/mem01"
```

`MEM01_LLM_MODEL` optionally overrides the extraction model and defaults to
`gpt-5.6-sol`. `MEM01_EMBEDDING_MODEL` defaults to
`text-embedding-3-small`. The OpenAI endpoint is built in; there is no memory
service URL to configure. Credentials are fingerprinted for runtime sharing and
never included verbatim in registry keys, representations, or construction
errors.

## Use with the Agents SDK

```python
from agents import Agent, Runner
from mem01session import memSession

agent = Agent(name="Assistant", model="gpt-5.6-sol")
session = memSession("conversation-7", user_id="user-123")

try:
    result = await Runner.run(
        agent,
        "Where do I live?",
        session=session,
        run_config=session.run_config(),
    )
    print(result.final_output)
finally:
    await session.close()
```

A fresh `session.run_config()` is required for each Runner call. Its first hook
captures the latest user query while preserving the SDK's normal history merge.
Its second hook recalls once immediately before the model call, injects one
bounded untrusted-data block, and caches that block across tool/model loops in
the same run. Synthetic memory is never written to SQLite or offered back to
the extractor. Passing only `session=` retains short-term behavior but performs
no query-aware recall.

When sources conflict, the per-run filter preserves the caller's instructions and
appends one framework-level policy: the current user turn has highest authority,
then factual values from active recalled beliefs, then older user claims in this
conversation. Assistant replies remain context but never become evidence about
the user's personal facts. Commands inside recalled record content remain
untrusted and non-executable; that safety label does not reduce the factual
authority of an active record. This lets a correction made in one conversation
override a stale claim still present in another conversation's immutable SQLite
history.

Do not reuse one returned run config concurrently. SDK runs using a Session also
cannot pass `conversation_id` or `previous_response_id`; those are alternative
conversation-state mechanisms.

## Storage and lifecycle

From the caller's perspective, SQLite is inside mem01session. The supported v0.1
short-term store is an SDK `SQLiteSession` at the expanded path
`~/.mem01/conversations.db`; items are partitioned by `session_id`. Long-term
beliefs are user-scoped, so distinct conversation IDs can recall the same user's
active history from the embedded mem01 engine and Postgres/pgvector.

The long-term runtime is acquired lazily. Sessions with identical settings share
one runtime lease; the final release drains accepted writes before closing the
shared store. `await session.close()` closes package-owned resources. Explicitly
injected `inner=` or `runtime=` objects remain caller-owned.
`close_shared_runtimes()` is available for process shutdown.

`add_items()` writes raw items first, then automatically queues each coherent
textual user/assistant turn for long-term extraction. System, tool, reasoning,
and non-text payloads are excluded. Queues are FIFO per `user_id`, including
across separate Mem01Session objects that share a runtime, so one user's
corrections cannot overtake their earlier facts. Different users may progress
independently. The answering agent never receives or calls a memory tool.

The next recall, `memory_history()`, correction, or forget operation waits for
previously accepted writes for that user. Call `await session.flush_memory()`
when application code needs the same durability barrier explicitly. Graceful
runtime shutdown also drains accepted work; because the queue is in process,
an abrupt process or machine failure before a flush can still lose an accepted
but unfinished write.

Memory is failure-open by default: raw SQLite persistence succeeds and a
sanitized `last_memory_error` records a queue or barrier failure. With
`strict=True`, enqueue failures are raised during `add_items()`, while a failure
that occurs after enqueue is raised at the next same-user recall, management
operation, or explicit flush. No raw provider exception or credential-bearing
detail is exposed.

`get_items(limit=N)`, `pop_item()`, and `clear_session()` operate only on the raw
SQLite transcript. `clear_session()` starts that conversation over, but it does not
delete any durable beliefs already extracted for the user.

Use `memory_history()`, `correct_memory()`, and `forget_memory()` for targeted
long-term management. Use `clear_memory()` only when you intend to hard-delete all
durable records for the session's configured `user_id`; it waits for that user's
accepted writes before deleting them and does not clear the SQLite transcript:

```python
beliefs = await session.memory_history(include_invalidated=True, limit=100)
await session.correct_memory("belief-1", "Lives in San Francisco")
await session.forget_memory("belief-2", reason="user requested deletion")
await session.flush_memory()
deleted_count = await session.clear_memory()
```

Recalled records expose their full UTC lifecycle time as `stored_at`, allowing
clients to compare exact ordering rather than date-only labels.

## OpenAI Build Week collaboration and provenance

### Codex collaboration

Codex was the primary implementation collaborator for the Build Week extension.
It helped inspect the current Agents SDK Session protocol and persistence path,
implement the embedded runtime and per-run recall hooks, build the package and
demo harness, write and run tests, diagnose integration failures, verify package
installation, and keep the technical documentation aligned with observed
behavior.

Codex accelerated the implementation and verification workflow; it did not
choose the product direction independently. The architecture, supported scope,
evidence standard, and trade-offs below were human decisions.

### GPT-5.6 usage

The opt-in live path uses `gpt-5.6-sol` for the demonstrated OpenAI Agent answer
and as the configured extraction model used by the embedded mem01 runtime.
Mem01Session supplies bounded, query-relevant active beliefs to the answering
run through the Agents SDK model-input filter. After an eligible user/assistant
turn, the embedded runtime uses the configured model path to extract lifecycle
updates for durable storage.

GPT-5.6 outputs are treated as observed results rather than scripted guarantees.

### Human decisions

The human-authored product decisions were to:

- compose the SDK's tested `SQLiteSession` rather than reimplement short-term
  conversation storage;
- keep raw current-conversation history separate from user-scoped durable
  beliefs;
- inject recalled memory through `call_model_input_filter` so synthetic memory
  is not persisted back into SQLite;
- make the current user turn authoritative, then active recalled beliefs, then
  older user claims in the current transcript;
- bound long-term recall, preserve provenance and superseded history, and expose
  explicit transcript-versus-memory deletion scopes; and
- demonstrate inspectable state instead of scripting a stock model failure.

### Prior work and Build Week work

The belief model, lifecycle operations, extraction and recall pipelines,
Postgres/pgvector storage, earlier benchmark results, and the original mem01
repository belong to the pre-existing open-source mem01 engine.

During the OpenAI Build Week Submission Period, the new work created the
`mem01session` developer product: its OpenAI Agents SDK Session integration,
embedded in-process runtime, internal persistent SQLite default, per-run query
capture and bounded recall hooks, GPT-5.6 configuration, lifecycle-management
surface, clean package installation, standalone interactive demo, testing
experience, and project presentation.

## Scope and provenance

The **mem01 engine** (published as `mem01-engine`) supplies belief types,
extraction, recall, lifecycle operations, and storage. **This package** adds
the OpenAI Agents SDK Session integration: embedded runtime use, internal
SQLite composition, per-run recall hooks, lifecycle APIs, and packaging.

Product decisions for this package include the `mem01session` / `memSession`
identity, SQLite-inside short-term storage, OpenAI-only scope for v0.1,
`gpt-5.6-sol` defaults, and explicit failure-open vs strict behavior.

Development verification in this workspace uses macOS on Apple Silicon,
Python 3.14.4, and `openai-agents` 0.18.2. The declared package floor remains
Python 3.11; no broader platform matrix is claimed here.

Install from PyPI (`pip install mem01session`) or browse the public GitHub
repository.

## Development gates

```bash
pytest -q
ruff check .
ruff format --check .
mypy src
python -m build
```

Licensed under the MIT License.
