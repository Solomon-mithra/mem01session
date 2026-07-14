# Mem01Session Embedded Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a locally buildable `mem01session` package that composes the OpenAI Agents SDK's SQLite session with the embedded mem01 engine, plus a deterministic three-lane demo and a truthful `/mem01session` product page, without creating a video or committing/pushing/deploying changes.

**Architecture:** `Mem01Session` delegates raw current-conversation storage to an internally created file-backed `SQLiteSession`, while a shared `EmbeddedMem01Runtime` calls the mem01 Python API across `asyncio.to_thread` boundaries for OpenAI extraction, embeddings, and Postgres/pgvector storage. A fresh per-run `RunConfig` captures the current query in `session_input_callback` and injects one cached, untrusted memory block only through `call_model_input_filter`, keeping synthetic memory out of session persistence and extraction.

**Tech Stack:** Python 3.11+, `openai-agents~=0.18.2`, mem01, OpenAI `gpt-5.6-sol`, Postgres/pgvector, pytest/pytest-asyncio, Ruff, mypy, Hatch/build, Next.js 16.2.10, React 19, TypeScript, ESLint.

**Repository rule:** The owner explicitly prohibited commits, pushes, deployments, and publication. Every normal commit checkpoint is replaced by a status/diff review; leave all changes uncommitted for owner approval.

---

## File map

### `mem01`

- `src/mem01/runtime.py`: construct the OpenAI-only in-process `MemoryClient` from explicit settings or environment.
- `src/mem01/store/postgres_store.py`: own a reusable psycopg connection pool and acquire a connection per store operation.
- `src/mem01/api/app.py`: reuse the public runtime builder while preserving health-only fake startup.
- `src/mem01/__init__.py`: export the runtime builder and settings.
- `tests/test_runtime.py`: environment validation, exact model/base URL defaults, injected dependency behavior, and secret-safe errors.
- `tests/test_postgres_store.py`: pool lifecycle and connection acquisition behavior.
- `LICENSE`: MIT text for the pre-existing public engine repository.

### `mem01session` (renamed from the former Agents adapter)

- `pyproject.toml`: distribution/import identity, direct `mem01[openai]` dependency, scripts, and build configuration.
- `src/mem01session/runtime.py`: async embedded runtime, process-shared runtime leases, ownership, and shutdown.
- `src/mem01session/session.py`: complete Agents SDK Session delegation, extraction policy, fresh run hooks, management methods, and close semantics.
- `src/mem01session/items.py`: eligible-text normalization and latest-query extraction.
- `src/mem01session/memory_block.py`: safe belief serialization and bounded model-only memory injection.
- `src/mem01session/metrics.py`: deterministic prepared-input measurement records.
- `src/mem01session/demo.py`: deterministic three-lane scenario and JSON artifact generation.
- `src/mem01session/__init__.py`: canonical `Mem01Session` exports.
- `examples/build_week_demo.py`: thin CLI over the package demo.
- `tests/test_runtime.py`, `tests/test_session.py`, `tests/test_runner_integration.py`, `tests/test_demo.py`, `tests/test_packaging.py`: red/green coverage for the final contract.
- `artifacts/prepared-input-scaling.json`: generated machine-readable 1/10/40-conversation receipt.
- `README.md`, `.env.example`: final install, environment, architecture, limitations, prior-work boundary, and testing instructions.

### `openai_hackathon_build_idea_docs`

- `09_website_mem01session_route.md`: renamed route specification.
- `README.md`, `07_demo_and_video_plan.md`, `08_execution_plan.md`: route links and implementation status synchronized to `/mem01session`.

### `mem01-site`

- `src/app/mem01session/page.tsx`: corrected metadata, quickstart, architecture, generated measurements, evidence, and FAQ.
- `src/app/mem01session/mem01session.module.css`: renamed route-scoped styles.
- `src/app/mem01session/opengraph-image.png`: renamed/generated route card.
- `scripts/check-mem01session.mjs`: exported-route acceptance check.
- `scripts/og-mem01session.html`, `scripts/generate-og.mjs`: renamed OG template/output.
- `src/app/sitemap.ts`, `package.json`: `/mem01session` route and check command.

---

### Task 1: Add the engine's OpenAI runtime and pooled Postgres ownership

**Files:**
- Create: `mem01/src/mem01/runtime.py`
- Create: `mem01/tests/test_runtime.py`
- Create: `mem01/LICENSE`
- Modify: `mem01/src/mem01/store/postgres_store.py`
- Modify: `mem01/src/mem01/api/app.py`
- Modify: `mem01/src/mem01/__init__.py`
- Modify: `mem01/pyproject.toml`
- Test: `mem01/tests/test_runtime.py`
- Test: `mem01/tests/test_postgres_store.py`

- [ ] **Step 1: Write failing runtime tests**

  Add tests that construct `OpenAIRuntimeSettings.from_env()` with an isolated environment and assert: missing `OPENAI_API_KEY` raises a secret-safe `RuntimeError`; missing `DATABASE_URL` raises a secret-safe `RuntimeError`; defaults are `gpt-5.6-sol`, `text-embedding-3-small`, and `https://api.openai.com/v1`; and explicit constructor values override environment values without logging credentials.

  The required public shape is:

  ```python
  settings = OpenAIRuntimeSettings.from_env()
  client = build_openai_memory_client(settings=settings)
  assert client.llm.model == "gpt-5.6-sol"
  assert client.embedder.model == "text-embedding-3-small"
  ```

- [ ] **Step 2: Verify the runtime tests fail for the missing module**

  Run: `mem01/.venv/bin/pytest mem01/tests/test_runtime.py -q`

  Expected: collection fails because `mem01.runtime` does not exist.

- [ ] **Step 3: Implement the settings and builder**

  Implement immutable settings with fields `api_key`, `database_url`, `llm_model`, `embedding_model`, `embedding_dimensions`, and `base_url`. `from_env()` loads `.env`, trims values, accepts `MEM01_LLM_MODEL` and `MEM01_EMBEDDING_MODEL`, and uses the standard OpenAI endpoint internally. `build_openai_memory_client()` constructs `PostgresBeliefStore`, `OpenAIEmbedder`, and `OpenAICompatLLM` directly and never chooses fake providers.

- [ ] **Step 4: Write failing pool tests**

  Patch `_require_psycopg()` with a recording fake pool module and assert one pool is created for a store, each operation uses `pool.connection()`, and `close()` closes exactly once. Preserve the existing public `PostgresBeliefStore` method behavior.

- [ ] **Step 5: Verify pool tests fail against the single-connection implementation**

  Run: `mem01/.venv/bin/pytest mem01/tests/test_postgres_store.py -q`

  Expected: pool assertions fail because the store still owns `_conn`.

- [ ] **Step 6: Convert the store to `psycopg_pool.ConnectionPool`**

  Add the pool dependency through `psycopg[binary,pool]>=3.1`; configure each acquired connection with `dict_row` and pgvector registration; run migration once; use connection context managers for all transactions; and make `close()` idempotent.

- [ ] **Step 7: Reuse the builder from the API and export it**

  Keep fake health-only API startup when no OpenAI key exists, but use `build_openai_memory_client()` for the real branch. Export `OpenAIRuntimeSettings` and `build_openai_memory_client` from `mem01`.

- [ ] **Step 8: Add the MIT license and verify the engine**

  Run: `mem01/.venv/bin/pytest -q`

  Expected: all engine tests pass. If the pre-existing factory test is affected by a real workspace `.env`, isolate `load_env()` in that test so its assertion is independent of developer secrets.

- [ ] **Step 9: Review status without committing**

  Run: `git -C mem01 status --short && git -C mem01 diff --check`

  Expected: only scoped engine files plus preserved pre-existing user changes are listed; `diff --check` exits zero.

### Task 2: Rename and package the embedded `mem01session` distribution

**Files:**
- Rename: the former adapter root to `mem01session/`
- Rename: its former import directory to `mem01session/src/mem01session/`
- Delete: `mem01session/src/mem01session/client.py`
- Create: `mem01session/src/mem01session/runtime.py`
- Modify: `mem01session/pyproject.toml`
- Modify: `mem01session/src/mem01session/__init__.py`
- Create: `mem01session/tests/test_runtime.py`
- Modify: all package tests/imports

- [ ] **Step 1: Perform only the mechanical directory/import rename**

  Rename the project and source directory, then replace distribution/import references with `mem01session`. Do not retain either legacy identity, a compatibility alias, or the HTTP client in the default package.

- [ ] **Step 2: Write failing embedded-runtime tests**

  Test an injected sync `MemoryClient` through `EmbeddedMem01Runtime` and assert `remember`, `recall`, `history`, `correct`, and `forget` run through `asyncio.to_thread`. Test that two default runtime acquisitions for identical settings return the same runtime object, reference counts release it only after the last lease, and an injected runtime remains caller-owned.

  Required API:

  ```python
  runtime = EmbeddedMem01Runtime(client=fake_memory_client)
  packed = await runtime.recall("where now?", user_id="alex", max_memory_tokens=800)
  await runtime.aclose()
  ```

- [ ] **Step 3: Verify the tests fail because the embedded runtime is absent**

  Run: `mem01session/.venv/bin/pytest mem01session/tests/test_runtime.py -q`

  Expected: import or behavior failure for the new runtime.

- [ ] **Step 4: Implement async runtime and shared leases**

  Use `asyncio.to_thread` for every sync engine call. Guard close with an async lock. Key the default registry by non-secret configuration values plus a one-way key fingerprint, not the raw API key or database URL in repr/errors. Return a lease object that releases the shared runtime; provide `close_shared_runtimes()` for application shutdown and tests.

- [ ] **Step 5: Configure the final distribution**

  Set `name = "mem01session"`, build `src/mem01session`, type-check package `mem01session`, retain `openai-agents~=0.18.2`, and depend on `mem01[openai]>=0.1.0`. Keep demo/build/test dependencies optional. Update every import to `from mem01session ...`.

- [ ] **Step 6: Verify package runtime tests and inspect for legacy names**

  Run: `mem01session/.venv/bin/pytest mem01session/tests/test_runtime.py -q`

  Run the legacy identity and sidecar-default scan specified in the task handoff.

  Expected: runtime tests pass and the legacy-name scan returns no product-code/documentation matches.

### Task 3: Make `Mem01Session` correct for SDK persistence and model input

**Files:**
- Modify: `mem01session/src/mem01session/session.py`
- Modify: `mem01session/src/mem01session/items.py`
- Modify: `mem01session/src/mem01session/memory_block.py`
- Modify: `mem01session/tests/test_session.py`
- Modify: `mem01session/tests/test_items.py`
- Modify: `mem01session/tests/test_memory_block.py`
- Modify: `mem01session/tests/test_runner_integration.py`

- [ ] **Step 1: Write failing constructor/delegation tests**

  Assert the final signature accepts `session_id`, `user_id`, `inner`, `runtime`, `conversation_db`, `max_memory_tokens`, and `strict`; a default inner session uses a persistent path under `~/.mem01/conversations.db` (with parent creation); injected inner/runtime objects remain caller-owned; `get_items(limit=N)` returns exactly the inner latest-N items with no memory block; and `session_id`, `session_settings`, `pop_item`, and `clear_session` delegate correctly.

- [ ] **Step 2: Write failing extraction-policy tests**

  Assert `add_items()` always persists raw items first, extracts only textual `user` and `assistant` message items, ignores system memory blocks/reasoning/tool metadata, performs no extraction for empty/non-text batches, passes `user_id` but not the conversation `session_id` for cross-session beliefs, records failure-open errors without secrets, and raises in strict mode.

- [ ] **Step 3: Write failing fresh-run hook tests**

  For every `run_config()` call, assert a distinct `RunConfig` and isolated run state. Its `session_input_callback` must capture the latest new user query and return only `[*history, *new_input]`. Its `call_model_input_filter` must recall once on the first model call, cache the block for later calls in the same run, prepend exactly one memory system item to model input, and never mutate the SDK-provided `ModelInputData`.

- [ ] **Step 4: Verify all new session tests fail against the HTTP/drop-in implementation**

  Run: `mem01session/.venv/bin/pytest mem01session/tests/test_session.py mem01session/tests/test_runner_integration.py -q`

  Expected: failures for constructor, no-memory `get_items`, runtime use, and the absent model-input filter.

- [ ] **Step 5: Implement the final Session behavior**

  Make the default inner store file-backed. Keep memory injection exclusively in `call_model_input_filter`. Serialize active beliefs as JSON-escaped untrusted records with provenance and explicit abstention guidance. Reuse the SDK's `ModelInputData` type and return a new instance. Preserve `max_memory_tokens` as a memory-block budget.

- [ ] **Step 6: Add public memory management methods**

  Implement and test:

  ```python
  beliefs = await session.memory_history(include_invalidated=True, limit=100)
  result = await session.correct_memory(memory_id, "Lives in San Francisco")
  result = await session.forget_memory(memory_id, reason="user requested deletion")
  ```

  Document that `pop_item()` changes only short-term raw history and does not automatically reverse extracted beliefs.

- [ ] **Step 7: Prove behavior with the real Runner and fake model**

  Use the actual Agents SDK `Agent` and `Runner`, an in-process fake runtime, and a recording fake model. Assert the model sees memory plus the current query while the inner session persists only the real user and generated items. Include a multi-model-call/tool case proving recall is cached and synthetic memory is never extracted.

- [ ] **Step 8: Verify the complete Session suite**

  Run: `mem01session/.venv/bin/pytest -q`

  Expected: all package tests pass.

### Task 4: Build deterministic demo, generated evidence, and package artifacts

**Files:**
- Create: `mem01session/src/mem01session/demo.py`
- Modify: `mem01session/src/mem01session/metrics.py`
- Modify: `mem01session/examples/build_week_demo.py`
- Modify: `mem01session/tests/test_demo.py`
- Create: `mem01session/tests/test_packaging.py`
- Create: `mem01session/artifacts/prepared-input-scaling.json`
- Modify: `mem01session/README.md`
- Modify: `mem01session/.env.example`

- [ ] **Step 1: Write failing deterministic three-lane tests**

  Use a checked-in 40-conversation fixture and assert: fresh stock sessions contain no earlier conversation facts; one reused stock session retains and grows raw history; Mem01Session uses distinct conversation IDs with one user and recalls active San Francisco plus earlier rent while NYC is superseded; the sister's name remains unsupported. Treat any model answer as an observation, never a scripted guarantee.

- [ ] **Step 2: Write failing artifact tests**

  Generate records at conversations 1, 10, and 40 with fields `strategy`, `conversation`, `prepared_input_items`, `prepared_input_characters`, `estimated_tokens`, `measurement`, `token_estimator`, `openai_agents_version`, `answer_model`, `extraction_model`, and `max_memory_tokens`. Assert the JSON file is exactly regenerated from the fixture and that the reused lane grows while the memory block remains bounded.

- [ ] **Step 3: Verify demo/artifact tests fail against hand-written output**

  Run: `mem01session/.venv/bin/pytest mem01session/tests/test_demo.py -q`

  Expected: failures for the missing three-lane artifact generator.

- [ ] **Step 4: Implement demo and artifact generation**

  Make `examples/build_week_demo.py` a thin CLI supporting deterministic key-free execution, `--json`, `--write-artifact`, and an explicit `--live` mode. The deterministic mode must exercise real Session/Runner preparation with fakes and must not require a source rebuild, OpenAI key, or database.

- [ ] **Step 5: Rewrite README and environment example**

  Document `pip install mem01session`, only `OPENAI_API_KEY` and `DATABASE_URL` as required normal variables, `gpt-5.6-sol`, internal SQLite, embedded mem01, Postgres/pgvector, fresh per-run `session.run_config()`, resource lifecycle, strict/failure-open modes, short-term-only `pop_item`, deterministic judge command, supported environment actually tested, pre-existing engine boundary, Codex collaboration, GPT-5.6 use, and no publication claim before publication.

- [ ] **Step 6: Build and inspect both local distributions**

  Run: `mem01/.venv/bin/python -m build mem01`

  Run: `mem01session/.venv/bin/python -m build mem01session`

  Create a clean temporary virtual environment, install the two prebuilt wheels with `--no-index --find-links`, import `Mem01Session`, and run deterministic mode. The packaging test must inspect wheel members and metadata for `mem01session`, absence of the legacy import package, and the declared mem01 dependency.

- [ ] **Step 7: Run package quality gates**

  Run from `mem01session`: `.venv/bin/pytest -q && .venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy src && .venv/bin/python -m build`

  Expected: every command exits zero.

### Task 5: Synchronize hackathon docs and implement `/mem01session`

**Files:**
- Rename: `openai_hackathon_build_idea_docs/09_website_openaisdk_route.md` to `openai_hackathon_build_idea_docs/09_website_mem01session_route.md`
- Modify: `openai_hackathon_build_idea_docs/README.md`
- Modify: `openai_hackathon_build_idea_docs/07_demo_and_video_plan.md`
- Modify: `openai_hackathon_build_idea_docs/08_execution_plan.md`
- Rename: `mem01-site/src/app/openaisdk/` to `mem01-site/src/app/mem01session/`
- Rename: `mem01-site/scripts/check-openaisdk.mjs` to `mem01-site/scripts/check-mem01session.mjs`
- Rename: `mem01-site/scripts/og-openaisdk.html` to `mem01-site/scripts/og-mem01session.html`
- Modify: `mem01-site/src/app/mem01session/page.tsx`
- Modify: `mem01-site/src/app/mem01session/mem01session.module.css`
- Modify: `mem01-site/scripts/generate-og.mjs`
- Modify: `mem01-site/src/app/sitemap.ts`
- Modify: `mem01-site/package.json`

- [ ] **Step 1: Rename and correct the hackathon route specification**

  Replace every target `/openaisdk` route reference with `/mem01session`, update the index filename/link, and mark package/runtime/site checklist items as implemented only when local verification supports them. Do not alter or create video claims.

- [ ] **Step 2: Read the installed Next.js 16 guides before editing**

  Read the relevant sections in `node_modules/next/dist/docs/01-app/02-guides/static-exports.md` and the metadata/sitemap/Open Graph file-convention docs. Preserve the existing static Server Component/export architecture.

- [ ] **Step 3: Rename route/check/OG files and make the acceptance check fail first**

  Rename the directories/files and change the checker to require `out/mem01session/index.html`, canonical `/mem01session/`, route OG image, sitemap entry, canonical import, `gpt-5.6-sol`, embedded architecture, generated artifact values, and no default localhost/base-URL instructions. Run `npm run export && npm run check:mem01session`; expect failure until page content is corrected.

- [ ] **Step 4: Implement the truthful final page**

  Use title `Mem01Session for the OpenAI Agents SDK | mem01` and the approved description. Show install/env/code quickstart, both run hooks via `session.run_config()`, internal SQLite plus embedded mem01/Postgres architecture, three truthful lanes, lifecycle/history API, generated scaling measurements, separate pre-existing/Build Week receipts, sourced comparisons, limitations, and separate future product-repository versus pre-existing-engine labels. Omit video CTA/section until a video exists.

- [ ] **Step 5: Update sitemap, OG generation, and command names**

  Remove `/openaisdk` from the exported route/sitemap, add `/mem01session`, generate the renamed OG asset, and expose only `check:mem01session`.

- [ ] **Step 6: Verify site acceptance**

  Run: `npm run lint && npm run og && npm run export && npm run check:mem01session`

  Expected: all commands exit zero and the export route table contains `/mem01session` but not `/openaisdk`.

### Task 6: End-to-end review and final uncommitted handoff

**Files:**
- Review all files changed by Tasks 1–5

- [ ] **Step 1: Run a legacy-contract scan**

  Run:

  ```bash
  rg -n "/openaisdk" \
    mem01session openai_hackathon_build_idea_docs mem01-site \
    --glob '!**/.venv/**' --glob '!**/node_modules/**' --glob '!**/out/**'
  ```

  Expected: no active final-product/default-route matches; any historical reference must be explicitly labeled as historical.

- [ ] **Step 2: Run fresh full verification**

  Run engine tests, package tests/lint/format/type/build, deterministic wheel install/demo, and site lint/OG/export/check in one fresh pass. Record exact counts and artifact paths.

- [ ] **Step 3: Run an optional live smoke test only when credentials are present**

  If both required variables exist, run one bounded live three-conversation smoke test with `gpt-5.6-sol`, redact all secret-bearing output, and record model/result observations. If they are absent or infrastructure is unreachable, report that external verification separately without weakening deterministic verification.

- [ ] **Step 4: Review every diff and preserve owner changes**

  Run `git status --short`, `git diff --check`, and scoped diffs for `mem01` and `mem01-site`; list the non-git `mem01session` files. Confirm no commit, push, deployment, publication, or video action occurred.

- [ ] **Step 5: Stop before integration actions**

  Leave the working trees uncommitted and hand the owner exact verification evidence, remaining external-only items (publication/hosting/video/Devpost), and the single next decision: whether to commit.
