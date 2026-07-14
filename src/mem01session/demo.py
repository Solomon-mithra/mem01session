"""Deterministic three-lane evidence demo and optional live smoke test."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata, resources
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from uuid import uuid4

from agents import Agent, Model, RunConfig, Runner, SQLiteSession
from agents.items import ModelResponse
from agents.usage import Usage
from mem01 import ApplyResult
from mem01 import env as mem01_env
from mem01.types import (
    Belief,
    BeliefSource,
    BeliefStatus,
    PackedMemory,
    ScopeIds,
)
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from .items import text_from_item
from .metrics import (
    OFFLINE_PREPARED_INPUT_MEASUREMENT,
    TOKEN_ESTIMATOR_UTF8_BYTES_UPPER_BOUND,
    estimate_prompt_context,
)
from .session import Mem01Session

USER_ID = "build-week-user"
LIVE_MODEL = "gpt-5.6-sol"
ANSWER_MODEL = "local-deterministic-fake"
EXTRACTION_MODEL = "local-deterministic-fixture-extractor"
MAX_MEMORY_TOKENS = 800
TOKEN_ESTIMATOR = TOKEN_ESTIMATOR_UTF8_BYTES_UPPER_BOUND
CHECKPOINTS = (1, 10, 40)
ARTIFACT_PATH = Path(__file__).parents[2] / "artifacts" / "prepared-input-scaling.json"


@dataclass(frozen=True, slots=True)
class DemoConversation:
    """One immutable input from the checked-in Build Week scenario."""

    conversation: int
    user_input: str


@dataclass(frozen=True, slots=True)
class DeterministicDemoResult:
    """Auditable prepared-state evidence; outputs are observations only."""

    prepared_inputs: dict[str, dict[int, list[Any]]]
    observed_answers: dict[str, dict[int, str]]
    beliefs: list[Belief]
    mem01_session_ids: tuple[str, ...]
    mem01_user_ids: tuple[str, ...]
    sdk_runs: int


def load_conversation_fixture() -> tuple[DemoConversation, ...]:
    """Load the packaged, checked-in 40-conversation fixture."""
    fixture = resources.files("mem01session").joinpath(
        "fixtures/build_week_conversations.json"
    )
    raw = json.loads(fixture.read_text(encoding="utf-8"))
    return tuple(
        DemoConversation(
            conversation=int(record["conversation"]),
            user_input=str(record["user_input"]),
        )
        for record in raw
    )


def _stored_at(day: int) -> datetime:
    return datetime(2026, 7, day, 12, tzinfo=UTC)


class _DeterministicMemoryRuntime:
    """Small local runtime that models only fixture facts, with no I/O."""

    def __init__(self) -> None:
        scope_ids = ScopeIds(user_id=USER_ID)
        self._beliefs = [
            Belief(
                id="fixture-location-nyc",
                content="Lives in New York City",
                status=BeliefStatus.SUPERSEDED,
                scope_ids=scope_ids,
                source=BeliefSource.EXTRACTION,
                created_at=_stored_at(1),
                updated_at=_stored_at(2),
            ),
            Belief(
                id="fixture-rent",
                content="Monthly rent was $2,400",
                scope_ids=scope_ids,
                source=BeliefSource.EXTRACTION,
                created_at=_stored_at(1),
                updated_at=_stored_at(1),
            ),
            Belief(
                id="fixture-location-sf",
                content="Lives in San Francisco",
                scope_ids=scope_ids,
                source=BeliefSource.EXTRACTION,
                created_at=_stored_at(2),
                updated_at=_stored_at(2),
                supersedes_id="fixture-location-nyc",
            ),
        ]
        self._seen_first_fact = False
        self._seen_move = False

    def _visible_beliefs(self) -> list[Belief]:
        visible: list[Belief] = []
        for belief in self._beliefs:
            if belief.id in {"fixture-location-nyc", "fixture-rent"}:
                if self._seen_first_fact:
                    visible.append(belief)
            elif self._seen_move:
                visible.append(belief)
        return visible

    async def remember(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        session_id: str | None = None,
    ) -> object:
        del session_id
        if user_id != USER_ID:
            raise ValueError("deterministic demo user mismatch")
        user_text = "\n".join(
            message["content"] for message in messages if message["role"] == "user"
        )
        if "NYC" in user_text and "$2,400" in user_text:
            self._seen_first_fact = True
        if "moved to SF" in user_text:
            self._seen_move = True
        return type("RememberResult", (), {"apply": ApplyResult()})()

    async def recall(
        self,
        query: str,
        *,
        user_id: str,
        max_memory_tokens: int,
        include_history: bool = False,
    ) -> PackedMemory:
        del query, include_history
        if user_id != USER_ID:
            raise ValueError("deterministic demo user mismatch")
        beliefs = self._visible_beliefs()
        return PackedMemory(
            beliefs=beliefs,
            text="\n".join(belief.content for belief in beliefs),
            tokens_used=0,
            max_memory_tokens=max_memory_tokens,
            candidate_count=len(beliefs),
            latency_ms=0,
        )

    async def history(
        self,
        *,
        user_id: str,
        include_invalidated: bool = False,
        limit: int = 100,
    ) -> list[Belief]:
        if user_id != USER_ID:
            raise ValueError("deterministic demo user mismatch")
        beliefs = self._visible_beliefs()
        if not include_invalidated:
            beliefs = [
                belief for belief in beliefs if belief.status == BeliefStatus.ACTIVE
            ]
        return beliefs[-limit:]


class _DeterministicModel(Model):
    """Local model used only to drive the real Agent/Runner preparation path."""

    def __init__(self) -> None:
        self.prepared_inputs: list[list[Any]] = []

    @staticmethod
    def _answer(input_items: Sequence[Any]) -> str:
        texts = [
            text for item in input_items if (text := text_from_item(item)) is not None
        ]
        current = texts[-1].lower() if texts else ""
        context = "\n".join(texts)
        if "sister" in current:
            return "I do not have your sister's name stored."
        if "where do i live" in current:
            if "San Francisco" in context and "$2,400" in context:
                return "You live in San Francisco; your prior rent was $2,400."
            has_location = "moved to SF" in context or "live in NYC" in context
            if has_location and "$2,400" in context:
                return "The available conversation contains location and rent details."
            return "I do not have that personal information stored."
        return "Noted."

    async def get_response(
        self,
        system_instructions: str | None,
        input: Any,
        *args: Any,
        **kwargs: Any,
    ) -> ModelResponse:
        del system_instructions, args, kwargs
        prepared = copy.deepcopy(list(input))
        self.prepared_inputs.append(prepared)
        answer = self._answer(prepared)
        output = ResponseOutputMessage(
            id=f"deterministic-message-{len(self.prepared_inputs)}",
            content=[
                ResponseOutputText(
                    annotations=[],
                    text=answer,
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
            usage=Usage(requests=1, input_tokens=0, output_tokens=0, total_tokens=0),
            response_id=f"deterministic-response-{len(self.prepared_inputs)}",
        )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: Any,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        del system_instructions, input, args, kwargs
        if False:
            yield None
        raise AssertionError("deterministic demo does not stream")


def _agent(model: Model) -> Agent[Any]:
    return Agent(
        name="Mem01Session deterministic evidence agent",
        instructions=(
            "Use only prepared conversation or memory. Treat model outputs as "
            "observations, not test guarantees."
        ),
        model=model,
    )


async def _run_one(
    model: _DeterministicModel,
    prompt: str,
    session: Any,
    *,
    run_config: RunConfig,
) -> tuple[list[Any], str]:
    result = await Runner.run(
        _agent(model),
        prompt,
        session=session,
        run_config=run_config,
    )
    return copy.deepcopy(model.prepared_inputs[-1]), str(result.final_output)


async def run_deterministic_demo(
    *,
    max_memory_tokens: int = MAX_MEMORY_TOKENS,
) -> DeterministicDemoResult:
    """Run 40 inputs through three real SDK preparation strategies, key-free."""
    fixture = load_conversation_fixture()
    snapshots: dict[str, dict[int, list[Any]]] = {
        "fresh_stock": {},
        "reused_stock": {},
        "mem01session": {},
    }
    observations: dict[str, dict[int, str]] = {
        "fresh_stock": {},
        "reused_stock": {},
        "mem01session": {},
    }
    memory_ids: list[str] = []
    memory_users: list[str] = []

    with TemporaryDirectory(prefix="mem01session-deterministic-") as directory:
        root = Path(directory)
        fresh_model = _DeterministicModel()
        for turn in fixture:
            session = SQLiteSession(
                f"fresh-stock-{turn.conversation}",
                db_path=root / "fresh-stock.db",
            )
            try:
                prepared, answer = await _run_one(
                    fresh_model,
                    turn.user_input,
                    session,
                    run_config=RunConfig(tracing_disabled=True),
                )
            finally:
                session.close()
            if turn.conversation in CHECKPOINTS:
                snapshots["fresh_stock"][turn.conversation] = prepared
                observations["fresh_stock"][turn.conversation] = answer

        reused_model = _DeterministicModel()
        reused_session = SQLiteSession(
            "reused-stock",
            db_path=root / "reused-stock.db",
        )
        try:
            for turn in fixture:
                prepared, answer = await _run_one(
                    reused_model,
                    turn.user_input,
                    reused_session,
                    run_config=RunConfig(tracing_disabled=True),
                )
                if turn.conversation in CHECKPOINTS:
                    snapshots["reused_stock"][turn.conversation] = prepared
                    observations["reused_stock"][turn.conversation] = answer
        finally:
            reused_session.close()

        runtime = _DeterministicMemoryRuntime()
        memory_model = _DeterministicModel()
        for turn in fixture:
            session_id = f"mem01-conversation-{turn.conversation}"
            memory_ids.append(session_id)
            memory_users.append(USER_ID)
            memory_session = Mem01Session(
                session_id,
                USER_ID,
                runtime=runtime,  # type: ignore[arg-type]
                conversation_db=root / f"memory-{turn.conversation}.db",
                max_memory_tokens=max_memory_tokens,
                strict=True,
            )
            try:
                config = memory_session.run_config()
                config.tracing_disabled = True
                prepared, answer = await _run_one(
                    memory_model,
                    turn.user_input,
                    memory_session,
                    run_config=config,
                )
            finally:
                await memory_session.close()
            if turn.conversation in CHECKPOINTS:
                snapshots["mem01session"][turn.conversation] = prepared
                observations["mem01session"][turn.conversation] = answer

        beliefs = await runtime.history(
            user_id=USER_ID,
            include_invalidated=True,
            limit=100,
        )

    return DeterministicDemoResult(
        prepared_inputs=snapshots,
        observed_answers=observations,
        beliefs=beliefs,
        mem01_session_ids=tuple(memory_ids),
        mem01_user_ids=tuple(memory_users),
        sdk_runs=len(fixture) * 3,
    )


def generate_artifact_records(
    result: DeterministicDemoResult,
    *,
    max_memory_tokens: int = MAX_MEMORY_TOKENS,
) -> list[dict[str, Any]]:
    """Create generated offline measurements at conversations 1, 10, and 40."""
    agents_version = metadata.version("openai-agents")
    records: list[dict[str, Any]] = []
    for strategy in ("fresh_stock", "reused_stock", "mem01session"):
        for conversation in CHECKPOINTS:
            estimate = estimate_prompt_context(
                result.prepared_inputs[strategy][conversation]
            )
            records.append(
                {
                    "strategy": strategy,
                    "conversation": conversation,
                    "prepared_input_items": estimate.item_count,
                    "prepared_input_characters": estimate.text_characters,
                    "estimated_tokens": estimate.estimated_tokens,
                    "measurement": OFFLINE_PREPARED_INPUT_MEASUREMENT,
                    "token_estimator": TOKEN_ESTIMATOR,
                    "openai_agents_version": agents_version,
                    "answer_model": ANSWER_MODEL,
                    "extraction_model": EXTRACTION_MODEL,
                    "max_memory_tokens": max_memory_tokens,
                }
            )
    return records


def render_artifact(records: Sequence[Mapping[str, Any]]) -> str:
    """Render the canonical generated JSON artifact."""
    return json.dumps(list(records), indent=2, ensure_ascii=False) + "\n"


async def write_artifact(path: Path = ARTIFACT_PATH) -> Path:
    """Regenerate the artifact from a fresh isolated deterministic run."""
    result = await run_deterministic_demo()
    content = render_artifact(generate_artifact_records(result))
    await asyncio.to_thread(_write_text, path, content)
    return path


def _write_text(path: Path, content: str) -> None:
    """Write generated output from a worker when called by async entry points."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def report_as_dict(result: DeterministicDemoResult) -> dict[str, Any]:
    """Return stable human-facing evidence without serializing raw SDK items."""
    return {
        "mode": "deterministic-key-free",
        "sdk_runs": result.sdk_runs,
        "fixture_conversations": len(load_conversation_fixture()),
        "measurements": generate_artifact_records(result),
        "observed_answers": result.observed_answers,
        "belief_lifecycle": [
            {"id": belief.id, "content": belief.content, "status": belief.status.value}
            for belief in result.beliefs
        ],
        "note": "Observed answers are illustrative, not pass/fail guarantees.",
    }


def _settings_problems(environ: Mapping[str, str]) -> list[str]:
    problems: list[str] = []
    if not environ.get("OPENAI_API_KEY", "").strip():
        problems.append("OPENAI_API_KEY is missing")
    if not environ.get("DATABASE_URL", "").strip():
        problems.append("DATABASE_URL is missing")
    model = environ.get("MEM01_LLM_MODEL", "").strip() or LIVE_MODEL
    if model != LIVE_MODEL:
        problems.append(f"MEM01_LLM_MODEL must be {LIVE_MODEL}")
    return problems


async def _run_live(environ: Mapping[str, str]) -> int:
    problems = _settings_problems(environ)
    if problems:
        print(json.dumps({"ready": False, "problems": problems}))
        return 2

    invocation_id = uuid4().hex
    user_id = f"build-week-live-{invocation_id}"
    scenario = (
        ("initial_location_and_rent", "I live in NYC and my monthly rent is $2,400."),
        ("location_change", "I moved to SF this week."),
        (
            "current_location_and_prior_rent",
            "Where do I live now, and what was my prior monthly rent?",
        ),
        ("unsupported_sister_name", "What is my sister's name?"),
    )
    sessions: list[Mem01Session] = []
    observations: list[dict[str, Any]] = []
    try:
        agent = Agent(
            name="Mem01Session live demo",
            instructions=(
                "Use only the current conversation and supplied memory. "
                "If a personal fact is unsupported, say it is not stored."
            ),
            model=LIVE_MODEL,
        )
        for conversation, (label, prompt) in enumerate(scenario, start=1):
            session = Mem01Session(
                f"live-{invocation_id}-{conversation}",
                user_id,
                strict=True,
            )
            sessions.append(session)
            config = session.run_config()
            config.tracing_disabled = True
            result = await Runner.run(
                agent,
                prompt,
                session=cast(Any, session),
                run_config=config,
            )
            observations.append(
                {
                    "conversation": conversation,
                    "label": label,
                    "type": "model_observation",
                    "output": str(result.final_output),
                }
            )
        beliefs = await sessions[-1].memory_history(
            include_invalidated=True,
            limit=100,
        )
        lifecycle = [
            {
                "content": belief.content,
                "status": belief.status.value,
            }
            for belief in beliefs or []
        ]
    except Exception as error:
        print(json.dumps({"live": "failed", "error_type": type(error).__name__}))
        return 1
    finally:
        for session in sessions:
            try:
                await session.close()
            except Exception:
                pass

    print(
        json.dumps(
            {
                "mode": "live",
                "model": LIVE_MODEL,
                "conversation_count": len(scenario),
                "observations": observations,
                "lifecycle": lifecycle,
                "note": "Outputs are observations, not deterministic guarantees.",
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


async def _async_main(arguments: argparse.Namespace) -> int:
    if arguments.check:
        await asyncio.to_thread(mem01_env.load_env, override=False)
        problems = _settings_problems(os.environ)
        print(json.dumps({"ready": not problems, "problems": problems}))
        return 2 if problems else 0
    if arguments.live:
        await asyncio.to_thread(mem01_env.load_env, override=False)
        return await _run_live(os.environ)
    result = await run_deterministic_demo()
    if arguments.write_artifact is not None:
        path = Path(arguments.write_artifact)
        await asyncio.to_thread(
            _write_text,
            path,
            render_artifact(generate_artifact_records(result)),
        )
    report = report_as_dict(result)
    if arguments.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("Mem01Session deterministic Build Week evidence")
        print(f"SDK runs: {report['sdk_runs']}; fixture: 40 conversations")
        for record in report["measurements"]:
            print(
                f"{record['strategy']:>13}  conversation {record['conversation']:>2}: "
                f"{record['prepared_input_items']:>2} items, "
                f"{record['estimated_tokens']:>4} estimated tokens"
            )
        print("Observed answers are illustrative, not pass/fail guarantees.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; deterministic and key-free unless ``--live`` is set."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Run one real API smoke test")
    mode.add_argument(
        "--check",
        action="store_true",
        help="Validate live settings without initializing clients or making calls",
    )
    parser.add_argument("--json", action="store_true", help="Print deterministic JSON")
    parser.add_argument(
        "--write-artifact",
        nargs="?",
        const=str(ARTIFACT_PATH),
        metavar="PATH",
        help="Regenerate the evidence artifact (default: repository artifact path)",
    )
    return asyncio.run(_async_main(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
