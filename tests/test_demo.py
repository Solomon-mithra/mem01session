from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from mem01.types import BeliefStatus

from mem01session.demo import (
    ANSWER_MODEL,
    ARTIFACT_PATH,
    EXTRACTION_MODEL,
    MAX_MEMORY_TOKENS,
    TOKEN_ESTIMATOR,
    generate_artifact_records,
    load_conversation_fixture,
    render_artifact,
    run_deterministic_demo,
)
from mem01session.memory_block import is_memory_item
from mem01session.metrics import OFFLINE_PREPARED_INPUT_MEASUREMENT


def _text(items: list[object]) -> str:
    return json.dumps(items, sort_keys=True)


def test_checked_in_fixture_has_40_numbered_conversations() -> None:
    fixture = load_conversation_fixture()

    assert len(fixture) == 40
    assert [turn.conversation for turn in fixture] == list(range(1, 41))
    assert "$2,400" in fixture[0].user_input
    assert "NYC" in fixture[0].user_input
    assert "SF" in fixture[1].user_input
    assert "sister" in fixture[-1].user_input.lower()


@pytest.mark.asyncio
async def test_three_lanes_prove_amnesia_growth_and_cross_session_memory() -> None:
    result = await run_deterministic_demo()

    assert result.sdk_runs == 120
    assert len(result.mem01_session_ids) == 40
    assert len(set(result.mem01_session_ids)) == 40
    assert set(result.mem01_user_ids) == {"build-week-user"}

    fresh_10 = result.prepared_inputs["fresh_stock"][10]
    reused_10 = result.prepared_inputs["reused_stock"][10]
    memory_10 = result.prepared_inputs["mem01session"][10]
    assert "NYC" not in _text(fresh_10)
    assert "$2,400" not in _text(fresh_10)
    assert "NYC" in _text(reused_10)
    assert "$2,400" in _text(reused_10)
    assert "San Francisco" in _text(memory_10)
    assert "$2,400" in _text(memory_10)
    assert "NYC" not in _text(memory_10)
    assert len([item for item in memory_10 if is_memory_item(item)]) == 1

    active = {
        belief.content
        for belief in result.beliefs
        if belief.status == BeliefStatus.ACTIVE
    }
    superseded = {
        belief.content
        for belief in result.beliefs
        if belief.status == BeliefStatus.SUPERSEDED
    }
    assert active == {"Lives in San Francisco", "Monthly rent was $2,400"}
    assert superseded == {"Lives in New York City"}

    assert all("sister" not in belief.content.lower() for belief in result.beliefs)
    assert all(
        isinstance(result.observed_answers[strategy][conversation], str)
        for strategy in result.observed_answers
        for conversation in (1, 10, 40)
    )


@pytest.mark.asyncio
async def test_artifact_is_exactly_regenerated_and_memory_is_bounded() -> None:
    result = await run_deterministic_demo()
    records = generate_artifact_records(result)

    assert json.loads(ARTIFACT_PATH.read_text()) == records
    assert ARTIFACT_PATH.read_text() == render_artifact(records)
    assert len(records) == 9
    assert {(record["strategy"], record["conversation"]) for record in records} == {
        (strategy, conversation)
        for strategy in ("fresh_stock", "reused_stock", "mem01session")
        for conversation in (1, 10, 40)
    }
    required = {
        "strategy",
        "conversation",
        "prepared_input_items",
        "prepared_input_characters",
        "estimated_tokens",
        "measurement",
        "token_estimator",
        "openai_agents_version",
        "answer_model",
        "extraction_model",
        "max_memory_tokens",
    }
    assert all(required <= record.keys() for record in records)
    assert all(
        record["measurement"] == OFFLINE_PREPARED_INPUT_MEASUREMENT
        for record in records
    )
    assert all(record["token_estimator"] == TOKEN_ESTIMATOR for record in records)
    assert all(record["answer_model"] == ANSWER_MODEL for record in records)
    assert all(record["extraction_model"] == EXTRACTION_MODEL for record in records)
    assert all(record["max_memory_tokens"] == MAX_MEMORY_TOKENS for record in records)

    reused = [record for record in records if record["strategy"] == "reused_stock"]
    assert [record["prepared_input_items"] for record in reused] == [1, 19, 79]
    assert reused[0]["estimated_tokens"] < reused[1]["estimated_tokens"]
    assert reused[1]["estimated_tokens"] < reused[2]["estimated_tokens"]

    for conversation in (10, 40):
        items = result.prepared_inputs["mem01session"][conversation]
        memory_item = next(item for item in items if is_memory_item(item))
        assert isinstance(memory_item, dict)
        assert len(memory_item["content"].encode("utf-8")) <= MAX_MEMORY_TOKENS


@pytest.mark.asyncio
async def test_deterministic_runs_are_repeatedly_isolated() -> None:
    first = generate_artifact_records(await run_deterministic_demo())
    second = generate_artifact_records(await run_deterministic_demo())

    assert first == second


def test_env_example_describes_only_embedded_runtime_settings() -> None:
    contents = (Path(__file__).parents[1] / ".env.example").read_text()

    assert "OPENAI_API_KEY=" in contents
    assert "DATABASE_URL=" in contents
    assert "MEM01_LLM_MODEL=gpt-5.6-sol" in contents
    assert "MEM01_EMBEDDING_MODEL=text-embedding-3-small" in contents
    forbidden_sidecar_setting = "_".join(("MEM01", "BASE", "URL"))
    assert forbidden_sidecar_setting not in contents
    assert "127.0.0.1" not in contents
    assert "localhost" not in contents


def test_example_is_a_thin_cli_over_the_packaged_demo() -> None:
    source = (Path(__file__).parents[1] / "examples" / "build_week_demo.py").read_text()

    assert "from mem01session.demo import main" in source
    assert len(source.splitlines()) <= 12


def test_readme_documents_judge_path_boundaries_and_truthful_status() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text()

    for required in (
        "python examples/build_week_demo.py --json",
        "python examples/build_week_demo.py --write-artifact",
        "SQLiteSession",
        "gpt-5.6-sol",
        "session.run_config()",
        "failure-open",
        "strict=True",
        "Pre-existing mem01 engine",
        "Build Week work",
        "Codex",
        "human",
        "not been published",
        "No deployment",
        "No video",
        "Python 3.14.4",
        "--no-index",
        "--find-links",
        ".venv/bin/python -m build ../mem01",
    ):
        assert required in readme


def test_check_is_secret_safe_and_does_not_run_the_demo(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mem01session import demo

    monkeypatch.setenv("OPENAI_API_KEY", "sk-never-print-this")
    monkeypatch.setenv("DATABASE_URL", "postgresql://secret-never-print-this")
    monkeypatch.setenv("MEM01_LLM_MODEL", "wrong-model")

    assert demo.main(["--check"]) == 2
    output = capsys.readouterr().out
    assert "wrong-model" not in output
    assert "sk-never-print-this" not in output
    assert "secret-never-print-this" not in output
    assert "MEM01_LLM_MODEL must be gpt-5.6-sol" in output


def test_check_loads_engine_dotenv_without_overriding_process_environment(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mem01session import demo

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    calls: list[bool] = []

    def load_dotenv(*, override: bool = False) -> list[Path]:
        calls.append(override)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-loaded-but-never-printed")
        monkeypatch.setenv("DATABASE_URL", "postgresql://loaded-but-never-printed")
        return [Path("mem01/.env")]

    monkeypatch.setattr(demo.mem01_env, "load_env", load_dotenv)

    assert demo.main(["--check"]) == 0
    assert calls == [False]
    output = capsys.readouterr().out
    assert '"ready": true' in output
    assert "loaded-but-never-printed" not in output


def test_deterministic_mode_does_not_load_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mem01session import demo

    def forbidden_load(*, override: bool = False) -> list[Path]:
        del override
        raise AssertionError("deterministic mode must not load .env")

    monkeypatch.setattr(demo.mem01_env, "load_env", forbidden_load)

    assert demo.main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "deterministic-key-free"


def test_live_loads_dotenv_before_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mem01session import demo

    events: list[str] = []

    def load_dotenv(*, override: bool = False) -> list[Path]:
        assert override is False
        events.append("load")
        return []

    async def fake_live(environ: object) -> int:
        assert environ is os.environ
        events.append("run")
        return 0

    monkeypatch.setattr(demo.mem01_env, "load_env", load_dotenv)
    monkeypatch.setattr(demo, "_run_live", fake_live)

    assert demo.main(["--live"]) == 0
    assert events == ["load", "run"]


@pytest.mark.asyncio
async def test_live_scenario_uses_fresh_sessions_and_reports_observations(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mem01.types import Belief

    from mem01session import demo

    sessions: list[SimpleNamespace] = []
    runner_prompts: list[str] = []

    class FakeLiveSession:
        def __init__(
            self,
            session_id: str,
            user_id: str,
            *,
            strict: bool,
        ) -> None:
            self.session_id = session_id
            self.user_id = user_id
            self.strict = strict
            self.closed = False
            sessions.append(self)

        def run_config(self) -> object:
            return SimpleNamespace(tracing_disabled=False)

        async def memory_history(
            self, *, include_invalidated: bool, limit: int
        ) -> list[Belief]:
            assert include_invalidated is True
            assert limit == 100
            return [
                Belief(
                    id="old-city",
                    content="Lives in New York City",
                    status=BeliefStatus.SUPERSEDED,
                ),
                Belief(id="current-city", content="Lives in San Francisco"),
            ]

        async def close(self) -> None:
            self.closed = True

    async def fake_run(
        agent: object,
        prompt: str,
        *,
        session: object,
        run_config: object,
    ) -> object:
        del agent, session, run_config
        runner_prompts.append(prompt)
        return SimpleNamespace(final_output=f"observation-{len(runner_prompts)}")

    monkeypatch.setattr(demo, "Mem01Session", FakeLiveSession)
    monkeypatch.setattr(demo.Runner, "run", fake_run)

    exit_code = await demo._run_live(
        {
            "OPENAI_API_KEY": "test-placeholder",
            "DATABASE_URL": "postgresql://test-placeholder",
            "MEM01_LLM_MODEL": "gpt-5.6-sol",
        }
    )

    assert exit_code == 0
    assert len(sessions) == 4
    assert len({session.session_id for session in sessions}) == 4
    assert len({session.user_id for session in sessions}) == 1
    assert all(session.user_id.startswith("build-week-live-") for session in sessions)
    assert all(session.strict is True and session.closed for session in sessions)
    assert "NYC" in runner_prompts[0] and "$2,400" in runner_prompts[0]
    assert "SF" in runner_prompts[1]
    assert "prior monthly rent" in runner_prompts[2]
    assert "sister" in runner_prompts[3].lower()

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "live"
    assert payload["model"] == "gpt-5.6-sol"
    assert len(payload["observations"]) == 4
    assert all(
        observation["type"] == "model_observation"
        for observation in payload["observations"]
    )
    assert [record["status"] for record in payload["lifecycle"]] == [
        "superseded",
        "active",
    ]
