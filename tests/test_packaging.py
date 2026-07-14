from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _wheel() -> Path:
    wheels = sorted((ROOT / "dist").glob("mem01session-*.whl"))
    assert wheels, "build the wheel before running packaging tests"
    return wheels[-1]


def test_wheel_contains_only_canonical_package_and_runtime_fixture() -> None:
    with zipfile.ZipFile(_wheel()) as archive:
        members = archive.namelist()

    assert "mem01session/demo.py" in members
    assert "mem01session/fixtures/build_week_conversations.json" in members
    assert "mem01session/py.typed" in members
    assert not any(
        "mem01_agents" in member or "mem01-agents" in member for member in members
    )
    assert not any(
        segment in member
        for member in members
        for segment in ("/__pycache__/", "/tests/", ".pytest_cache")
    )


def test_wheel_metadata_declares_canonical_identity_and_mem01_dependency() -> None:
    with zipfile.ZipFile(_wheel()) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser(policy=default).parsebytes(archive.read(metadata_name))

    assert metadata["Name"] == "mem01session"
    assert metadata["Version"] == "0.1.2"
    requirements = metadata.get_all("Requires-Dist")
    assert requirements is not None
    assert "mem01-engine[openai]>=0.1.0" in requirements
    assert "openai-agents~=0.18.2" in requirements


def test_clean_preprovisioned_venv_installs_local_wheels_offline(
    tmp_path: Path,
) -> None:
    """Smoke a complete pre-provisioned wheelhouse with the index disabled."""
    wheelhouse_setting = os.environ.get("MEM01SESSION_WHEELHOUSE")
    if not wheelhouse_setting:
        pytest.skip(
            "prepare a wheelhouse with scripts/prepare_offline_wheelhouse.py and "
            "set MEM01SESSION_WHEELHOUSE"
        )
    wheelhouse = Path(wheelhouse_setting).expanduser().resolve()
    assert wheelhouse.is_dir()
    assert list(wheelhouse.glob("mem01_engine-*.whl")) or list(
        wheelhouse.glob("mem01-engine-*.whl")
    )
    assert list(wheelhouse.glob("mem01session-*.whl"))

    environment = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", environment],
        check=True,
        capture_output=True,
        text=True,
    )
    environment_python = environment / "bin" / "python"
    environment_pip = environment / "bin" / "pip"
    subprocess.run(
        [
            environment_pip,
            "install",
            "--no-index",
            "--find-links",
            wheelhouse,
            "mem01session==0.1.2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    imported = subprocess.run(
        [
            environment_python,
            "-c",
            (
                "from pathlib import Path; import mem01, mem01session; "
                "print(Path(mem01.__file__).resolve()); "
                "print(Path(mem01session.__file__).resolve())"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    imported_paths = imported.splitlines()
    assert len(imported_paths) == 2
    assert all(str(environment.resolve()) in path for path in imported_paths)

    completed = subprocess.run(
        [environment / "bin" / "mem01session-demo", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["sdk_runs"] == 120


def test_wheelhouse_preparation_script_documents_direct_local_wheels() -> None:
    source = (ROOT / "scripts" / "prepare_offline_wheelhouse.py").read_text()

    assert "mem01_engine-0.1.0-py3-none-any.whl" in source
    assert "mem01session-0.1.2-py3-none-any.whl" in source
    assert '"download"' in source
    assert '"--only-binary=:all:"' in source
