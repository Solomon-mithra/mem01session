#!/usr/bin/env python3
"""Resolve both local project wheels and all dependencies into a wheelhouse."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).parents[1]
ENGINE_WHEEL = (
    ROOT.parent / "mem01" / "dist" / "mem01_engine-0.1.0-py3-none-any.whl"
)
SESSION_WHEEL = ROOT / "dist" / "mem01session-0.1.3-py3-none-any.whl"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheelhouse", type=Path)
    arguments = parser.parse_args(argv)
    missing = [path for path in (ENGINE_WHEEL, SESSION_WHEEL) if not path.is_file()]
    if missing:
        parser.error("build both local wheels before preparing the wheelhouse")
    arguments.wheelhouse.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--dest",
            arguments.wheelhouse,
            "--only-binary=:all:",
            ENGINE_WHEEL,
            SESSION_WHEEL,
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
