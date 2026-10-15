"""Every script in examples/ runs to the end, so an API change cannot leave one broken."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).parent.parent / "examples").glob("*.py"))


def test_there_are_examples() -> None:
    assert EXAMPLES


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(script: Path) -> None:
    done = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
