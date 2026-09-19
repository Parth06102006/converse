"""Test verifying ruff linter passes cleanly on the asl-vision package."""

import subprocess
import sys
from pathlib import Path


def test_ruff_linter_clean() -> None:
    """Run ruff check on the asl-vision codebase and assert zero errors."""
    pkg_dir = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "."],
        cwd=pkg_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Ruff check failed:\n{result.stdout}\n{result.stderr}"
