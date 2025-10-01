# this_file: tests/test_metadata_markers.py
"""Ensure project metadata conventions stay enforced."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIRS = (
    PROJECT_ROOT / "src" / "uutel",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "examples",
)
ALLOWLIST = {Path("src/uutel/_version.py")}
MAX_HEADER_LINES = 40


def _read_head(path: Path, max_lines: int) -> str:
    """Return the top of a file for marker inspection."""

    with path.open(encoding="utf-8") as handle:
        lines = []
        for _, line in zip(range(max_lines), handle, strict=False):
            lines.append(line)
        return "".join(lines)


@pytest.mark.parametrize(
    "path", sorted({p for directory in TARGET_DIRS for p in directory.rglob("*.py")})
)
def test_python_files_include_this_file_marker(path: Path) -> None:
    """All tracked Python files should declare their relative path."""

    relative = path.relative_to(PROJECT_ROOT)
    if relative in ALLOWLIST:
        pytest.skip("Generated artefact without marker requirement")

    header = _read_head(path, MAX_HEADER_LINES)

    assert "this_file:" in header, f"Missing this_file marker near top of {relative}"
