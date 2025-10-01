# this_file: tests/test_documentation_aliases.py
"""Documentation lint tests for engine alias usage."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from uutel.__main__ import validate_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOC_PATHS = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "TROUBLESHOOTING.md",
    PROJECT_ROOT / "AGENTS.md",
    PROJECT_ROOT / "GEMINI.md",
    PROJECT_ROOT / "LLXPRT.md",
    PROJECT_ROOT / "QWEN.md",
]

ENGINE_PATTERN = re.compile(r"--engine\s+([A-Za-z0-9_./-]+)")


@pytest.mark.parametrize("doc_path", DOC_PATHS)
def test_documented_engine_aliases_resolve(doc_path: Path) -> None:
    """Every documented `--engine` value should resolve via `validate_engine`."""

    text = doc_path.read_text(encoding="utf-8")
    matches = ENGINE_PATTERN.findall(text)
    if not matches:
        pytest.skip(f"No --engine references to validate in {doc_path.name}.")

    failures: list[str] = []
    for raw_value in matches:
        candidate = raw_value.strip("'\"`.,)")
        try:
            validate_engine(candidate)
        except (
            ValueError
        ) as exc:  # pragma: no cover - failure path exercised in test assertions
            failures.append(f"{candidate}: {exc}")

    assert not failures, (
        f"Documentation in {doc_path.name} references invalid engines:\n"
        + "\n".join(failures)
    )
