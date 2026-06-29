# this_file: tests/test_documentation_aliases.py
"""Documentation lint tests for engine alias usage."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from uutel.__main__ import validate_engine
from uutel.docs import recorded_examples

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


def test_readme_quick_usage_includes_recorded_hints() -> None:
    """README quick usage section should surface every recorded live hint."""

    readme_path = PROJECT_ROOT / "README.md"
    text = readme_path.read_text(encoding="utf-8")

    missing: list[str] = []
    for fixture in recorded_examples.RECORDED_FIXTURES:
        hint = fixture["live_hint"]
        if hint not in text:
            missing.append(hint)

    assert not missing, (
        "README quick usage is missing recorded live hints: " + ", ".join(missing)
    )


def test_recorded_fixtures_engines_resolve_via_validate_engine() -> None:
    """Recorded fixture engine identifiers should resolve through CLI validation."""

    failures: list[str] = []

    for fixture in recorded_examples.RECORDED_FIXTURES:
        alias = fixture.get("engine", "")
        try:
            validate_engine(alias)
        except ValueError as exc:  # pragma: no cover - error path captured in assertion
            failures.append(f"{alias}: {exc}")

    assert not failures, (
        "Recorded fixtures reference unknown engines:\n" + "\n".join(failures)
    )


def test_recorded_fixtures_live_hint_matches_engine_alias() -> None:
    """Ensure every live hint reflects the documented alias and command format."""

    pattern = re.compile(r'^uutel complete --prompt "(.+?)" --engine ([A-Za-z0-9-]+)$')
    mismatches: list[str] = []

    for fixture in recorded_examples.RECORDED_FIXTURES:
        live_hint = fixture.get("live_hint", "")
        match = pattern.match(live_hint)
        if not match:
            mismatches.append(f"{fixture.get('label', live_hint)}: format mismatch")
            continue
        alias = match.group(2)
        expected_alias = fixture.get("engine")
        if alias != expected_alias:
            mismatches.append(
                f"{fixture.get('label', live_hint)}: live hint alias '{alias}'"
                f" does not match metadata '{expected_alias}'"
            )

    assert not mismatches, (
        "Recorded fixture live hints are inconsistent:\n" + "\n".join(mismatches)
    )
