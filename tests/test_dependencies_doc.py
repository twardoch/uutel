# this_file: tests/test_dependencies_doc.py
"""Guardrail tests keeping dependency docs aligned with pyproject."""

from __future__ import annotations

import re
from pathlib import Path

import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
DEPENDENCIES_DOC = PROJECT_ROOT / "DEPENDENCIES.md"

_REQUIREMENT_SPLIT = re.compile(r"[\[<>=!~ ;]")
_DOC_ENTRY = re.compile(r"- \*\*(?P<requirement>[^*]+)\*\*")


def _normalise_requirement(raw: str) -> str:
    """Return the package name portion of a dependency string."""

    requirement = raw.split(";", 1)[0].strip()
    parts = _REQUIREMENT_SPLIT.split(requirement, 1)
    return parts[0].strip().lower()


def _pyproject_packages() -> set[str]:
    with PYPROJECT_PATH.open("rb") as handle:
        data = tomllib.load(handle)

    packages: set[str] = set()
    for requirement in data.get("project", {}).get("dependencies", []):
        packages.add(_normalise_requirement(requirement))

    optional = data.get("project", {}).get("optional-dependencies", {})
    for group in optional.values():
        for requirement in group:
            packages.add(_normalise_requirement(requirement))

    dependency_groups = data.get("dependency-groups", {})
    for group in dependency_groups.values():
        for requirement in group:
            packages.add(_normalise_requirement(requirement))

    return packages


def _documented_packages() -> set[str]:
    text = DEPENDENCIES_DOC.read_text(encoding="utf-8")
    packages: set[str] = set()
    for match in _DOC_ENTRY.finditer(text):
        requirement = match.group("requirement").strip()
        if not requirement or " " in requirement:
            continue
        packages.add(_normalise_requirement(requirement))
    return packages


def test_dependencies_documentation_matches_pyproject() -> None:
    """Every documented dependency should exist in pyproject.toml."""

    doc_packages = _documented_packages()
    declared_packages = _pyproject_packages()

    missing_from_doc = sorted(declared_packages - doc_packages)
    undocumented = sorted(doc_packages - declared_packages)
    assert not missing_from_doc, (
        "pyproject.toml declares packages missing from DEPENDENCIES.md: "
        f"{missing_from_doc}"
    )
    assert not undocumented, (
        f"Dependencies.md lists packages missing from pyproject: {undocumented}"
    )
