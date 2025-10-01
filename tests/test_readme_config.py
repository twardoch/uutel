# this_file: tests/test_readme_config.py
"""Ensure README configuration snippet stays in sync with defaults."""

from __future__ import annotations

import re
from pathlib import Path

from uutel.core.config import create_default_config

README_PATH = Path(__file__).resolve().parents[1] / "README.md"
CONFIG_SNIPPET_PATTERN = re.compile(
    r"```(?:toml)?\s*\n(?P<body># UUTEL Configuration[\s\S]*?)```",
    re.MULTILINE,
)


def test_readme_config_snippet_matches_default() -> None:
    """README should present the current `create_default_config` template."""

    readme_text = README_PATH.read_text(encoding="utf-8")
    match = CONFIG_SNIPPET_PATTERN.search(readme_text)
    assert match, (
        "README.md is missing a fenced code block starting with '# UUTEL Configuration'."
    )

    snippet = match.group("body").strip()
    default_config = create_default_config().strip()

    assert snippet == default_config, (
        "README config snippet drifted from create_default_config(); update documentation or default template."
    )
