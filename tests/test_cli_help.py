# this_file: tests/test_cli_help.py
"""Snapshot coverage for Fire-powered CLI help output."""

from __future__ import annotations

import subprocess
import sys
from textwrap import dedent

from uutel.__main__ import ENGINE_ALIASES, UUTELCLI


def _run_help(*args: str) -> str:
    """Execute `python -m uutel <args> --help` and return stdout."""

    command = [sys.executable, "-m", "uutel", *args, "--help"]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    return output.replace("\r\n", "\n").strip()


def test_top_level_help_snapshot() -> None:
    """Top-level CLI help should surface alias and guidance text."""

    output = _run_help()

    expected = dedent(
        """
        INFO: Showing help with the command '__main__.py -- --help'.

        NAME
            __main__.py - UUTEL Command Line Interface.

        SYNOPSIS
            __main__.py -

        DESCRIPTION
            Alias-first engines:
              - codex -> my-custom-llm/codex-large
              - claude -> uutel-claude/claude-sonnet-4
              - gemini -> uutel-gemini/gemini-2.5-pro
              - cloud -> uutel-cloud/gemini-2.5-pro

            Run `uutel list_engines` to review mappings and provider prerequisites.
            Use `uutel complete --help` or `uutel test --help` for command-specific flags.
        """
    ).strip()

    assert output == expected, (
        "Top-level help output drifted; update docstring or snapshot expectations."
    )


def test_complete_help_snapshot() -> None:
    """`uutel complete --help` should advertise alias usage and defaults."""

    output = _run_help("complete")

    expected = dedent(
        """
        INFO: Showing help with the command '__main__.py complete -- --help'.

        NAME
            __main__.py complete - Complete a prompt using the configured engine.

        SYNOPSIS
            __main__.py complete PROMPT <flags>

        DESCRIPTION
            Defaults to the codex alias (my-custom-llm/codex-large).
            Use --engine <alias> to target claude, gemini, or cloud from `uutel list_engines`.
            Enable --stream true to print incremental output when providers support streaming.

        POSITIONAL ARGUMENTS
            PROMPT
                Type: 'str'

        FLAGS
            -e, --engine=ENGINE
                Type: Optional['str | None']
                Default: None
            -m, --max_tokens=MAX_TOKENS
                Type: Optional['int | None']
                Default: None
            -t, --temperature=TEMPERATURE
                Type: Optional['float | None']
                Default: None
            --system=SYSTEM
                Type: Optional['str | None']
                Default: None
            --stream=STREAM
                Type: Optional['bool | None']
                Default: None
            -v, --verbose=VERBOSE
                Type: Optional['bool | None']
                Default: None

        NOTES
            You can also use flags syntax for POSITIONAL ARGUMENTS
        """
    ).strip()

    assert output == expected, (
        "`uutel complete --help` output drifted; update docstring or snapshot expectations."
    )


def test_test_help_snapshot() -> None:
    """`uutel test --help` should surface alias guidance."""

    output = _run_help("test")

    expected = dedent(
        """
        INFO: Showing help with the command '__main__.py test -- --help'.

        NAME
            __main__.py test - Quick readiness probe for provider aliases.

        SYNOPSIS
            __main__.py test <flags>

        DESCRIPTION
            Validates codex, claude, gemini, or cloud using validate_engine before running tests.
            Displays provider prerequisites when credentials or CLIs are missing.

        FLAGS
            -e, --engine=ENGINE
                Type: 'str'
                Default: 'my-custom-llm/codex-large'
            -v, --verbose=VERBOSE
                Type: 'bool'
                Default: True
        """
    ).strip()

    assert output == expected, (
        "`uutel test --help` output drifted; update docstring or snapshot expectations."
    )


def test_cli_docstring_alias_guidance_matches_engine_aliases() -> None:
    """Docstring alias summary should stay in sync with canonical ENGINE_ALIASES data."""

    docstring = UUTELCLI.__doc__ or ""
    parsed: dict[str, str] = {}
    for raw_line in docstring.splitlines():
        line = raw_line.strip()
        if not line.startswith("- ") or "->" not in line:
            continue
        alias, target = (part.strip() for part in line[2:].split("->", 1))
        parsed[alias] = target

    expected = {
        alias: ENGINE_ALIASES[alias] for alias in ("codex", "claude", "gemini", "cloud")
    }

    assert parsed == expected, (
        "UUTELCLI docstring alias lines drifted from ENGINE_ALIASES entries"
    )
