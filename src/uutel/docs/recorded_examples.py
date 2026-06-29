# this_file: src/uutel/docs/recorded_examples.py
"""Recorded example metadata shared between CLI and documentation."""

from __future__ import annotations

from typing import List, MutableMapping

RecordedFixture = MutableMapping[str, str]

RECORDED_FIXTURES: List[RecordedFixture] = [
    {
        "label": "Codex (GPT-4o)",
        "key": "codex",
        "engine": "codex",
        "prompt": "Write a sorter",
        "fixture_path": "codex/simple_completion.json",
        "live_hint": 'uutel complete --prompt "Write a sorter" --engine codex',
    },
    {
        "label": "Claude Code (Sonnet 4)",
        "key": "claude",
        "engine": "claude",
        "prompt": "Say hello",
        "fixture_path": "claude/simple_completion.json",
        "live_hint": 'uutel complete --prompt "Say hello" --engine claude',
    },
    {
        "label": "Gemini CLI (2.5 Pro)",
        "key": "gemini",
        "engine": "gemini",
        "prompt": "Summarise Gemini API",
        "fixture_path": "gemini/simple_completion.json",
        "live_hint": 'uutel complete --prompt "Summarise Gemini API" --engine gemini',
    },
    {
        "label": "Cloud Code (Gemini 2.5 Pro)",
        "key": "cloud_code",
        "engine": "cloud",
        "prompt": "Deployment checklist",
        "fixture_path": "cloud_code/simple_completion.json",
        "live_hint": 'uutel complete --prompt "Deployment checklist" --engine cloud',
    },
]

__all__ = ["RECORDED_FIXTURES", "RecordedFixture"]
