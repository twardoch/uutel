# this_file: tests/test_provider_fixtures.py
"""Verify recorded provider fixtures exist and contain expected fields."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_ROOT = Path(__file__).parent / "data" / "providers"


def _load_fixture(provider: str) -> dict:
    """Load the standard completion fixture for the given provider."""

    fixture_path = FIXTURE_ROOT / provider / "simple_completion.json"
    if not fixture_path.exists():
        pytest.fail(f"Missing fixture: {fixture_path}")
    with fixture_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_claude_fixture_contains_result_and_usage() -> None:
    """Claude fixture should include result text and usage accounting."""

    payload = _load_fixture("claude")
    assert isinstance(payload.get("result"), str) and payload["result"].strip(), (
        "Claude result missing"
    )
    assert "usage" in payload and "input_tokens" in payload["usage"], (
        "Claude usage missing token counts"
    )


def test_codex_fixture_contains_choices_and_usage() -> None:
    """Codex fixture should resemble ChatGPT backend payload."""

    payload = _load_fixture("codex")
    choices = payload.get("choices")
    assert isinstance(choices, list) and choices, "Codex choices missing"
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    assert isinstance(message.get("content"), str) and message["content"].strip(), (
        "Codex message content missing"
    )
    assert "usage" in payload and payload["usage"].get("total_tokens"), (
        "Codex usage missing"
    )


def test_gemini_fixture_contains_candidates_and_usage() -> None:
    """Gemini fixture should follow generateContent schema."""

    payload = _load_fixture("gemini")
    candidates = payload.get("candidates")
    assert isinstance(candidates, list) and candidates, "Gemini candidates missing"
    parts = candidates[0].get("content", {}).get("parts", [])
    assert (
        parts and isinstance(parts[0].get("text"), str) and parts[0]["text"].strip()
    ), "Gemini text missing"
    usage = payload.get("usageMetadata")
    assert usage and usage.get("totalTokenCount"), "Gemini usage metadata missing"


def test_cloud_code_fixture_contains_candidates_and_usage() -> None:
    """Cloud Code fixture should surface text and usage metadata."""

    payload = _load_fixture("cloud_code")
    response = payload.get("response", {})
    candidates = response.get("candidates")
    assert isinstance(candidates, list) and candidates, "Cloud Code candidates missing"
    text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text")
    assert isinstance(text, str) and text.strip(), "Cloud Code text missing"
    usage = response.get("usageMetadata")
    assert usage and usage.get("totalTokenCount"), "Cloud Code usage metadata missing"
