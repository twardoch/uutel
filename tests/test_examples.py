# this_file: tests/test_examples.py
"""Regression tests for example scripts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import examples.basic_usage as basic_usage
import pytest
from examples.basic_usage import (
    RECORDED_FIXTURES,
    _gather_live_runs,
    _load_stub_payload,
    _normalise_structured_content,
    _resolve_stub_dir,
    _sum_positive_tokens,
    extract_recorded_text,
    truncate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_env_flag_accepts_single_character_truthy_values(monkeypatch) -> None:
    """Single-character affirmatives should be treated as truthy environment flags."""

    env_name = "UUTEL_TEST_FLAG"
    monkeypatch.setenv(env_name, "Y")
    assert basic_usage._env_flag(env_name) is True, (
        "Uppercase Y should be accepted as truthy"
    )

    monkeypatch.setenv(env_name, "t")
    assert basic_usage._env_flag(env_name) is True, (
        "Lowercase t should be accepted as truthy"
    )

    monkeypatch.setenv(env_name, "0")
    assert basic_usage._env_flag(env_name) is False, "Numeric zero should remain false"

    monkeypatch.delenv(env_name, raising=False)


def test_truncate_when_under_limit_then_returns_trimmed_text() -> None:
    """Helper should strip surrounding whitespace when length fits within limit."""

    result = truncate("  keep me  ", limit=50)

    assert result == "keep me", "truncate should strip leading/trailing whitespace"


def test_truncate_when_whitespace_sequences_then_collapses_to_single_space() -> None:
    """Newlines and tabs should be normalised into single spaces for display."""

    messy = "first line\n\nsecond\tline  third"

    result = truncate(messy, limit=80)

    assert result == "first line second line third", (
        "truncate should collapse whitespace sequences into single spaces"
    )


def test_truncate_when_exceeds_limit_then_appends_ellipsis() -> None:
    """Longer strings should be shortened with an ellipsis marker."""

    text = "abcdefghijk"

    result = truncate(text, limit=10)

    assert result == "abcdefg...", (
        "truncate should reserve three characters for the ellipsis"
    )


def test_truncate_when_limit_leq_three_then_returns_prefix_without_ellipsis() -> None:
    """Very small limits should return the leading characters without an ellipsis."""

    text = "truncate"

    assert truncate(text, limit=3) == "tru", (
        "Limit of three should return first three characters"
    )
    assert truncate(text, limit=1) == "t", "Limit of one should return first character"


def test_truncate_when_limit_non_positive_then_returns_empty_string() -> None:
    """Non-positive limits should return an empty preview instead of slicing strangely."""

    assert truncate("example", limit=0) == "", "Zero limit should produce empty string"
    assert truncate("example", limit=-5) == "", (
        "Negative limit should produce empty string"
    )


def test_sum_positive_tokens_when_positive_ints_then_returns_total() -> None:
    """Helper should add positive integer values together."""

    result = _sum_positive_tokens(3, 7, 0)

    assert result == 10, "Positive integers should be summed"


def test_sum_positive_tokens_when_no_positive_values_then_returns_none() -> None:
    """Helper should return None when no positive numeric values provided."""

    result = _sum_positive_tokens(None, 0, -4, "text")

    assert result is None, "Non-positive inputs should not produce a total"


def test_sum_positive_tokens_when_booleans_then_ignore_values() -> None:
    """Boolean values should be skipped instead of counting as integers."""

    result = _sum_positive_tokens(True, False, 5)

    assert result == 5, "Booleans should not contribute to the sum"


def test_sum_positive_tokens_when_positive_floats_then_truncate_to_int() -> None:
    """Positive floats should be cast to integers before summing."""

    result = _sum_positive_tokens(2.9, 3.1)

    assert result == 5, "Floats should be truncated to integer totals"


def test_basic_usage_example_replays_recorded_outputs():
    """The basic usage example should print recorded provider snippets."""

    result = subprocess.run(
        [sys.executable, "examples/basic_usage.py"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        check=True,
    )

    output = result.stdout
    assert "UUTEL Basic Usage Example" in output
    assert "Codex (GPT-4o):" in output
    assert "Claude Code (Sonnet 4):" in output
    assert "Gemini CLI (2.5 Pro):" in output
    assert "Cloud Code (Gemini 2.5 Pro):" in output

    snippets = [
        "safe sorter that keeps the original list untouched",
        "Claude Code is ready to jump into refactors, flaky tests",
        "Gemini 2.5 Pro CLI mirrors google-generativeai: use `--session`",
        "Before deploying, confirm `gcloud auth application-default print-access-token` works",
    ]

    for snippet in snippets:
        assert snippet in output


def test_basic_usage_example_live_mode_uses_stub_directory(tmp_path):
    """When live mode is toggled with a fixture dir, the script consumes stub data."""

    stub_root = tmp_path / "live"
    (stub_root / "codex").mkdir(parents=True)
    (stub_root / "claude").mkdir(parents=True)
    (stub_root / "gemini").mkdir(parents=True)
    (stub_root / "cloud_code").mkdir(parents=True)

    (stub_root / "codex" / "simple_completion.json").write_text(
        '{"choices": [{"message": {"content": "Live Codex stub"}}], "usage": {"total_tokens": 9}}',
        encoding="utf-8",
    )
    (stub_root / "claude" / "simple_completion.json").write_text(
        '{"result": "Live Claude stub", "usage": {"total_tokens": 7}}',
        encoding="utf-8",
    )
    (stub_root / "gemini" / "simple_completion.json").write_text(
        '{"candidates": [{"content": {"parts": [{"text": "Live Gemini stub"}]}}], "usageMetadata": {"totalTokenCount": 5}}',
        encoding="utf-8",
    )
    (stub_root / "cloud_code" / "simple_completion.json").write_text(
        '{"response": {"candidates": [{"content": {"parts": [{"text": "Live Cloud stub"}]}}], "usageMetadata": {"totalTokenCount": 4}}}',
        encoding="utf-8",
    )

    env = {
        **dict(**os.environ),
        "UUTEL_LIVE_EXAMPLE": "1",
        "UUTEL_LIVE_FIXTURES_DIR": str(stub_root),
    }

    result = subprocess.run(
        [sys.executable, "examples/basic_usage.py"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        check=True,
        env=env,
    )

    output = result.stdout
    assert "Live Provider Runs" in output
    assert "Live Codex stub" in output
    assert "Live Claude stub" in output
    assert "Live Gemini stub" in output
    assert "Live Cloud stub" in output


def test_resolve_stub_dir_requires_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """File paths should not be treated as valid stub directories."""

    file_path = tmp_path / "stub.json"
    file_path.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("UUTEL_LIVE_FIXTURES_DIR", str(file_path))

    resolved = _resolve_stub_dir()

    assert resolved is None, (
        "File paths should be ignored when resolving stub directories"
    )


def test_load_stub_payload_ignores_directory_entries(tmp_path: Path) -> None:
    """Directories matching fixture filenames should be skipped gracefully."""

    fixture = RECORDED_FIXTURES[0]
    provider_dir = tmp_path / fixture["path"].parent.name
    provider_dir.mkdir(parents=True)
    (provider_dir / fixture["path"].name).mkdir()

    payload, candidate = _load_stub_payload(fixture, tmp_path)

    assert payload is None, "Directory stubs should be ignored instead of raising"
    assert candidate is None, "Directory stubs should not return a candidate path"


def test_load_stub_payload_reads_stubs_with_utf8_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stub JSON should be read using explicit UTF-8 encoding."""

    fixture = RECORDED_FIXTURES[0]
    provider_dir = tmp_path / fixture["path"].parent.name
    provider_dir.mkdir(parents=True)
    stub_file = provider_dir / fixture["path"].name
    stub_file.write_text(
        '{"choices": [{"message": {"content": "stub"}}]}',
        encoding="utf-8",
    )

    original_read_text = Path.read_text

    def guarded_read_text(self: Path, *args, **kwargs):
        if self.resolve() == stub_file.resolve() and "encoding" not in kwargs:
            raise AssertionError("expected UTF-8 encoding when reading stub file")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    payload, candidate = _load_stub_payload(fixture, tmp_path)

    assert payload is not None, "Stub payload should load successfully"
    assert candidate == stub_file, "Candidate path should point to the stub file"
    assert payload["choices"][0]["message"]["content"] == "stub"


def test_load_stub_payload_returns_error_when_unicode_decode_fails(
    tmp_path: Path,
) -> None:
    """Stub loader should surface structured errors when decoding fails."""

    fixture = {
        "path": Path("codex/simple_completion.json"),
        "key": "codex",
    }
    stub_root = tmp_path
    target_dir = stub_root / fixture["path"].parent.name
    target_dir.mkdir(parents=True)
    target_file = target_dir / fixture["path"].name
    target_file.write_bytes(b"\xff\xfe\x00 invalid utf-8")

    payload, candidate = _load_stub_payload(fixture, stub_root)

    assert payload is not None, "Stub loader should return structured error payload"
    assert "error" in payload, "Error details should be provided when decoding fails"
    message = payload["error"]
    assert "decode" in message.lower(), "Error message should mention decoding issues"
    assert candidate == target_file, (
        "Candidate path should point to the unreadable stub"
    )


def test_load_stub_payload_returns_error_when_read_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stub loader should handle filesystem errors without raising."""

    fixture = {
        "path": Path("codex/simple_completion.json"),
        "key": "codex",
    }
    stub_root = tmp_path
    target_dir = stub_root / fixture["path"].parent.name
    target_dir.mkdir(parents=True)
    target_file = target_dir / fixture["path"].name
    target_file.write_text("{}", encoding="utf-8")

    original_read_text = Path.read_text

    def raising_read_text(self: Path, *args, **kwargs):
        if self.resolve() == target_file.resolve():
            raise PermissionError("permission denied")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", raising_read_text)

    payload, candidate = _load_stub_payload(fixture, stub_root)

    assert payload is not None, "Stub loader should capture permission errors"
    assert "error" in payload, "Permission errors should produce structured payload"
    assert "permission" in payload["error"].lower(), (
        "Error message should mention permission issues"
    )
    assert candidate == target_file, "Candidate should reflect the unreadable stub path"


def test_gather_live_runs_surfaces_stub_load_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live run gathering should return error entries when stub loading fails."""

    fixture = {
        "label": "Codex",
        "key": "codex",
        "engine": "codex",
        "prompt": "Summarise reliability",
        "path": Path("codex/simple_completion.json"),
    }
    stub_root = tmp_path
    target_dir = stub_root / "codex"
    target_dir.mkdir(parents=True)
    target_file = target_dir / "simple_completion.json"
    target_file.write_text("{}", encoding="utf-8")

    original_read_text = Path.read_text

    def raising_read_text(self: Path, *args, **kwargs):
        if self.resolve() == target_file.resolve():
            raise PermissionError("permission denied")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", raising_read_text)

    entries = _gather_live_runs([fixture], stub_root)

    assert entries, "Gathered entries should include the error result"
    entry = entries[0]
    assert entry["status"] == "error", "Stub load failure should emit error status"
    assert "permission" in entry["message"].lower(), (
        "Error message should reference permission issue"
    )


def test_recorded_fixtures_align_with_canonical_engines() -> None:
    """Recorded fixtures should reference canonical engines recognised by the CLI."""

    from uutel.__main__ import validate_engine

    for fixture in RECORDED_FIXTURES:
        validate_engine(fixture["engine"])
        assert "canonical_engine" not in fixture, (
            "Canonical engine should be derived dynamically instead of stored in fixtures"
        )
        path = fixture["path"]
        assert path.is_file(), "Recorded fixture path should exist"
        content = path.read_text(encoding="utf-8")
        assert content.strip(), (
            "Recorded fixture should contain UTF-8 decodable content"
        )


def test_extract_recorded_text_handles_missing_token_metadata() -> None:
    """Extraction should degrade gracefully when usage metadata is absent."""

    codex_payload = {
        "choices": [
            {
                "message": {"content": "Sorted copy response"},
                "finish_reason": "stop",
            }
        ],
    }

    text, tokens = extract_recorded_text("codex", codex_payload)

    assert text == "Sorted copy response"
    assert tokens == 0, (
        "Codex extraction should default tokens to zero when usage missing"
    )

    claude_payload = {
        "result": "Claude helper",
        "usage": {
            "input_tokens": 5,
            "output_tokens": 7,
        },
    }

    text, tokens = extract_recorded_text("claude", claude_payload)

    assert text == "Claude helper"
    assert tokens == 12, (
        "Claude extraction should sum input/output tokens when total missing"
    )

    gemini_payload = {
        "candidates": [{"content": {"parts": [{"text": "Gemini helper"}]}}]
    }

    text, tokens = extract_recorded_text("gemini", gemini_payload)

    assert text == "Gemini helper"
    assert tokens == 0, (
        "Gemini extraction should default token count to zero when metadata missing"
    )


def test_extract_recorded_text_handles_structured_message_content() -> None:
    """Recorded Codex payloads with structured message content should flatten gracefully."""

    codex_payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "Alpha"},
                        {"type": "text", "text": " "},
                        {"type": "paragraph", "content": ["Beta"]},
                        "!",
                    ]
                }
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }

    text, tokens = extract_recorded_text("codex", codex_payload)

    assert text == "Alpha Beta!", (
        "Structured content lists should collapse into a single readable string"
    )
    assert tokens == 7, (
        "Token fallback should sum prompt and completion tokens when total missing"
    )


def test_normalise_structured_content_skips_function_calls() -> None:
    """Structured content should drop function/tool call payloads while preserving text order."""

    content = [
        {"text": "Hello "},
        {"functionCall": {"name": "search", "args": {"query": "uutel"}}},
        {"type": "tool_call", "id": "call-1", "text": "should be ignored"},
        {"text": "world"},
    ]

    flattened = _normalise_structured_content(content)

    assert flattened == "Hello world", (
        "Flattening should remove function/tool call nodes while concatenating textual parts"
    )


def test_extract_recorded_text_skips_function_call_parts() -> None:
    """Gemini extractions should ignore functionCall parts and only return human-readable text."""

    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Plan step one."},
                        {
                            "functionCall": {
                                "name": "google.search",
                                "args": {"query": "uutel"},
                            }
                        },
                        {"text": "Plan step two."},
                    ]
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 7,
            "totalTokenCount": 12,
        },
    }

    text, tokens = extract_recorded_text("gemini", payload)

    assert text == "Plan step one.Plan step two.", (
        "FunctionCall payloads should not leak into recorded text output"
    )
    assert tokens == 12, "Token counts should stay intact for valid payloads"


def test_extract_recorded_text_rejects_unknown_provider_key() -> None:
    """Unknown provider keys should raise to expose fixture misconfigurations."""

    with pytest.raises(ValueError) as exc_info:
        extract_recorded_text("unknown", {})

    assert "unknown provider key" in str(exc_info.value).lower()


def test_load_stub_payload_returns_none_for_missing_file(tmp_path: Path) -> None:
    """Absent stub files should produce a (None, None) tuple."""

    fixture = RECORDED_FIXTURES[0]
    payload, candidate = _load_stub_payload(fixture, tmp_path)

    assert payload is None
    assert candidate is None


def test_load_stub_payload_reports_invalid_json(tmp_path: Path) -> None:
    """Malformed JSON fixtures should surface a descriptive error payload."""

    fixture = RECORDED_FIXTURES[0]
    stub_path = tmp_path / fixture["path"].parent.name / fixture["path"].name
    stub_path.parent.mkdir(parents=True)
    stub_path.write_text("{not valid json}", encoding="utf-8")

    payload, candidate = _load_stub_payload(fixture, tmp_path)

    assert payload is not None, "Invalid JSON should return an error payload"
    assert candidate == stub_path, "Returned path should point to the malformed stub"
    assert payload.get("error", "").startswith("Invalid stub JSON:"), (
        "Error message should highlight invalid JSON"
    )


def test_live_runs_surface_readiness_guidance(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """When providers are unready, the example should print readiness guidance."""

    monkeypatch.setenv("UUTEL_LIVE_EXAMPLE", "1")
    monkeypatch.delenv("UUTEL_LIVE_FIXTURES_DIR", raising=False)

    def fake_readiness(self, engine: str) -> tuple[bool, list[str]]:
        return False, [
            "⚠️ Fake credentials missing",
            "💡 Run fake login command",
        ]

    monkeypatch.setattr(
        "uutel.__main__.UUTELCLI._check_provider_readiness",
        fake_readiness,
        raising=False,
    )
    monkeypatch.setattr("uutel.__main__.setup_providers", lambda: None, raising=False)

    basic_usage.demonstrate_core_functionality()

    captured = capsys.readouterr()
    assert "⚠️ Fake credentials missing" in captured.out
    assert "💡 Run fake login command" in captured.out


def test_demonstrate_core_functionality_reads_fixtures_with_utf8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recorded fixture playback should request UTF-8 encoded content."""

    target_paths = {fixture["path"].resolve() for fixture in RECORDED_FIXTURES}
    original_read_text = Path.read_text

    def guarded_read_text(self: Path, *args, **kwargs):
        if self.resolve() in target_paths and "encoding" not in kwargs:
            raise AssertionError(
                "expected UTF-8 encoding when reading recorded fixture"
            )
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.delenv("UUTEL_LIVE_EXAMPLE", raising=False)

    basic_usage.demonstrate_core_functionality()


def test_recorded_fixtures_align_with_engine_aliases() -> None:
    """Fixture engines should resolve through CLI aliases to their canonical counterparts."""

    from uutel.__main__ import AVAILABLE_ENGINES, ENGINE_ALIASES, validate_engine

    for fixture in RECORDED_FIXTURES:
        alias = fixture["engine"]
        resolved = validate_engine(alias)
        assert alias in ENGINE_ALIASES, (
            f"Fixture {fixture['label']} should reference a defined CLI alias"
        )
        assert resolved in AVAILABLE_ENGINES, (
            f"Canonical engine {resolved} missing from AVAILABLE_ENGINES"
        )
        assert ENGINE_ALIASES[alias] == resolved, (
            f"Alias {alias} should map to {resolved} in ENGINE_ALIASES"
        )


def test_recorded_fixtures_live_hint_uses_documented_alias() -> None:
    """Live hint commands should advertise supported engine aliases for copy/paste usage."""

    for fixture in RECORDED_FIXTURES:
        alias = fixture["engine"]
        hint = fixture["live_hint"]
        assert hint.startswith("uutel complete"), (
            f"Live hint for {fixture['label']} should start with 'uutel complete'"
        )
        assert f"--engine {alias}" in hint, (
            f"Live hint for {fixture['label']} should reuse alias {alias}"
        )


def test_recorded_fixtures_use_realistic_transcripts() -> None:
    """Recorded transcripts should match the curated provider responses used in docs."""

    expected_text = {
        "codex": (
            "Absolutely—here's a safe sorter that keeps the original list untouched, "
            "filters out None entries, and returns a stable sorted copy.\n\n"
            "```python\nfrom collections.abc import Sequence\n\n"
            "def tidy_sort(values: Sequence[int | float | None]) -> list[int | float]:\n"
            "    clean = [value for value in values if value is not None]\n"
            "    return sorted(clean)\n\n"
            "print(tidy_sort([5, None, 2, 1]))\n```"
        ),
        "claude": (
            "Hi! Claude Code is ready to jump into refactors, flaky tests, or release "
            "tooling. Share the failing snippet and I'll propose the next reliable change."
        ),
        "gemini": (
            "Gemini 2.5 Pro CLI mirrors google-generativeai: use `--session` to reuse "
            "context, `--response-schema` for JSON, and `--tool` to match Python "
            "function-calling workflows."
        ),
        "cloud_code": (
            "Before deploying, confirm `gcloud auth application-default print-access-token` "
            "works for the active project, export `CLOUD_CODE_PROJECT`, run integration "
            "tests, snapshot Terraform state, and watch Cloud Monitoring during rollout."
        ),
    }

    for fixture in RECORDED_FIXTURES:
        payload = json.loads(fixture["path"].read_text(encoding="utf-8"))
        text, _ = extract_recorded_text(fixture["key"], payload)
        assert text == expected_text[fixture["key"]], (
            f"Recorded text for {fixture['label']} should match curated transcript"
        )
