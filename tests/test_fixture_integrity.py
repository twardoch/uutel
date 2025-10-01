# this_file: tests/test_fixture_integrity.py
"""Guard recorded provider fixtures and docs against placeholder content."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from examples.basic_usage import extract_recorded_text
from jsonschema import exceptions as jsonschema_exceptions

from uutel.core.fixture_validation import validate_completion_fixture

FIXTURE_ROOT = Path(__file__).resolve().parents[0] / "data" / "providers"

_PROVIDER_KEY = {
    "codex": "codex",
    "claude": "claude",
    "gemini": "gemini",
    "cloud_code": "cloud_code",
}

_PLACEHOLDER_PHRASES = {
    "this is a mock response",
    "dummy response",
    "placeholder output",
    "in a real implementation",
    "sample response",
}


@pytest.mark.parametrize(
    "fixture_path", sorted(FIXTURE_ROOT.glob("*/simple_completion.json"))
)
def test_recorded_provider_fixtures_contain_realistic_text(fixture_path: Path) -> None:
    """Simple completions must contain non-placeholder text and token usage."""

    provider_folder = fixture_path.parent.name
    provider_key = _PROVIDER_KEY.get(provider_folder)
    assert provider_key, f"Unknown provider folder {provider_folder}"

    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    text, tokens = extract_recorded_text(provider_key, payload)

    assert text.strip(), f"Fixture {fixture_path} has empty response text"
    lowered = text.lower()
    assert not any(phrase in lowered for phrase in _PLACEHOLDER_PHRASES), (
        f"Fixture {fixture_path} contains placeholder text"
    )
    assert tokens and tokens > 0, f"Fixture {fixture_path} missing usage tokens"


@pytest.mark.parametrize(
    "fixture_path", sorted(FIXTURE_ROOT.glob("*/simple_completion.json"))
)
def test_recorded_provider_fixtures_match_schema(fixture_path: Path) -> None:
    """Provider fixtures must satisfy the shared completion schema."""

    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    validate_completion_fixture(payload)


def test_validate_completion_fixture_missing_total_tokens_flagged() -> None:
    """Missing usage totals should raise a validation error with dotted path details."""

    invalid_payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7},
    }

    with pytest.raises(jsonschema_exceptions.ValidationError) as exc_info:
        validate_completion_fixture(invalid_payload)

    message = str(exc_info.value)
    assert "usage.total_tokens" in message, (
        "Error message should pinpoint missing usage.total_tokens field"
    )


def test_validate_completion_fixture_rejects_inconsistent_totals() -> None:
    """Mismatched token totals should be surfaced with informative guidance."""

    codex_payload = {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{"message": {"role": "assistant", "content": "Sorted copy"}}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 25,
        },
    }

    with pytest.raises(jsonschema_exceptions.ValidationError) as exc_info:
        validate_completion_fixture(codex_payload)

    message = str(exc_info.value)
    assert "usage.total_tokens" in message, (
        "Error message should mention the inconsistent total token field"
    )
    assert "prompt_tokens" in message and "completion_tokens" in message, (
        "Guidance should reference the component fields contributing to the mismatch"
    )

    gemini_payload = {
        "candidates": [{"content": {"parts": [{"text": "Gemini reply"}]}}],
        "usageMetadata": {
            "promptTokenCount": 4,
            "candidatesTokenCount": 6,
            "totalTokenCount": 20,
        },
    }

    with pytest.raises(jsonschema_exceptions.ValidationError) as exc_info:
        validate_completion_fixture(gemini_payload)

    message = str(exc_info.value)
    assert "usageMetadata.totalTokenCount" in message, (
        "Gemini mismatch should mention totalTokenCount"
    )
    assert "promptTokenCount" in message and "candidatesTokenCount" in message, (
        "Gemini guidance should reference prompt and candidate counts"
    )


def test_validate_completion_fixture_rejects_whitespace_only_text() -> None:
    """Whitespace-only content should fail fixture validation."""

    codex_payload = {
        "id": "chatcmpl-white",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{"message": {"role": "assistant", "content": "   "}}],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 6,
            "total_tokens": 10,
        },
    }

    with pytest.raises(jsonschema_exceptions.ValidationError) as codex_exc:
        validate_completion_fixture(codex_payload)
    assert "choices.0.message.content" in str(codex_exc.value)

    claude_payload = {
        "result": "     ",
        "usage": {"input_tokens": 3, "output_tokens": 5},
    }

    with pytest.raises(jsonschema_exceptions.ValidationError) as claude_exc:
        validate_completion_fixture(claude_payload)
    assert "result" in str(claude_exc.value)

    gemini_payload = {
        "candidates": [{"content": {"parts": [{"text": "	"}]}}],
        "usageMetadata": {
            "promptTokenCount": 2,
            "candidatesTokenCount": 3,
            "totalTokenCount": 5,
        },
    }

    with pytest.raises(jsonschema_exceptions.ValidationError) as gemini_exc:
        validate_completion_fixture(gemini_payload)
    assert "candidates.0.content.parts.0.text" in str(gemini_exc.value)

    cloud_payload = {
        "response": {
            "candidates": [{"content": {"parts": [{"text": " "}]}}],
            "usageMetadata": {
                "promptTokenCount": 2,
                "candidatesTokenCount": 3,
                "totalTokenCount": 5,
            },
        }
    }

    with pytest.raises(jsonschema_exceptions.ValidationError) as cloud_exc:
        validate_completion_fixture(cloud_payload)
    assert "response.candidates.0.content.parts.0.text" in str(cloud_exc.value)


def test_validate_completion_fixture_when_non_mapping_then_raises_type_error() -> None:
    """Non-dict payloads should raise a descriptive TypeError."""

    with pytest.raises(TypeError) as exc_info:
        validate_completion_fixture(["invalid", "payload"])

    message = str(exc_info.value)
    assert "mapping" in message
    assert "list" in message


def test_validate_completion_fixture_accepts_trimmed_text() -> None:
    """Legitimate non-empty text should continue to validate."""

    payload = {
        "id": "chatcmpl-safe",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{"message": {"role": "assistant", "content": " Answer "}}],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 6,
            "total_tokens": 10,
        },
    }

    validate_completion_fixture(payload)


def test_validate_completion_fixture_rejects_non_mapping_payload() -> None:
    """Non-dict payloads should raise TypeError with helpful messaging."""

    with pytest.raises(TypeError) as exc_info:
        validate_completion_fixture(["not", "a", "mapping"])  # type: ignore[arg-type]

    message = str(exc_info.value)
    assert "Fixture payload must be a mapping" in message, (
        "TypeError should explain payload must originate from JSON object"
    )


def test_docs_no_longer_reference_mock_responses() -> None:
    """Documentation should not advertise mock responses."""

    project_root = Path(__file__).resolve().parents[1]
    doc_paths = [
        project_root / "README.md",
        project_root / "TROUBLESHOOTING.md",
        project_root / "AGENTS.md",
        project_root / "GEMINI.md",
        project_root / "QWEN.md",
        project_root / "LLXPRT.md",
    ]

    for doc_path in doc_paths:
        content = doc_path.read_text(encoding="utf-8").lower()
        for phrase in _PLACEHOLDER_PHRASES:
            assert phrase not in content, (
                f"Remove placeholder phrase '{phrase}' from {doc_path}"
            )
