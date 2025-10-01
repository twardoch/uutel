# this_file: tests/test_cli_validators.py
"""Unit tests for CLI validation helpers."""

from __future__ import annotations

import math

import pytest

from uutel.__main__ import validate_engine, validate_parameters


class TestValidateEngine:
    """Engine validation should normalise aliases and guard unknown inputs."""

    def test_validate_engine_resolves_alias_case_insensitively(self) -> None:
        """Alias inputs should resolve to canonical engine identifiers regardless of case."""
        engine = validate_engine("Claude")

        assert engine == "uutel-claude/claude-sonnet-4", (
            "Alias should resolve to Claude Sonnet canonical engine"
        )

    @pytest.mark.parametrize(
        "alias, expected",
        [
            ("codex", "my-custom-llm/codex-large"),
            ("gemini", "uutel-gemini/gemini-2.5-pro"),
            ("cloud", "uutel-cloud/gemini-2.5-pro"),
        ],
    )
    def test_validate_engine_resolves_primary_aliases(
        self, alias: str, expected: str
    ) -> None:
        """Primary shortcuts should resolve to their canonical engine identifiers."""

        engine = validate_engine(alias)

        assert engine == expected, f"Alias {alias} should resolve to {expected}"

    def test_validate_engine_trims_whitespace_before_resolution(self) -> None:
        """Leading and trailing whitespace should not break alias resolution."""

        engine = validate_engine("  gemini   ")

        assert engine == "uutel-gemini/gemini-2.5-pro", (
            "Whitespace should be stripped before alias lookup"
        )

    def test_validate_engine_accepts_uutel_slash_shortcut(self) -> None:
        """Inputs using the `uutel/<alias>` shorthand should resolve to canonical engines."""

        engine = validate_engine("uutel/claude")

        assert engine == "uutel-claude/claude-sonnet-4", (
            "uutel/claude should map to Claude Sonnet canonical engine"
        )

    def test_validate_engine_accepts_uutel_hyphen_shortcut(self) -> None:
        """Inputs using the `uutel-<alias>` shorthand should resolve to canonical engines."""

        engine = validate_engine("uutel-codex")

        assert engine == "my-custom-llm/codex-large", (
            "uutel-codex should map to default Codex engine"
        )

    def test_validate_engine_accepts_canonical_engine_case_insensitively(self) -> None:
        """Canonical engine identifiers with different casing should resolve to stored casing."""

        engine = validate_engine("UUTEL-CODEX/GPT-4O")

        assert engine == "uutel-codex/gpt-4o", (
            "Case variants of canonical engines should resolve to their canonical identifier"
        )

    def test_validate_engine_preserves_error_guidance_for_unknown_uutel_alias(
        self,
    ) -> None:
        """Unknown uutel shortcuts should still raise with guidance for supported aliases."""

        with pytest.raises(ValueError) as exc_info:
            validate_engine("uutel/unknown")

        message = str(exc_info.value)
        assert "Unknown engine 'uutel/unknown'" in message
        assert "Alias shortcuts" in message
        assert "uutel list_engines" in message

    def test_validate_engine_rejects_whitespace_only_inputs(self) -> None:
        """Whitespace-only values should raise a guidance-rich ValueError."""

        with pytest.raises(ValueError) as exc_info:
            validate_engine("   ")

        message = str(exc_info.value)
        assert "Engine name is required" in message, (
            "Whitespace-only engine should be rejected explicitly"
        )

    def test_validate_engine_rejects_non_string_inputs(self) -> None:
        """Non-string values should raise the standard guidance message."""

        with pytest.raises(ValueError) as exc_info:
            validate_engine(42)  # type: ignore[arg-type]

        message = str(exc_info.value)
        assert "Engine name is required" in message, (
            "Non-string inputs should be rejected explicitly"
        )

    def test_validate_engine_raises_with_suggestions_for_unknown(self) -> None:
        """Unknown engines should raise ValueError with helpful suggestions."""
        with pytest.raises(ValueError) as exc_info:
            validate_engine("unknown-engine")

        message = str(exc_info.value)
        assert "Unknown engine 'unknown-engine'" in message, (
            "Error message should include failing engine"
        )
        assert "Alias shortcuts" in message, (
            "Error message should list alias suggestions"
        )
        assert "uutel list_engines" in message, "Guidance should suggest list command"

    def test_validate_engine_suggests_similar_names(self) -> None:
        """Typo'd engines should surface the closest known engine or alias in guidance."""

        with pytest.raises(ValueError) as exc_info:
            validate_engine("codxe")

        message = str(exc_info.value)
        assert "Unknown engine 'codxe'" in message, (
            "Error message should echo the invalid input"
        )
        assert "Did you mean" in message, (
            "Guidance should include a closest-match suggestion"
        )
        assert "codex" in message, "Suggestion should highlight the canonical alias"


class TestValidateParameters:
    """Completion parameter validation edge cases."""

    def test_validate_parameters_accepts_boundary_values(self) -> None:
        """Minimum and maximum supported values should pass validation."""
        validate_parameters(1, 0.0)
        validate_parameters(8000, 2.0)

    def test_validate_parameters_rejects_invalid_ranges(self) -> None:
        """Out-of-range tokens or temperature should raise ValueError with guidance."""
        with pytest.raises(ValueError) as exc_tokens:
            validate_parameters(0, 0.5)
        assert "max_tokens must be an integer between 1 and 8000" in str(
            exc_tokens.value
        ), "Token guidance should be present"

        with pytest.raises(ValueError) as exc_temp:
            validate_parameters(100, -0.1)
        assert "temperature must be a finite number between 0.0 and 2.0" in str(
            exc_temp.value
        ), "Temperature guidance should be present"

        with pytest.raises(ValueError) as exc_temp_high:
            validate_parameters(100, 2.5)
        assert "temperature must be a finite number between 0.0 and 2.0" in str(
            exc_temp_high.value
        ), "High temperature guidance should be present"

    def test_validate_parameters_rejects_boolean_inputs(self) -> None:
        """Boolean inputs should not be treated as valid integers/floats."""

        with pytest.raises(ValueError) as exc_tokens:
            validate_parameters(True, 0.5)

        message = str(exc_tokens.value)
        assert "max_tokens" in message, "Error message should reference max_tokens"
        assert "integer between 1 and 8000" in message

        with pytest.raises(ValueError) as exc_temp:
            validate_parameters(100, True)

        temp_message = str(exc_temp.value)
        assert "temperature" in temp_message, (
            "Error message should reference temperature"
        )
        assert "finite number between 0.0 and 2.0" in temp_message

    def test_validate_parameters_rejects_nan_temperature(self) -> None:
        """NaN temperatures should be rejected with helpful guidance."""

        with pytest.raises(ValueError) as exc_info:
            validate_parameters(100, float("nan"))

        message = str(exc_info.value)
        assert "temperature" in message, "Error message should reference temperature"
        assert "finite number between 0.0 and 2.0" in message, (
            "Guidance should mention finite numeric range"
        )

    def test_validate_parameters_rejects_none_inputs(self) -> None:
        """None inputs should produce familiar validation errors."""

        with pytest.raises(ValueError) as exc_tokens:
            validate_parameters(None, 0.5)  # type: ignore[arg-type]

        message_tokens = str(exc_tokens.value)
        assert "max_tokens" in message_tokens
        assert "integer between 1 and 8000" in message_tokens

        with pytest.raises(ValueError) as exc_temp:
            validate_parameters(100, None)  # type: ignore[arg-type]

        message_temp = str(exc_temp.value)
        assert "temperature" in message_temp
        assert "finite number between 0.0 and 2.0" in message_temp

    def test_validate_parameters_rejects_infinite_temperature(self) -> None:
        """Infinite temperatures should raise with the finite-range guidance."""

        with pytest.raises(ValueError) as exc_info:
            validate_parameters(100, math.inf)

        message = str(exc_info.value)
        assert "temperature" in message
        assert "finite number between 0.0 and 2.0" in message
