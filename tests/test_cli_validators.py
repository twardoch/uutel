# this_file: tests/test_cli_validators.py
"""Unit tests for CLI validation helpers."""

from __future__ import annotations

import math

import pytest

from uutel.__main__ import (
    ENGINE_ALIASES,
    MODEL_NAME_LOOKUP,
    _validate_engine_aliases,
    validate_engine,
    validate_parameters,
)


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

    @pytest.mark.parametrize(
        "alias, expected",
        [
            ("claude-code", "uutel-claude/claude-sonnet-4"),
            ("gemini-cli", "uutel-gemini/gemini-2.5-pro"),
            ("cloud-code", "uutel-cloud/gemini-2.5-pro"),
            ("codex-large", "my-custom-llm/codex-large"),
        ],
    )
    def test_validate_engine_accepts_common_provider_synonyms(
        self, alias: str, expected: str
    ) -> None:
        """Common provider shorthand strings should resolve to canonical engines."""

        resolved = validate_engine(alias)

        assert resolved == expected, (
            f"Synonym '{alias}' should resolve to canonical engine '{expected}'"
        )

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

    @pytest.mark.parametrize(
        "nested_alias, expected",
        [
            ("uutel/claude/claude-sonnet-4", "uutel-claude/claude-sonnet-4"),
            ("uutel/codex/gpt-4o", "uutel-codex/gpt-4o"),
            ("uutel/gemini/gemini-2.5-pro", "uutel-gemini/gemini-2.5-pro"),
        ],
    )
    def test_validate_engine_accepts_nested_uutel_model_shorthand(
        self, nested_alias: str, expected: str
    ) -> None:
        """Nested `uutel/<alias>/<model>` inputs should resolve via `MODEL_NAME_LOOKUP`."""

        resolved = validate_engine(nested_alias)

        assert resolved == expected, (
            f"Nested shorthand '{nested_alias}' should resolve to '{expected}'"
        )

    def test_validate_engine_rejects_cross_provider_nested_shorthand(self) -> None:
        """Mixed provider nested shorthands should raise helpful guidance."""

        with pytest.raises(ValueError) as exc_info:
            validate_engine("uutel/claude/gemini-2.5-pro")

        message = str(exc_info.value)
        assert "Cross-provider nested shorthand" in message
        assert "uutel/claude/gemini-2.5-pro" in message
        assert "claude" in message and "gemini-2.5-pro" in message

    def test_validate_engine_accepts_model_shorthand_alias(self) -> None:
        """uutel-prefixed model identifiers should resolve via MODEL_NAME_LOOKUP."""

        engine = validate_engine("uutel/gpt-4o")

        assert engine == "uutel-codex/gpt-4o", (
            "uutel/gpt-4o should map to the canonical GPT-4o engine"
        )

    def test_validate_engine_accepts_uutel_hyphen_shortcut(self) -> None:
        """Inputs using the `uutel-<alias>` shorthand should resolve to canonical engines."""

        engine = validate_engine("uutel-codex")

        assert engine == "my-custom-llm/codex-large", (
            "uutel-codex should map to default Codex engine"
        )

    def test_validate_engine_normalises_underscore_alias(self) -> None:
        """Underscore variants of aliases should resolve after normalisation."""

        engine = validate_engine("gemini_cli")

        assert engine == "uutel-gemini/gemini-2.5-pro", (
            "Underscore alias should resolve to Gemini canonical engine"
        )

    def test_validate_engine_normalises_whitespace_alias(self) -> None:
        """Whitespace within aliases should collapse to single hyphen segments."""

        engine = validate_engine("gemini 2.5 pro")

        assert engine == "uutel-gemini/gemini-2.5-pro", (
            "Whitespace alias should resolve to Gemini canonical engine"
        )

    @pytest.mark.parametrize(
        "model_name, expected",
        [
            ("gpt-4o", "uutel-codex/gpt-4o"),
            ("gpt-4o-mini", "my-custom-llm/codex-mini"),
            ("codex-large", "my-custom-llm/codex-large"),
            ("claude-sonnet-4", "uutel-claude/claude-sonnet-4"),
            ("gemini-2.5-pro", "uutel-gemini/gemini-2.5-pro"),
        ],
    )
    def test_validate_engine_accepts_bare_model_identifiers(
        self, model_name: str, expected: str
    ) -> None:
        """Bare model identifiers should resolve to documented canonical engines."""

        resolved = validate_engine(model_name)

        assert resolved == expected, (
            f"Bare model '{model_name}' should resolve to canonical engine '{expected}'"
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

    @pytest.mark.parametrize(
        "alias, expected",
        [
            ("--claude--", "uutel-claude/claude-sonnet-4"),
            ("codex-", "my-custom-llm/codex-large"),
            ("__gemini__", "uutel-gemini/gemini-2.5-pro"),
        ],
    )
    def test_validate_engine_trims_punctuation_aliases(
        self, alias: str, expected: str
    ) -> None:
        """Leading/trailing punctuation should be ignored during alias normalisation."""

        resolved = validate_engine(alias)

        assert resolved == expected, (
            f"Alias '{alias}' should resolve to canonical engine '{expected}'"
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

    def test_validate_engine_error_guidance_sorted(self) -> None:
        """Error guidance should list engines and aliases sorted for deterministic output."""

        with pytest.raises(ValueError) as exc_info:
            validate_engine("unknown-engine")

        message = str(exc_info.value)

        def _extract_block(prefix: str) -> list[str]:
            start = f"{prefix}\n"
            try:
                _, remainder = message.split(start, 1)
            except ValueError:
                pytest.fail(f"Missing '{prefix}' block in error guidance")
            collected: list[str] = []
            for raw_line in remainder.splitlines():
                if not raw_line.strip():
                    break
                collected.append(raw_line.strip())
            return collected

        engine_lines = _extract_block("Available engines:")
        alias_lines = _extract_block("Alias shortcuts:")

        assert engine_lines == sorted(engine_lines), (
            "Engine list should be alphabetically sorted"
        )
        assert alias_lines == sorted(alias_lines), (
            "Alias list should be alphabetically sorted"
        )

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

    def test_build_model_alias_map_raises_on_duplicate_tail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Duplicate tail aliases should raise to prevent ambiguous resolution."""

        import uutel.__main__ as cli

        monkeypatch.setitem(
            cli.AVAILABLE_ENGINES,
            "uutel-clone/claude-sonnet-4",
            "Duplicate entry for testing",
        )

        with pytest.raises(RuntimeError) as exc_info:
            cli._build_model_alias_map()

        message = str(exc_info.value)
        assert "claude-sonnet-4" in message
        assert "uutel-clone/claude-sonnet-4" in message


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


class TestEngineAliasInvariants:
    """Alias invariants should guard against configuration drift."""

    def test_validate_engine_aliases_passes_for_current_configuration(self) -> None:
        """Current alias map should satisfy invariants."""

        assert _validate_engine_aliases() is None

    def test_every_canonical_engine_has_at_least_one_alias(self) -> None:
        """Each canonical engine should be reachable through at least one CLI alias."""

        from uutel.__main__ import AVAILABLE_ENGINES

        alias_targets: dict[str, set[str]] = {}
        for alias, target in ENGINE_ALIASES.items():
            alias_targets.setdefault(target, set()).add(alias)

        model_targets: dict[str, set[str]] = {}
        for shorthand, target in MODEL_NAME_LOOKUP.items():
            model_targets.setdefault(target, set()).add(shorthand)

        missing_cover = []
        for engine_name in AVAILABLE_ENGINES:
            aliases = alias_targets.get(engine_name, set())
            shorthands = model_targets.get(engine_name, set())
            if not aliases and not shorthands:
                missing_cover.append(engine_name)

        assert not missing_cover, (
            "Each canonical engine should be reachable via CLI alias or model shorthand; "
            f"missing coverage for: {missing_cover}"
        )

    def test_validate_engine_aliases_detects_missing_target(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing alias targets should raise a descriptive RuntimeError."""

        monkeypatch.setattr(
            "uutel.__main__.AVAILABLE_ENGINES",
            {"engine-a": "demo"},
            raising=False,
        )
        monkeypatch.setattr(
            "uutel.__main__.ENGINE_ALIASES",
            {"alias": "missing-engine"},
            raising=False,
        )

        with pytest.raises(RuntimeError) as exc_info:
            _validate_engine_aliases()

        message = str(exc_info.value)
        assert "missing-engine" in message
        assert "ENGINE_ALIASES" in message
