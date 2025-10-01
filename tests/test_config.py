# this_file: tests/test_config.py
"""Regression tests for uutel.core.config helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from uutel.core.config import (
    UUTELConfig,
    create_default_config,
    load_config,
    save_config,
    validate_config,
)


class TestConfigMerge:
    """Tests covering merge_with_args precedence rules."""

    def test_merge_with_args_prioritises_cli_over_config(self) -> None:
        """CLI arguments should override persisted configuration values."""
        config = UUTELConfig(
            engine="uutel-gemini/gemini-2.5-pro", max_tokens=200, temperature=0.5
        )

        merged = config.merge_with_args(
            engine="codex", max_tokens=1000, temperature=1.2, stream=True
        )

        assert merged["engine"] == "codex", "CLI engine should override config engine"
        assert merged["max_tokens"] == 1000, (
            "CLI max_tokens should override config max_tokens"
        )
        assert merged["temperature"] == 1.2, (
            "CLI temperature should override config temperature"
        )
        assert merged["stream"] is True, "Stream flag should accept CLI override"

    def test_merge_with_args_uses_config_when_cli_absent(self) -> None:
        """Persisted configuration should populate default CLI settings."""
        config = UUTELConfig(engine="uutel-claude/claude-sonnet-4", verbose=True)

        merged = config.merge_with_args()

        assert merged["engine"] == "uutel-claude/claude-sonnet-4", (
            "Config engine should populate defaults when CLI omits engine"
        )
        assert merged["verbose"] is True, (
            "Config verbose flag should persist when CLI omits value"
        )


class TestConfigValidation:
    """Validation should flag out-of-range values."""

    def test_validate_config_returns_errors_for_out_of_range_values(self) -> None:
        """Invalid numeric ranges must produce helpful validation errors."""
        config = UUTELConfig(max_tokens=0, temperature=2.5)

        errors = validate_config(config)

        assert "max_tokens must be an integer between 1 and 8000" in errors, (
            "Out-of-range max_tokens should be reported"
        )
        assert "temperature must be a number between 0.0 and 2.0" in errors, (
            "Out-of-range temperature should be reported"
        )

    def test_validate_config_rejects_boolean_inputs(self) -> None:
        """Boolean values should be treated as invalid config entries."""

        config = UUTELConfig(max_tokens=True, temperature=False)

        errors = validate_config(config)

        assert errors, "Validation should reject boolean values masquerading as numeric"
        assert any("max_tokens" in error for error in errors), (
            "Error messages should reference max_tokens when rejecting bool"
        )
        assert any("temperature" in error for error in errors), (
            "Error messages should reference temperature when rejecting bool"
        )

    def test_validate_config_rejects_nan_and_inf_temperature(self) -> None:
        """NaN/inf temperatures should be surfaced as invalid config values."""

        config_nan = UUTELConfig(temperature=float("nan"))
        config_inf = UUTELConfig(temperature=float("inf"))

        nan_errors = validate_config(config_nan)
        inf_errors = validate_config(config_inf)

        assert any("temperature" in error for error in nan_errors), (
            "NaN temperature should trigger validation error"
        )
        assert any("temperature" in error for error in inf_errors), (
            "Infinite temperature should trigger validation error"
        )

    def test_validate_config_flags_invalid_engine(self) -> None:
        """Unknown engines should be rejected with descriptive guidance."""

        config = UUTELConfig(engine="uutel/unknown-model")

        errors = validate_config(config)

        assert errors, "Validation should reject unsupported engine identifiers"
        assert any("engine" in error.lower() for error in errors), (
            "Error messages should mention engine validation when rejecting invalid values"
        )

    def test_validate_config_allows_alias_engine_values(self) -> None:
        """Known aliases such as 'codex' should validate successfully."""

        config = UUTELConfig(engine="codex")

        errors = validate_config(config)

        assert errors == [], "Recognised alias engines should pass validation"

    def test_validate_config_rejects_invalid_system_value(self) -> None:
        """System prompt must be a non-empty string when provided."""

        config_whitespace = UUTELConfig(system="   ")
        config_non_string = UUTELConfig(system=123)  # type: ignore[arg-type]

        whitespace_errors = validate_config(config_whitespace)
        non_string_errors = validate_config(config_non_string)

        assert any("system" in error.lower() for error in whitespace_errors), (
            "Whitespace-only system prompt should trigger validation error"
        )
        assert any("system" in error.lower() for error in non_string_errors), (
            "Non-string system prompt should trigger validation error"
        )

    def test_validate_config_rejects_non_boolean_stream(self) -> None:
        """Stream flag must be boolean when persisted to config."""

        config = UUTELConfig(stream="true")  # type: ignore[arg-type]

        errors = validate_config(config)

        assert errors, "Stream flag should reject non-boolean values"
        assert any("stream" in error.lower() for error in errors), (
            "Validation error should reference stream flag"
        )

    def test_validate_config_rejects_non_boolean_verbose(self) -> None:
        """Verbose flag must be boolean when persisted to config."""

        config = UUTELConfig(verbose="yes")  # type: ignore[arg-type]

        errors = validate_config(config)

        assert errors, "Verbose flag should reject non-boolean values"
        assert any("verbose" in error.lower() for error in errors), (
            "Validation error should reference verbose flag"
        )


class TestConfigPersistence:
    """Roundtrip persistence helpers should preserve values."""

    def test_save_and_load_config_roundtrip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Saving then loading a config should retain all fields."""
        config_path = tmp_path / "uutel.toml"
        config = UUTELConfig(
            engine="uutel-codex/gpt-4o",
            max_tokens=750,
            temperature=0.9,
            system="system prompt",
            stream=True,
            verbose=False,
        )

        save_config(config, config_path=str(config_path))

        monkeypatch.setattr("uutel.core.config.get_config_path", lambda: config_path)

        loaded = load_config()

        assert loaded == config, "Loaded config should match saved config exactly"

        # File content should contain lowercase booleans for TOML compatibility
        content = config_path.read_text(encoding="utf-8")
        assert "stream = true" in content, (
            "Stream flag should serialize as lowercase true"
        )
        assert "verbose = false" in content, (
            "Verbose flag should serialize as lowercase false"
        )

    def test_load_config_returns_defaults_when_toml_invalid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid TOML should not crash and should fall back to default config."""

        config_path = tmp_path / "uutel.toml"
        config_path.write_text("invalid = [", encoding="utf-8")
        monkeypatch.setattr("uutel.core.config.get_config_path", lambda: config_path)

        loaded = load_config()

        assert loaded == UUTELConfig(), (
            "Invalid TOML should return default configuration"
        )

    def test_load_config_warns_on_unknown_keys(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Unexpected config keys should emit a warning for early typo detection."""

        config_path = tmp_path / "uutel.toml"
        config_path.write_text(
            """
engine = "uutel-gemini/gemini-2.5-pro"
unknown_flag = true
extraneous = "value"
            """.strip(),
            encoding="utf-8",
        )
        monkeypatch.setattr("uutel.core.config.get_config_path", lambda: config_path)

        with caplog.at_level("WARNING"):
            loaded = load_config()

        assert loaded.engine == "uutel-gemini/gemini-2.5-pro", (
            "Known keys should still populate the configuration"
        )
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        assert warning_messages, (
            "Load should warn when encountering unknown configuration keys"
        )
        combined = "\n".join(warning_messages)
        assert "unknown_flag" in combined and "extraneous" in combined, (
            "Warning should list all unexpected configuration keys"
        )

    def test_save_config_handles_quotes_and_newlines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Saving configs with quotes/newlines should remain valid TOML."""

        config_path = tmp_path / "uutel.toml"
        config = UUTELConfig(
            engine="my-custom-llm/codex-large",
            max_tokens=600,
            temperature=0.85,
            system='She said "Hello"\nand waved.',
        )

        save_config(config, config_path=str(config_path))
        monkeypatch.setattr("uutel.core.config.get_config_path", lambda: config_path)

        import tomllib

        with open(config_path, "rb") as handle:
            payload = tomllib.load(handle)

        assert payload["system"] == 'She said "Hello"\nand waved.', (
            "System prompt should round-trip with quotes and newlines intact"
        )

    def test_create_default_config_contains_expected_defaults(self) -> None:
        """Default config snippet should mention canonical engine and defaults."""
        default_snippet = create_default_config()

        assert 'engine = "my-custom-llm/codex-large"' in default_snippet, (
            "Default engine should mention codex large"
        )
        assert "max_tokens = 500" in default_snippet, "Default max_tokens should be 500"
        assert "temperature = 0.7" in default_snippet, (
            "Default temperature should be 0.7"
        )
        assert "stream = false" in default_snippet, "Stream default should be false"
        assert "verbose = false" in default_snippet, "Verbose default should be false"

    def test_load_config_coerces_numeric_strings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Numeric strings in the config should become ints/floats on load."""

        config_path = tmp_path / "uutel.toml"
        config_path.write_text(
            'engine = "uutel-codex/gpt-4o"\nmax_tokens = "600"\ntemperature = "0.81"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr("uutel.core.config.get_config_path", lambda: config_path)

        loaded = load_config()

        assert isinstance(loaded.max_tokens, int), (
            "max_tokens string should coerce to int"
        )
        assert loaded.max_tokens == 600
        assert isinstance(loaded.temperature, float), (
            "temperature string should coerce to float"
        )
        assert loaded.temperature == pytest.approx(0.81)

    def test_load_config_leaves_invalid_numeric_strings_for_validation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Malformed numeric strings should remain strings so validation flags them."""

        config_path = tmp_path / "uutel.toml"
        config_path.write_text(
            'max_tokens = "many"\ntemperature = "scalding"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr("uutel.core.config.get_config_path", lambda: config_path)

        loaded = load_config()

        assert loaded.max_tokens == "many"
        assert loaded.temperature == "scalding"

        errors = validate_config(loaded)

        assert any("max_tokens" in error for error in errors), (
            "Invalid max_tokens string should trigger validation error"
        )
        assert any("temperature" in error for error in errors), (
            "Invalid temperature string should trigger validation error"
        )

    @pytest.mark.parametrize(
        "raw_value, expected",
        [("+750", 750), ("1_024", 1024), ("-5", -5)],
    )
    def test_load_config_when_max_tokens_signed_string_then_int_coercion(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        raw_value: str,
        expected: int,
    ) -> None:
        """Signed and underscore-separated max_tokens strings should coerce to ints."""

        config_path = tmp_path / "uutel.toml"
        config_path.write_text(f'max_tokens = "{raw_value}"\n', encoding="utf-8")
        monkeypatch.setattr("uutel.core.config.get_config_path", lambda: config_path)

        loaded = load_config()

        assert isinstance(loaded.max_tokens, int), (
            "max_tokens strings should coerce to integers"
        )
        assert loaded.max_tokens == expected, (
            "Coerced max_tokens should match parsed integer"
        )

        if expected < 1:
            errors = validate_config(loaded)
            assert any("max_tokens" in error for error in errors), (
                "Negative max_tokens should still surface validation errors"
            )

    def test_load_config_when_max_tokens_integral_float_then_int_coercion(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Integral float values for max_tokens should coerce to integers on load."""

        config_path = tmp_path / "uutel.toml"
        config_path.write_text("max_tokens = 2048.0\n", encoding="utf-8")
        monkeypatch.setattr("uutel.core.config.get_config_path", lambda: config_path)

        loaded = load_config()

        assert isinstance(loaded.max_tokens, int), (
            "Integral floats should downcast to int"
        )
        assert loaded.max_tokens == 2048, (
            "Coerced max_tokens should match original float value"
        )

    @pytest.mark.parametrize(
        "raw_value, expected",
        [
            ("TRUE", True),
            ("yes", True),
            ("On", True),
            ("0", False),
            ("off", False),
            ("No", False),
        ],
    )
    def test_load_config_when_boolean_strings_then_flags_coerced(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        raw_value: str,
        expected: bool,
    ) -> None:
        """Common boolean string literals should coerce to bool flags on load."""

        config_path = tmp_path / "uutel.toml"
        config_path.write_text(
            f'stream = "{raw_value}"\nverbose = "{raw_value}"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr("uutel.core.config.get_config_path", lambda: config_path)

        loaded = load_config()

        assert isinstance(loaded.stream, bool), (
            "Stream string literal should coerce to bool"
        )
        assert isinstance(loaded.verbose, bool), (
            "Verbose string literal should coerce to bool"
        )
        assert loaded.stream is expected, "Stream flag should match expected boolean"
        assert loaded.verbose is expected, "Verbose flag should match expected boolean"

    def test_load_config_when_boolean_integers_then_flags_coerced(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Integer literals for stream/verbose should coerce to boolean flags."""

        config_path = tmp_path / "uutel.toml"
        config_path.write_text("stream = 1\nverbose = 0\n", encoding="utf-8")
        monkeypatch.setattr("uutel.core.config.get_config_path", lambda: config_path)

        loaded = load_config()

        assert loaded.stream is True, "Integer one should coerce stream to True"
        assert loaded.verbose is False, "Integer zero should coerce verbose to False"

    def test_load_config_when_engine_or_system_whitespace_then_trimmed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Engine/system strings should trim outer whitespace during load."""

        config_path = tmp_path / "uutel.toml"
        config_path.write_text(
            'engine = "  codex  "\nsystem = "  Keep concise  "\n',
            encoding="utf-8",
        )
        monkeypatch.setattr("uutel.core.config.get_config_path", lambda: config_path)

        loaded = load_config()

        assert loaded.engine == "codex", "Engine value should be trimmed when loading"
        assert loaded.system == "Keep concise", (
            "System prompt should be trimmed on load"
        )

    def test_load_config_when_engine_or_system_blank_then_none(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Blank engine/system strings should collapse to None so defaults apply."""

        config_path = tmp_path / "uutel.toml"
        config_path.write_text('engine = "   "\nsystem = ""\n', encoding="utf-8")
        monkeypatch.setattr("uutel.core.config.get_config_path", lambda: config_path)

        loaded = load_config()

        assert loaded.engine is None, "Engine blank string should convert to None"
        assert loaded.system is None, "System blank string should convert to None"


class TestConfigLoadFailures:
    """Configuration loader failure scenarios."""

    def test_load_config_returns_defaults_when_permission_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Permission errors should log a warning and fall back to defaults."""

        config_path = tmp_path / "uutel.toml"
        config_path.write_text('engine = "uutel-codex/gpt-4o"', encoding="utf-8")
        original_open = open

        def raising_open(path, mode="r", *args, **kwargs):
            from pathlib import Path as _Path

            if _Path(path) == config_path and "r" in mode:
                raise PermissionError("permission denied")
            return original_open(path, mode, *args, **kwargs)

        monkeypatch.setattr("builtins.open", raising_open)
        monkeypatch.setattr("uutel.core.config.get_config_path", lambda: config_path)

        with caplog.at_level("WARNING"):
            result = load_config()

        assert result == UUTELConfig(), (
            "Permission error should return default configuration"
        )
        assert any(
            "unable to read configuration file" in record.message.lower()
            for record in caplog.records
        ), "Warning message should mention inability to read the config file"

    def test_load_config_returns_defaults_when_root_not_table(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Non-table TOML roots should log a warning and return defaults."""

        config_path = tmp_path / "uutel.toml"
        config_path.write_text('engine = "uutel-codex/gpt-4o"', encoding="utf-8")
        monkeypatch.setattr("uutel.core.config.get_config_path", lambda: config_path)

        import tomllib

        def fake_load(*_args, **_kwargs):
            return ["not", "a", "table"]

        monkeypatch.setattr(tomllib, "load", fake_load)

        with caplog.at_level("WARNING"):
            result = load_config()

        assert result == UUTELConfig(), (
            "Loader should fall back to defaults on non-table roots"
        )
        messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        assert any("top-level table" in message for message in messages), (
            "Warning should explain that the TOML root must be a table"
        )
