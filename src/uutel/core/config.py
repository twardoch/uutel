# this_file: src/uutel/core/config.py
"""UUTEL configuration management - simplified core functionality.

This module provides basic configuration file support for UUTEL CLI.
"""

from __future__ import annotations

# Standard library imports
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Third-party imports
import tomli_w

# Local imports
from uutel.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class UUTELConfig:
    """Basic UUTEL configuration class."""

    engine: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    system: str | None = None
    stream: bool | None = None
    verbose: bool | None = None

    def merge_with_args(self, **kwargs) -> dict[str, Any]:
        """Merge configuration with command-line arguments."""
        result = {}

        # Start with config values
        if self.engine is not None:
            result["engine"] = self.engine
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.system is not None:
            result["system"] = self.system
        if self.stream is not None:
            result["stream"] = self.stream
        if self.verbose is not None:
            result["verbose"] = self.verbose

        # Override with CLI arguments
        for key, value in kwargs.items():
            if value is not None:
                result[key] = value

        return result


def get_config_path() -> Path:
    """Get the path to the configuration file."""
    return Path.home() / ".uutel.toml"


def load_config() -> UUTELConfig:
    """Load configuration from file or return defaults."""
    config_path = get_config_path()

    if not config_path.exists():
        logger.debug("No configuration file found, using defaults")
        return UUTELConfig()

    import tomllib

    try:
        with open(config_path, "rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        reason = getattr(exc, "strerror", None) or str(exc)
        logger.warning(
            "Unable to read configuration file %s: %s",
            config_path,
            reason,
        )
        return UUTELConfig()
    except tomllib.TOMLDecodeError as exc:
        logger.warning(f"Failed to parse configuration {config_path}: {exc}")
        return UUTELConfig()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Failed to load configuration: {exc}")
        return UUTELConfig()

    logger.debug(f"Configuration loaded from {config_path}")

    if not isinstance(data, dict):
        logger.warning(
            "Configuration file %s must declare a top-level table; got %s. Using defaults instead.",
            config_path,
            type(data).__name__,
        )
        return UUTELConfig()

    if isinstance(data, dict):
        allowed_keys = {
            "engine",
            "max_tokens",
            "temperature",
            "system",
            "stream",
            "verbose",
        }
        unexpected_keys = sorted(key for key in data.keys() if key not in allowed_keys)
        if unexpected_keys:
            logger.warning(
                "Unknown configuration keys in %s: %s",
                config_path,
                ", ".join(unexpected_keys),
            )

    engine_value = data.get("engine")
    max_tokens_value = data.get("max_tokens")
    temperature_value = data.get("temperature")
    system_value = data.get("system")
    stream_value = data.get("stream")
    verbose_value = data.get("verbose")

    if isinstance(engine_value, str):
        trimmed_engine = engine_value.strip()
        engine_value = trimmed_engine or None

    if isinstance(max_tokens_value, str):
        stripped_tokens = max_tokens_value.strip()
        if not stripped_tokens:
            max_tokens_value = None
        else:
            try:
                max_tokens_value = int(stripped_tokens)
            except ValueError:
                pass
    elif isinstance(max_tokens_value, float) and not math.isnan(max_tokens_value):
        if math.isfinite(max_tokens_value) and max_tokens_value.is_integer():
            max_tokens_value = int(max_tokens_value)

    if isinstance(temperature_value, str):
        stripped_temperature = temperature_value.strip()
        if not stripped_temperature:
            temperature_value = None
        else:
            try:
                temperature_value = float(stripped_temperature)
            except ValueError:
                pass

    if isinstance(system_value, str):
        trimmed_system = system_value.strip()
        system_value = trimmed_system or None

    def _coerce_bool_literal(value: Any) -> Any:
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                return None
            lowered = trimmed.lower()
            if lowered in {"none", "null", "default"}:
                return None
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
            return value
        if isinstance(value, int | float):
            if isinstance(value, float):
                if not math.isfinite(value):
                    return value
                if value.is_integer():
                    value = int(value)
                else:
                    return value
            if value == 0:
                return False
            if value == 1:
                return True
            return value
        return value

    stream_value = _coerce_bool_literal(stream_value)
    verbose_value = _coerce_bool_literal(verbose_value)

    return UUTELConfig(
        engine=engine_value,
        max_tokens=max_tokens_value,
        temperature=temperature_value,
        system=system_value,
        stream=stream_value,
        verbose=verbose_value,
    )


def save_config(config: UUTELConfig, config_path: str | None = None) -> None:
    """Save configuration to file."""
    if config_path is None:
        config_path = get_config_path()
    else:
        config_path = Path(config_path)

    config_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {}
    if config.engine is not None:
        payload["engine"] = config.engine
    if config.max_tokens is not None:
        payload["max_tokens"] = config.max_tokens
    if config.temperature is not None:
        payload["temperature"] = config.temperature
    if config.system is not None:
        payload["system"] = config.system
    if config.stream is not None:
        payload["stream"] = config.stream
    if config.verbose is not None:
        payload["verbose"] = config.verbose

    body = tomli_w.dumps(payload) if payload else ""
    content = "# UUTEL Configuration\n\n" + body
    if not content.endswith("\n"):
        content += "\n"

    try:
        with open(config_path, "w", encoding="utf-8") as handle:
            handle.write(content)
        logger.debug(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise


def validate_config(config: UUTELConfig) -> list[str]:
    """Basic configuration validation."""
    errors = []

    if config.engine:
        try:
            from uutel.__main__ import (
                validate_engine,  # Local import to avoid circular dependency
            )

            validate_engine(config.engine)
        except ValueError as exc:
            errors.append(str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"engine validation failed: {exc}")

    if config.max_tokens is not None:
        if (
            isinstance(config.max_tokens, bool)
            or not isinstance(config.max_tokens, int)
            or config.max_tokens < 1
            or config.max_tokens > 8000
        ):
            errors.append("max_tokens must be an integer between 1 and 8000")

    if config.temperature is not None:
        temperature = config.temperature
        if isinstance(temperature, bool) or not isinstance(temperature, int | float):
            errors.append("temperature must be a number between 0.0 and 2.0")
        else:
            numeric_temperature = float(temperature)
            if math.isnan(numeric_temperature) or math.isinf(numeric_temperature):
                errors.append("temperature must be a number between 0.0 and 2.0")
            elif numeric_temperature < 0.0 or numeric_temperature > 2.0:
                errors.append("temperature must be a number between 0.0 and 2.0")

    if config.system is not None:
        if not isinstance(config.system, str):
            errors.append("system prompt must be a string value")
        elif not config.system.strip():
            errors.append("system prompt must not be empty or whitespace")

    if config.stream is not None and not isinstance(config.stream, bool):
        errors.append("stream must be a boolean value")

    if config.verbose is not None and not isinstance(config.verbose, bool):
        errors.append("verbose must be a boolean value")

    return errors


def create_default_config() -> str:
    """Create default configuration file content."""
    return """# UUTEL Configuration

engine = "my-custom-llm/codex-large"
max_tokens = 500
temperature = 0.7
stream = false
verbose = false
"""
