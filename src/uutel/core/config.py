# this_file: src/uutel/core/config.py
"""UUTEL configuration management - simplified core functionality.

This module provides basic configuration file support for UUTEL CLI.
"""

from __future__ import annotations

# Standard library imports
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

    try:
        import tomllib

        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        logger.debug(f"Configuration loaded from {config_path}")

        return UUTELConfig(
            engine=data.get("engine"),
            max_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
            system=data.get("system"),
            stream=data.get("stream"),
            verbose=data.get("verbose"),
        )
    except Exception as e:
        logger.warning(f"Failed to load configuration: {e}")
        return UUTELConfig()


def save_config(config: UUTELConfig, config_path: str | None = None) -> None:
    """Save configuration to file."""
    if config_path is None:
        config_path = get_config_path()
    else:
        config_path = Path(config_path)

    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Build TOML content
    lines = ["# UUTEL Configuration", ""]

    if config.engine is not None:
        lines.append(f'engine = "{config.engine}"')
    if config.max_tokens is not None:
        lines.append(f"max_tokens = {config.max_tokens}")
    if config.temperature is not None:
        lines.append(f"temperature = {config.temperature}")
    if config.system is not None:
        lines.append(f'system = "{config.system}"')
    if config.stream is not None:
        lines.append(f"stream = {str(config.stream).lower()}")
    if config.verbose is not None:
        lines.append(f"verbose = {str(config.verbose).lower()}")

    content = "\n".join(lines) + "\n"

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.debug(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise


def validate_config(config: UUTELConfig) -> list[str]:
    """Basic configuration validation."""
    errors = []

    if config.max_tokens is not None:
        if (
            not isinstance(config.max_tokens, int)
            or config.max_tokens < 1
            or config.max_tokens > 8000
        ):
            errors.append("max_tokens must be an integer between 1 and 8000")

    if config.temperature is not None:
        if (
            not isinstance(config.temperature, int | float)
            or config.temperature < 0.0
            or config.temperature > 2.0
        ):
            errors.append("temperature must be a number between 0.0 and 2.0")

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
