#!/usr/bin/env python3
# this_file: src/uutel/__main__.py
"""UUTEL CLI - Simple Fire-based CLI for single-turn inference."""

from __future__ import annotations

import sys
from typing import Any

import fire
import litellm

from uutel.core.config import (
    UUTELConfig,
    create_default_config,
    load_config,
    save_config,
    validate_config,
)
from uutel.core.logging_config import get_logger
from uutel.providers.codex.custom_llm import CodexCustomLLM

logger = get_logger(__name__)

# Available engines for validation
AVAILABLE_ENGINES = {
    "my-custom-llm/codex-large": "Large Codex model (default)",
    "my-custom-llm/codex-mini": "Mini Codex model - faster responses",
    "my-custom-llm/codex-turbo": "Turbo Codex model - balanced speed/quality",
    "my-custom-llm/codex-fast": "Fast Codex model - quick responses",
    "my-custom-llm/codex-preview": "Preview Codex model - latest features",
}


def setup_providers() -> None:
    """Setup UUTEL providers with LiteLLM."""
    try:
        logger.debug("Setting up UUTEL providers...")
        litellm.custom_provider_map = [
            {"provider": "my-custom-llm", "custom_handler": CodexCustomLLM()},
        ]
        logger.debug("UUTEL providers registered with LiteLLM")
    except Exception as e:
        logger.error(f"Failed to setup providers: {e}")
        print(f"Warning: Provider setup failed: {e}", file=sys.stderr)


def validate_engine(engine: str) -> str:
    """Validate engine name."""
    if not engine or not isinstance(engine, str):
        raise ValueError("Engine name is required and must be a string")

    if engine not in AVAILABLE_ENGINES:
        available = "\n  ".join(f"{k}: {v}" for k, v in AVAILABLE_ENGINES.items())
        raise ValueError(
            f"Unknown engine '{engine}'.\n\n"
            f"Available engines:\n  {available}\n\n"
            f"💡 Try: uutel list_engines to see all options"
        )
    return engine


def validate_parameters(max_tokens: int, temperature: float) -> None:
    """Validate completion parameters."""
    if not isinstance(max_tokens, int) or max_tokens < 1 or max_tokens > 8000:
        raise ValueError(
            f"max_tokens must be an integer between 1 and 8000, got: {max_tokens}\n"
            f"💡 Typical values: 50 (short), 500 (medium), 2000 (long)"
        )

    if (
        not isinstance(temperature, int | float)
        or temperature < 0.0
        or temperature > 2.0
    ):
        raise ValueError(
            f"temperature must be a number between 0.0 and 2.0, got: {temperature}\n"
            f"💡 0.0 = deterministic, 0.7 = balanced, 1.5 = creative"
        )


def format_error_message(error: Exception, context: str = "") -> str:
    """Format error messages with basic suggestions."""
    error_msg = str(error)

    if "rate limit" in error_msg.lower():
        return f"❌ Rate limit exceeded{f' in {context}' if context else ''}\n💡 Try again in a few seconds"
    elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
        return f"❌ Authentication failed{f' in {context}' if context else ''}\n💡 Check your API keys"
    elif "network" in error_msg.lower() or "connection" in error_msg.lower():
        return f"❌ Network error{f' in {context}' if context else ''}\n💡 Check your internet connection"
    elif "timeout" in error_msg.lower():
        return f"❌ Request timeout{f' in {context}' if context else ''}\n💡 Try reducing max_tokens"
    else:
        return f"❌ Error{f' in {context}' if context else ''}: {error_msg}\n💡 Use --verbose for more details"


class UUTELCLI:
    """UUTEL Command Line Interface."""

    def __init__(self) -> None:
        """Initialize the CLI."""
        setup_providers()
        try:
            self.config = load_config()
            logger.debug("Configuration loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}")
            self.config = UUTELConfig()

    def complete(
        self,
        prompt: str,
        engine: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        stream: bool | None = None,
        verbose: bool | None = None,
    ) -> str:
        """Complete a prompt using the specified engine."""
        # Merge configuration with CLI arguments
        merged_args = self.config.merge_with_args(
            engine=engine,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            stream=stream,
            verbose=verbose,
        )

        # Apply merged values with fallback defaults
        engine = merged_args.get("engine") or "my-custom-llm/codex-large"
        max_tokens = merged_args.get("max_tokens") or 500
        temperature = (
            merged_args.get("temperature")
            if merged_args.get("temperature") is not None
            else 0.7
        )
        system = merged_args.get("system")
        stream = merged_args.get("stream") or False
        verbose = merged_args.get("verbose") or False

        if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            error_msg = '❌ Prompt is required and cannot be empty\n💡 Try: uutel complete "Your prompt here"'
            print(error_msg, file=sys.stderr)
            return error_msg

        # Configure logging
        if verbose:
            import logging

            logging.getLogger("uutel").setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
            litellm.set_verbose = True
            print("🔧 Verbose mode enabled", file=sys.stderr)
        else:
            import logging

            logging.getLogger("uutel").setLevel(logging.WARNING)
            logger.setLevel(logging.WARNING)
            litellm.set_verbose = False

        try:
            # Validate parameters
            engine = validate_engine(engine)
            validate_parameters(max_tokens, temperature)

            if verbose:
                print(f"🎯 Using engine: {engine}", file=sys.stderr)
                print(
                    f"⚙️  Parameters: max_tokens={max_tokens}, temperature={temperature}",
                    file=sys.stderr,
                )

            # Build messages
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            if stream:
                if verbose:
                    print("📡 Starting streaming response...", file=sys.stderr)
                return self._stream_completion(
                    messages, engine, max_tokens, temperature
                )
            else:
                if verbose:
                    print("⏳ Generating completion...", file=sys.stderr)

                response = litellm.completion(
                    model=engine,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                result = response.choices[0].message.content
                print(result)

                if verbose:
                    print(
                        f"✅ Completion successful ({len(result)} characters)",
                        file=sys.stderr,
                    )
                return result

        except ValueError as e:
            error_msg = str(e)
            print(error_msg, file=sys.stderr)
            return error_msg
        except Exception as e:
            error_msg = format_error_message(e, "completion")
            print(error_msg, file=sys.stderr)
            return error_msg

    def _stream_completion(
        self,
        messages: list[dict[str, Any]],
        engine: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Stream a completion response."""
        try:
            response = litellm.completion(
                model=engine,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

            full_content = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    print(content, end="", flush=True)

            print()  # Add final newline
            return full_content

        except Exception as e:
            error_msg = format_error_message(e, "streaming")
            print(error_msg, file=sys.stderr)
            return error_msg

    def list_engines(self) -> None:
        """List available engines/providers."""
        print("🔧 UUTEL Available Engines")
        print("=" * 50)
        print()

        for engine, description in AVAILABLE_ENGINES.items():
            print(f"  {engine}")
            print(f"    {description}")
            print()

        print("📝 Usage Examples:")
        print('  uutel complete "Hello" --engine my-custom-llm/codex-mini')
        print("  uutel test --engine my-custom-llm/codex-fast")

    def test(
        self, engine: str = "my-custom-llm/codex-large", verbose: bool = True
    ) -> str:
        """Test an engine with a simple prompt."""
        try:
            engine = validate_engine(engine)
            print(f"🧪 Testing engine: {engine}")
            print("─" * 40)

            result = self.complete(
                prompt="Hello! Can you respond with a brief greeting?",
                engine=engine,
                max_tokens=50,
                verbose=verbose,
            )

            if result and not result.startswith("❌"):
                print("─" * 40)
                print("✅ Test completed successfully!")
                print(f"💡 Engine '{engine}' is working correctly")
            else:
                print("─" * 40)
                print("❌ Test failed - see error details above")

            return result

        except Exception as e:
            error_msg = format_error_message(e, "testing")
            print(error_msg, file=sys.stderr)
            return error_msg

    def config(self, action: str = "show", **kwargs: Any) -> str:
        """Manage UUTEL configuration file."""
        try:
            if action == "show":
                return self._config_show()
            elif action == "init":
                return self._config_init()
            elif action == "set":
                return self._config_set(**kwargs)
            elif action.startswith("get"):
                key = action[4:] if action != "get" else kwargs.get("key", "")
                if not key:
                    return "❌ Key name required\n💡 Usage: uutel config get engine"
                return self._config_get(key)
            else:
                return (
                    "❌ Invalid config action\n"
                    "💡 Available actions: show, init, set, get\n"
                    "💡 Try: uutel config show"
                )

        except Exception as e:
            logger.error(f"Configuration error: {e}")
            return f"❌ Configuration operation failed: {e}"

    def _config_show(self) -> str:
        """Show current configuration."""
        try:
            from uutel.core.config import get_config_path

            config_path = get_config_path()

            if not config_path.exists():
                return "📝 No configuration file found\n💡 Create one with: uutel config init"

            print(f"📁 Configuration file: {config_path}")
            print("📋 Current settings:")

            if self.config.engine:
                print(f"  engine = {self.config.engine}")
            if self.config.max_tokens:
                print(f"  max_tokens = {self.config.max_tokens}")
            if self.config.temperature is not None:
                print(f"  temperature = {self.config.temperature}")
            if self.config.system:
                print(f"  system = {self.config.system}")
            print(f"  stream = {self.config.stream}")
            print(f"  verbose = {self.config.verbose}")

            return "✅ Configuration displayed"

        except Exception as e:
            return f"❌ Failed to show configuration: {e}"

    def _config_init(self) -> str:
        """Initialize default configuration file."""
        try:
            from uutel.core.config import get_config_path

            config_path = get_config_path()

            if config_path.exists():
                return f"❌ Configuration file already exists: {config_path}\n💡 Use 'uutel config show' to view"

            default_content = create_default_config()
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, "w", encoding="utf-8") as f:
                f.write(default_content)

            return (
                f"✅ Created default configuration file: {config_path}\n"
                f"💡 Edit the file or use 'uutel config set' to customize settings"
            )

        except Exception as e:
            return f"❌ Failed to initialize configuration: {e}"

    def _config_set(self, **kwargs: Any) -> str:
        """Set configuration values."""
        try:
            updated_config = UUTELConfig(
                engine=kwargs.get("engine") or self.config.engine,
                max_tokens=kwargs.get("max_tokens") or self.config.max_tokens,
                temperature=kwargs.get("temperature")
                if kwargs.get("temperature") is not None
                else self.config.temperature,
                system=kwargs.get("system") or self.config.system,
                stream=kwargs.get("stream")
                if kwargs.get("stream") is not None
                else self.config.stream,
                verbose=kwargs.get("verbose")
                if kwargs.get("verbose") is not None
                else self.config.verbose,
            )

            errors = validate_config(updated_config)
            if errors:
                return "❌ Invalid configuration:\n" + "\n".join(
                    f"  • {error}" for error in errors
                )

            save_config(updated_config)
            self.config = updated_config

            changed = [
                f"{key} = {value}" for key, value in kwargs.items() if value is not None
            ]
            return f"✅ Configuration updated: {', '.join(changed)}\n💡 Use 'uutel config show' to see all settings"

        except Exception as e:
            return f"❌ Failed to set configuration: {e}"

    def _config_get(self, key: str) -> str:
        """Get specific configuration value."""
        try:
            value = getattr(self.config, key, None)
            if value is None:
                return f"❌ Configuration key '{key}' not set or invalid"
            return str(value)
        except Exception as e:
            return f"❌ Failed to get configuration value: {e}"


def main() -> None:
    """Main entry point for the CLI."""
    fire.Fire(UUTELCLI)


if __name__ == "__main__":
    main()
