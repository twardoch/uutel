#!/usr/bin/env python3
# this_file: src/uutel/__main__.py
"""UUTEL CLI - Simple Fire-based CLI for single-turn inference."""

from __future__ import annotations

import errno
import json
import math
import os
import re
import shutil
import sys
from difflib import get_close_matches
from pathlib import Path
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
from uutel.providers.claude_code import ClaudeCodeUU
from uutel.providers.cloud_code import CloudCodeUU
from uutel.providers.codex.custom_llm import CodexCustomLLM
from uutel.providers.gemini_cli import GeminiCLIUU

logger = get_logger(__name__)

# Available engines for validation
AVAILABLE_ENGINES = {
    "my-custom-llm/codex-large": "OpenAI GPT-4o via Codex session tokens (default)",
    "my-custom-llm/codex-mini": "GPT-4o-mini via Codex session tokens",
    "my-custom-llm/codex-turbo": "GPT-4 Turbo via Codex session tokens",
    "my-custom-llm/codex-fast": "GPT-3.5 Turbo via Codex session tokens",
    "my-custom-llm/codex-preview": "o1-preview via Codex session tokens",
    "uutel-codex/gpt-4o": "Direct GPT-4o using Codex provider (OpenAI API compatible)",
    "uutel-claude/claude-sonnet-4": "Claude Code Sonnet 4 via claude CLI",
    "uutel-gemini/gemini-2.5-pro": "Gemini 2.5 Pro via google-generativeai or gemini CLI",
    "uutel-cloud/gemini-2.5-pro": "Cloud Code Gemini 2.5 Pro via OAuth credentials",
}

ENGINE_ALIASES = {
    "codex": "my-custom-llm/codex-large",
    "claude": "uutel-claude/claude-sonnet-4",
    "gemini": "uutel-gemini/gemini-2.5-pro",
    "cloud": "uutel-cloud/gemini-2.5-pro",
}

CANONICAL_ENGINE_LOOKUP = {name.lower(): name for name in AVAILABLE_ENGINES}

PROVIDER_REQUIREMENTS = [
    (
        "Claude Code",
        "Requires @anthropic-ai/claude-code CLI (npm install -g @anthropic-ai/claude-code) and claude login",
    ),
    (
        "Gemini CLI",
        "Install @google/gemini-cli, run gemini login or set GOOGLE_API_KEY",
    ),
    (
        "Cloud Code",
        "Share Gemini OAuth (gemini login) or GOOGLE_API_KEY, plus CLOUD_CODE_PROJECT",
    ),
]


def _read_gcloud_default_project(home_path: Path | None = None) -> str | None:
    """Return default gcloud project id when configured locally."""

    base = home_path or Path.home()
    config_path = base / ".config" / "gcloud" / "configurations" / "config_default"

    if not config_path.exists():
        return None

    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None

    project_id: str | None = None
    in_core_section = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_core_section = line.lower() == "[core]"
            continue
        if not in_core_section:
            continue
        if line.lower().startswith("project"):
            parts = line.split("=", 1)
            if len(parts) == 2:
                candidate = parts[1].strip()
                if candidate:
                    project_id = candidate
                    break

    return project_id


def setup_providers() -> None:
    """Setup UUTEL providers with LiteLLM."""
    try:
        logger.debug("Setting up UUTEL providers...")
        codex_handler = CodexCustomLLM()
        existing_map = getattr(litellm, "custom_provider_map", [])
        preserved_entries: list[dict[str, Any]] = []
        uutel_providers = {
            "my-custom-llm",
            "uutel-codex",
            "uutel-claude",
            "uutel-gemini",
            "uutel-cloud",
        }

        normalised_map: list[Any]
        if existing_map is None:
            normalised_map = []
        elif isinstance(existing_map, dict):
            normalised_map = [
                {"provider": provider, "custom_handler": handler}
                for provider, handler in existing_map.items()
            ]
        elif isinstance(existing_map, list | tuple | set):
            normalised_map = list(existing_map)
        else:
            normalised_map = [existing_map]

        for entry in normalised_map:
            if isinstance(entry, dict):
                provider_name = entry.get("provider")
                if provider_name not in uutel_providers:
                    preserved_entries.append(entry)
            else:  # pragma: no cover - defensive guard for legacy tuples
                preserved_entries.append(entry)

        uutel_entries = [
            {"provider": "my-custom-llm", "custom_handler": codex_handler},
            {"provider": "uutel-codex", "custom_handler": codex_handler},
            {"provider": "uutel-claude", "custom_handler": ClaudeCodeUU()},
            {"provider": "uutel-gemini", "custom_handler": GeminiCLIUU()},
            {"provider": "uutel-cloud", "custom_handler": CloudCodeUU()},
        ]

        litellm.custom_provider_map = [*preserved_entries, *uutel_entries]
        logger.debug("UUTEL providers registered with LiteLLM")
    except Exception as e:
        logger.error(f"Failed to setup providers: {e}")
        print(f"Warning: Provider setup failed: {e}", file=sys.stderr)


def validate_engine(engine: str) -> str:
    """Validate engine name."""
    if not engine or not isinstance(engine, str):
        raise ValueError("Engine name is required and must be a string")

    engine_key = engine.strip()
    if not engine_key:
        raise ValueError(
            "Engine name is required and must be a non-empty string (received whitespace-only input)"
        )
    normalised_alias = engine_key.lower()
    alias_target = ENGINE_ALIASES.get(normalised_alias)
    if not alias_target:
        shorthand_target: str | None = None
        if normalised_alias.startswith("uutel/"):
            candidate = normalised_alias.split("/", 1)[1]
            shorthand_target = ENGINE_ALIASES.get(candidate)
        elif normalised_alias.startswith("uutel-"):
            candidate = normalised_alias.split("-", 1)[1]
            if "/" not in candidate:
                shorthand_target = ENGINE_ALIASES.get(candidate)
        if shorthand_target:
            alias_target = shorthand_target
    if alias_target:
        engine_key = alias_target

    canonical_target = CANONICAL_ENGINE_LOOKUP.get(engine_key.lower())
    if canonical_target:
        engine_key = canonical_target

    if engine_key not in AVAILABLE_ENGINES:
        available = "\n  ".join(f"{k}: {v}" for k, v in AVAILABLE_ENGINES.items())
        aliases = "\n  ".join(
            f"{alias} -> {target}" for alias, target in ENGINE_ALIASES.items()
        )

        candidate_map = {
            **CANONICAL_ENGINE_LOOKUP,
            **{alias.lower(): alias for alias in ENGINE_ALIASES},
        }
        close_matches = get_close_matches(
            engine_key.lower(), candidate_map.keys(), n=3, cutoff=0.6
        )
        suggestion_lines = ""
        if close_matches:
            suggestions = ", ".join(candidate_map[match] for match in close_matches)
            suggestion_lines = f"Did you mean: {suggestions}?\n\n"

        raise ValueError(
            f"Unknown engine '{engine}'.\n\n"
            f"{suggestion_lines}"
            f"Available engines:\n  {available}\n\n"
            f"Alias shortcuts:\n  {aliases}\n\n"
            f"💡 Try: uutel list_engines to see all options"
        )
    return engine_key


def validate_parameters(max_tokens: int, temperature: float) -> None:
    """Validate completion parameters."""
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise ValueError(
            f"max_tokens must be an integer between 1 and 8000, got: {max_tokens}\n"
            f"💡 Typical values: 50 (short), 500 (medium), 2000 (long)"
        )

    if max_tokens < 1 or max_tokens > 8000:
        raise ValueError(
            f"max_tokens must be an integer between 1 and 8000, got: {max_tokens}\n"
            f"💡 Typical values: 50 (short), 500 (medium), 2000 (long)"
        )

    def _temperature_error(value: Any) -> ValueError:
        return ValueError(
            f"temperature must be a finite number between 0.0 and 2.0, got: {value}\n"
            f"💡 0.0 = deterministic, 0.7 = balanced, 1.5 = creative"
        )

    if isinstance(temperature, bool) or not isinstance(temperature, int | float):
        raise _temperature_error(temperature)

    numeric_temperature = float(temperature)
    if not math.isfinite(numeric_temperature):
        raise _temperature_error(temperature)

    if not 0.0 <= numeric_temperature <= 2.0:
        raise _temperature_error(temperature)


def _extract_provider_metadata(error: Exception) -> tuple[str | None, str | None]:
    """Attempt to pull provider and model details from LiteLLM exceptions."""

    provider = getattr(error, "provider", None) or getattr(error, "llm_provider", None)
    model = getattr(error, "model", None) or getattr(error, "model_name", None)

    if hasattr(error, "__dict__"):
        data = error.__dict__
        provider = provider or data.get("provider") or data.get("llm_provider")
        model = model or data.get("model") or data.get("model_name") or data.get("llm")

    return (str(provider) if provider else None, str(model) if model else None)


def format_error_message(error: Exception, context: str = "") -> str:
    """Format error messages with basic suggestions."""

    provider, model = _extract_provider_metadata(error)
    error_msg = str(error)
    context_suffix = f" in {context}" if context else ""

    if provider or model:
        details_parts = [value for value in (provider, model) if value]
        detail_suffix = f" ({' | '.join(details_parts)})" if details_parts else ""
        return (
            f"❌ Provider error{context_suffix}{detail_suffix}: {error_msg}\n"
            "💡 Run uutel diagnostics to review provider setup before retrying"
        )

    lowered = error_msg.lower()
    if "rate limit" in lowered:
        return f"❌ Rate limit exceeded{context_suffix}\n💡 Try again in a few seconds"
    if "authentication" in lowered or "unauthorized" in lowered:
        return f"❌ Authentication failed{context_suffix}\n💡 Check your API keys"
    if "network" in lowered or "connection" in lowered:
        return f"❌ Network error{context_suffix}\n💡 Check your internet connection"
    if "timeout" in lowered:
        return f"❌ Request timeout{context_suffix}\n💡 Try reducing max_tokens"

    return f"❌ Error{context_suffix}: {error_msg}\n💡 Use --verbose for more details"


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_PRINTABLE_THRESHOLD = 32
_ALLOWED_CONTROL_CODES = {9, 10, 13}  # \t, \n, \r


def _scrub_control_sequences(message: str) -> str:
    """Remove ANSI escape codes and non-printable control characters."""

    if not message:
        return ""

    without_ansi = _ANSI_ESCAPE_RE.sub("", message)
    cleaned_chars = []
    for char in without_ansi:
        codepoint = ord(char)
        if codepoint >= _PRINTABLE_THRESHOLD or codepoint in _ALLOWED_CONTROL_CODES:
            cleaned_chars.append(char)
    return "".join(cleaned_chars)


def _safe_output(
    message: str = "",
    *,
    target: str = "stdout",
    end: str = "\n",
    flush: bool = False,
) -> None:
    """Write output while suppressing BrokenPipeError/EPIPE issues."""

    stream = sys.stdout if target == "stdout" else sys.stderr
    text = _scrub_control_sequences(str(message))
    try:
        print(text, end=end, file=stream, flush=flush)
    except BrokenPipeError:
        return
    except OSError as exc:  # pragma: no cover - defensive guard
        if getattr(exc, "errno", None) == errno.EPIPE:
            return
        raise


class UUTELCLI:
    """UUTEL Command Line Interface.

    Alias-first engines:
      - codex -> my-custom-llm/codex-large
      - claude -> uutel-claude/claude-sonnet-4
      - gemini -> uutel-gemini/gemini-2.5-pro
      - cloud -> uutel-cloud/gemini-2.5-pro

    Run `uutel list_engines` to review mappings and provider prerequisites.
    Use `uutel complete --help` or `uutel test --help` for command-specific flags.
    """

    _PLACEHOLDER_PHRASES = (
        "mock response",
        "placeholder output",
        "dummy response",
        "in a real implementation",
        "sample response",
    )
    _CANCELLATION_MESSAGE = "⚪ Operation cancelled by user"

    def __init__(self) -> None:
        """Initialize the CLI."""
        setup_providers()
        try:
            self.config = load_config()
            logger.debug("Configuration loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}")
            self.config = UUTELConfig()

    def _looks_like_placeholder(self, result: Any) -> bool:
        """Return True if completion output appears to be a canned placeholder."""

        if not isinstance(result, str):
            return False

        collapsed = result.strip().lower()
        if not collapsed:
            return True

        return any(phrase in collapsed for phrase in self._PLACEHOLDER_PHRASES)

    def _safe_print(
        self,
        message: str = "",
        *,
        target: str = "stdout",
        end: str = "\n",
        flush: bool = False,
    ) -> None:
        """Print text while swallowing BrokenPipeError cases."""

        _safe_output(message, target=target, end=end, flush=flush)

    def _check_provider_readiness(self, engine: str) -> tuple[bool, list[str]]:
        """Verify provider prerequisites before issuing live requests."""

        try:
            canonical_engine = validate_engine(engine)
        except ValueError as exc:
            return False, [str(exc)]

        guidance: list[str] = []
        ready = True

        def _get_env(name: str) -> str | None:
            raw = os.environ.get(name)
            if raw is None:
                return None
            trimmed = raw.strip()
            return trimmed or None

        codex_prefixes = ("my-custom-llm/codex", "uutel-codex/")
        gemini_prefixes = ("uutel-gemini/",)
        cloud_prefix = "uutel-cloud/"

        if canonical_engine.startswith(codex_prefixes):
            has_api_token = any(
                _get_env(var) for var in ("OPENAI_API_KEY", "OPENAI_SESSION_TOKEN")
            )
            auth_path = Path.home() / ".codex" / "auth.json"
            if not has_api_token and not auth_path.exists():
                ready = False
                guidance.append("⚠️ Codex credentials missing")
                guidance.append("💡 Run codex login or set OPENAI_API_KEY")
        elif canonical_engine.startswith("uutel-claude/"):
            if not any(shutil.which(cmd) for cmd in ("claude", "claude-code")):
                ready = False
                guidance.append("⚠️ Claude CLI not found in PATH")
                guidance.append(
                    "💡 Install @anthropic-ai/claude-code and run claude login"
                )
        elif canonical_engine.startswith(gemini_prefixes):
            has_api_key = any(
                _get_env(var)
                for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY")
            )
            has_cli = shutil.which("gemini") is not None
            if not has_api_key and not has_cli:
                ready = False
                guidance.append("⚠️ Gemini credentials not detected")
                guidance.append(
                    "💡 Run gemini login or set GOOGLE_API_KEY / GEMINI_API_KEY"
                )
        elif canonical_engine.startswith(cloud_prefix):
            project_id: str | None = None
            project_source: str | None = None
            for env_var in (
                "CLOUD_CODE_PROJECT",
                "GOOGLE_CLOUD_PROJECT",
                "GOOGLE_PROJECT",
            ):
                value = _get_env(env_var)
                if value:
                    project_id = value
                    project_source = "env"
                    break
            if not project_id:
                gcloud_project = _read_gcloud_default_project(Path.home())
                if gcloud_project:
                    project_id = gcloud_project
                    project_source = "gcloud"
            has_api_key = any(
                _get_env(var)
                for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY")
            )
            credential_paths = [
                Path.home() / ".gemini" / "oauth_creds.json",
                Path.home() / ".config" / "gemini" / "oauth_creds.json",
                Path.home() / ".google-cloud-code" / "credentials.json",
            ]
            has_oauth = any(path.exists() for path in credential_paths)

            service_account_env = _get_env("GOOGLE_APPLICATION_CREDENTIALS") or ""
            has_service_account = False
            service_account_project: str | None = None
            if service_account_env:
                service_account_path = Path(service_account_env).expanduser()
                if service_account_path.exists():
                    try:
                        raw_payload = service_account_path.read_text(encoding="utf-8")
                        payload = json.loads(raw_payload)
                        if not isinstance(payload, dict):
                            guidance.append(
                                f"⚠️ Service account file at {service_account_path} must contain a JSON object"
                            )
                        elif not payload.get("client_email"):
                            guidance.append(
                                f"⚠️ Service account file at {service_account_path} is missing client_email"
                            )
                        else:
                            has_service_account = True
                            candidate_project = (
                                payload.get("project_id")
                                or payload.get("projectId")
                                or payload.get("project")
                            )
                            if candidate_project:
                                service_account_project = str(candidate_project)
                    except UnicodeDecodeError as exc:
                        guidance.append(
                            f"⚠️ Unable to decode service account file at {service_account_path}: {exc.reason}"
                        )
                    except json.JSONDecodeError as exc:
                        guidance.append(
                            f"⚠️ Invalid service account JSON at {service_account_path}: {exc.msg}"
                        )
                    except OSError as exc:
                        guidance.append(
                            f"⚠️ Unable to read service account file at {service_account_path}: {exc.strerror}"
                        )
                else:
                    guidance.append(
                        f"⚠️ Service account file not found at {service_account_path}"
                    )

            if not project_id and service_account_project:
                project_id = service_account_project
                project_source = "service_account"

            if not project_id:
                ready = False
                guidance.append("⚠️ Cloud Code project ID missing")
                guidance.append(
                    "💡 Set CLOUD_CODE_PROJECT or supply --project_id for Cloud Code engines"
                )
            elif project_source == "gcloud":
                guidance.append(f"ℹ️ Using gcloud config project '{project_id}'")
            elif project_source == "service_account":
                guidance.append(f"ℹ️ Using service account project '{project_id}'")
            if not (has_api_key or has_oauth or has_service_account):
                ready = False
                guidance.append("⚠️ Cloud Code credentials not detected")
                guidance.append(
                    "💡 Run gemini login to create oauth_creds.json or set GOOGLE_API_KEY"
                )

        return ready, guidance

    def _format_empty_response_message(self, engine: str) -> str:
        """Return a guidance banner for empty LiteLLM responses."""

        return (
            f"❌ Received empty response from engine '{engine}'.\n"
            "💡 Enable --verbose to inspect LiteLLM logs before retrying."
        )

    def _normalise_message_content(self, content: Any) -> str:
        """Normalise structured message content into a plain string."""

        def _append_parts(result: list[str], part: Any) -> None:
            if part is None:
                return
            if isinstance(part, str):
                result.append(part)
                return
            if isinstance(part, dict):
                part_type = part.get("type")
                if part_type in {"tool_call", "tool_calls"}:
                    return
                if "text" in part:
                    _append_parts(result, part.get("text"))
                    return
                if "content" in part:
                    _append_parts(result, part.get("content"))
                    return
            if isinstance(part, list):
                for item in part:
                    _append_parts(result, item)
                return
            text_value = str(part)
            if text_value.strip():
                result.append(text_value)

        parts: list[str] = []
        _append_parts(parts, content)
        combined = "".join(parts)
        cleaned = _scrub_control_sequences(combined)
        return cleaned if cleaned.strip() else ""

    def _extract_completion_text(self, response: Any) -> str | None:
        """Extract assistant text from a LiteLLM completion response."""

        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")

        if not choices:
            return None

        for choice in choices:
            message = getattr(choice, "message", None)
            if message is None and isinstance(choice, dict):
                message = choice.get("message")

            content = getattr(message, "content", None) if message is not None else None
            if content is None and isinstance(message, dict):
                content = message.get("content")

            text_value = self._normalise_message_content(content)
            if text_value:
                return text_value

        return None

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
        """Complete a prompt using the configured engine.

        Defaults to the codex alias (my-custom-llm/codex-large).
        Use --engine <alias> to target claude, gemini, or cloud from `uutel list_engines`.
        Enable --stream true to print incremental output when providers support streaming.
        """
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
        max_tokens = merged_args.get("max_tokens")
        if max_tokens is None:
            max_tokens = 500
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
            self._safe_print(error_msg, target="stderr")
            return error_msg

        # Configure logging
        import logging

        previous_env = os.environ.get("LITELLM_LOG")
        uutel_logger = logging.getLogger("uutel")
        previous_uutel_level = uutel_logger.level
        previous_cli_level = logger.level

        if verbose:
            os.environ["LITELLM_LOG"] = "DEBUG"
            uutel_logger.setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
            self._safe_print("🔧 Verbose mode enabled", target="stderr")
        else:
            uutel_logger.setLevel(logging.WARNING)
            logger.setLevel(logging.WARNING)

        try:
            ready, guidance = self._check_provider_readiness(engine)
            if not ready:
                guidance_lines = (
                    list(guidance)
                    if guidance
                    else [
                        "⚠️ Provider prerequisites missing",
                        "💡 Review engine credentials before retrying",
                    ]
                )
                if not any("uutel diagnostics" in line for line in guidance_lines):
                    guidance_lines.append(
                        "💡 Run uutel diagnostics to review provider setup before retrying"
                    )
                for line in guidance_lines:
                    self._safe_print(line, target="stderr")
                return "\n".join(guidance_lines)

            # Validate parameters
            engine = validate_engine(engine)
            validate_parameters(max_tokens, temperature)

            if verbose:
                self._safe_print(f"🎯 Using engine: {engine}", target="stderr")
                self._safe_print(
                    f"⚙️  Parameters: max_tokens={max_tokens}, temperature={temperature}",
                    target="stderr",
                )

            # Build messages
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            if stream:
                if verbose:
                    self._safe_print(
                        "📡 Starting streaming response...", target="stderr"
                    )
                result = self._stream_completion(
                    messages, engine, max_tokens, temperature
                )
                if self._looks_like_placeholder(result):
                    placeholder_message = (
                        f"❌ Placeholder output detected for engine '{engine}'."
                        "\n💡 Use a live provider or refresh your credentials before retrying."
                    )
                    self._safe_print(placeholder_message, target="stderr")
                    return placeholder_message
                return result
            else:
                if verbose:
                    self._safe_print("⏳ Generating completion...", target="stderr")

                response = litellm.completion(
                    model=engine,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                extracted = self._extract_completion_text(response)
                if extracted is None:
                    empty_message = self._format_empty_response_message(engine)
                    self._safe_print(empty_message, target="stderr")
                    return empty_message

                result = extracted

                if self._looks_like_placeholder(result):
                    placeholder_message = (
                        f"❌ Placeholder output detected for engine '{engine}'."
                        "\n💡 Use a live provider or refresh your credentials before retrying."
                    )
                    self._safe_print(placeholder_message, target="stderr")
                    return placeholder_message

                self._safe_print(result)

                if verbose:
                    self._safe_print(
                        f"✅ Completion successful ({len(result)} characters)",
                        target="stderr",
                    )
                return result

        except KeyboardInterrupt:
            cancellation_message = self._CANCELLATION_MESSAGE
            self._safe_print(cancellation_message, target="stderr")
            return cancellation_message
        except ValueError as e:
            error_msg = str(e)
            self._safe_print(error_msg, target="stderr")
            return error_msg
        except Exception as e:
            error_msg = format_error_message(e, "completion")
            self._safe_print(error_msg, target="stderr")
            return error_msg
        finally:
            if previous_env is None:
                os.environ.pop("LITELLM_LOG", None)
            else:
                os.environ["LITELLM_LOG"] = previous_env

            uutel_logger.setLevel(previous_uutel_level)
            logger.setLevel(previous_cli_level)

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

            parts: list[str] = []
            for chunk in response:
                choices = getattr(chunk, "choices", None)
                if choices is None and isinstance(chunk, dict):
                    choices = chunk.get("choices")
                if not choices:
                    continue

                for choice in choices:
                    delta = getattr(choice, "delta", None)
                    if delta is None and isinstance(choice, dict):
                        delta = choice.get("delta")
                    if delta is None:
                        continue

                    content = (
                        getattr(delta, "content", None) if delta is not None else None
                    )
                    if content is None and isinstance(delta, dict):
                        content = delta.get("content")

                    text_piece = self._normalise_message_content(content)
                    if not text_piece:
                        continue

                    self._safe_print(text_piece, end="", flush=True)
                    parts.append(text_piece)

            self._safe_print()
            if not parts:
                empty_message = self._format_empty_response_message(engine)
                self._safe_print(empty_message, target="stderr")
                return empty_message
            return "".join(parts)

        except KeyboardInterrupt:
            cancellation_message = self._CANCELLATION_MESSAGE
            self._safe_print(cancellation_message, target="stderr")
            return cancellation_message
        except Exception as e:
            error_msg = format_error_message(e, "streaming")
            self._safe_print(error_msg, target="stderr")
            return error_msg

    def list_engines(self) -> None:
        """List available engines/providers."""
        self._safe_print("🔧 UUTEL Available Engines")
        self._safe_print("=" * 50)
        self._safe_print()

        for engine, description in AVAILABLE_ENGINES.items():
            self._safe_print(f"  {engine}")
            self._safe_print(f"    {description}")
        self._safe_print()

        self._safe_print("📝 Usage Examples:")
        self._safe_print('  uutel complete --prompt "Write a sorter" --engine codex')
        self._safe_print('  uutel complete --prompt "Say hello" --engine claude')
        self._safe_print(
            '  uutel complete --prompt "Summarise Gemini API" --engine gemini'
        )
        self._safe_print(
            '  uutel complete --prompt "Deployment checklist" --engine cloud'
        )
        self._safe_print("  uutel test --engine codex")
        self._safe_print("  uutel test --engine claude")
        self._safe_print()
        self._safe_print("🔐 Provider Requirements:")
        for name, guidance in PROVIDER_REQUIREMENTS:
            self._safe_print(f"  {name}: {guidance}")
        self._safe_print()
        self._safe_print("Aliases:")
        for alias, target in ENGINE_ALIASES.items():
            self._safe_print(f"  {alias} -> {target}")

    def test(
        self, engine: str = "my-custom-llm/codex-large", verbose: bool = True
    ) -> str:
        """Quick readiness probe for provider aliases.

        Validates codex, claude, gemini, or cloud using validate_engine before running tests.
        Displays provider prerequisites when credentials or CLIs are missing.
        """
        try:
            engine = validate_engine(engine)
            self._safe_print(f"🧪 Testing engine: {engine}")
            self._safe_print("─" * 40)

            ready, guidance = self._check_provider_readiness(engine)
            if not ready:
                guidance_lines = (
                    list(guidance)
                    if guidance
                    else [
                        "⚠️ Provider prerequisites missing",
                        "💡 Review engine credentials before retrying",
                    ]
                )
                if not any("uutel diagnostics" in line for line in guidance_lines):
                    guidance_lines.append(
                        "💡 Run uutel diagnostics to review provider setup before retrying"
                    )
                for line in guidance_lines:
                    self._safe_print(line, target="stderr")
                return "\n".join(guidance_lines)

            result = self.complete(
                prompt="Hello! Can you respond with a brief greeting?",
                engine=engine,
                max_tokens=50,
                verbose=verbose,
            )

            if self._looks_like_placeholder(result):
                placeholder_message = (
                    f"❌ Placeholder output detected for engine '{engine}'."
                    "\n💡 Use a live provider or refresh your credentials before retrying."
                )
                self._safe_print(placeholder_message, target="stderr")
                result = placeholder_message

            if result and not result.startswith("❌"):
                self._safe_print("─" * 40)
                self._safe_print("✅ Test completed successfully!")
                self._safe_print(f"💡 Engine '{engine}' is working correctly")
            else:
                self._safe_print("─" * 40)
                self._safe_print("❌ Test failed - see error details above")

            return result

        except KeyboardInterrupt:
            cancellation_message = self._CANCELLATION_MESSAGE
            self._safe_print(cancellation_message, target="stderr")
            return cancellation_message
        except Exception as e:
            error_msg = format_error_message(e, "testing")
            self._safe_print(error_msg, target="stderr")
            return error_msg

    def diagnostics(self) -> str:
        """Summarise provider readiness across registered aliases."""

        self._safe_print("🩺 UUTEL Diagnostics")
        self._safe_print("─" * 40)

        ready_count = 0
        issue_count = 0

        for alias, engine in ENGINE_ALIASES.items():
            ready, guidance = self._check_provider_readiness(engine)
            status_icon = "✅" if ready else "⚠️"
            self._safe_print(f"{status_icon} {alias} ({engine})")
            if guidance:
                for hint in guidance:
                    self._safe_print(f"   {hint}")
            if ready:
                ready_count += 1
            else:
                issue_count += 1

        summary = (
            f"Diagnostics complete: {ready_count} ready, {issue_count} need attention"
        )
        self._safe_print(summary)
        return summary

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

            try:
                self.config = load_config()
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning(f"Failed to refresh configuration before show: {exc}")

            self._safe_print(f"📁 Configuration file: {config_path}")
            self._safe_print("📋 Current settings:")

            if self.config.engine:
                self._safe_print(f"  engine = {self.config.engine}")
            max_tokens_value = self.config.max_tokens
            max_tokens_display = (
                str(max_tokens_value)
                if max_tokens_value is not None
                else "default (500)"
            )
            self._safe_print(f"  max_tokens = {max_tokens_display}")

            temperature_value = self.config.temperature
            temperature_display = (
                str(temperature_value)
                if temperature_value is not None
                else "default (0.7)"
            )
            self._safe_print(f"  temperature = {temperature_display}")
            if self.config.system:
                self._safe_print(f"  system = {self.config.system}")
            stream_display = (
                "default (False)"
                if self.config.stream is None
                else str(self.config.stream)
            )
            verbose_display = (
                "default (False)"
                if self.config.verbose is None
                else str(self.config.verbose)
            )
            self._safe_print(f"  stream = {stream_display}")
            self._safe_print(f"  verbose = {verbose_display}")

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

            try:
                self.config = load_config()
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning(f"Failed to refresh configuration after init: {exc}")

            return (
                f"✅ Created default configuration file: {config_path}\n"
                f"💡 Edit the file or use 'uutel config set' to customize settings"
            )

        except Exception as e:
            return f"❌ Failed to initialize configuration: {e}"

    def _config_set(self, **kwargs: Any) -> str:
        """Set configuration values with CLI-friendly coercion."""
        try:
            allowed_keys = {
                "engine",
                "max_tokens",
                "temperature",
                "system",
                "stream",
                "verbose",
            }
            unknown_fields = sorted(key for key in kwargs if key not in allowed_keys)
            if unknown_fields:
                allowed_display = ", ".join(sorted(allowed_keys))
                return (
                    f"❌ Unknown configuration fields: {', '.join(unknown_fields)}\n"
                    f"💡 Allowed keys: {allowed_display}"
                )

            current = self.config
            manual_errors: list[str] = []

            def _provided(key: str) -> bool:
                return key in kwargs

            def _coerce_int(raw: Any) -> int | None:
                if raw is None:
                    return None
                if isinstance(raw, bool):
                    raise ValueError("max_tokens must be an integer between 1 and 8000")
                if isinstance(raw, int):
                    return raw
                if isinstance(raw, str):
                    candidate = raw.strip()
                    if candidate.lower() in {"", "none", "null", "default"}:
                        return None
                    try:
                        return int(candidate)
                    except ValueError as exc:  # pragma: no cover - error flow exercised via ValueError path
                        raise ValueError(
                            "max_tokens must be an integer between 1 and 8000"
                        ) from exc
                raise ValueError("max_tokens must be an integer between 1 and 8000")

            def _coerce_float(raw: Any) -> float | None:
                if raw is None:
                    return None
                if isinstance(raw, bool):
                    raise ValueError("temperature must be a number between 0.0 and 2.0")
                if isinstance(raw, int | float):
                    return float(raw)
                if isinstance(raw, str):
                    candidate = raw.strip()
                    if candidate.lower() in {"", "none", "null", "default"}:
                        return None
                    try:
                        return float(candidate)
                    except ValueError as exc:  # pragma: no cover - error flow exercised via ValueError path
                        raise ValueError(
                            "temperature must be a number between 0.0 and 2.0"
                        ) from exc
                raise ValueError("temperature must be a number between 0.0 and 2.0")

            def _coerce_bool(raw: Any, field: str) -> bool | None:
                if raw is None:
                    return None
                if isinstance(raw, bool):
                    return raw
                if isinstance(raw, str):
                    candidate = raw.strip().lower()
                    if candidate in {"", "none", "null", "default"}:
                        return None
                    if candidate in {"true", "1", "yes", "y", "on"}:
                        return True
                    if candidate in {"false", "0", "no", "n", "off"}:
                        return False
                raise ValueError(f"{field} must be a boolean value")

            def _coerce_system(raw: Any) -> str | None:
                if raw is None:
                    return None
                if isinstance(raw, str):
                    candidate = raw.strip()
                    if candidate == "" or candidate.lower() in {
                        "none",
                        "null",
                        "default",
                    }:
                        return None
                    return candidate
                return raw

            def _record_error(message: str) -> None:
                if message not in manual_errors:
                    manual_errors.append(message)

            def _format_invalid(messages: list[str]) -> str:
                bullet_list = "\n".join(f"  • {error}" for error in messages)
                return f"❌ Invalid configuration:\n{bullet_list}"

            engine_value = current.engine
            if _provided("engine"):
                raw_engine = kwargs["engine"]
                if raw_engine is None:
                    engine_value = None
                elif isinstance(raw_engine, str) and raw_engine.strip().lower() in {
                    "",
                    "none",
                    "null",
                    "default",
                }:
                    engine_value = None
                else:
                    try:
                        engine_value = validate_engine(raw_engine)
                    except ValueError as exc:
                        return str(exc)

            if _provided("max_tokens"):
                try:
                    max_tokens_value = _coerce_int(kwargs["max_tokens"])
                except ValueError as exc:
                    _record_error(str(exc))
                    max_tokens_value = current.max_tokens
            else:
                max_tokens_value = current.max_tokens

            if _provided("temperature"):
                try:
                    temperature_value = _coerce_float(kwargs["temperature"])
                except ValueError as exc:
                    _record_error(str(exc))
                    temperature_value = current.temperature
            else:
                temperature_value = current.temperature

            if _provided("stream"):
                try:
                    stream_value = _coerce_bool(kwargs["stream"], "stream")
                except ValueError as exc:
                    _record_error(str(exc))
                    stream_value = current.stream
            else:
                stream_value = current.stream

            if _provided("verbose"):
                try:
                    verbose_value = _coerce_bool(kwargs["verbose"], "verbose")
                except ValueError as exc:
                    _record_error(str(exc))
                    verbose_value = current.verbose
            else:
                verbose_value = current.verbose

            system_value = (
                _coerce_system(kwargs["system"])
                if _provided("system")
                else current.system
            )

            if manual_errors:
                return _format_invalid(manual_errors)

            updated_config = UUTELConfig(
                engine=engine_value,
                max_tokens=max_tokens_value,
                temperature=temperature_value,
                system=system_value,
                stream=stream_value,
                verbose=verbose_value,
            )

            errors = validate_config(updated_config)
            if errors:
                return _format_invalid(errors)

            if updated_config == current:
                return "ℹ️ No configuration changes provided; existing settings kept."

            save_config(updated_config)
            self.config = updated_config

            changes: list[str] = []
            for field in (
                "engine",
                "max_tokens",
                "temperature",
                "system",
                "stream",
                "verbose",
            ):
                before = getattr(current, field)
                after = getattr(updated_config, field)
                if before == after:
                    continue
                display = "default" if after is None else after
                changes.append(f"{field} = {display}")

            change_summary = ", ".join(changes) if changes else "updated values"
            return f"✅ Configuration updated: {change_summary}\n💡 Use 'uutel config show' to see all settings"

        except Exception as e:
            return f"❌ Failed to set configuration: {e}"

    def _config_get(self, key: str) -> str:
        """Get specific configuration value."""
        try:
            try:
                self.config = load_config()
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning(f"Failed to refresh configuration before get: {exc}")
            normalized_key = key.strip()
            value = getattr(self.config, normalized_key, None)
            if value is None:
                return f"❌ Configuration key '{normalized_key}' not set or invalid"
            return str(value)
        except Exception as e:
            return f"❌ Failed to get configuration value: {e}"


def main() -> None:
    """Main entry point for the CLI."""

    try:
        fire.Fire(UUTELCLI)
    except KeyboardInterrupt:
        _safe_output(UUTELCLI._CANCELLATION_MESSAGE, target="stderr")
    except BrokenPipeError:
        return
    except OSError as exc:  # pragma: no cover - defensive fallback
        if getattr(exc, "errno", None) == errno.EPIPE:
            return
        raise


if __name__ == "__main__":
    main()
