# this_file: tests/test_cli.py
"""Comprehensive test suite for UUTEL CLI functionality.

This module tests all CLI commands, parameter validation, error handling,
and integration with providers to ensure CLI reliability.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import litellm
import pytest
import tomli_w
from examples import basic_usage

import uutel.__main__ as cli_module
from uutel.__main__ import UUTELCLI, _read_gcloud_default_project, main, setup_providers
from uutel.core.config import UUTELConfig


class TestUUTELCLI:
    """Test the main UUTELCLI class and its functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.cli = UUTELCLI()

    def test_cli_initialization(self) -> None:
        """Test CLI initializes correctly and sets up providers."""
        assert self.cli is not None
        # Verify providers are set up during initialization
        # This is implicitly tested by successful CLI creation

    def test_list_engines_outputs_sorted_sections(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Engine and alias listings should be alphabetically ordered for deterministic output."""

        self.cli.list_engines()
        captured = capsys.readouterr()

        lines = captured.out.splitlines()
        engine_keys = sorted(cli_module.AVAILABLE_ENGINES.keys())
        expected_engine_lines = [f"  {engine}" for engine in engine_keys]
        engine_candidates = {f"  {engine}" for engine in cli_module.AVAILABLE_ENGINES}
        printed_engine_lines = [line for line in lines if line in engine_candidates]

        assert printed_engine_lines == expected_engine_lines, (
            "Engine list should be alphabetically ordered"
        )

        alias_items = sorted(cli_module.ENGINE_ALIASES.items())
        expected_alias_lines = [
            f"  {alias} -> {target}" for alias, target in alias_items
        ]
        alias_candidates = {
            f"  {alias} -> {target}"
            for alias, target in cli_module.ENGINE_ALIASES.items()
        }
        printed_alias_lines = [line for line in lines if line in alias_candidates]

        assert printed_alias_lines == expected_alias_lines, (
            "Alias list should be alphabetically ordered"
        )

    def test_list_engines_provider_requirements_cover_all_entries(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The provider requirements block should enumerate every configured entry in order."""

        self.cli.list_engines()
        captured = capsys.readouterr()

        lines = captured.out.splitlines()
        try:
            start_index = lines.index("🔐 Provider Requirements:") + 1
        except ValueError:  # pragma: no cover - diagnostic clarity
            pytest.fail("Provider requirements header missing from list_engines output")

        requirement_lines: list[str] = []
        for line in lines[start_index:]:
            if not line.strip():
                break
            requirement_lines.append(line)

        expected_lines = [
            f"  {name}: {guidance}"
            for name, guidance in cli_module.PROVIDER_REQUIREMENTS
        ]

        assert requirement_lines == expected_lines, (
            "Provider requirements output drifted from configured guidance list"
        )

    def test_list_engines_usage_includes_recorded_live_hints(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Usage examples should surface every recorded fixture live hint."""

        self.cli.list_engines()
        captured = capsys.readouterr()

        usage_block = captured.out.split("🔐 Provider Requirements:", 1)[0]

        missing: list[str] = []
        for fixture in basic_usage.RECORDED_FIXTURES:
            hint = fixture["live_hint"]
            expected_line = f"  {hint}"
            if expected_line not in usage_block:
                missing.append(hint)

        assert not missing, (
            "list_engines usage examples missing recorded live hints: "
            + ", ".join(missing)
        )

    def test_list_engines_usage_reflects_runtime_recorded_hints(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Usage block should reflect current RECORDED_FIXTURES without manual updates."""

        patched_fixtures = list(basic_usage.RECORDED_FIXTURES)
        patched_fixtures.append(
            {
                "label": "Patched Codex",
                "key": "codex-patched",
                "engine": "codex",
                "prompt": "Check sanitisation",
                "path": patched_fixtures[0]["path"],
                "live_hint": 'uutel complete --prompt "Check sanitisation" --engine codex',
            }
        )
        monkeypatch.setattr(
            basic_usage, "RECORDED_FIXTURES", patched_fixtures, raising=False
        )

        self.cli.list_engines()
        captured = capsys.readouterr()

        usage_block = captured.out.split("🔐 Provider Requirements:", 1)[0]

        assert (
            '  uutel complete --prompt "Check sanitisation" --engine codex'
            in usage_block
        ), "Usage block should include dynamically patched live hint"

    def test_list_engines_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test list_engines command output."""

        self.cli.list_engines()
        captured = capsys.readouterr()

        # Check that engines are listed with new format
        assert "UUTEL Available Engines" in captured.out
        assert "my-custom-llm/codex-large" in captured.out
        assert "my-custom-llm/codex-mini" in captured.out
        assert "my-custom-llm/codex-turbo" in captured.out
        assert "my-custom-llm/codex-fast" in captured.out
        assert "my-custom-llm/codex-preview" in captured.out
        assert "uutel-claude/claude-sonnet-4" in captured.out
        assert "uutel-gemini/gemini-2.5-pro" in captured.out
        assert "uutel-cloud/gemini-2.5-pro" in captured.out

        # Check descriptions are present
        assert "OpenAI GPT-4o via Codex session tokens" in captured.out
        assert "GPT-4o-mini via Codex session tokens" in captured.out
        assert "GPT-4 Turbo via Codex session tokens" in captured.out
        assert "GPT-3.5 Turbo via Codex session tokens" in captured.out
        assert "o1-preview via Codex session tokens" in captured.out
        assert "Usage Examples" in captured.out
        assert "Provider Requirements" in captured.out
        assert "Claude Code" in captured.out
        assert "Gemini CLI" in captured.out

        # Usage examples should promote alias-first commands
        assert (
            'uutel complete --prompt "Write a sorter" --engine codex' in captured.out
        ), "Usage examples should advertise codex alias"
        assert 'uutel complete --prompt "Say hello" --engine claude' in captured.out, (
            "Usage examples should advertise claude alias"
        )
        assert "uutel test --engine codex" in captured.out, (
            "Test usage should show codex alias"
        )
        assert "uutel test --engine claude" in captured.out, (
            "Test usage should show claude alias"
        )
        assert "uutel test --engine gemini" in captured.out, (
            "Test usage should show gemini alias"
        )
        assert "uutel test --engine cloud" in captured.out, (
            "Test usage should show cloud alias"
        )

        # Alias summary is shown for quick selection
        assert "Aliases:" in captured.out
        assert "codex -> my-custom-llm/codex-large" in captured.out
        assert "codex-large -> my-custom-llm/codex-large" in captured.out
        assert "openai-codex -> my-custom-llm/codex-large" in captured.out
        assert "claude -> uutel-claude/claude-sonnet-4" in captured.out
        assert "gemini -> uutel-gemini/gemini-2.5-pro" in captured.out
        assert "cloud -> uutel-cloud/gemini-2.5-pro" in captured.out
        assert "claude-code -> uutel-claude/claude-sonnet-4" in captured.out
        assert "gemini-cli -> uutel-gemini/gemini-2.5-pro" in captured.out
        assert "cloud-code -> uutel-cloud/gemini-2.5-pro" in captured.out

    @patch("litellm.completion")
    def test_complete_command_basic(self, mock_completion, capsys):
        """Test basic complete command functionality."""
        # Mock successful completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_completion.return_value = mock_response

        result = self.cli.complete("Test prompt")

        # Verify completion was called with correct parameters
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        assert call_args[1]["model"] == "my-custom-llm/codex-large"
        assert call_args[1]["messages"] == [{"role": "user", "content": "Test prompt"}]
        assert call_args[1]["max_tokens"] == 500
        assert call_args[1]["temperature"] == 0.7

        # Verify response handling
        assert result == "Test response"
        captured = capsys.readouterr()
        assert "Test response" in captured.out

    @patch("litellm.completion")
    def test_complete_command_filters_control_sequences(
        self, mock_completion, capsys
    ) -> None:
        """ANSI/OSC/C1 bytes from providers should be scrubbed before display and return."""

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "Result \x1b[31mtext\x1b[0m\x90ignored payload\x9c!\x1b]2;title\x07"
        )
        mock_completion.return_value = mock_response

        result = self.cli.complete("Sanitise this")

        assert result == "Result text!"
        captured = capsys.readouterr()
        assert "Result text!" in captured.out
        assert "\x1b" not in captured.out
        assert "\x90" not in captured.out

    @patch("litellm.completion")
    def test_complete_command_with_system_message(self, mock_completion):
        """Test complete command with system message."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "System response"
        mock_completion.return_value = mock_response

        self.cli.complete("User prompt", system="You are helpful")

        # Verify system message is included
        call_args = mock_completion.call_args
        expected_messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "User prompt"},
        ]
        assert call_args[1]["messages"] == expected_messages

    @patch("litellm.completion")
    def test_complete_command_with_custom_parameters(self, mock_completion):
        """Test complete command with custom parameters."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Custom response"
        mock_completion.return_value = mock_response

        self.cli.complete(
            "Test prompt",
            engine="my-custom-llm/codex-mini",
            max_tokens=200,
            temperature=0.5,
        )

        # Verify custom parameters are used
        call_args = mock_completion.call_args
        assert call_args[1]["model"] == "my-custom-llm/codex-mini"
        assert call_args[1]["max_tokens"] == 200
        assert call_args[1]["temperature"] == 0.5

    @patch("litellm.completion")
    def test_complete_command_with_alias(self, mock_completion):
        """Alias names resolve to their canonical engine strings."""

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Alias response"
        mock_completion.return_value = mock_response

        self.cli.complete("Alias prompt", engine="codex")

        call_args = mock_completion.call_args
        assert call_args[1]["model"] == "my-custom-llm/codex-large"

    @patch("litellm.completion")
    def test_complete_command_accepts_punctuated_alias(
        self, mock_completion: MagicMock
    ) -> None:
        """Messy alias input with punctuation should resolve to canonical engine."""

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Punctuated alias"
        mock_completion.return_value = mock_response

        self.cli.complete("Alias prompt", engine="--codex--")

        call_args = mock_completion.call_args
        assert call_args[1]["model"] == "my-custom-llm/codex-large"

    @patch("litellm.completion")
    def test_complete_command_streaming(self, mock_completion, capsys):
        """Test complete command with streaming enabled."""
        # Mock streaming response
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello "

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = "world!"

        mock_completion.return_value = [mock_chunk1, mock_chunk2]

        result = self.cli.complete("Test prompt", stream=True)

        # Verify streaming parameters
        call_args = mock_completion.call_args
        assert call_args[1]["stream"] is True

        # Verify streaming output
        assert result == "Hello world!"
        captured = capsys.readouterr()
        assert "Hello world!" in captured.out

    @patch("litellm.completion")
    def test_complete_command_streaming_filters_control_sequences(
        self, mock_completion, capsys
    ) -> None:
        """Streaming paths should strip control bytes across incremental chunks."""

        chunk_one = MagicMock()
        chunk_one.choices = [MagicMock()]
        chunk_one.choices[0].delta.content = "Hello\x1b]0;title\x07 "

        chunk_two = MagicMock()
        chunk_two.choices = [MagicMock()]
        chunk_two.choices[0].delta.content = "\x9b31mworld!"

        mock_completion.return_value = [chunk_one, chunk_two]

        result = self.cli.complete("Stream sanitisation", stream=True)

        assert result == "Hello world!"
        captured = capsys.readouterr()
        assert "Hello world!" in captured.out
        assert "\x1b" not in captured.out
        assert "\x9b" not in captured.out

    @patch("litellm.completion")
    def test_complete_command_streaming_handles_missing_choices(
        self, mock_completion, capsys
    ):
        """Streaming should skip chunks that do not include choices without raising."""

        empty_chunk = MagicMock()
        empty_chunk.choices = []

        valid_choice = MagicMock()
        valid_choice.delta.content = "usable text"
        valid_chunk = MagicMock()
        valid_chunk.choices = [valid_choice]

        mock_completion.return_value = [empty_chunk, valid_chunk]

        result = self.cli.complete("Test prompt", stream=True)

        assert result == "usable text"
        captured = capsys.readouterr()
        assert "usable text" in captured.out

    @patch("litellm.completion")
    def test_complete_command_streaming_handles_structured_delta_content(
        self, mock_completion, capsys
    ):
        """Structured delta content arrays should be flattened into plain text."""

        choice = MagicMock()
        choice.delta.content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": " there"},
        ]
        chunk = MagicMock()
        chunk.choices = [choice]

        mock_completion.return_value = [chunk]

        result = self.cli.complete("Test prompt", stream=True)

        assert result == "Hello there"
        captured = capsys.readouterr()
        assert "Hello there" in captured.out

    @patch("litellm.completion")
    def test_complete_command_streaming_returns_empty_banner(
        self, mock_completion, capsys
    ):
        """Streaming runs with no emitted text should reuse the empty-response banner."""

        choice = MagicMock()
        choice.delta.content = None
        chunk = MagicMock()
        chunk.choices = [choice]

        mock_completion.return_value = [chunk]

        result = self.cli.complete("Test prompt", stream=True)

        expected = self.cli._format_empty_response_message("my-custom-llm/codex-large")
        assert result == expected
        captured = capsys.readouterr()
        assert expected in captured.err

    @patch("litellm.completion")
    def test_complete_command_error_handling(self, mock_completion, capsys):
        """Test complete command error handling."""
        # Mock completion error
        mock_completion.side_effect = Exception("API Error")

        result = self.cli.complete("Test prompt")

        # Verify error is handled gracefully with enhanced format
        assert "❌ Error in completion: API Error" in result
        captured = capsys.readouterr()
        assert "❌ Error in completion: API Error" in captured.err
        assert "💡 Use --verbose for more details" in captured.err

    @patch("uutel.__main__.UUTELCLI._check_provider_readiness")
    @patch("litellm.completion")
    def test_complete_command_handles_provider_exception(
        self,
        mock_completion,
        mock_readiness,
        capsys,
    ) -> None:
        """LiteLLM provider failures should surface provider context."""

        mock_readiness.return_value = (True, [])

        class FakeLiteLLMException(Exception):
            def __init__(
                self, message: str, provider: str | None, model: str | None
            ) -> None:
                super().__init__(message)
                self.provider = provider
                self.model = model

        mock_completion.side_effect = FakeLiteLLMException(
            "Upstream provider outage",
            provider="uutel-claude",
            model="uutel-claude/claude-sonnet-4",
        )

        result = self.cli.complete(
            "Hello there",
            engine="uutel-claude/claude-sonnet-4",
            stream=False,
        )

        stderr = capsys.readouterr().err

        assert "uutel-claude" in result
        assert "Upstream provider outage" in result
        assert "uutel diagnostics" in result.lower()
        assert "uutel-claude" in stderr

    @patch("litellm.completion")
    def test_complete_command_streaming_error(self, mock_completion, capsys):
        """Test streaming error handling."""
        # Mock streaming error
        mock_completion.side_effect = Exception("Streaming Error")

        result = self.cli.complete("Test prompt", stream=True)

        # Verify streaming error is handled with enhanced format
        assert "❌ Error in streaming: Streaming Error" in result
        captured = capsys.readouterr()
        assert "❌ Error in streaming: Streaming Error" in captured.err
        assert "💡 Use --verbose for more details" in captured.err

    @patch("uutel.__main__.UUTELCLI._check_provider_readiness")
    @patch("litellm.completion")
    def test_complete_command_blocks_when_provider_not_ready(
        self, mock_completion, mock_readiness, capsys
    ):
        """uutel complete should surface readiness guidance before issuing requests."""

        mock_readiness.return_value = (
            False,
            [
                "⚠️ Codex credentials missing",
                "💡 Run codex login or set OPENAI_API_KEY",
            ],
        )

        result = self.cli.complete("Test prompt", engine="claude")

        mock_completion.assert_not_called()

        expected_lines = [
            "⚠️ Codex credentials missing",
            "💡 Run codex login or set OPENAI_API_KEY",
            "💡 Run uutel diagnostics to review provider setup before retrying",
        ]
        expected = "\n".join(expected_lines)
        assert result == expected

        captured = capsys.readouterr()
        for line in expected_lines:
            assert line in captured.err

    @patch("uutel.__main__.UUTELCLI._check_provider_readiness")
    @patch("litellm.completion")
    def test_complete_command_warns_on_placeholder_result(
        self, mock_completion, mock_readiness, capsys
    ):
        """Placeholder responses should emit the same warning banner as uutel test."""

        mock_readiness.return_value = (True, [])

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "This is a mock response from Codex provider for model codex-large."
            " Received 1 messages."
        )
        mock_completion.return_value = mock_response

        result = self.cli.complete("Test prompt", engine="codex")

        placeholder_message = (
            "❌ Placeholder output detected for engine 'my-custom-llm/codex-large'."
            "\n💡 Use a live provider or refresh your credentials before retrying."
        )

        assert result == placeholder_message
        captured = capsys.readouterr()
        assert placeholder_message in captured.err
        assert "Placeholder output detected" in captured.err

    @patch("uutel.__main__.UUTELCLI._stream_completion")
    @patch("uutel.__main__.UUTELCLI._check_provider_readiness")
    def test_complete_streaming_warns_on_placeholder_result(
        self, mock_readiness, mock_stream, capsys
    ):
        """Streaming completions should also emit placeholder warnings."""

        mock_readiness.return_value = (True, [])
        mock_stream.return_value = (
            "This is a mock response from Codex provider for model codex-large."
            " Received 1 messages."
        )

        result = self.cli.complete("Test prompt", engine="codex", stream=True)

        placeholder_message = (
            "❌ Placeholder output detected for engine 'my-custom-llm/codex-large'."
            "\n💡 Use a live provider or refresh your credentials before retrying."
        )

        assert result == placeholder_message
        captured = capsys.readouterr()
        assert placeholder_message in captured.err

    @patch("uutel.__main__.UUTELCLI._check_provider_readiness", return_value=(True, []))
    @patch("litellm.completion")
    def test_complete_preserves_existing_litellm_log_value_when_not_verbose(
        self,
        mock_completion: MagicMock,
        _mock_readiness: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """CLI should not clear user-defined LITELLM_LOG when verbose is disabled."""

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_completion.return_value = mock_response

        with patch.dict(os.environ, {"LITELLM_LOG": "INFO"}, clear=True):
            result = self.cli.complete("Test prompt", verbose=False)
            captured = capsys.readouterr()
            assert "Test response" in captured.out
            assert os.environ["LITELLM_LOG"] == "INFO", (
                "CLI should preserve user LITELLM_LOG value when verbose mode is off"
            )

        assert result == "Test response"

    @patch("uutel.__main__.UUTELCLI._check_provider_readiness", return_value=(True, []))
    @patch("litellm.completion")
    def test_complete_restores_logging_state_after_verbose_run(
        self,
        mock_completion: MagicMock,
        _mock_readiness: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Verbose runs should restore env vars and logger levels after completion."""

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Verbose response"
        mock_completion.return_value = mock_response

        uutel_logger = logging.getLogger("uutel")
        original_uutel_level = uutel_logger.level
        original_cli_level = cli_module.logger.level

        with patch.dict(os.environ, {}, clear=True):
            result = self.cli.complete("Verbose prompt", verbose=True)
            captured = capsys.readouterr()
            assert "🔧 Verbose mode enabled" in captured.err
            assert "🎯 Using engine" in captured.err
            assert "⚙️  Parameters" in captured.err
            assert "LITELLM_LOG" not in os.environ, (
                "Verbose run should leave LITELLM_LOG unset after completion"
            )

        assert result == "Verbose response"
        assert logging.getLogger("uutel").level == original_uutel_level, (
            "uutel logger level should be restored after verbose completion"
        )
        assert cli_module.logger.level == original_cli_level, (
            "CLI module logger level should be restored after verbose completion"
        )

    @patch("litellm.completion")
    def test_complete_command_handles_empty_choices(self, mock_completion, capsys):
        """LiteLLM responses without choices should return a friendly error."""

        mock_completion.return_value = MagicMock(choices=[])

        result = self.cli.complete("Test prompt", engine="codex")

        expected = (
            "❌ Received empty response from engine 'my-custom-llm/codex-large'."
            "\n💡 Enable --verbose to inspect LiteLLM logs before retrying."
        )

        assert result == expected
        captured = capsys.readouterr()
        assert expected in captured.err

    @patch("litellm.completion")
    def test_complete_command_handles_missing_message_content(
        self, mock_completion, capsys
    ):
        """Choices lacking message content should also surface the new guidance."""

        choice = MagicMock()
        choice.message = MagicMock(content=None)
        mock_completion.return_value = MagicMock(choices=[choice])

        result = self.cli.complete("Test prompt", engine="codex")

        expected = (
            "❌ Received empty response from engine 'my-custom-llm/codex-large'."
            "\n💡 Enable --verbose to inspect LiteLLM logs before retrying."
        )

        assert result == expected
        captured = capsys.readouterr()
        assert expected in captured.err

    @patch("uutel.__main__.UUTELCLI._check_provider_readiness")
    @patch("litellm.completion", side_effect=KeyboardInterrupt)
    def test_complete_command_handles_keyboard_interrupt(
        self,
        mock_completion,
        mock_readiness,
        capsys,
    ):
        """User cancellations should surface a friendly message instead of a traceback."""

        mock_readiness.return_value = (True, [])

        result = self.cli.complete("Test prompt", engine="codex")

        expected = "⚪ Operation cancelled by user"
        assert result == expected

        captured = capsys.readouterr()
        assert expected in captured.err
        assert "Traceback" not in captured.err

    @patch("uutel.__main__.UUTELCLI._check_provider_readiness")
    @patch("litellm.completion", side_effect=KeyboardInterrupt)
    def test_complete_command_streaming_handles_keyboard_interrupt(
        self,
        mock_completion,
        mock_readiness,
        capsys,
    ):
        """Streaming cancellations should also return the friendly cancellation message."""

        mock_readiness.return_value = (True, [])

        result = self.cli.complete("Test prompt", engine="codex", stream=True)

        expected = "⚪ Operation cancelled by user"
        assert result == expected

        captured = capsys.readouterr()
        assert expected in captured.err
        assert "Traceback" not in captured.err

    def test_safe_print_swallows_broken_pipe_stdout(self, monkeypatch):
        """_safe_print should absorb BrokenPipeError when writing to stdout."""

        class BrokenStdout:
            def write(self, _text: str) -> None:
                raise BrokenPipeError

            def flush(self) -> None:
                raise BrokenPipeError

        monkeypatch.setattr(sys, "stdout", BrokenStdout())

        self.cli._safe_print("hello world")

    def test_safe_print_swallows_broken_pipe_stderr(self, monkeypatch):
        """_safe_print should absorb BrokenPipeError for stderr as well."""

        class BrokenStderr:
            def write(self, _text: str) -> None:
                raise BrokenPipeError

            def flush(self) -> None:
                raise BrokenPipeError

        monkeypatch.setattr(sys, "stderr", BrokenStderr())

        self.cli._safe_print("oops", target="stderr")

    @patch("uutel.__main__.UUTELCLI.complete")
    def test_test_command(self, mock_complete):
        """Test the test command functionality."""
        mock_complete.return_value = "Test response"

        result = self.cli.test("my-custom-llm/codex-fast")

        # Verify test command calls complete with correct parameters
        mock_complete.assert_called_once_with(
            prompt="Hello! Can you respond with a brief greeting?",
            engine="my-custom-llm/codex-fast",
            max_tokens=50,
            verbose=True,
        )
        assert result == "Test response"

    @patch("uutel.__main__.UUTELCLI.complete")
    def test_test_command_default_engine(self, mock_complete):
        """Test test command with default engine."""
        mock_complete.return_value = "Default test response"

        self.cli.test()

        # Verify default engine is used
        call_args = mock_complete.call_args
        assert call_args[1]["engine"] == "my-custom-llm/codex-large"

    @patch("uutel.__main__.UUTELCLI.complete")
    def test_test_command_alias_engine(self, mock_complete):
        """Alias inputs should resolve to canonical engines in test command."""

        mock_complete.return_value = "Alias test response"

        self.cli.test("claude")

        call_args = mock_complete.call_args
        assert call_args[1]["engine"] == "uutel-claude/claude-sonnet-4"

    @patch("uutel.__main__.UUTELCLI.complete")
    def test_test_command_accepts_punctuated_alias(self, mock_complete):
        """Messy alias input for test command should resolve via alias normalisation."""

        mock_complete.return_value = "Alias test response"

        self.cli.test("__gemini__")

        call_args = mock_complete.call_args
        assert call_args[1]["engine"] == "uutel-gemini/gemini-2.5-pro"

    @patch("uutel.__main__.UUTELCLI.complete")
    @patch("uutel.__main__.UUTELCLI._check_provider_readiness")
    def test_test_command_rejects_placeholder_output(
        self, mock_readiness, mock_complete, capsys
    ):
        """uutel test should fail when the provider returns mock placeholder text."""

        mock_readiness.return_value = (True, [])
        mock_complete.return_value = "This is a mock response from Codex provider for model codex-large. Received 1 messages."

        result = self.cli.test("codex", verbose=False)

        mock_complete.assert_called_once()

        captured = capsys.readouterr()
        assert "Placeholder output detected" in result
        assert "Placeholder output detected" in captured.err
        assert "Use a live provider" in captured.err

    @patch("uutel.__main__.UUTELCLI._check_provider_readiness")
    @patch("litellm.completion")
    def test_complete_command_swallows_broken_pipe_when_printing_result(
        self,
        mock_completion,
        mock_readiness,
        monkeypatch,
    ):
        """Printing the final result should not crash when stdout closes early."""

        class BrokenStdout:
            def write(self, _text: str) -> None:
                raise BrokenPipeError

            def flush(self) -> None:
                raise BrokenPipeError

        monkeypatch.setattr(sys, "stdout", BrokenStdout())

        mock_readiness.return_value = (True, [])

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Result text"
        mock_completion.return_value = mock_response

        result = self.cli.complete("Prompt", engine="codex")

        assert result == "Result text"

    def test_looks_like_placeholder_detects_legacy_phrases(self) -> None:
        """Legacy canned strings should be flagged as placeholders."""
        legacy_text = "In a real implementation this would call the API"

        assert self.cli._looks_like_placeholder(legacy_text) is True, (
            "Legacy phrasing should be treated as placeholder output"
        )

    def test_looks_like_placeholder_allows_realistic_output(self) -> None:
        """Recorded fixture text should not be misclassified as placeholder."""
        recorded_text = "Here is a Python function that sorts a list using the built-in sorted call."

        assert self.cli._looks_like_placeholder(recorded_text) is False, (
            "Realistic recorded content should not trigger placeholder detection"
        )

    @patch("uutel.__main__.UUTELCLI.complete")
    @patch("uutel.__main__.UUTELCLI._check_provider_readiness")
    def test_test_command_blocks_when_preflight_fails(
        self, mock_readiness, mock_complete, capsys
    ):
        """uutel test should surface readiness warnings instead of calling complete."""

        mock_readiness.return_value = (
            False,
            [
                "⚠️ Codex credentials missing",
                "💡 Run codex login or set OPENAI_API_KEY",
            ],
        )

        result = self.cli.test("my-custom-llm/codex-large", verbose=False)

        mock_complete.assert_not_called()

        expected_lines = [
            "⚠️ Codex credentials missing",
            "💡 Run codex login or set OPENAI_API_KEY",
            "💡 Run uutel diagnostics to review provider setup before retrying",
        ]

        assert result == "\n".join(expected_lines)

        captured = capsys.readouterr()
        for line in expected_lines:
            assert line in captured.err

    @patch("uutel.__main__.UUTELCLI.complete", side_effect=KeyboardInterrupt)
    @patch("uutel.__main__.UUTELCLI._check_provider_readiness")
    def test_test_command_handles_keyboard_interrupt(
        self,
        mock_readiness,
        mock_complete,
        capsys,
    ):
        """The test command should translate cancellations into guidance."""

        mock_readiness.return_value = (True, [])

        result = self.cli.test("codex", verbose=False)

        expected = "⚪ Operation cancelled by user"
        assert result == expected

        captured = capsys.readouterr()
        assert expected in captured.err
        assert "Traceback" not in captured.err

    @patch("litellm.completion")
    def test_verbose_logging_restores_env_flag(self, mock_completion):
        """Verbose mode should not leave LITELLM_LOG mutated after completion."""
        import litellm

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Verbose response"
        mock_completion.return_value = mock_response

        original_set_verbose = litellm.set_verbose

        with patch.dict(os.environ, {}, clear=True):
            self.cli.complete("Test prompt", verbose=True)

            assert "LITELLM_LOG" not in os.environ
            assert litellm.set_verbose is original_set_verbose

            mock_completion.reset_mock()

            self.cli.complete("Test prompt", verbose=False)

            assert "LITELLM_LOG" not in os.environ

    def test_streaming_chunk_processing(self, capsys):
        """Test streaming chunk processing with various chunk types."""
        # Create mock chunks with different content patterns
        chunk_with_content = MagicMock()
        chunk_with_content.choices = [MagicMock()]
        chunk_with_content.choices[0].delta.content = "Content"

        chunk_without_content = MagicMock()
        chunk_without_content.choices = [MagicMock()]
        chunk_without_content.choices[0].delta.content = None

        # Test the internal streaming method
        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = [
                chunk_with_content,
                chunk_without_content,
                chunk_with_content,
            ]

            result = self.cli._stream_completion(
                messages=[{"role": "user", "content": "test"}],
                engine="my-custom-llm/codex-large",
                max_tokens=100,
                temperature=0.7,
            )

            assert result == "ContentContent"


class TestReadGcloudDefaultProject:
    """Unit tests for the gcloud project discovery helper."""

    def test_read_gcloud_default_project_returns_none_when_missing(
        self, tmp_path: Path
    ) -> None:
        """Missing configuration files should yield None."""

        project = _read_gcloud_default_project(tmp_path)

        assert project is None

    def test_read_gcloud_default_project_ignores_malformed_file(
        self, tmp_path: Path
    ) -> None:
        """Configs lacking a core project entry should be ignored."""

        config_dir = tmp_path / ".config" / "gcloud" / "configurations"
        config_dir.mkdir(parents=True)
        (config_dir / "config_default").write_text(
            "[other]\nproject = nope", encoding="utf-8"
        )

        project = _read_gcloud_default_project(tmp_path)

        assert project is None

    def test_read_gcloud_default_project_handles_decode_error(
        self, tmp_path: Path
    ) -> None:
        """Unreadable config files should be treated as missing."""

        config_dir = tmp_path / ".config" / "gcloud" / "configurations"
        config_dir.mkdir(parents=True)
        (config_dir / "config_default").write_bytes(b"\xff\xfe\x00broken")

        project = _read_gcloud_default_project(tmp_path)

        assert project is None, "Decode errors should fall back to no project"

    def test_read_gcloud_default_project_parses_core_project(
        self, tmp_path: Path
    ) -> None:
        """A core project entry should be returned as the project id."""

        config_dir = tmp_path / ".config" / "gcloud" / "configurations"
        config_dir.mkdir(parents=True)
        (config_dir / "config_default").write_text(
            "# gcloud config\n[core]\nproject = my-gcp-project\n", encoding="utf-8"
        )

        project = _read_gcloud_default_project(tmp_path)

        assert project == "my-gcp-project"


class TestCLIParameterValidation:
    """Test CLI parameter validation and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cli = UUTELCLI()

    def test_empty_prompt_handling(self, capsys):
        """Test handling of empty prompt."""
        result = self.cli.complete("")

        # Verify empty prompt is rejected with helpful message
        assert "❌ Prompt is required and cannot be empty" in result
        captured = capsys.readouterr()
        assert "❌ Prompt is required and cannot be empty" in captured.err
        assert '💡 Try: uutel complete "Your prompt here"' in captured.err

    @patch("litellm.completion")
    def test_long_prompt_handling(self, mock_completion):
        """Test handling of very long prompts."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Long response"
        mock_completion.return_value = mock_response

        long_prompt = "x" * 10000  # 10k characters
        self.cli.complete(long_prompt)

        # Verify long prompt is handled
        call_args = mock_completion.call_args
        assert call_args[1]["messages"] == [{"role": "user", "content": long_prompt}]

    @patch("litellm.completion")
    def test_boundary_token_values(self, mock_completion):
        """Test boundary values for token parameters."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Boundary response"
        mock_completion.return_value = mock_response

        # Test minimum and maximum reasonable values
        self.cli.complete("Test", max_tokens=1)
        call_args = mock_completion.call_args
        assert call_args[1]["max_tokens"] == 1

        self.cli.complete("Test", max_tokens=4000)
        call_args = mock_completion.call_args
        assert call_args[1]["max_tokens"] == 4000

    @patch("litellm.completion")
    def test_complete_command_rejects_zero_max_tokens(self, mock_completion, capsys):
        """Zero max_tokens should trigger validation error instead of defaulting."""

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Zero tokens response"
        mock_completion.return_value = mock_response

        error = self.cli.complete("Test prompt", max_tokens=0)
        captured = capsys.readouterr()

        assert "max_tokens must be an integer between 1 and 8000" in error
        assert "max_tokens must be an integer between 1 and 8000" in captured.err
        mock_completion.assert_not_called()

    @patch("litellm.completion")
    def test_complete_command_uses_config_defaults(self, mock_completion):
        """Config-sourced defaults should propagate into completion requests."""

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Config default response"
        mock_completion.return_value = mock_response
        self.cli.config = UUTELConfig(max_tokens=750, temperature=1.1)

        self.cli.complete("Config prompt")

        call_args = mock_completion.call_args
        assert call_args[1]["max_tokens"] == 750
        assert call_args[1]["temperature"] == 1.1

    @patch("litellm.completion")
    def test_complete_command_invalid_config_max_tokens_surfaces_error(
        self, mock_completion, capsys
    ):
        """Invalid persisted max_tokens should bubble validation errors to the user."""

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Invalid config response"
        mock_completion.return_value = mock_response
        self.cli.config = UUTELConfig(max_tokens=0)

        error = self.cli.complete("Config prompt")
        captured = capsys.readouterr()

        assert "max_tokens must be an integer between 1 and 8000" in error
        assert "max_tokens must be an integer between 1 and 8000" in captured.err
        mock_completion.assert_not_called()

    @patch("litellm.completion")
    def test_boundary_temperature_values(self, mock_completion):
        """Test boundary values for temperature parameter."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Temperature response"
        mock_completion.return_value = mock_response

        # Test minimum and maximum temperature values
        self.cli.complete("Test", temperature=0.0)
        call_args = mock_completion.call_args
        assert call_args[1]["temperature"] == 0.0

        self.cli.complete("Test", temperature=2.0)
        call_args = mock_completion.call_args
        assert call_args[1]["temperature"] == 2.0

    def test_invalid_engine_handling(self, capsys):
        """Test handling of invalid engine names."""
        result = self.cli.complete("Test", engine="invalid/model")

        # Verify invalid engine is rejected with helpful message
        assert "Unknown engine 'invalid/model'" in result
        captured = capsys.readouterr()
        assert "Unknown engine 'invalid/model'" in captured.err


class TestCLIIntegration:
    """Integration tests for CLI with actual provider interactions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cli = UUTELCLI()

    def test_provider_setup_during_initialization(self):
        """Test that providers are properly set up during CLI initialization."""
        # Verify that litellm.custom_provider_map is configured
        # This is tested implicitly by successful CLI operations
        assert hasattr(self.cli, "complete")
        assert hasattr(self.cli, "list_engines")
        assert hasattr(self.cli, "test")

    @patch("litellm.completion")
    def test_multiple_engine_support(self, mock_completion):
        """Test that CLI works with different engines."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Engine response"
        mock_completion.return_value = mock_response

        engines_to_test = [
            "my-custom-llm/codex-large",
            "my-custom-llm/codex-mini",
            "my-custom-llm/codex-turbo",
            "my-custom-llm/codex-fast",
            "my-custom-llm/codex-preview",
        ]

        for engine in engines_to_test:
            result = self.cli.complete("Test", engine=engine)
            assert result == "Engine response"

            # Verify correct engine was used
            call_args = mock_completion.call_args
            assert call_args[1]["model"] == engine

    @patch("litellm.completion")
    def test_cli_state_persistence(self, mock_completion):
        """Test that CLI maintains state properly across multiple calls."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "State response"
        mock_completion.return_value = mock_response

        # Make multiple calls to verify state is maintained
        self.cli.complete("First call")
        self.cli.complete("Second call")
        self.cli.list_engines()
        self.cli.complete("Third call")

        # Verify all calls succeeded (no state corruption)
        assert mock_completion.call_count == 3  # complete calls only

    def test_help_system_integration(self):
        """Test that Fire's help system works with CLI."""
        # This tests that the CLI is properly configured for Fire
        # Fire automatically generates help based on method signatures

        # Test that methods have proper docstrings for help generation
        assert self.cli.complete.__doc__ is not None
        assert self.cli.list_engines.__doc__ is not None
        assert self.cli.test.__doc__ is not None

        # Test that methods have reasonable parameter annotations
        import inspect

        sig = inspect.signature(self.cli.complete)
        assert "prompt" in sig.parameters
        assert "engine" in sig.parameters
        assert "max_tokens" in sig.parameters


class TestCLIErrorScenarios:
    """Test CLI behavior in various error scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cli = UUTELCLI()

    @patch("litellm.completion")
    def test_network_error_handling(self, mock_completion, capsys):
        """Test handling of network-related errors."""
        mock_completion.side_effect = ConnectionError("Network unavailable")

        result = self.cli.complete("Test prompt")

        assert "❌ Network error in completion" in result
        captured = capsys.readouterr()
        assert "❌ Network error in completion" in captured.err
        assert "💡 Check your internet connection" in captured.err

    @patch("litellm.completion")
    def test_timeout_error_handling(self, mock_completion, capsys):
        """Test handling of timeout errors."""
        mock_completion.side_effect = TimeoutError("Request timeout")

        result = self.cli.complete("Test prompt")

        assert "❌ Request timeout in completion" in result
        captured = capsys.readouterr()
        assert "❌ Request timeout in completion" in captured.err
        assert "💡 Try reducing max_tokens" in captured.err

    @patch("litellm.completion")
    def test_authentication_error_handling(self, mock_completion, capsys):
        """Test handling of authentication errors."""
        mock_completion.side_effect = Exception("Authentication failed")

        result = self.cli.complete("Test prompt")

        assert "❌ Authentication failed in completion" in result
        captured = capsys.readouterr()
        assert "❌ Authentication failed in completion" in captured.err
        assert "💡 Check your API keys" in captured.err

    @patch("litellm.completion")
    def test_rate_limit_error_handling(self, mock_completion, capsys):
        """Test handling of rate limit errors."""
        mock_completion.side_effect = Exception("Rate limit exceeded")

        result = self.cli.complete("Test prompt")

        assert "❌ Rate limit exceeded in completion" in result
        captured = capsys.readouterr()
        assert "❌ Rate limit exceeded in completion" in captured.err
        assert "💡 Try again in a few seconds" in captured.err

    @patch("uutel.__main__.setup_providers")
    def test_provider_setup_error_handling(self, mock_setup):
        """Test handling of provider setup errors."""
        mock_setup.side_effect = Exception("Provider setup failed")

        # This should not prevent CLI creation but might affect functionality
        try:
            cli = UUTELCLI()
            # CLI should still be created even if provider setup fails
            assert cli is not None
        except Exception:
            # If it does fail, it should fail gracefully
            pass


class TestCLIConfigCommands:
    """Regression coverage for config subcommands."""

    def setup_method(self) -> None:
        self.cli = UUTELCLI()

    def test_config_set_engine_alias_canonicalises_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setting engine via alias should persist the canonical identifier."""

        captured: dict[str, UUTELConfig] = {}

        def fake_save_config(config: UUTELConfig) -> None:
            captured["config"] = config

        monkeypatch.setattr("uutel.__main__.save_config", fake_save_config)

        result = UUTELCLI.config(self.cli, "set", engine="claude")

        assert result.startswith("✅ Configuration updated"), (
            "Config command should report success"
        )
        assert self.cli.config.engine == "uutel-claude/claude-sonnet-4", (
            "CLI state should use canonical engine identifier"
        )
        assert "config" in captured, "save_config should be invoked with updated config"
        assert captured["config"].engine == "uutel-claude/claude-sonnet-4", (
            "Persisted config should store canonical engine"
        )

    def test_config_set_engine_synonym_canonicalises_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Common provider synonyms should resolve before persisting engine."""

        captured: dict[str, UUTELConfig] = {}

        def fake_save_config(config: UUTELConfig) -> None:
            captured["config"] = config

        monkeypatch.setattr("uutel.__main__.save_config", fake_save_config)

        result = UUTELCLI.config(self.cli, "set", engine="gemini-cli")

        assert result.startswith("✅ Configuration updated"), (
            "Config command should report success for recognised synonyms"
        )
        assert self.cli.config.engine == "uutel-gemini/gemini-2.5-pro", (
            "CLI state should use canonical engine after synonym resolution"
        )
        assert captured["config"].engine == "uutel-gemini/gemini-2.5-pro", (
            "Persisted config should store canonical engine after synonym resolution"
        )

    def test_config_set_rejects_unknown_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid engine names should not be persisted."""

        def fail_save_config(
            config: UUTELConfig,
        ) -> None:  # pragma: no cover - should not run
            raise AssertionError("save_config should not be called for invalid engine")

        monkeypatch.setattr("uutel.__main__.save_config", fail_save_config)

        result = UUTELCLI.config(self.cli, "set", engine="invalid-engine")

        assert "Unknown engine" in result, (
            "Response should explain engine validation failure"
        )

    def test_config_set_clears_system_prompt_when_blank(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Blank system inputs should clear the persisted prompt."""

        self.cli.config = UUTELConfig(system="You are helpful")
        captured: dict[str, UUTELConfig] = {}

        def fake_save_config(config: UUTELConfig) -> None:
            captured["config"] = config

        monkeypatch.setattr("uutel.__main__.save_config", fake_save_config)

        result = UUTELCLI.config(self.cli, "set", system="")

        assert result.startswith("✅ Configuration updated"), (
            "Config set should report success after clearing system"
        )
        assert "config" in captured, (
            "save_config should persist the cleared system prompt"
        )
        assert captured["config"].system is None, (
            "Persisted config should drop system when blank string provided"
        )
        assert self.cli.config.system is None, (
            "CLI state should drop system when blank string provided"
        )

    def test_config_set_trims_system_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """System prompts should persist without incidental leading/trailing whitespace."""

        captured: dict[str, UUTELConfig] = {}

        def fake_save_config(config: UUTELConfig) -> None:
            captured["config"] = config

        monkeypatch.setattr("uutel.__main__.save_config", fake_save_config)

        result = UUTELCLI.config(self.cli, "set", system="   Be concise.  ")

        assert result.startswith("✅ Configuration updated"), (
            "Config set should report success for trimmed system"
        )
        assert "config" in captured, "save_config should persist trimmed system prompt"
        assert captured["config"].system == "Be concise.", (
            "Persisted system prompt should be trimmed"
        )
        assert self.cli.config.system == "Be concise.", (
            "CLI state should store trimmed system prompt"
        )

    def test_config_set_coerces_string_inputs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """String literals from Fire should coerce into proper numeric/boolean types."""

        captured: dict[str, UUTELConfig] = {}

        def fake_save_config(config: UUTELConfig) -> None:
            captured["config"] = config

        monkeypatch.setattr("uutel.__main__.save_config", fake_save_config)

        result = UUTELCLI.config(
            self.cli,
            "set",
            max_tokens="750",
            temperature="0.5",
            stream="false",
            verbose="TRUE",
        )

        assert result.startswith("✅ Configuration updated"), (
            "Config set should accept string inputs from Fire"
        )
        assert captured["config"].max_tokens == 750, (
            "max_tokens string should coerce to int"
        )
        assert abs(captured["config"].temperature - 0.5) < 1e-9, (
            "temperature string should coerce to float"
        )
        assert captured["config"].stream is False, (
            "stream string should coerce to boolean"
        )
        assert captured["config"].verbose is True, (
            "verbose string should coerce to boolean"
        )

    def test_config_set_no_changes_skips_save(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Config set without explicit updates should return guidance instead of rewriting."""

        def fail_save_config(
            config: UUTELConfig,
        ) -> None:  # pragma: no cover - should not run
            raise AssertionError(
                "save_config should not be invoked when no changes provided"
            )

        monkeypatch.setattr("uutel.__main__.save_config", fail_save_config)

        result = UUTELCLI.config(self.cli, "set")

        assert result.startswith("ℹ️ No configuration changes provided"), (
            "CLI should surface no-op guidance"
        )

    def test_config_set_rejects_unknown_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Typos in config set arguments should surface explicit guidance."""

        def fail_save_config(
            config: UUTELConfig,
        ) -> None:  # pragma: no cover - should not run
            raise AssertionError(
                "save_config should not be invoked when unknown keys are provided"
            )

        monkeypatch.setattr("uutel.__main__.save_config", fail_save_config)

        result = UUTELCLI.config(self.cli, "set", max_toknes=500)  # Deliberate typo

        assert result.startswith("❌ Unknown configuration fields"), (
            "CLI should flag unknown config keys explicitly"
        )
        assert "max_toknes" in result, "Error message should list the unexpected key"
        assert "engine" in result and "verbose" in result, (
            "Guidance should enumerate allowed configuration keys for correction"
        )

    def test_config_set_invalid_numeric_inputs_return_bulleted_guidance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid numeric literals should surface bullet guidance and skip persistence."""

        def fail_save_config(
            config: UUTELConfig,
        ) -> None:  # pragma: no cover - should not run
            raise AssertionError(
                "save_config should not be invoked when validation fails"
            )

        monkeypatch.setattr("uutel.__main__.save_config", fail_save_config)

        result = UUTELCLI.config(
            self.cli,
            "set",
            max_tokens="abc",
            temperature="foo",
        )

        assert result.startswith("❌ Invalid configuration:"), (
            "CLI should reuse the invalid-configuration heading for coercion failures"
        )
        assert "• max_tokens must be an integer between 1 and 8000" in result, (
            "Guidance should include max_tokens validation details"
        )
        assert "• temperature must be a number between 0.0 and 2.0" in result, (
            "Guidance should include temperature validation details"
        )

    def test_config_set_default_boolean_clears_overrides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Passing default sentinels should clear persisted boolean overrides."""

        self.cli.config = UUTELConfig(stream=True, verbose=True)
        captured: dict[str, UUTELConfig] = {}

        def fake_save_config(config: UUTELConfig) -> None:
            captured["config"] = config

        monkeypatch.setattr("uutel.__main__.save_config", fake_save_config)

        result = UUTELCLI.config(
            self.cli,
            "set",
            stream="default",
            verbose="none",
        )

        assert result.startswith("✅ Configuration updated"), (
            "Config set should report success after clearing defaults"
        )
        assert "stream = default" in result, (
            "Change summary should mark stream as default"
        )
        assert "verbose = default" in result, (
            "Change summary should mark verbose as default"
        )
        assert captured["config"].stream is None, (
            "Persisted config should clear stream override"
        )
        assert captured["config"].verbose is None, (
            "Persisted config should clear verbose override"
        )
        assert self.cli.config.stream is None, "CLI state should clear stream override"
        assert self.cli.config.verbose is None, (
            "CLI state should clear verbose override"
        )

    def test_config_set_accepts_numeric_literals_with_underscores(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Underscore-separated integers and mixed-case booleans should persist correctly."""

        captured: dict[str, UUTELConfig] = {}

        def fake_save_config(config: UUTELConfig) -> None:
            captured["config"] = config

        monkeypatch.setattr("uutel.__main__.save_config", fake_save_config)

        result = UUTELCLI.config(
            self.cli,
            "set",
            max_tokens="1_250",
            temperature="1.25",
            stream="On",
            verbose="Off",
        )

        assert result.startswith("✅ Configuration updated"), (
            "Config set should succeed with underscore literals"
        )
        assert "max_tokens = 1250" in result, (
            "Change summary should display canonical integer"
        )
        assert captured["config"].max_tokens == 1250, (
            "Persisted config should normalise integer literal"
        )
        assert captured["config"].temperature == pytest.approx(1.25), (
            "Temperature should coerce to float"
        )
        assert captured["config"].stream is True, (
            "Stream flag should honour mixed-case true keyword"
        )
        assert captured["config"].verbose is False, (
            "Verbose flag should honour mixed-case false keyword"
        )

    def test_config_init_creates_file_and_refreshes_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Initialising config should write defaults and refresh CLI state immediately."""

        config_path = tmp_path / "uutel_config.toml"
        monkeypatch.setattr("uutel.core.config.get_config_path", lambda: config_path)

        cli = UUTELCLI()

        assert not config_path.exists(), "Precondition: config file should start absent"

        result = UUTELCLI.config(cli, "init")

        assert result.startswith("✅"), "Config init should report successful creation"
        assert config_path.exists(), "Config init must create the configuration file"
        assert cli.config.engine == "my-custom-llm/codex-large", (
            "CLI state should refresh to default engine after init"
        )
        assert cli.config.max_tokens == 500, (
            "Default max_tokens should populate CLI state"
        )
        assert cli.config.temperature == pytest.approx(0.7), (
            "Default temperature should populate CLI state"
        )
        assert cli.config.stream is False, (
            "Default stream flag should populate CLI state"
        )
        assert cli.config.verbose is False, (
            "Default verbose flag should populate CLI state"
        )

    def test_config_init_writes_canonical_snippet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Config init should persist the exact default snippet for documentation parity."""

        from uutel.core.config import create_default_config

        config_path = tmp_path / "uutel_config.toml"
        monkeypatch.setattr("uutel.core.config.get_config_path", lambda: config_path)

        cli = UUTELCLI()

        result = UUTELCLI.config(cli, "init")

        assert result.startswith("✅"), (
            "Config init should report success before snippet check"
        )
        content = config_path.read_text(encoding="utf-8")
        expected = create_default_config()
        assert content == expected, (
            "Written config should match create_default_config exactly"
        )
        assert content.endswith("\n"), (
            "Default snippet should end with a trailing newline for POSIX tooling"
        )

    def test_config_show_reloads_config_from_disk(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Config show should reload the file so printed values reflect on-disk edits."""

        config_path = tmp_path / "uutel_config.toml"
        monkeypatch.setattr("uutel.core.config.get_config_path", lambda: config_path)

        cli = UUTELCLI()

        payload = {
            "engine": "uutel-claude/claude-sonnet-4",
            "max_tokens": 620,
            "temperature": 0.3,
            "stream": True,
            "verbose": True,
        }
        config_path.write_text(
            "# UUTEL Configuration\n\n" + tomli_w.dumps(payload),
            encoding="utf-8",
        )

        result = UUTELCLI.config(cli, "show")
        captured = capsys.readouterr()

        assert result == "✅ Configuration displayed", (
            "Config show should report success"
        )
        assert "uutel-claude/claude-sonnet-4" in captured.out, (
            "Printed config should include refreshed engine field"
        )
        assert "max_tokens = 620" in captured.out, (
            "Printed config should include refreshed max_tokens"
        )
        assert "temperature = 0.3" in captured.out, (
            "Printed config should include refreshed temperature"
        )
        assert "stream = True" in captured.out, (
            "Printed config should include refreshed stream flag"
        )
        assert "verbose = True" in captured.out, (
            "Printed config should include refreshed verbose flag"
        )

    def test_config_show_when_missing_file_returns_guidance(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Config show should guide the user to init when no file exists."""

        config_path = tmp_path / "uutel_config.toml"
        monkeypatch.setattr("uutel.core.config.get_config_path", lambda: config_path)

        cli = UUTELCLI()

        assert not config_path.exists(), "Precondition: config file should be absent"

        result = UUTELCLI.config(cli, "show")
        captured = capsys.readouterr()

        assert (
            result
            == "📝 No configuration file found\n💡 Create one with: uutel config init"
        ), "Missing config should return guidance to run config init"
        assert captured.out == "" and captured.err == "", (
            "No config file message should not emit additional stdout/stderr"
        )

    def test_config_show_displays_default_booleans(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Unset boolean flags should render with explicit default guidance."""

        config_path = tmp_path / "uutel_config.toml"
        monkeypatch.setattr("uutel.core.config.get_config_path", lambda: config_path)

        cli = UUTELCLI()

        payload = {"engine": "my-custom-llm/codex-large"}
        config_path.write_text(
            "# UUTEL Configuration\n\n" + tomli_w.dumps(payload),
            encoding="utf-8",
        )

        result = UUTELCLI.config(cli, "show")
        captured = capsys.readouterr()

        assert result == "✅ Configuration displayed", (
            "Config show should report success"
        )
        assert "max_tokens = default (500)" in captured.out, (
            "Unset max_tokens should surface default guidance"
        )
        assert "temperature = default (0.7)" in captured.out, (
            "Unset temperature should surface default guidance"
        )
        assert "stream = default (False)" in captured.out, (
            "Unset stream flag should surface default guidance"
        )
        assert "verbose = default (False)" in captured.out, (
            "Unset verbose flag should surface default guidance"
        )

    def test_config_get_reloads_config_before_returning_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Config get should refresh stored values before reading attribute content."""

        config_path = tmp_path / "uutel_config.toml"
        monkeypatch.setattr("uutel.core.config.get_config_path", lambda: config_path)

        cli = UUTELCLI()

        payload = {"engine": "uutel-gemini/gemini-2.5-pro"}
        config_path.write_text(
            "# UUTEL Configuration\n\n" + tomli_w.dumps(payload),
            encoding="utf-8",
        )

        value = UUTELCLI.config(cli, "get_engine")

        assert value == "uutel-gemini/gemini-2.5-pro", (
            "Config get should surface refreshed engine value"
        )


class TestCLIProviderReadiness:
    """Unit tests for CLI provider readiness preflight checks."""

    def setup_method(self):
        """Initialise CLI instance for readiness tests."""
        self.cli = UUTELCLI()

    def test_check_provider_readiness_codex_without_creds(self, tmp_path, monkeypatch):
        """Codex engines should surface warnings when no credentials are available."""
        monkeypatch.setattr("uutel.__main__.Path.home", lambda: tmp_path)

        with patch.dict(os.environ, {}, clear=True):
            ready, hints = self.cli._check_provider_readiness(
                "my-custom-llm/codex-large"
            )

        assert ready is False
        assert any("codex login" in hint.lower() for hint in hints)

    def test_check_provider_readiness_codex_with_openai_key(
        self, tmp_path, monkeypatch
    ):
        """OPENAI_API_KEY should satisfy Codex readiness checks."""
        monkeypatch.setattr("uutel.__main__.Path.home", lambda: tmp_path)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            ready, hints = self.cli._check_provider_readiness(
                "my-custom-llm/codex-large"
            )

        assert ready is True
        assert hints == []

    def test_check_provider_readiness_codex_ignores_blank_env(
        self, tmp_path, monkeypatch
    ):
        """Whitespace-only env vars should not satisfy Codex credential checks."""

        monkeypatch.setattr("uutel.__main__.Path.home", lambda: tmp_path)

        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "   ", "OPENAI_SESSION_TOKEN": "\t"},
            clear=True,
        ):
            ready, hints = self.cli._check_provider_readiness(
                "my-custom-llm/codex-large"
            )

        assert ready is False
        assert any("codex login" in hint.lower() for hint in hints)


class TestCLIDiagnostics:
    """Tests for the diagnostics command summarising provider readiness."""

    def setup_method(self) -> None:
        self.cli = UUTELCLI()

    def test_diagnostics_reports_ready_and_missing(self, capsys, monkeypatch):
        """Diagnostics should surface readiness per alias with guidance lines."""

        def fake_readiness(engine: str):
            if engine.startswith("my-custom-llm/codex"):
                return True, ["Using OPENAI_API_KEY"]
            if engine.startswith("uutel-claude/"):
                return False, ["⚠️ Claude CLI missing"]
            if engine.startswith("uutel-gemini/"):
                return False, ["⚠️ Gemini credentials missing"]
            return False, ["⚠️ CLOUD_CODE_PROJECT not configured"]

        monkeypatch.setattr(self.cli, "_check_provider_readiness", fake_readiness)

        summary = self.cli.diagnostics()
        captured = capsys.readouterr()

        assert "🩺 UUTEL Diagnostics" in captured.out
        assert (
            "✅ codex, codex-large, openai-codex (my-custom-llm/codex-large)"
            in captured.out
        ), "Codex diagnostics should list all aliases"
        assert "⚠️ claude, claude-code (uutel-claude/claude-sonnet-4)" in captured.out, (
            "Claude diagnostics should group claude aliases"
        )
        assert "⚠️ gemini, gemini-cli (uutel-gemini/gemini-2.5-pro)" in captured.out, (
            "Gemini diagnostics should group gemini aliases"
        )
        assert "⚠️ cloud, cloud-code (uutel-cloud/gemini-2.5-pro)" in captured.out, (
            "Cloud diagnostics should group cloud aliases"
        )
        assert "Using OPENAI_API_KEY" in captured.out
        assert "⚠️ Claude CLI missing" in captured.out
        assert "⚠️ Gemini credentials missing" in captured.out
        assert "⚠️ CLOUD_CODE_PROJECT not configured" in captured.out
        assert summary == "Diagnostics complete: 1 ready, 3 need attention"

    def test_check_provider_readiness_claude_without_cli(self, monkeypatch):
        """Missing claude CLI should produce actionable readiness guidance."""
        monkeypatch.setattr("uutel.__main__.shutil.which", lambda _: None)

        ready, hints = self.cli._check_provider_readiness(
            "uutel-claude/claude-sonnet-4"
        )

        assert ready is False
        assert any("@anthropic-ai/claude-code" in hint for hint in hints)

    def test_check_provider_readiness_claude_with_cli(self, monkeypatch):
        """Presence of claude CLI binary should pass readiness checks."""
        monkeypatch.setattr(
            "uutel.__main__.shutil.which", lambda name: "/usr/bin/claude"
        )

        ready, hints = self.cli._check_provider_readiness(
            "uutel-claude/claude-sonnet-4"
        )

        assert ready is True
        assert hints == []

    def test_check_provider_readiness_gemini_without_creds(self, monkeypatch) -> None:
        """Gemini engines should warn when neither API key nor CLI is available."""

        monkeypatch.setattr("uutel.__main__.shutil.which", lambda name: None)

        with patch.dict(os.environ, {}, clear=True):
            ready, hints = self.cli._check_provider_readiness(
                "uutel-gemini/gemini-2.5-pro"
            )

        assert ready is False
        assert any("gemini credentials" in hint.lower() for hint in hints)

    def test_check_provider_readiness_gemini_ignores_blank_api_key(
        self, monkeypatch
    ) -> None:
        """Whitespace API key values should not satisfy Gemini readiness checks."""

        monkeypatch.setattr("uutel.__main__.shutil.which", lambda name: None)

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "  \n"}, clear=True):
            ready, hints = self.cli._check_provider_readiness(
                "uutel-gemini/gemini-2.5-pro"
            )

        assert ready is False
        assert any("gemini credentials" in hint.lower() for hint in hints)

    def test_check_provider_readiness_cloud_without_project(
        self, tmp_path, monkeypatch
    ) -> None:
        """Cloud Code engines should require project id or credentials."""

        monkeypatch.setattr("uutel.__main__.Path.home", lambda: tmp_path)
        monkeypatch.setattr("uutel.__main__.shutil.which", lambda name: None)

        with patch.dict(os.environ, {}, clear=True):
            ready, hints = self.cli._check_provider_readiness(
                "uutel-cloud/gemini-2.5-pro"
            )

        assert ready is False
        assert any("project" in hint.lower() for hint in hints)

    def test_check_provider_readiness_cloud_ignores_blank_api_key(
        self, tmp_path, monkeypatch
    ) -> None:
        """Whitespace API keys should not be treated as valid Cloud Code credentials."""

        monkeypatch.setattr("uutel.__main__.Path.home", lambda: tmp_path)
        monkeypatch.setattr("uutel.__main__.shutil.which", lambda name: None)

        with patch.dict(
            os.environ,
            {"GOOGLE_API_KEY": "\t", "CLOUD_CODE_PROJECT": "proj"},
            clear=True,
        ):
            ready, hints = self.cli._check_provider_readiness(
                "uutel-cloud/gemini-2.5-pro"
            )

        assert ready is False
        assert any("credentials" in hint.lower() for hint in hints)

    def test_check_provider_readiness_cloud_with_api_key(
        self, tmp_path, monkeypatch
    ) -> None:
        """Cloud Code readiness should pass with API key and project id."""

        monkeypatch.setattr("uutel.__main__.Path.home", lambda: tmp_path)
        monkeypatch.setattr("uutel.__main__.shutil.which", lambda name: None)

        with patch.dict(
            os.environ,
            {"GOOGLE_API_KEY": "test", "CLOUD_CODE_PROJECT": "demo-project"},
            clear=True,
        ):
            ready, hints = self.cli._check_provider_readiness(
                "uutel-cloud/gemini-2.5-pro"
            )

        assert ready is True
        assert hints == []

    def test_check_provider_readiness_cloud_with_service_account(
        self, tmp_path, monkeypatch
    ) -> None:
        """Service-account credentials should satisfy Cloud Code requirements."""

        monkeypatch.setattr("uutel.__main__.Path.home", lambda: tmp_path)
        monkeypatch.setattr("uutel.__main__.shutil.which", lambda name: None)

        service_account = tmp_path / "service-account.json"
        service_account.write_text(
            json.dumps(
                {
                    "type": "service_account",
                    "client_email": "uutel-cloud@example.com",
                    "project_id": "demo-project",
                    "private_key": "-----BEGIN PRIVATE KEY-----\nREDACTED\n-----END PRIVATE KEY-----\n",
                }
            ),
            encoding="utf-8",
        )

        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": str(service_account),
                "CLOUD_CODE_PROJECT": "demo-project",
            },
            clear=True,
        ):
            ready, hints = self.cli._check_provider_readiness(
                "uutel-cloud/gemini-2.5-pro"
            )

        assert ready is True
        assert hints == []

    def test_check_provider_readiness_cloud_warns_on_missing_service_account(
        self, tmp_path, monkeypatch
    ) -> None:
        """Missing service-account files should emit explicit guidance."""

        monkeypatch.setattr("uutel.__main__.Path.home", lambda: tmp_path)
        monkeypatch.setattr("uutel.__main__.shutil.which", lambda name: None)

        missing_path = tmp_path / "missing-service-account.json"

        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": str(missing_path),
                "CLOUD_CODE_PROJECT": "demo-project",
            },
            clear=True,
        ):
            ready, hints = self.cli._check_provider_readiness(
                "uutel-cloud/gemini-2.5-pro"
            )

        assert ready is False
        assert any(str(missing_path) in hint for hint in hints), (
            "Guidance should mention the missing service account path"
        )
        assert any("credentials not detected" in hint.lower() for hint in hints), (
            "General credential guidance should still appear"
        )

    def test_check_provider_readiness_cloud_rejects_invalid_service_account(
        self, tmp_path, monkeypatch
    ) -> None:
        """Unreadable service-account payloads should fail readiness."""

        monkeypatch.setattr("uutel.__main__.Path.home", lambda: tmp_path)
        monkeypatch.setattr("uutel.__main__.shutil.which", lambda name: None)

        service_account = tmp_path / "broken.json"
        service_account.write_text("not-json", encoding="utf-8")

        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": str(service_account),
                "CLOUD_CODE_PROJECT": "demo-project",
            },
            clear=True,
        ):
            ready, hints = self.cli._check_provider_readiness(
                "uutel-cloud/gemini-2.5-pro"
            )

        assert ready is False, "Invalid service account should block readiness"
        assert any(
            "invalid service account" in hint.lower() or "parse" in hint.lower()
            for hint in hints
        ), "Guidance should flag broken service-account content"

    def test_check_provider_readiness_cloud_handles_decode_error(
        self, tmp_path, monkeypatch
    ) -> None:
        """Binary service-account files should surface a readable guidance message."""

        monkeypatch.setattr("uutel.__main__.Path.home", lambda: tmp_path)
        monkeypatch.setattr("uutel.__main__.shutil.which", lambda name: None)

        service_account = tmp_path / "binary.json"
        service_account.write_bytes(b"\xff\xfe\x00broken")

        with patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": str(service_account),
                "CLOUD_CODE_PROJECT": "demo-project",
            },
            clear=True,
        ):
            ready, hints = self.cli._check_provider_readiness(
                "uutel-cloud/gemini-2.5-pro"
            )

        assert ready is False, "Decode failures should block readiness"
        assert any(
            "decode" in hint.lower() or "read" in hint.lower() for hint in hints
        ), "Guidance should mention the unreadable service-account file"

    def test_check_provider_readiness_cloud_uses_gcloud_config(
        self, tmp_path, monkeypatch
    ) -> None:
        """Default gcloud project should satisfy readiness when env vars absent."""

        monkeypatch.setattr("uutel.__main__.Path.home", lambda: tmp_path)
        monkeypatch.setattr("uutel.__main__.shutil.which", lambda name: None)

        gcloud_dir = tmp_path / ".config" / "gcloud" / "configurations"
        gcloud_dir.mkdir(parents=True)
        (gcloud_dir / "config_default").write_text(
            "[core]\nproject = gcloud-project\n",
            encoding="utf-8",
        )

        gemini_config = tmp_path / ".config" / "gemini"
        gemini_config.mkdir(parents=True)
        (gemini_config / "oauth_creds.json").write_text("{}", encoding="utf-8")

        with patch.dict(os.environ, {}, clear=True):
            ready, hints = self.cli._check_provider_readiness(
                "uutel-cloud/gemini-2.5-pro"
            )

        assert ready is True
        assert any("gcloud config" in hint.lower() for hint in hints)


class TestCompletionTextExtraction:
    """Edge cases for `_extract_completion_text`."""

    def setup_method(self) -> None:
        """Create a fresh CLI instance for extraction tests."""
        self.cli = UUTELCLI()

    def test_extract_completion_text_skips_empty_first_choice(self) -> None:
        """When the first choice lacks content, the helper should fall back to later choices."""
        response = MagicMock()
        empty_choice = MagicMock()
        empty_choice.message = MagicMock(content="")
        filled_choice = MagicMock()
        filled_choice.message = MagicMock(content="Second choice text")
        response.choices = [empty_choice, filled_choice]

        extracted = self.cli._extract_completion_text(response)

        assert extracted == "Second choice text", (
            "Helper should return first non-empty choice content"
        )

    def test_extract_completion_text_handles_structured_content_list(self) -> None:
        """Structured content arrays with dictionaries should be flattened into plain text."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "Hello"},
                            {"type": "text", "text": ", world"},
                        ]
                    }
                }
            ]
        }

        extracted = self.cli._extract_completion_text(response)

        assert extracted == "Hello, world", (
            "Helper should join structured text segments in order"
        )


class TestSetupProviders:
    """Ensure provider registration coexists with existing LiteLLM handlers."""

    def test_setup_providers_preserves_existing_entries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing custom_provider_map entries should survive setup."""

        sentinel = {"provider": "existing-provider", "custom_handler": object()}
        shared_map = [sentinel.copy()]

        monkeypatch.setattr(
            "uutel.__main__.litellm.custom_provider_map", shared_map, raising=False
        )
        monkeypatch.setattr("litellm.custom_provider_map", shared_map, raising=False)

        setup_providers()

        providers = litellm.custom_provider_map
        assert any(
            entry.get("provider") == "existing-provider" for entry in providers
        ), "Existing provider entries should remain after setup"
        assert any(entry.get("provider") == "uutel-claude" for entry in providers), (
            "UUTEL providers should still be registered"
        )

    def test_setup_providers_is_idempotent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repeated setup invocations should not duplicate UUTEL handlers."""

        shared_map: list[dict[str, Any]] = []

        monkeypatch.setattr(
            "uutel.__main__.litellm.custom_provider_map", shared_map, raising=False
        )
        monkeypatch.setattr("litellm.custom_provider_map", shared_map, raising=False)

        setup_providers()

        initial_counts: dict[str, int] = {}
        for entry in litellm.custom_provider_map:
            provider = entry.get("provider")
            if provider in {
                "my-custom-llm",
                "uutel-codex",
                "uutel-claude",
                "uutel-gemini",
                "uutel-cloud",
            }:
                initial_counts[provider] = initial_counts.get(provider, 0) + 1

        setup_providers()

        for provider, count in initial_counts.items():
            final_count = sum(
                1
                for entry in litellm.custom_provider_map
                if entry.get("provider") == provider
            )
            assert final_count == count, (
                f"Provider {provider} should remain at {count} entries, found {final_count}"
            )

    def test_setup_providers_handles_none_existing_map(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """None-valued custom_provider_map should initialise cleanly."""

        monkeypatch.setattr(
            "uutel.__main__.litellm.custom_provider_map", None, raising=False
        )
        monkeypatch.setattr("litellm.custom_provider_map", None, raising=False)

        setup_providers()

        providers = litellm.custom_provider_map
        assert isinstance(providers, list), (
            "Provider map should normalise None into a list"
        )
        assert any(entry.get("provider") == "uutel-claude" for entry in providers), (
            "UUTEL handlers should still register after normalisation"
        )

    def test_setup_providers_preserves_tuple_entries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tuple-based provider maps should retain non-UUTEL entries."""

        sentinel_entry = {"provider": "external-provider", "custom_handler": object()}
        tuple_map = (sentinel_entry.copy(),)

        monkeypatch.setattr(
            "uutel.__main__.litellm.custom_provider_map", tuple_map, raising=False
        )
        monkeypatch.setattr("litellm.custom_provider_map", tuple_map, raising=False)

        setup_providers()

        providers = litellm.custom_provider_map
        assert isinstance(providers, list), "Tuple input should normalise to list"
        assert any(
            entry.get("provider") == "external-provider" for entry in providers
        ), "Existing tuple entries should survive normalisation"

    def test_setup_providers_converts_dict_map(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dict-style provider maps should be converted while preserving handlers."""

        handler = object()
        dict_map = {"external-provider": handler}

        monkeypatch.setattr(
            "uutel.__main__.litellm.custom_provider_map", dict_map, raising=False
        )
        monkeypatch.setattr("litellm.custom_provider_map", dict_map, raising=False)

        setup_providers()

        providers = litellm.custom_provider_map
        assert isinstance(providers, list), "Dict input should normalise to list"
        assert any(
            entry.get("provider") == "external-provider"
            and entry.get("custom_handler") is handler
            for entry in providers
        ), "Dict entries should be preserved as provider/custom_handler pairs"


class TestCLIStreamingSanitisation:
    """Validate sanitisation of streamed provider output."""

    def setup_method(self) -> None:
        self.cli = UUTELCLI()

    @patch("litellm.completion")
    def test_streaming_output_sanitises_control_sequences(
        self, mock_completion: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """ANSI and control bytes should be stripped from streamed output."""

        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = "\x1b[31mHello\x1b[0m\x07 world"
        mock_completion.return_value = [chunk]

        result = self.cli.complete("Prompt", stream=True)

        captured = capsys.readouterr()
        assert "\x1b" not in captured.out, "ANSI escape sequences should be removed"
        assert "\x07" not in captured.out, "Bell characters should be removed"
        assert "Hello world" in captured.out, "Sanitised text should remain visible"
        assert result == "Hello world", "Return value should reflect sanitised text"


@patch("uutel.__main__.fire.Fire")
def test_main_handles_keyboard_interrupt(mock_fire, capsys) -> None:
    """main() should translate KeyboardInterrupt into a friendly cancellation message."""

    mock_fire.side_effect = KeyboardInterrupt

    main()

    captured = capsys.readouterr()
    assert "⚪ Operation cancelled by user" in captured.err


@patch("uutel.__main__.fire.Fire")
def test_main_handles_broken_pipe_without_crash(mock_fire, capsys) -> None:
    """BrokenPipeError from Fire should be swallowed without stderr chatter."""

    mock_fire.side_effect = BrokenPipeError

    main()

    captured = capsys.readouterr()
    assert captured.err == ""
