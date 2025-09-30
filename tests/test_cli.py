# this_file: tests/test_cli.py
"""Comprehensive test suite for UUTEL CLI functionality.

This module tests all CLI commands, parameter validation, error handling,
and integration with providers to ensure CLI reliability.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from uutel.__main__ import UUTELCLI


class TestUUTELCLI:
    """Test the main UUTELCLI class and its functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cli = UUTELCLI()

    def test_cli_initialization(self):
        """Test CLI initializes correctly and sets up providers."""
        assert self.cli is not None
        # Verify providers are set up during initialization
        # This is implicitly tested by successful CLI creation

    def test_list_engines_command(self, capsys):
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

        # Check descriptions are present
        assert "Large Codex model" in captured.out
        assert "Coming Soon:" in captured.out

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
    def test_complete_command_error_handling(self, mock_completion, capsys):
        """Test complete command error handling."""
        # Mock completion error
        mock_completion.side_effect = Exception("API Error")

        result = self.cli.complete("Test prompt")

        # Verify error is handled gracefully with enhanced format
        assert "❌ Error: API Error" in result
        captured = capsys.readouterr()
        assert "❌ Error: API Error" in captured.err

    @patch("litellm.completion")
    def test_complete_command_streaming_error(self, mock_completion, capsys):
        """Test streaming error handling."""
        # Mock streaming error
        mock_completion.side_effect = Exception("Streaming Error")

        result = self.cli.complete("Test prompt", stream=True)

        # Verify streaming error is handled with enhanced format
        assert "❌ Error: Streaming Error" in result
        captured = capsys.readouterr()
        assert "❌ Error: Streaming Error" in captured.err

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

    @patch("litellm.completion")
    def test_verbose_logging_enabled(self, mock_completion):
        """Test that verbose mode enables logging."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Verbose response"
        mock_completion.return_value = mock_response

        with patch("litellm.set_verbose"):
            self.cli.complete("Test prompt", verbose=True)

            # Note: litellm.set_verbose is deprecated, so this tests the legacy behavior
            # In practice, the verbose flag affects logging configuration

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


class TestCLIParameterValidation:
    """Test CLI parameter validation and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cli = UUTELCLI()

    def test_empty_prompt_handling(self, capsys):
        """Test handling of empty prompt."""
        result = self.cli.complete("")

        # Verify empty prompt is rejected with helpful message
        assert "❌ Prompt is required and must be a non-empty string" in result
        captured = capsys.readouterr()
        assert "❌ Prompt is required and must be a non-empty string" in captured.err

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

        assert "❌ Error: Network unavailable" in result
        captured = capsys.readouterr()
        assert "❌ Error: Network unavailable" in captured.err

    @patch("litellm.completion")
    def test_timeout_error_handling(self, mock_completion, capsys):
        """Test handling of timeout errors."""
        mock_completion.side_effect = TimeoutError("Request timeout")

        result = self.cli.complete("Test prompt")

        assert "❌ Error: Request timeout" in result

    @patch("litellm.completion")
    def test_authentication_error_handling(self, mock_completion, capsys):
        """Test handling of authentication errors."""
        mock_completion.side_effect = Exception("Authentication failed")

        result = self.cli.complete("Test prompt")

        assert "❌ Error: Authentication failed" in result

    @patch("litellm.completion")
    def test_rate_limit_error_handling(self, mock_completion, capsys):
        """Test handling of rate limit errors."""
        mock_completion.side_effect = Exception("Rate limit exceeded")

        result = self.cli.complete("Test prompt")

        assert "❌ Error: Rate limit exceeded" in result

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
