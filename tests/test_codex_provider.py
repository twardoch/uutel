# this_file: tests/test_codex_provider.py
"""Test suite for Codex provider functionality."""

from unittest.mock import Mock, patch

import pytest
from litellm.types.utils import ModelResponse

from uutel.core.exceptions import UUTELError
from uutel.providers.codex import CodexUU


class TestCodexUUBasics:
    """Test basic CodexUU functionality."""

    def test_codex_uu_initialization(self) -> None:
        """Test CodexUU initializes correctly."""
        codex = CodexUU()

        assert codex.provider_name == "codex"
        assert isinstance(codex.supported_models, list)
        assert len(codex.supported_models) > 0
        assert "gpt-4o" in codex.supported_models

    def test_codex_uu_supported_models(self) -> None:
        """Test CodexUU has expected supported models."""
        codex = CodexUU()
        expected_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "o1-preview",
            "o1-mini",
        ]

        for model in expected_models:
            assert model in codex.supported_models


class TestCodexUUCompletion:
    """Test CodexUU completion functionality."""

    def test_completion_basic_functionality(self) -> None:
        """Test basic completion functionality."""
        codex = CodexUU()
        model_response = ModelResponse()
        model_response.choices = [Mock()]
        model_response.choices[0].message = Mock()

        # Mock the required parameters
        result = codex.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            api_base="https://api.example.com",
            custom_prompt_dict={},
            model_response=model_response,
            print_verbose=Mock(),
            encoding="utf-8",
            api_key="test-key",
            logging_obj=Mock(),
            optional_params={},
        )

        assert result == model_response
        assert result.model == "gpt-4o"
        assert "mock response" in result.choices[0].message.content.lower()
        assert result.choices[0].finish_reason == "stop"

    def test_completion_with_empty_model(self) -> None:
        """Test completion fails with empty model."""
        codex = CodexUU()
        model_response = ModelResponse()

        with pytest.raises(UUTELError) as exc_info:
            codex.completion(
                model="",
                messages=[{"role": "user", "content": "Hello"}],
                api_base="https://api.example.com",
                custom_prompt_dict={},
                model_response=model_response,
                print_verbose=Mock(),
                encoding="utf-8",
                api_key="test-key",
                logging_obj=Mock(),
                optional_params={},
            )

        assert "Model name is required" in str(exc_info.value)
        assert exc_info.value.provider == "codex"

    def test_completion_with_empty_messages(self) -> None:
        """Test completion fails with empty messages."""
        codex = CodexUU()
        model_response = ModelResponse()

        with pytest.raises(UUTELError) as exc_info:
            codex.completion(
                model="gpt-4o",
                messages=[],
                api_base="https://api.example.com",
                custom_prompt_dict={},
                model_response=model_response,
                print_verbose=Mock(),
                encoding="utf-8",
                api_key="test-key",
                logging_obj=Mock(),
                optional_params={},
            )

        assert "Messages are required" in str(exc_info.value)

    def test_completion_error_handling(self) -> None:
        """Test completion error handling."""
        codex = CodexUU()
        model_response = Mock()
        # Make model_response.model assignment fail
        type(model_response).model = Mock(side_effect=Exception("Test error"))

        with pytest.raises(UUTELError) as exc_info:
            codex.completion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                api_base="https://api.example.com",
                custom_prompt_dict={},
                model_response=model_response,
                print_verbose=Mock(),
                encoding="utf-8",
                api_key="test-key",
                logging_obj=Mock(),
                optional_params={},
            )

        assert "Codex completion failed" in str(exc_info.value)


class TestCodexUUAsyncCompletion:
    """Test CodexUU async completion functionality."""

    def test_acompletion_basic_functionality(self) -> None:
        """Test basic async completion functionality."""
        # For now, test that acompletion is callable and returns expected result
        # In a full implementation, this would be properly async
        import asyncio

        codex = CodexUU()
        model_response = ModelResponse()
        model_response.choices = [Mock()]
        model_response.choices[0].message = Mock()

        async def run_test():
            return await codex.acompletion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                api_base="https://api.example.com",
                custom_prompt_dict={},
                model_response=model_response,
                print_verbose=Mock(),
                encoding="utf-8",
                api_key="test-key",
                logging_obj=Mock(),
                optional_params={},
            )

        result = asyncio.run(run_test())
        assert result == model_response
        assert result.model == "gpt-4o"


class TestCodexUUStreaming:
    """Test CodexUU streaming functionality."""

    def test_streaming_basic_functionality(self) -> None:
        """Test basic streaming functionality."""
        codex = CodexUU()
        model_response = ModelResponse()

        chunks = list(
            codex.streaming(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                api_base="https://api.example.com",
                custom_prompt_dict={},
                model_response=model_response,
                print_verbose=Mock(),
                encoding="utf-8",
                api_key="test-key",
                logging_obj=Mock(),
                optional_params={},
            )
        )

        assert len(chunks) > 0
        assert all(isinstance(chunk, dict) for chunk in chunks)

        # Check that chunks have the expected GenericStreamingChunk structure
        for chunk in chunks:
            assert "text" in chunk
            assert "finish_reason" in chunk
            assert "index" in chunk
            assert "is_finished" in chunk
            assert "tool_use" in chunk
            assert "usage" in chunk

        # Check that the last chunk has finish_reason "stop"
        last_chunk = chunks[-1]
        assert last_chunk["finish_reason"] == "stop"
        assert last_chunk["is_finished"] is True

    def test_astreaming_basic_functionality(self) -> None:
        """Test basic async streaming functionality."""
        import asyncio

        codex = CodexUU()
        model_response = ModelResponse()

        async def run_test():
            chunks = []
            async for chunk in codex.astreaming(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                api_base="https://api.example.com",
                custom_prompt_dict={},
                model_response=model_response,
                print_verbose=Mock(),
                encoding="utf-8",
                api_key="test-key",
                logging_obj=Mock(),
                optional_params={},
            ):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run_test())
        assert len(chunks) > 0
        assert all(isinstance(chunk, dict) for chunk in chunks)

        # Check that chunks have the expected GenericStreamingChunk structure
        for chunk in chunks:
            assert "text" in chunk
            assert "finish_reason" in chunk
            assert "index" in chunk
            assert "is_finished" in chunk
            assert "tool_use" in chunk
            assert "usage" in chunk


class TestCodexUULogging:
    """Test CodexUU logging functionality."""

    @patch("uutel.providers.codex.provider.logger")
    def test_completion_logging(self, mock_logger) -> None:
        """Test completion request logging."""
        codex = CodexUU()
        model_response = ModelResponse()
        model_response.choices = [Mock()]
        model_response.choices[0].message = Mock()

        codex.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            api_base="https://api.example.com",
            custom_prompt_dict={},
            model_response=model_response,
            print_verbose=Mock(),
            encoding="utf-8",
            api_key="test-key",
            logging_obj=Mock(),
            optional_params={},
        )

        # Verify debug logging was called
        mock_logger.debug.assert_called()
        debug_calls = [call.args[0] for call in mock_logger.debug.call_args_list]
        assert any("Codex completion request" in call for call in debug_calls)
        assert any("completed successfully" in call for call in debug_calls)

    @patch("uutel.providers.codex.provider.logger")
    def test_completion_error_logging(self, mock_logger) -> None:
        """Test completion error logging."""
        codex = CodexUU()
        model_response = Mock()
        type(model_response).model = Mock(side_effect=Exception("Test error"))

        with pytest.raises(UUTELError):
            codex.completion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                api_base="https://api.example.com",
                custom_prompt_dict={},
                model_response=model_response,
                print_verbose=Mock(),
                encoding="utf-8",
                api_key="test-key",
                logging_obj=Mock(),
                optional_params={},
            )

        # Verify error logging was called
        mock_logger.error.assert_called()
        error_call = mock_logger.error.call_args[0][0]
        assert "Codex completion failed" in error_call


class TestCodexUUEdgeCases:
    """Test CodexUU edge cases and robustness."""

    def test_completion_with_various_models(self) -> None:
        """Test completion works with different supported models."""
        codex = CodexUU()

        for model in codex.supported_models:
            model_response = ModelResponse()
            model_response.choices = [Mock()]
            model_response.choices[0].message = Mock()

            result = codex.completion(
                model=model,
                messages=[{"role": "user", "content": "Test"}],
                api_base="https://api.example.com",
                custom_prompt_dict={},
                model_response=model_response,
                print_verbose=Mock(),
                encoding="utf-8",
                api_key="test-key",
                logging_obj=Mock(),
                optional_params={},
            )

            assert result.model == model

    def test_completion_with_complex_messages(self) -> None:
        """Test completion with complex message structures."""
        codex = CodexUU()
        model_response = ModelResponse()
        model_response.choices = [Mock()]
        model_response.choices[0].message = Mock()

        complex_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        result = codex.completion(
            model="gpt-4o",
            messages=complex_messages,
            api_base="https://api.example.com",
            custom_prompt_dict={},
            model_response=model_response,
            print_verbose=Mock(),
            encoding="utf-8",
            api_key="test-key",
            logging_obj=Mock(),
            optional_params={},
        )

        assert result == model_response
        assert str(len(complex_messages)) in result.choices[0].message.content

    def test_completion_with_optional_params(self) -> None:
        """Test completion with various optional parameters."""
        codex = CodexUU()
        model_response = ModelResponse()
        model_response.choices = [Mock()]
        model_response.choices[0].message = Mock()

        optional_params = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "stream": False,
        }

        result = codex.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Test"}],
            api_base="https://api.example.com",
            custom_prompt_dict={},
            model_response=model_response,
            print_verbose=Mock(),
            encoding="utf-8",
            api_key="test-key",
            logging_obj=Mock(),
            optional_params=optional_params,
            timeout=30.0,
            headers={"Custom-Header": "test"},
        )

        assert result == model_response
