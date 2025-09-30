# this_file: tests/test_codex_provider.py
"""Test suite for Codex provider functionality."""

import json
from collections.abc import Callable
from typing import Any
from unittest.mock import Mock, patch

import pytest
from litellm.types.utils import ModelResponse

from uutel.core.exceptions import UUTELError
from uutel.providers.codex import CodexUU
from uutel.providers.codex.custom_llm import CodexCustomLLM


class DummyStreamResponse:
    """Synchronous mock response yielding pre-defined SSE lines."""

    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code

    def __enter__(self) -> "DummyStreamResponse":
        return self

    def __exit__(
        self, exc_type, exc, tb
    ) -> None:  # pragma: no cover - nothing to clean
        return None

    def iter_lines(self) -> list[str]:
        yield from self._lines


class DummyClient:
    """Minimal HTTP client stub returning DummyStreamResponse."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.last_request: dict[str, Any] | None = None

    def stream(self, method: str, url: str, **kwargs) -> DummyStreamResponse:
        self.last_request = {"method": method, "url": url, **kwargs}
        return DummyStreamResponse(self.lines)


class AsyncDummyStreamResponse:
    """Async mock response yielding SSE lines."""

    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code

    async def __aenter__(self) -> "AsyncDummyStreamResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class AsyncDummyClient:
    """Minimal async HTTP client stub returning AsyncDummyStreamResponse."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.last_request: dict[str, Any] | None = None

    def stream(self, method: str, url: str, **kwargs) -> AsyncDummyStreamResponse:
        self.last_request = {"method": method, "url": url, **kwargs}
        return AsyncDummyStreamResponse(self.lines)


@pytest.fixture
def codex_client_factory() -> Callable[..., tuple[Mock, dict]]:
    """Provide a factory for creating stubbed Codex HTTP clients."""

    def _factory(
        *,
        content: str = "Mock response from Codex",
        finish_reason: str = "stop",
        usage: dict | None = None,
        tool_calls: list[dict] | None = None,
        raw_response: dict | None = None,
    ) -> tuple[Mock, dict]:
        captured: dict[str, object] = {}

        client = Mock(name="codex_http_client")

        def _post(url: str, *, headers=None, json=None) -> Mock:
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json

            response = Mock(name="codex_http_response")
            response.raise_for_status.return_value = None
            if raw_response is not None:
                response.json.return_value = raw_response
            else:
                message_payload: dict[str, Any] = {"content": content}
                if tool_calls is not None:
                    message_payload["tool_calls"] = tool_calls
                response.json.return_value = {
                    "choices": [
                        {
                            "message": message_payload,
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": usage
                    or {
                        "prompt_tokens": 12,
                        "completion_tokens": 24,
                        "total_tokens": 36,
                    },
                }
            return response

        client.post.side_effect = _post
        client.close = Mock(name="close")

        return client, captured

    return _factory


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

    def test_completion_basic_functionality(
        self, codex_client_factory: Callable[..., tuple[Mock, dict]]
    ) -> None:
        """Test basic completion functionality."""
        codex = CodexUU()
        model_response = ModelResponse()
        model_response.choices = [Mock()]
        model_response.choices[0].message = Mock()

        client, captured = codex_client_factory()

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
            client=client,
        )

        assert result == model_response
        assert result.model == "gpt-4o"
        assert "mock response" in result.choices[0].message.content.lower()
        assert result.choices[0].finish_reason == "stop"
        assert captured["url"] == "https://api.example.com/chat/completions"
        assert captured["json"]["messages"][0]["content"] == "Hello"
        client.close.assert_not_called()

    def test_completion_translates_tool_calls(
        self, codex_client_factory: Callable[..., tuple[Mock, dict]]
    ) -> None:
        """Codex completion should normalise tool call payloads."""

        codex = CodexUU()
        model_response = ModelResponse()
        model_response.choices = [Mock()]
        model_response.choices[0].message = Mock()

        client, _ = codex_client_factory(
            content="",
            tool_calls=[
                {"id": "item_1", "name": "search", "arguments": {"query": "docs"}}
            ],
        )

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
            client=client,
        )

        tool_calls = result.choices[0].message.tool_calls
        assert tool_calls is not None
        assert tool_calls[0]["function"]["name"] == "search"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {"query": "docs"}

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

    def test_completion_error_handling(
        self, codex_client_factory: Callable[..., tuple[Mock, dict]]
    ) -> None:
        """Test completion error handling."""
        codex = CodexUU()
        model_response = Mock()
        # Make model_response.model assignment fail
        type(model_response).model = Mock(side_effect=Exception("Test error"))

        client, _ = codex_client_factory()

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
                client=client,
            )

        assert "Codex completion failed" in str(exc_info.value)


class TestCodexUUAsyncCompletion:
    """Test CodexUU async completion functionality."""

    def test_acompletion_basic_functionality(
        self, codex_client_factory: Callable[..., tuple[Mock, dict]]
    ) -> None:
        """Test basic async completion functionality."""
        # For now, test that acompletion is callable and returns expected result
        # In a full implementation, this would be properly async
        import asyncio

        codex = CodexUU()
        model_response = ModelResponse()
        model_response.choices = [Mock()]
        model_response.choices[0].message = Mock()
        client, captured = codex_client_factory(content="Async mock response")

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
                client=client,
            )

        result = asyncio.run(run_test())
        assert result == model_response
        assert result.model == "gpt-4o"
        assert "async mock response" in result.choices[0].message.content.lower()
        assert captured["json"]["model"] == "gpt-4o"


class TestCodexUUStreaming:
    """Test CodexUU streaming functionality."""

    def test_streaming_basic_functionality(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Codex streaming should parse text deltas and emit finish chunk."""

        codex = CodexUU()
        monkeypatch.setattr(codex, "_load_codex_auth", lambda: ("token", "account"))

        lines = [
            'data: {"type": "response.output_text.delta", "delta": "Hello"}',
            "",
            'data: {"type": "response.output_text.delta", "delta": " world"}',
            "",
            'data: {"type": "response.completed", "response": {"status": "stop", "usage": {"input_tokens": 3, "output_tokens": 4}}}',
            "",
            "data: [DONE]",
            "",
        ]

        client = DummyClient(lines)
        chunks = list(
            codex.streaming(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                api_base="https://chatgpt.com/backend-api",
                custom_prompt_dict={},
                model_response=ModelResponse(),
                print_verbose=Mock(),
                encoding="utf-8",
                api_key=None,
                logging_obj=Mock(),
                optional_params={},
                client=client,
            )
        )

        assert [chunk["text"] for chunk in chunks[:-1]] == ["Hello", " world"]
        finish_chunk = chunks[-1]
        assert finish_chunk["is_finished"] is True
        assert finish_chunk["usage"] == {
            "input_tokens": 3,
            "output_tokens": 4,
            "total_tokens": 7,
        }

    def test_streaming_handles_tool_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Codex streaming should emit tool use chunks from SSE events."""

        codex = CodexUU()
        monkeypatch.setattr(codex, "_load_codex_auth", lambda: ("token", "account"))

        lines = [
            'data: {"type": "response.output_item.added", "item": {"type": "function_call", "name": "search", "id": "item_1"}}',
            "",
            'data: {"type": "response.function_call_arguments.done", "item_id": "item_1", "arguments": {"query": "docs"}}',
            "",
            'data: {"type": "response.completed", "response": {"status": "stop"}}',
            "",
            "data: [DONE]",
            "",
        ]

        client = DummyClient(lines)
        chunks = list(
            codex.streaming(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                api_base="https://chatgpt.com/backend-api",
                custom_prompt_dict={},
                model_response=ModelResponse(),
                print_verbose=Mock(),
                encoding="utf-8",
                api_key=None,
                logging_obj=Mock(),
                optional_params={},
                client=client,
            )
        )

        tool_chunk = next(chunk for chunk in chunks if chunk["tool_use"] is not None)
        assert tool_chunk["tool_use"]["id"] == "item_1"
        assert tool_chunk["tool_use"]["name"] == "search"
        assert json.loads(tool_chunk["tool_use"]["arguments"]) == {"query": "docs"}

    def test_astreaming_basic_functionality(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test basic async streaming functionality."""
        import asyncio

        codex = CodexUU()

        lines = [
            'data: {"type": "response.output_text.delta", "delta": "Hello"}',
            "",
            'data: {"type": "response.completed", "response": {"status": "stop"}}',
            "",
            "data: [DONE]",
            "",
        ]

        async def run_test() -> list[dict]:
            monkeypatch.setattr(codex, "_load_codex_auth", lambda: ("token", "account"))
            client = AsyncDummyClient(lines)
            chunks: list[dict] = []
            async for chunk in codex.astreaming(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                api_base="https://chatgpt.com/backend-api",
                custom_prompt_dict={},
                model_response=ModelResponse(),
                print_verbose=Mock(),
                encoding="utf-8",
                api_key=None,
                logging_obj=Mock(),
                optional_params={},
                client=client,
            ):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run_test())
        assert [chunk["text"] for chunk in chunks[:-1]] == ["Hello"]
        assert chunks[-1]["is_finished"] is True


class TestCodexUULogging:
    """Test CodexUU logging functionality."""

    @patch("uutel.providers.codex.provider.logger")
    def test_completion_logging(
        self,
        mock_logger,
        codex_client_factory: Callable[..., tuple[Mock, dict]],
    ) -> None:
        """Test completion request logging."""
        codex = CodexUU()
        model_response = ModelResponse()
        model_response.choices = [Mock()]
        model_response.choices[0].message = Mock()
        client, _ = codex_client_factory()

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
            client=client,
        )

        # Verify debug logging was called
        mock_logger.debug.assert_called()
        debug_calls = [call.args[0] for call in mock_logger.debug.call_args_list]
        assert any("Codex completion request" in call for call in debug_calls)
        assert any("completed successfully" in call for call in debug_calls)

    @patch("uutel.providers.codex.provider.logger")
    def test_completion_error_logging(
        self,
        mock_logger,
        codex_client_factory: Callable[..., tuple[Mock, dict]],
    ) -> None:
        """Test completion error logging."""
        codex = CodexUU()
        model_response = Mock()
        type(model_response).model = Mock(side_effect=Exception("Test error"))
        client, _ = codex_client_factory()

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
                client=client,
            )

        # Verify error logging was called
        mock_logger.error.assert_called()
        error_call = mock_logger.error.call_args[0][0]
        assert "Codex completion failed" in error_call


class TestCodexUUEdgeCases:
    """Test CodexUU edge cases and robustness."""

    def test_completion_with_various_models(
        self, codex_client_factory: Callable[..., tuple[Mock, dict]]
    ) -> None:
        """Test completion works with different supported models."""
        codex = CodexUU()

        for model in codex.supported_models:
            model_response = ModelResponse()
            model_response.choices = [Mock()]
            model_response.choices[0].message = Mock()
            client, captured = codex_client_factory(content=f"Response for {model}")

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
                client=client,
            )

            assert result.model == model
            assert captured["json"]["model"] == model
            assert f"response for {model}" in result.choices[0].message.content.lower()

    def test_completion_with_complex_messages(
        self, codex_client_factory: Callable[..., tuple[Mock, dict]]
    ) -> None:
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

        client, captured = codex_client_factory(content="Structured mock response")

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
            client=client,
        )

        assert result == model_response
        assert captured["json"]["messages"] == complex_messages
        assert "structured mock response" in result.choices[0].message.content.lower()

    def test_completion_with_optional_params(
        self, codex_client_factory: Callable[..., tuple[Mock, dict]]
    ) -> None:
        """Test completion with various optional parameters."""
        codex = CodexUU()
        model_response = ModelResponse()
        model_response.choices = [Mock()]
        model_response.choices[0].message = Mock()

        client, captured = codex_client_factory(content="Optional response")

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
            client=client,
        )

        assert result == model_response
        assert result.choices[0].finish_reason == "stop"
        assert captured["headers"]["Authorization"] == "Bearer test-key"
        assert captured["headers"]["Custom-Header"] == "test"
        assert captured["json"]["temperature"] == pytest.approx(0.7)
        assert captured["json"]["max_tokens"] == 100
        assert captured["json"]["top_p"] == pytest.approx(0.9)


class TestCodexCustomLLMDelegation:
    """Ensure the CustomLLM adapter delegates to CodexUU."""

    def _minimal_model_response(self) -> ModelResponse:
        response = ModelResponse()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        return response

    @patch("uutel.providers.codex.custom_llm.CodexUU")
    def test_completion_delegates_to_provider(self, mock_codex_uu: Mock) -> None:
        provider_instance = mock_codex_uu.return_value
        expected_response = self._minimal_model_response()
        provider_instance.completion.return_value = expected_response

        custom_llm = CodexCustomLLM()
        result = custom_llm.completion(
            model="uutel-codex/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            api_base="",
            custom_prompt_dict={},
            model_response=self._minimal_model_response(),
            print_verbose=Mock(),
            encoding="utf-8",
            api_key="api-key",
            logging_obj=Mock(),
            optional_params={},
        )

        provider_instance.completion.assert_called_once()
        assert result is expected_response

    @patch("uutel.providers.codex.custom_llm.CodexUU")
    def test_streaming_delegates_to_provider(self, mock_codex_uu: Mock) -> None:
        provider_instance = mock_codex_uu.return_value
        provider_instance.streaming.return_value = iter([{"text": "chunk"}])

        custom_llm = CodexCustomLLM()
        iterator = custom_llm.streaming(
            model="uutel-codex/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            api_base="",
            custom_prompt_dict={},
            model_response=self._minimal_model_response(),
            print_verbose=Mock(),
            encoding="utf-8",
            api_key="api-key",
            logging_obj=Mock(),
            optional_params={},
        )

        assert list(iterator) == [{"text": "chunk"}]
        provider_instance.streaming.assert_called_once()

    @pytest.mark.asyncio
    @patch("uutel.providers.codex.custom_llm.CodexUU")
    async def test_async_completion_delegates_to_provider(
        self, mock_codex_uu: Mock
    ) -> None:
        provider_instance = mock_codex_uu.return_value
        expected_response = self._minimal_model_response()
        provider_instance.acompletion.return_value = expected_response

        custom_llm = CodexCustomLLM()
        result = await custom_llm.acompletion(
            model="uutel-codex/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            api_base="",
            custom_prompt_dict={},
            model_response=self._minimal_model_response(),
            print_verbose=Mock(),
            encoding="utf-8",
            api_key="api-key",
            logging_obj=Mock(),
            optional_params={},
        )

        provider_instance.acompletion.assert_called_once()
        assert result is expected_response

    @pytest.mark.asyncio
    @patch("uutel.providers.codex.custom_llm.CodexUU")
    async def test_async_streaming_delegates_to_provider(
        self, mock_codex_uu: Mock
    ) -> None:
        async def _agen():
            yield {"text": "async"}

        provider_instance = mock_codex_uu.return_value
        provider_instance.astreaming.return_value = _agen()

        custom_llm = CodexCustomLLM()
        result = []
        async for chunk in custom_llm.astreaming(
            model="uutel-codex/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            api_base="",
            custom_prompt_dict={},
            model_response=self._minimal_model_response(),
            print_verbose=Mock(),
            encoding="utf-8",
            api_key="api-key",
            logging_obj=Mock(),
            optional_params={},
        ):
            result.append(chunk)

        provider_instance.astreaming.assert_called_once()
        assert result == [{"text": "async"}]
