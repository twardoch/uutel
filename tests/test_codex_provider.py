# this_file: tests/test_codex_provider.py
"""Test suite for Codex provider functionality."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import httpx
import litellm
import pytest
from litellm.types.utils import GenericStreamingChunk, ModelResponse

from uutel.core.exceptions import UUTELError
from uutel.providers.codex import CodexUU
from uutel.providers.codex.custom_llm import CodexCustomLLM


class DummyStreamResponse:
    """Synchronous sample response yielding pre-defined SSE lines."""

    def __init__(
        self,
        lines: list[str],
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._lines = lines
        self.status_code = status_code
        self.headers = headers or {}

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

    def __init__(
        self,
        lines: list[str],
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.lines = lines
        self.last_request: dict[str, Any] | None = None
        self.status_code = status_code
        self.headers = headers or {}

    def stream(self, method: str, url: str, **kwargs) -> DummyStreamResponse:
        self.last_request = {"method": method, "url": url, **kwargs}
        return DummyStreamResponse(
            self.lines, status_code=self.status_code, headers=self.headers
        )


class AsyncDummyStreamResponse:
    """Async sample response yielding SSE lines."""

    def __init__(
        self,
        lines: list[str],
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._lines = lines
        self.status_code = status_code
        self.headers = headers or {}

    async def __aenter__(self) -> "AsyncDummyStreamResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class AsyncDummyClient:
    """Minimal async HTTP client stub returning AsyncDummyStreamResponse."""

    def __init__(
        self,
        lines: list[str],
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.lines = lines
        self.last_request: dict[str, Any] | None = None
        self.status_code = status_code
        self.headers = headers or {}

    def stream(self, method: str, url: str, **kwargs) -> AsyncDummyStreamResponse:
        self.last_request = {"method": method, "url": url, **kwargs}
        return AsyncDummyStreamResponse(
            self.lines,
            status_code=self.status_code,
            headers=self.headers,
        )


@pytest.fixture
def codex_client_factory() -> Callable[..., tuple[Mock, dict]]:
    """Provide a factory for creating stubbed Codex HTTP clients."""

    def _factory(
        *,
        content: str = "Codex sample completion output",
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


class TestCodexCustomLLMModelMapping:
    """Regression coverage for CodexCustomLLM model resolution."""

    def test_map_model_name_when_missing_model_then_bad_request(self) -> None:
        """An empty model should raise a litellm.BadRequestError."""

        custom_llm = CodexCustomLLM()

        with pytest.raises(litellm.BadRequestError) as exc_info:
            custom_llm._map_model_name("")

        error = exc_info.value
        assert error.model == ""
        assert error.llm_provider == "codex"
        assert "required" in str(error)

    def test_map_model_name_when_unknown_model_then_bad_request(self) -> None:
        """Unknown models should raise a BadRequestError referencing the provider."""

        custom_llm = CodexCustomLLM()

        with pytest.raises(litellm.BadRequestError) as exc_info:
            custom_llm._map_model_name("unknown-model")

        error = exc_info.value
        assert error.model == "unknown-model"
        assert error.llm_provider == "codex"
        assert "Unsupported" in str(error)

    def test_map_model_name_when_codex_alias_then_resolves_supported_model(
        self,
    ) -> None:
        """Alias names should resolve to actual Codex backend models."""

        custom_llm = CodexCustomLLM()

        resolved = custom_llm._map_model_name("codex-large")

        assert resolved == "gpt-4o"

    def test_map_model_name_when_my_custom_alias_then_resolves_supported_model(
        self,
    ) -> None:
        """my-custom-llm aliases should also resolve to canonical backend IDs."""

        custom_llm = CodexCustomLLM()

        resolved = custom_llm._map_model_name("my-custom-llm/codex-mini")

        assert resolved == "gpt-4o-mini"

    def test_map_model_name_when_alias_uppercase_then_resolves(self) -> None:
        """Alias resolution should be case-insensitive for partner integrations."""

        custom_llm = CodexCustomLLM()

        resolved = custom_llm._map_model_name("CODEX-LARGE")

        assert resolved == "gpt-4o"

    def test_map_model_name_when_unknown_mixed_case_then_bad_request(self) -> None:
        """Mixed-case unknown aliases should still raise informative errors."""

        custom_llm = CodexCustomLLM()

        with pytest.raises(litellm.BadRequestError) as exc_info:
            custom_llm._map_model_name("Codex-Unknown")

        error = exc_info.value
        assert error.model == "Codex-Unknown"
        assert "Unsupported" in str(error)

    def test_map_model_name_when_blank_input_then_requires_model(self) -> None:
        """Whitespace-only inputs should raise a required-model error."""

        custom_llm = CodexCustomLLM()

        with pytest.raises(litellm.BadRequestError) as exc_info:
            custom_llm._map_model_name("   \n  ")

        error = exc_info.value
        assert error.llm_provider == "codex"
        assert "Model name is required" in str(error)

    def test_map_model_name_when_non_string_then_type_error(self) -> None:
        """Non-string inputs should raise a deterministic bad request error."""

        custom_llm = CodexCustomLLM()

        candidates = [None, 123]

        for candidate in candidates:
            with pytest.raises(litellm.BadRequestError) as exc_info:
                custom_llm._map_model_name(candidate)  # type: ignore[arg-type]

            error = exc_info.value
            assert error.llm_provider == "codex"
            assert "Model must be a string" in str(error)

    def test_map_model_name_when_unknown_then_suggests_matches(self) -> None:
        """Unknown aliases should surface close-match suggestions for remediation."""

        custom_llm = CodexCustomLLM()

        with pytest.raises(litellm.BadRequestError) as exc_info:
            custom_llm._map_model_name("codex-lagre")

        message = str(exc_info.value)
        assert "Did you mean" in message
        assert "codex-large" in message
        assert "codex-mini" in message

    def test_map_model_name_when_backend_model_then_passthrough(self) -> None:
        """Direct backend model names should pass straight through."""

        custom_llm = CodexCustomLLM()

        resolved = custom_llm._map_model_name("gpt-4o")

        assert resolved == "gpt-4o"


class TestCodexCustomLLMErrorHandling:
    """Tests for API error translation in CodexCustomLLM."""

    def test_completion_when_provider_raises_uutel_error_then_api_error_has_metadata(
        self,
    ) -> None:
        """CodexCustomLLM should raise APIConnectionError with provider/model info."""

        class _FailingProvider:
            provider_name = "codex"
            supported_models = ["gpt-4o"]

            def completion(self, *args: Any, **kwargs: Any) -> None:
                raise UUTELError("boom", provider="codex")

        custom_llm = CodexCustomLLM(provider=_FailingProvider())

        with pytest.raises(litellm.APIConnectionError) as exc_info:
            custom_llm.completion(model="codex-large")

        error = exc_info.value
        assert error.llm_provider == "codex"
        assert error.model == "gpt-4o"

    def test_streaming_when_provider_raises_uutel_error_then_api_error_has_metadata(
        self,
    ) -> None:
        """Streaming errors should include provider/model metadata."""

        class _FailingProvider:
            provider_name = "codex"
            supported_models = ["gpt-4o"]

            def streaming(
                self, *args: Any, **kwargs: Any
            ) -> Iterator[GenericStreamingChunk]:  # type: ignore[override]
                raise UUTELError("boom", provider="codex")

        custom_llm = CodexCustomLLM(provider=_FailingProvider())

        with pytest.raises(litellm.APIConnectionError) as exc_info:
            next(custom_llm.streaming(model="codex-large"))

        error = exc_info.value
        assert error.llm_provider == "codex"
        assert error.model == "gpt-4o"

    def test_astreaming_when_provider_raises_uutel_error_then_api_error_has_metadata(
        self,
    ) -> None:
        """Async streaming errors should include provider/model metadata."""

        class _FailingProvider:
            provider_name = "codex"
            supported_models = ["gpt-4o"]

            async def astreaming(
                self, *args: Any, **kwargs: Any
            ) -> AsyncIterator[GenericStreamingChunk]:  # type: ignore[override]
                raise UUTELError("boom", provider="codex")
                if False:  # pragma: no cover - appease type checkers
                    yield None

        custom_llm = CodexCustomLLM(provider=_FailingProvider())

        async def _consume() -> None:
            async for _ in custom_llm.astreaming(model="codex-large"):
                pass

        with pytest.raises(litellm.APIConnectionError) as exc_info:
            asyncio.run(_consume())

        error = exc_info.value
        assert error.llm_provider == "codex"
        assert error.model == "gpt-4o"


class TestCodexCustomLLMModelResponseNormalisation:
    """Tests for normalising LiteLLM model responses."""

    def test_prepare_kwargs_when_model_response_missing_then_inserts_placeholder(
        self,
    ) -> None:
        """Providers returning None should receive a minimal ModelResponse."""

        class _NoopProvider(CodexUU):
            def completion(self, *args: Any, **kwargs: Any) -> ModelResponse:  # type: ignore[override]
                return kwargs["model_response"]

        custom_llm = CodexCustomLLM(provider=_NoopProvider())

        response = custom_llm.completion(model="codex-mini")

        assert response.choices is not None
        assert response.choices[0].message is not None

    def test_prepare_kwargs_when_choices_missing_message_then_backfills_message(
        self,
    ) -> None:
        """Existing ModelResponse objects missing messages should be normalised."""

        custom_llm = CodexCustomLLM()
        model_response = litellm.ModelResponse()
        choice = litellm.utils.Choices()
        choice.message = None
        model_response.choices = [choice]

        prepared = custom_llm._prepare_kwargs(
            {"model": "codex-mini", "model_response": model_response}
        )

        response = prepared["model_response"]
        assert response.choices is not None
        assert response.choices[0].message is not None


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
        assert (
            "codex sample completion output"
            in result.choices[0].message.content.lower()
        )
        assert result.choices[0].finish_reason == "stop"
        assert captured["url"] == "https://api.example.com/chat/completions"
        assert captured["json"]["messages"][0]["content"] == "Hello"
        client.close.assert_not_called()

    @pytest.mark.parametrize(
        ("status", "expected_substrings", "headers"),
        [
            (
                403,
                ["forbidden", "codex login"],
                {},
            ),
            (
                429,
                ["rate limit", "retry after 12s"],
                {"Retry-After": "12"},
            ),
            (
                503,
                ["service unavailable", "try again"],
                {},
            ),
        ],
    )
    def test_completion_http_errors_emit_guidance(
        self,
        status: int,
        expected_substrings: list[str],
        headers: dict[str, str],
    ) -> None:
        """HTTP failures should raise UUTELError with actionable guidance."""

        codex = CodexUU()
        model_response = ModelResponse()
        model_response.choices = [Mock()]
        model_response.choices[0].message = Mock()

        request = httpx.Request("POST", "https://api.example.com/chat/completions")
        response = httpx.Response(status, request=request, headers=headers)
        http_error = httpx.HTTPStatusError(
            "failure", request=request, response=response
        )

        client = Mock()
        client.post.side_effect = http_error

        with pytest.raises(UUTELError) as excinfo:
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

        message = str(excinfo.value).lower()
        for expected in expected_substrings:
            assert expected in message

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

    def test_completion_returns_credential_guidance_on_401(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Codex completion should surface actionable guidance on HTTP 401."""

        codex = CodexUU()
        monkeypatch.setattr(codex, "_load_codex_auth", lambda: ("token", "account"))

        request = httpx.Request(
            "POST", "https://chatgpt.com/backend-api/codex/responses"
        )
        response = httpx.Response(401, request=request)
        http_error = httpx.HTTPStatusError(
            "Unauthorized", request=request, response=response
        )

        failing_response = Mock()
        failing_response.raise_for_status.side_effect = http_error

        client = Mock()
        client.post.return_value = failing_response

        model_response = ModelResponse()
        model_response.choices = [Mock()]
        model_response.choices[0].message = Mock()

        with pytest.raises(UUTELError) as exc_info:
            codex.completion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                api_base="https://chatgpt.com/backend-api",
                custom_prompt_dict={},
                model_response=model_response,
                print_verbose=Mock(),
                encoding="utf-8",
                api_key=None,
                logging_obj=Mock(),
                optional_params={},
                client=client,
            )

        message = str(exc_info.value).lower()
        assert "codex credentials" in message
        assert "codex login" in message
        assert exc_info.value.provider == "codex"

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

    def test_completion_with_api_key_uses_openai_payload(
        self, codex_client_factory: Callable[..., tuple[Mock, dict]]
    ) -> None:
        """Supplying an API key should target OpenAI chat completions endpoint."""

        codex = CodexUU()
        model_response = ModelResponse()
        model_response.choices = [Mock()]
        model_response.choices[0].message = Mock()

        client, captured = codex_client_factory()

        codex.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            api_base="https://api.openai.com/v1",
            custom_prompt_dict={},
            model_response=model_response,
            print_verbose=Mock(),
            encoding="utf-8",
            api_key="test-key",
            logging_obj=Mock(),
            optional_params={"temperature": 0.35, "max_tokens": 128},
            client=client,
        )

        assert captured["url"] == "https://api.openai.com/v1/chat/completions", (
            "Expected OpenAI endpoint when API key supplied"
        )
        headers = captured["headers"]
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"
        payload = captured["json"]
        assert payload["model"] == "gpt-4o"
        assert payload["messages"][0]["content"] == "Hello"
        assert payload["temperature"] == 0.35
        assert payload["max_tokens"] == 128


class TestCodexUUAsyncCompletion:
    """Test CodexUU async completion functionality."""

    def test_acompletion_basic_functionality(self) -> None:
        """Async completion should await an async client rather than sync fallback."""
        import asyncio

        codex = CodexUU()
        codex.completion = Mock(
            side_effect=AssertionError("Sync completion must not run")
        )

        model_response = ModelResponse()
        model_response.choices = [Mock()]
        model_response.choices[0].message = Mock()

        response_payload = {
            "choices": [
                {
                    "message": {"content": "Async sample completion output"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 20,
                "total_tokens": 32,
            },
        }

        http_response = Mock()
        http_response.raise_for_status.return_value = None
        http_response.json.return_value = response_payload

        client = AsyncMock()
        client.post.return_value = http_response

        async def run_test() -> ModelResponse:
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

        assert result is model_response
        assert result.model == "gpt-4o"
        assert (
            "async sample completion output"
            in result.choices[0].message.content.lower()
        )

        client.post.assert_awaited_once()
        await_args = client.post.await_args
        assert await_args.args[0] == "https://api.example.com/chat/completions"
        assert await_args.kwargs["json"]["model"] == "gpt-4o"

    def test_acompletion_returns_credential_guidance_on_401(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Async completion should raise actionable guidance on HTTP 401."""

        codex = CodexUU()
        monkeypatch.setattr(codex, "_load_codex_auth", lambda: ("token", "account"))

        request = httpx.Request(
            "POST", "https://chatgpt.com/backend-api/codex/responses"
        )
        response = httpx.Response(401, request=request)
        http_error = httpx.HTTPStatusError(
            "Unauthorized", request=request, response=response
        )

        failing_response = Mock()
        failing_response.raise_for_status.side_effect = http_error

        client = AsyncMock()
        client.post.return_value = failing_response

        async def run() -> ModelResponse:
            return await codex.acompletion(
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

        with pytest.raises(UUTELError) as exc_info:
            asyncio.run(run())

        message = str(exc_info.value).lower()
        assert "codex credentials" in message
        assert "codex login" in message
        assert exc_info.value.provider == "codex"


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

    def test_streaming_status_error_maps_to_guidance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Streaming failures should reuse HTTP guidance messaging."""

        codex = CodexUU()
        monkeypatch.setattr(codex, "_load_codex_auth", lambda: ("token", "account"))

        client = DummyClient([], status_code=429, headers={"retry-after": "10"})

        with pytest.raises(UUTELError) as excinfo:
            list(
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

        message = str(excinfo.value).lower()
        assert "rate limit" in message
        assert "retry after 10s" in message
        assert excinfo.value.provider == "codex"

    def test_parse_retry_after_supports_http_date_header(self) -> None:
        """HTTP-date retry headers should convert into second deltas."""

        codex = CodexUU()
        future = datetime.now(timezone.utc) + timedelta(seconds=42)
        header_value = format_datetime(future)

        result = codex._parse_retry_after({"retry-after": header_value})

        assert result is not None, "HTTP-date retry header should return seconds"
        assert 40 <= result <= 42, "Parsed retry seconds should approximate 42s"

    def test_format_status_guidance_includes_seconds_for_http_date(self) -> None:
        """Guidance message should embed parsed seconds from HTTP-date headers."""

        codex = CodexUU()
        future = datetime.now(timezone.utc) + timedelta(seconds=30)
        header_value = format_datetime(future)

        seconds = codex._parse_retry_after({"retry-after": header_value})

        message = codex._format_status_guidance(
            status=429,
            headers={"retry-after": header_value},
            fallback="rate limit",
        ).lower()

        assert seconds is not None, "Retry-after parsing should produce integer seconds"
        assert f"retry after {seconds}s" in message

    def test_streaming_handles_name_and_tool_argument_deltas(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Codex streaming should capture name/argument deltas for tool calls."""

        codex = CodexUU()
        monkeypatch.setattr(codex, "_load_codex_auth", lambda: ("token", "account"))

        lines = [
            'data: {"type": "response.output_item.added", "item": {"type": "function_call", "id": "item_42"}}',
            "",
            'data: {"type": "response.function_call_name.delta", "item_id": "item_42", "delta": "search"}',
            "",
            'data: {"type": "response.tool_call_arguments.delta", "item_id": "item_42", "delta": "{\\"query\\": "}',
            "",
            'data: {"type": "response.tool_call_arguments.delta", "item_id": "item_42", "delta": "\\"docs\\"}"}',
            "",
            'data: {"type": "response.function_call_arguments.done", "item_id": "item_42"}',
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
        assert tool_chunk["tool_use"]["id"] == "item_42"
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

        client, captured = codex_client_factory(content="Structured sample completion")

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
        assert (
            "structured sample completion" in result.choices[0].message.content.lower()
        )

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

    @patch("uutel.providers.codex.custom_llm.CodexUU")
    def test_async_completion_delegates_to_provider(self, mock_codex_uu: Mock) -> None:
        provider_instance = mock_codex_uu.return_value
        expected_response = self._minimal_model_response()
        provider_instance.acompletion = AsyncMock(return_value=expected_response)

        custom_llm = CodexCustomLLM()

        async def _run() -> ModelResponse:
            return await custom_llm.acompletion(
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

        result = asyncio.run(_run())

        provider_instance.acompletion.assert_called_once()
        assert result is expected_response

    @patch("uutel.providers.codex.custom_llm.CodexUU")
    def test_async_streaming_delegates_to_provider(self, mock_codex_uu: Mock) -> None:
        async def _agen():
            yield {"text": "async"}

        provider_instance = mock_codex_uu.return_value
        provider_instance.astreaming = Mock(return_value=_agen())

        custom_llm = CodexCustomLLM()

        async def _collect() -> list[dict[str, str]]:
            chunks: list[dict[str, str]] = []
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
                chunks.append(chunk)
            return chunks

        result = asyncio.run(_collect())

        provider_instance.astreaming.assert_called_once()
        assert result == [{"text": "async"}]


class TestCodexAuthLoader:
    """Unit tests for Codex auth file parsing."""

    def setup_method(self) -> None:
        self.provider = CodexUU()

    def _create_auth_file(self, tmp_path: Path, payload: dict[str, object]) -> None:
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir(parents=True)
        (codex_dir / "auth.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_load_codex_auth_supports_nested_tokens_structure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Legacy auth layout in tokens{} should be accepted."""

        payload = {
            "tokens": {
                "access_token": "legacy-access",
                "account_id": "acct-123",
            }
        }
        self._create_auth_file(tmp_path, payload)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        access_token, account_id = self.provider._load_codex_auth()

        assert access_token == "legacy-access"
        assert account_id == "acct-123"

    def test_load_codex_auth_supports_flat_structure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Modern auth layout with top-level tokens should be accepted."""

        payload = {
            "access_token": "flat-access",
            "refresh_token": "refresh-token",
            "workspace_id": "workspace-456",
        }
        self._create_auth_file(tmp_path, payload)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        access_token, account_id = self.provider._load_codex_auth()

        assert access_token == "flat-access"
        assert account_id == "workspace-456"

    def test_load_codex_auth_raises_when_access_token_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing access token should raise a UUTELError with guidance."""

        payload = {"workspace_id": "workspace-789"}
        self._create_auth_file(tmp_path, payload)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        with pytest.raises(UUTELError) as exc:
            self.provider._load_codex_auth()

        message = str(exc.value)
        assert "access token" in message.lower()
        assert exc.value.provider == "codex"
