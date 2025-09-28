# this_file: tests/conftest.py
"""Pytest configuration and fixtures for UUTEL tests.

This module provides shared fixtures and configuration for testing UUTEL
providers, authentication, and core functionality.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from litellm.types.utils import ModelResponse

from uutel.core import BaseAuth, BaseUU, RetryConfig


@pytest.fixture
def mock_provider() -> str:
    """Return a mock provider name for testing."""
    return "test-provider"


@pytest.fixture
def mock_model_name() -> str:
    """Return a mock model name for testing."""
    return "test-model-1.0"


@pytest.fixture
def sample_messages() -> list[dict[str, str]]:
    """Return sample OpenAI-format messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello, world!"},
        {"role": "assistant", "content": "Hello! How can I help you today?"},
    ]


@pytest.fixture
def sample_openai_request() -> dict[str, Any]:
    """Return a sample OpenAI-compatible request payload."""
    return {
        "model": "test-model-1.0",
        "messages": [{"role": "user", "content": "Test message"}],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False,
    }


@pytest.fixture
def sample_model_response() -> ModelResponse:
    """Return a sample LiteLLM ModelResponse for testing."""
    response = ModelResponse()
    response.choices = []
    response.usage = {}
    return response


@pytest.fixture
def retry_config() -> RetryConfig:
    """Return a test RetryConfig with reduced timeouts."""
    return RetryConfig(
        max_retries=2,
        backoff_factor=1.5,
        retry_on_status=[429, 502, 503, 504],
        retry_on_exceptions=[ConnectionError, TimeoutError],
    )


@pytest.fixture
def mock_http_client() -> MagicMock:
    """Return a mock HTTP client for testing network operations."""
    client = MagicMock()

    # Mock sync response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "success"}
    mock_response.text = '{"status": "success"}'
    client.get.return_value = mock_response
    client.post.return_value = mock_response

    return client


@pytest.fixture
def mock_async_http_client() -> AsyncMock:
    """Return a mock async HTTP client for testing async operations."""
    client = AsyncMock()

    # Mock async response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "success"}
    mock_response.text = '{"status": "success"}'
    client.get.return_value = mock_response
    client.post.return_value = mock_response

    return client


@pytest.fixture
def mock_auth_result() -> dict[str, Any]:
    """Return a mock authentication result for testing."""
    return {
        "success": True,
        "token": "mock-token-12345",
        "expires_at": None,
        "error": None,
    }


@pytest.fixture
def mock_base_auth() -> BaseAuth:
    """Return a mock BaseAuth instance for testing."""
    auth = BaseAuth()
    auth.provider_name = "mock-provider"
    auth.auth_type = "mock-auth"
    return auth


@pytest.fixture
def mock_base_uu() -> BaseUU:
    """Return a mock BaseUU instance for testing."""
    uu = BaseUU()
    uu.provider_name = "mock-provider"
    uu.supported_models = ["mock-model-1.0", "mock-model-2.0"]
    return uu


@pytest.fixture
def mock_streaming_response() -> list[dict[str, Any]]:
    """Return mock streaming response chunks for testing."""
    return [
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "choices": [
                {"delta": {"content": "Hello"}, "index": 0, "finish_reason": None}
            ],
        },
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "choices": [
                {"delta": {"content": " world"}, "index": 0, "finish_reason": None}
            ],
        },
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        },
    ]


@pytest.fixture
def mock_tool_definition() -> dict[str, Any]:
    """Return a mock tool/function definition for testing."""
    return {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }


@pytest.fixture
def environment_variables(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set up mock environment variables for testing."""
    env_vars = {
        "CLAUDE_CODE_TOKEN": "mock-claude-token",
        "GEMINI_API_KEY": "mock-gemini-key",
        "GOOGLE_APPLICATION_CREDENTIALS": "mock-credentials.json",
        "CODEX_TOKEN": "mock-codex-token",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_provider_response() -> dict[str, Any]:
    """Return a mock provider-specific response for testing."""
    return {
        "id": "resp-123",
        "model": "test-model-1.0",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Test response from mock provider",
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }


@pytest.fixture
def mock_error_response() -> dict[str, Any]:
    """Return a mock error response for testing error handling."""
    return {
        "error": {
            "message": "Test error message",
            "type": "test_error",
            "code": "TEST_001",
        }
    }


class MockAsyncIterator:
    """Mock async iterator for testing streaming responses."""

    def __init__(self, items: list[Any]) -> None:
        self.items = items
        self.index = 0

    def __aiter__(self) -> MockAsyncIterator:
        return self

    async def __anext__(self) -> Any:
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


@pytest.fixture
def mock_async_iterator() -> MockAsyncIterator:
    """Return a mock async iterator for testing streaming."""
    chunks = [
        {"delta": {"content": "Hello"}},
        {"delta": {"content": " world"}},
        {"delta": {"content": "!"}},
    ]
    return MockAsyncIterator(chunks)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (may require API keys)"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "network: mark test as requiring network access")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark slow tests
        if "slow" in item.nodeid or hasattr(item.function, "slow"):
            item.add_marker(pytest.mark.slow)

        # Mark network tests
        if "network" in item.nodeid or "http" in item.nodeid:
            item.add_marker(pytest.mark.network)
