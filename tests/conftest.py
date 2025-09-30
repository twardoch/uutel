# this_file: tests/conftest.py
"""Simple pytest configuration for UUTEL tests.

This module provides basic fixtures for testing UUTEL core functionality.
"""

from __future__ import annotations

# Standard library imports
from typing import Any
from unittest.mock import MagicMock

# Third-party imports
import pytest
from litellm.types.utils import ModelResponse

# Local imports
from uutel.core import BaseAuth, BaseUU


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
    return response


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
