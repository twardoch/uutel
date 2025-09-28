# this_file: tests/test_utils.py
"""Tests for UUTEL core utilities."""

from __future__ import annotations

from uutel.core.utils import (
    RetryConfig,
    create_http_client,
    extract_provider_from_model,
    format_error_message,
    transform_openai_to_provider,
    transform_provider_to_openai,
    validate_model_name,
)


class TestMessageTransformation:
    """Test message transformation utilities."""

    def test_transform_openai_to_provider_basic(self) -> None:
        """Test basic OpenAI to provider message transformation."""
        openai_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, world!"},
        ]

        transformed = transform_openai_to_provider(openai_messages, "test-provider")

        assert isinstance(transformed, list)
        assert len(transformed) == 2
        assert transformed[0]["role"] == "system"
        assert transformed[0]["content"] == "You are a helpful assistant"

    def test_transform_provider_to_openai_basic(self) -> None:
        """Test basic provider to OpenAI message transformation."""
        provider_messages = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ]

        transformed = transform_provider_to_openai(provider_messages, "test-provider")

        assert isinstance(transformed, list)
        assert len(transformed) == 1
        assert transformed[0]["role"] == "assistant"
        assert transformed[0]["content"] == "Hello! How can I help you today?"

    def test_transform_empty_messages(self) -> None:
        """Test transformation of empty message list."""
        assert transform_openai_to_provider([], "test") == []
        assert transform_provider_to_openai([], "test") == []


class TestRetryConfig:
    """Test retry configuration."""

    def test_retry_config_defaults(self) -> None:
        """Test RetryConfig default values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.backoff_factor == 2.0
        assert 429 in config.retry_on_status
        assert 502 in config.retry_on_status

    def test_retry_config_custom(self) -> None:
        """Test RetryConfig with custom values."""
        config = RetryConfig(
            max_retries=5, backoff_factor=1.5, retry_on_status=[429, 503]
        )
        assert config.max_retries == 5
        assert config.backoff_factor == 1.5
        assert config.retry_on_status == [429, 503]


class TestHttpClient:
    """Test HTTP client utilities."""

    def test_create_http_client_sync(self) -> None:
        """Test creating synchronous HTTP client."""
        client = create_http_client(async_client=False)
        assert client is not None
        assert hasattr(client, "get")
        assert hasattr(client, "post")

    def test_create_http_client_async(self) -> None:
        """Test creating asynchronous HTTP client."""
        client = create_http_client(async_client=True)
        assert client is not None
        assert hasattr(client, "get")
        assert hasattr(client, "post")


class TestModelValidation:
    """Test model name validation utilities."""

    def test_validate_model_name_valid(self) -> None:
        """Test validation of valid model names."""
        valid_models = ["claude-3-5-sonnet-20241022", "gemini-2.0-flash-exp", "gpt-4o"]

        for model in valid_models:
            assert validate_model_name(model) is True

    def test_validate_model_name_invalid(self) -> None:
        """Test validation of invalid model names."""
        invalid_models = [
            "",
            None,
            "invalid/model",
            "model with spaces",
        ]

        for model in invalid_models:
            assert validate_model_name(model) is False

    def test_validate_model_name_with_provider_prefix(self) -> None:
        """Test validation of model names with provider prefixes."""
        prefixed_models = [
            "uutel/claude-code/claude-3-5-sonnet",
            "uutel/gemini-cli/gemini-2.0-flash",
        ]

        for model in prefixed_models:
            assert validate_model_name(model) is True

    def test_transform_messages_with_invalid_format(self) -> None:
        """Test transformation with invalid message formats."""
        invalid_messages = [
            {"invalid": "format"},  # Missing role/content
            {"role": "user"},  # Missing content
            {"content": "test"},  # Missing role
            "not a dict",  # Not a dictionary
            None,  # None value
        ]

        # These should be filtered out in transformation
        result = transform_openai_to_provider(invalid_messages, "test")
        assert len(result) == 0

        result = transform_provider_to_openai(invalid_messages, "test")
        assert len(result) == 0

    def test_transform_messages_mixed_valid_invalid(self) -> None:
        """Test transformation with mix of valid and invalid messages."""
        mixed_messages = [
            {"role": "user", "content": "Valid message"},
            {"invalid": "format"},
            {"role": "assistant", "content": "Another valid message"},
            None,
        ]

        result = transform_openai_to_provider(mixed_messages, "test")
        assert len(result) == 2
        assert result[0]["content"] == "Valid message"
        assert result[1]["content"] == "Another valid message"

    def test_create_http_client_with_custom_timeout(self) -> None:
        """Test HTTP client creation with custom timeout."""
        client = create_http_client(async_client=False, timeout=5.0)
        assert client is not None

        async_client = create_http_client(async_client=True, timeout=15.0)
        assert async_client is not None

    def test_create_http_client_with_retry_config(self) -> None:
        """Test HTTP client creation with retry configuration."""
        retry_config = RetryConfig(max_retries=5, backoff_factor=1.5)

        client = create_http_client(
            async_client=False, timeout=10.0, retry_config=retry_config
        )
        assert client is not None

        async_client = create_http_client(
            async_client=True, timeout=10.0, retry_config=retry_config
        )
        assert async_client is not None


class TestProviderExtraction:
    """Test provider and model extraction utilities."""

    def test_extract_provider_from_model_simple(self) -> None:
        """Test extraction from simple model names."""
        provider, model = extract_provider_from_model("simple-model")
        assert provider == "unknown"
        assert model == "simple-model"

    def test_extract_provider_from_model_with_uutel_prefix(self) -> None:
        """Test extraction from UUTEL prefixed models."""
        provider, model = extract_provider_from_model(
            "uutel/claude-code/claude-3-5-sonnet"
        )
        assert provider == "claude-code"
        assert model == "claude-3-5-sonnet"

    def test_extract_provider_from_model_nested_model_names(self) -> None:
        """Test extraction with nested model names."""
        provider, model = extract_provider_from_model(
            "uutel/gemini-cli/models/gemini-2.0-flash"
        )
        assert provider == "gemini-cli"
        assert model == "models/gemini-2.0-flash"

    def test_extract_provider_from_model_invalid_prefix(self) -> None:
        """Test extraction from invalid prefixes."""
        # Invalid prefix (not uutel)
        provider, model = extract_provider_from_model("other/provider/model")
        assert provider == "unknown"
        assert model == "other/provider/model"

        # Too few parts
        provider, model = extract_provider_from_model("uutel/provider")
        assert provider == "unknown"
        assert model == "uutel/provider"

    def test_extract_provider_from_model_edge_cases(self) -> None:
        """Test extraction edge cases."""
        # Empty model name with slashes
        provider, model = extract_provider_from_model("uutel//")
        assert provider == ""
        assert model == ""

        # Single slash
        provider, model = extract_provider_from_model("model/name")
        assert provider == "unknown"
        assert model == "model/name"


class TestErrorFormatting:
    """Test error message formatting utilities."""

    def test_format_error_message_basic(self) -> None:
        """Test basic error message formatting."""
        error = ValueError("Test error message")
        formatted = format_error_message(error, "test-provider")

        assert "[test-provider]" in formatted
        assert "ValueError" in formatted
        assert "Test error message" in formatted
        assert formatted == "[test-provider] ValueError: Test error message"

    def test_format_error_message_different_exception_types(self) -> None:
        """Test formatting with different exception types."""
        # Test with different exception types
        exceptions = [
            (ConnectionError("Network failed"), "ConnectionError"),
            (TimeoutError("Request timeout"), "TimeoutError"),
            (KeyError("Missing key"), "KeyError"),
            (RuntimeError("Runtime issue"), "RuntimeError"),
        ]

        for exc, exc_type in exceptions:
            formatted = format_error_message(exc, "provider")
            assert f"[provider] {exc_type}:" in formatted
            assert str(exc) in formatted

    def test_format_error_message_empty_provider(self) -> None:
        """Test formatting with empty provider name."""
        error = ValueError("Test error")
        formatted = format_error_message(error, "")
        assert formatted == "[] ValueError: Test error"

    def test_format_error_message_complex_error(self) -> None:
        """Test formatting with complex error message."""
        complex_message = (
            "Error occurred: status=500, details={'error': 'Internal server error'}"
        )
        error = Exception(complex_message)
        formatted = format_error_message(error, "complex-provider")

        assert "[complex-provider]" in formatted
        assert "Exception" in formatted
        assert complex_message in formatted


class TestRetryConfigEdgeCases:
    """Test retry configuration edge cases."""

    def test_retry_config_with_empty_lists(self) -> None:
        """Test RetryConfig with empty lists."""
        config = RetryConfig(retry_on_status=[], retry_on_exceptions=[])
        assert config.retry_on_status == []
        assert config.retry_on_exceptions == []

    def test_retry_config_extreme_values(self) -> None:
        """Test RetryConfig with extreme values."""
        config = RetryConfig(max_retries=0, backoff_factor=0.1)
        assert config.max_retries == 0
        assert config.backoff_factor == 0.1

        config = RetryConfig(max_retries=100, backoff_factor=10.0)
        assert config.max_retries == 100
        assert config.backoff_factor == 10.0
