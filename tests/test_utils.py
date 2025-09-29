# this_file: tests/test_utils.py
"""Tests for UUTEL core utilities."""

from __future__ import annotations

from typing import Any

from uutel.core.utils import (
    RetryConfig,
    create_http_client,
    create_tool_call_response,
    extract_provider_from_model,
    extract_tool_calls_from_response,
    format_error_message,
    get_error_debug_info,
    transform_openai_to_provider,
    transform_provider_to_openai,
    transform_provider_tools_to_openai,
    validate_model_name,
    validate_tool_schema,
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
        assert config.max_retries == 3, (
            f"Default max_retries should be 3, got {config.max_retries}"
        )
        assert config.backoff_factor == 2.0, (
            f"Default backoff_factor should be 2.0, got {config.backoff_factor}"
        )
        assert 429 in config.retry_on_status, (
            f"Status 429 should be in retry list, got {config.retry_on_status}"
        )
        assert 502 in config.retry_on_status, (
            f"Status 502 should be in retry list, got {config.retry_on_status}"
        )

    def test_retry_config_custom(self) -> None:
        """Test RetryConfig with custom values."""
        config = RetryConfig(
            max_retries=5, backoff_factor=1.5, retry_on_status=[429, 503]
        )
        assert config.max_retries == 5, (
            f"Custom max_retries should be 5, got {config.max_retries}"
        )
        assert config.backoff_factor == 1.5, (
            f"Custom backoff_factor should be 1.5, got {config.backoff_factor}"
        )
        assert config.retry_on_status == [429, 503], (
            f"Custom retry_on_status should be [429, 503], got {config.retry_on_status}"
        )


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
        assert client is not None, (
            "Sync HTTP client creation failed with custom timeout"
        )

        async_client = create_http_client(async_client=True, timeout=15.0)
        assert async_client is not None, (
            "Async HTTP client creation failed with custom timeout"
        )

    def test_create_http_client_with_retry_config(self) -> None:
        """Test HTTP client creation with retry configuration."""
        retry_config = RetryConfig(max_retries=5, backoff_factor=1.5)

        client = create_http_client(
            async_client=False, timeout=10.0, retry_config=retry_config
        )
        assert client is not None, "Sync HTTP client creation failed with retry config"

        async_client = create_http_client(
            async_client=True, timeout=10.0, retry_config=retry_config
        )
        assert async_client is not None, (
            "Async HTTP client creation failed with retry config"
        )


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
        # Empty model name with slashes - enhanced to handle gracefully
        provider, model = extract_provider_from_model("uutel//")
        assert provider == "unknown"  # Enhanced to return "unknown" for edge cases
        assert model == "uutel//"  # Enhanced to return original for edge cases

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
        assert (
            formatted == "[unknown] ValueError: Test error"
        )  # Enhanced to use "unknown" for empty provider

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


class TestModelValidationEdgeCases:
    """Test model validation edge cases to cover uncovered lines."""

    def test_validate_model_name_invalid_regex_parts(self) -> None:
        """Test model validation with invalid regex parts (line 168)."""
        # Test with special characters in parts that fail regex
        invalid_models = [
            "uutel/provider/@invalid",  # @ is not allowed
            "uutel/provider/model$name",  # $ is not allowed
            "uutel/provider/model name",  # spaces in parts
            "uutel//empty-provider",  # empty provider part
            "uutel/provider/",  # empty model part
            "uutel/provider with spaces/model",  # spaces in provider
            "uutel/provider/model#tag",  # # not allowed
        ]

        for model in invalid_models:
            assert not validate_model_name(model), f"Should be invalid: {model}"

    def test_validate_model_name_complex_edge_cases(self) -> None:
        """Test additional model validation edge cases."""
        # Valid models should pass
        valid_models = [
            "uutel/claude-code/claude-3-5-sonnet",
            "uutel/gemini_cli/gemini-2.0.flash",
            "uutel/provider123/model_name-v1.0",
            "simple-model-name",
            "model.with.dots",
            "model_with_underscores",
        ]

        for model in valid_models:
            assert validate_model_name(model), f"Should be valid: {model}"


class TestErrorHandlingEdgeCases:
    """Test error handling utilities with edge cases."""

    def test_format_error_message_with_uutel_error(self) -> None:
        """Test format_error_message with UUTEL error (line 220)."""
        from uutel.core.exceptions import UUTELError

        uutel_error = UUTELError(
            "Test UUTEL error", provider="test-provider", error_code="TEST_001"
        )

        formatted = format_error_message(uutel_error, "should-be-ignored")

        # Should use UUTELError's __str__ method, not the basic formatting
        assert "Test UUTEL error" in formatted
        assert "Provider: test-provider" in formatted
        assert "Code: TEST_001" in formatted
        assert "[should-be-ignored]" not in formatted  # Basic format not used

    def test_get_error_debug_info_with_uutel_error(self) -> None:
        """Test get_error_debug_info with UUTEL error (lines 238-241)."""
        from uutel.core.exceptions import ValidationError

        validation_error = ValidationError(
            "Invalid input",
            provider="test-provider",
            field_name="test_field",
            field_value="invalid_value",
        )

        debug_info = get_error_debug_info(validation_error)

        # Should return UUTEL error's debug info
        assert debug_info["error_type"] == "ValidationError"
        assert debug_info["message"] == "Invalid input"
        assert debug_info["provider"] == "test-provider"
        assert "field_name" in debug_info["debug_context"]

    def test_get_error_debug_info_with_standard_error(self) -> None:
        """Test get_error_debug_info with standard error (lines 242-253)."""
        standard_error = ValueError("Standard error message")

        debug_info = get_error_debug_info(standard_error)

        # Should return enhanced debug info for non-UUTEL error
        assert debug_info["error_type"] == "ValueError"
        assert debug_info["message"] == "Standard error message"
        assert debug_info["provider"] is None
        assert debug_info["error_code"] is None
        # Enhanced to include args in debug context
        assert "args" in debug_info["debug_context"]
        assert debug_info["debug_context"]["args"] == ["Standard error message"]


class TestToolValidationEdgeCases:
    """Test tool validation edge cases."""

    def test_validate_tool_schema_invalid_parameters_type(self) -> None:
        """Test tool validation with invalid parameters type (line 299)."""
        # Tool with parameters that don't have type: object
        invalid_tool = {
            "type": "function",
            "function": {
                "name": "test_function",
                "description": "Test function",
                "parameters": {
                    "type": "string"  # Should be "object"
                },
            },
        }

        assert not validate_tool_schema(invalid_tool)

    def test_validate_tool_schema_missing_parameters_type(self) -> None:
        """Test tool validation with missing parameters type."""
        invalid_tool = {
            "type": "function",
            "function": {
                "name": "test_function",
                "description": "Test function",
                "parameters": {
                    "properties": {}  # Missing "type": "object"
                },
            },
        }

        assert not validate_tool_schema(invalid_tool)

    def test_transform_provider_tools_to_openai_empty(self) -> None:
        """Test provider tools transformation with empty input (line 349)."""
        # Test with None
        result = transform_provider_tools_to_openai(None, "test-provider")
        assert result == []

        # Test with empty list
        result = transform_provider_tools_to_openai([], "test-provider")
        assert result == []


class TestToolCallResponseEdgeCases:
    """Test tool call response creation edge cases."""

    def test_create_tool_call_response_non_serializable_result(self) -> None:
        """Test tool response with non-JSON serializable result (lines 389-391)."""

        # Create a non-serializable object
        class NonSerializable:
            def __init__(self) -> None:
                self.circular_ref = self

        non_serializable = NonSerializable()

        response = create_tool_call_response(
            tool_call_id="test_id",
            function_name="test_function",
            function_result=non_serializable,
        )

        # Should fallback to string representation
        assert response["tool_call_id"] == "test_id"
        assert response["role"] == "tool"
        assert "NonSerializable" in response["content"]  # String representation

    def test_create_tool_call_response_none_result(self) -> None:
        """Test tool response with None result."""
        response = create_tool_call_response(
            tool_call_id="test_id", function_name="test_function", function_result=None
        )

        assert response["content"] == "null"


class TestToolCallExtractionEdgeCases:
    """Test tool call extraction edge cases."""

    def test_extract_tool_calls_non_dict_response(self) -> None:
        """Test extraction with non-dict response (line 409)."""
        # Test with various non-dict types
        non_dict_responses = ["string response", ["list", "response"], 123, None, True]

        for response in non_dict_responses:
            result = extract_tool_calls_from_response(response)
            assert result == []

    def test_extract_tool_calls_malformed_choices(self) -> None:
        """Test extraction with malformed choices (line 418)."""
        # Non-dict choice
        response = {"choices": ["not a dict"]}
        result = extract_tool_calls_from_response(response)
        assert result == []

        # Choice that's not a dict
        response2: dict[str, Any] = {"choices": [123]}
        result = extract_tool_calls_from_response(response2)
        assert result == []

    def test_extract_tool_calls_malformed_message(self) -> None:
        """Test extraction with malformed message (line 422)."""
        # Non-dict message
        response = {"choices": [{"message": "not a dict"}]}
        result = extract_tool_calls_from_response(response)
        assert result == []

        # Message that's not a dict
        response2: dict[str, Any] = {"choices": [{"message": 123}]}
        result = extract_tool_calls_from_response(response2)
        assert result == []

    def test_extract_tool_calls_malformed_tool_calls(self) -> None:
        """Test extraction with malformed tool_calls (line 426)."""
        # Non-list tool_calls
        response = {"choices": [{"message": {"tool_calls": "not a list"}}]}
        result = extract_tool_calls_from_response(response)
        assert result == []

        # tool_calls that's not a list
        response2: dict[str, Any] = {"choices": [{"message": {"tool_calls": 123}}]}
        result = extract_tool_calls_from_response(response2)
        assert result == []

    def test_extract_tool_calls_empty_choices(self) -> None:
        """Test extraction with empty choices."""
        response: dict[str, Any] = {"choices": []}
        result = extract_tool_calls_from_response(response)
        assert result == []

    def test_extract_tool_calls_missing_fields(self) -> None:
        """Test extraction with missing fields."""
        # Missing choices
        response: dict[str, Any] = {}
        result = extract_tool_calls_from_response(response)
        assert result == []

        # Missing message
        response = {"choices": [{}]}
        result = extract_tool_calls_from_response(response)
        assert result == []

        # Missing tool_calls
        response = {"choices": [{"message": {}}]}
        result = extract_tool_calls_from_response(response)
        assert result == []
