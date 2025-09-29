# this_file: tests/test_exceptions.py
"""Tests for UUTEL exception framework."""

from __future__ import annotations

from uutel.core.exceptions import (
    AuthenticationError,
    ConfigurationError,
    ModelError,
    ModelNotFoundError,
    NetworkError,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    StreamingError,
    TimeoutError,
    TokenLimitError,
    ToolCallError,
    UUTELError,
    ValidationError,
    create_configuration_error,
    create_model_not_found_error,
    create_network_error_with_retry_info,
    create_token_limit_error,
)


class TestUUTELBaseException:
    """Test the base UUTEL exception class."""

    def test_uutel_error_creation(self) -> None:
        """Test that UUTELError can be created with message."""
        error = UUTELError("Test error message")
        assert str(error) == "Test error message"
        assert error.provider is None
        assert error.error_code is None

    def test_uutel_error_with_provider(self) -> None:
        """Test UUTELError with provider information."""
        error = UUTELError("Test error", provider="test-provider")
        assert str(error) == "Test error | Provider: test-provider", (
            f"Error string format incorrect: {error!s}"
        )
        assert error.provider == "test-provider", (
            f"Provider should be 'test-provider', got {error.provider}"
        )

    def test_uutel_error_with_error_code(self) -> None:
        """Test UUTELError with error code."""
        error = UUTELError("Test error", error_code="TEST_001")
        assert str(error) == "Test error | Code: TEST_001", (
            f"Error string format incorrect: {error!s}"
        )
        assert error.error_code == "TEST_001", (
            f"Error code should be 'TEST_001', got {error.error_code}"
        )

    def test_uutel_error_inheritance(self) -> None:
        """Test that UUTELError inherits from Exception."""
        error = UUTELError("Test error")
        assert isinstance(error, Exception), (
            f"UUTELError should inherit from Exception, got type {type(error)}"
        )

    def test_uutel_error_with_request_id(self) -> None:
        """Test UUTELError with request_id parameter."""
        error = UUTELError("Test error", request_id="req_123")
        assert "Request: req_123" in str(error)
        assert error.request_id == "req_123"

    def test_uutel_error_get_debug_info(self) -> None:
        """Test UUTELError get_debug_info method."""
        error = UUTELError(
            "Test error",
            provider="test-provider",
            error_code="TEST_001",
            request_id="req_123",
            debug_context={"custom": "value"},
        )

        debug_info = error.get_debug_info()

        assert debug_info["error_type"] == "UUTELError"
        assert debug_info["message"] == "Test error"
        assert debug_info["provider"] == "test-provider"
        assert debug_info["error_code"] == "TEST_001"
        assert debug_info["request_id"] == "req_123"
        assert "timestamp" in debug_info
        assert debug_info["debug_context"] == {"custom": "value"}

    def test_uutel_error_add_context(self) -> None:
        """Test UUTELError add_context method."""
        error = UUTELError("Test error")
        error.add_context("test_key", "test_value")

        debug_info = error.get_debug_info()
        assert debug_info["debug_context"]["test_key"] == "test_value"


class TestAuthenticationError:
    """Test authentication-specific errors."""

    def test_authentication_error_creation(self) -> None:
        """Test AuthenticationError creation."""
        error = AuthenticationError("Invalid credentials")
        assert str(error) == "Invalid credentials"
        assert isinstance(error, UUTELError)

    def test_authentication_error_with_provider(self) -> None:
        """Test AuthenticationError with provider context."""
        error = AuthenticationError(
            "Token expired", provider="claude-code", error_code="AUTH_001"
        )
        assert error.provider == "claude-code"
        assert error.error_code == "AUTH_001"

    def test_authentication_error_with_auth_method(self) -> None:
        """Test AuthenticationError with auth_method parameter."""
        error = AuthenticationError(
            "OAuth flow failed",
            provider="claude-code",
            auth_method="oauth2",
            debug_context={"flow": "authorization_code"},
        )
        debug_info = error.get_debug_info()
        assert debug_info["debug_context"]["auth_method"] == "oauth2"
        assert debug_info["debug_context"]["flow"] == "authorization_code"


class TestRateLimitError:
    """Test rate limiting errors."""

    def test_rate_limit_error_creation(self) -> None:
        """Test RateLimitError creation."""
        error = RateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert isinstance(error, UUTELError)

    def test_rate_limit_error_with_retry_after(self) -> None:
        """Test RateLimitError with retry information."""
        error = RateLimitError(
            "Too many requests", provider="gemini-cli", retry_after=60
        )
        assert error.retry_after == 60
        assert error.provider == "gemini-cli"

    def test_rate_limit_error_with_quota_type(self) -> None:
        """Test RateLimitError with quota_type in debug context."""
        error = RateLimitError(
            "Rate limit exceeded",
            provider="gemini-cli",
            retry_after=30,
            debug_context={"quota_type": "requests"},
        )
        debug_info = error.get_debug_info()
        assert error.retry_after == 30
        assert debug_info["debug_context"]["quota_type"] == "requests"


class TestModelError:
    """Test model-specific errors."""

    def test_model_error_creation(self) -> None:
        """Test ModelError creation."""
        error = ModelError("Model not found")
        assert str(error) == "Model not found"
        assert isinstance(error, UUTELError)

    def test_model_error_with_model_name(self) -> None:
        """Test ModelError with model information."""
        error = ModelError(
            "Model overloaded", provider="cloud-code", model_name="gemini-2.5-pro"
        )
        assert error.model_name == "gemini-2.5-pro"

    def test_model_error_with_model_name_in_debug(self) -> None:
        """Test ModelError with model_name parameter in debug context."""
        error = ModelError(
            "Model not available",
            provider="cloud-code",
            model_name="gemini-2.5-pro",
            debug_context={"region": "us-central1"},
        )
        debug_info = error.get_debug_info()
        assert debug_info["debug_context"]["model_name"] == "gemini-2.5-pro"
        assert debug_info["debug_context"]["region"] == "us-central1"


class TestNetworkError:
    """Test network-related errors."""

    def test_network_error_creation(self) -> None:
        """Test NetworkError creation."""
        error = NetworkError("Connection timeout")
        assert str(error) == "Connection timeout"
        assert isinstance(error, UUTELError)

    def test_network_error_with_status_code(self) -> None:
        """Test NetworkError with HTTP status."""
        error = NetworkError("Service unavailable", provider="codex", status_code=503)
        assert error.status_code == 503

    def test_network_error_with_url(self) -> None:
        """Test NetworkError with URL parameter."""
        error = NetworkError(
            "Connection timeout",
            provider="codex",
            status_code=503,
            url="https://api.openai.com/v1/chat/completions",
        )
        debug_info = error.get_debug_info()
        assert debug_info["debug_context"]["status_code"] == 503
        assert (
            debug_info["debug_context"]["url"]
            == "https://api.openai.com/v1/chat/completions"
        )


class TestValidationError:
    """Test validation errors."""

    def test_validation_error_creation(self) -> None:
        """Test ValidationError creation."""
        error = ValidationError("Invalid message format")
        assert str(error) == "Invalid message format"
        assert isinstance(error, UUTELError)

    def test_validation_error_with_field(self) -> None:
        """Test ValidationError with field information."""
        error = ValidationError(
            "Field is required", field_name="messages", field_value=None
        )
        assert error.field_name == "messages"
        assert error.field_value is None

    def test_validation_error_with_field_in_debug(self) -> None:
        """Test ValidationError with field_name and field_value parameters."""
        error = ValidationError(
            "Invalid model name",
            provider="gemini-cli",
            field_name="model",
            field_value="invalid-model-123",
        )
        debug_info = error.get_debug_info()
        assert debug_info["debug_context"]["field_name"] == "model"
        assert debug_info["debug_context"]["field_value"] == "invalid-model-123"


class TestProviderError:
    """Test provider-specific errors."""

    def test_provider_error_creation(self) -> None:
        """Test ProviderError creation."""
        error = ProviderError("Provider internal error")
        assert str(error) == "Provider internal error"
        assert isinstance(error, UUTELError)

    def test_provider_error_with_details(self) -> None:
        """Test ProviderError with detailed information."""
        error = ProviderError(
            "API error",
            provider="claude-code",
            error_code="PROVIDER_001",
            original_error="Invalid request format",
        )
        assert error.original_error == "Invalid request format"

    def test_provider_error_with_provider_details(self) -> None:
        """Test ProviderError with provider_details in debug context."""
        error = ProviderError(
            "Provider unavailable",
            provider="claude-code",
            debug_context={
                "provider_details": {"service": "anthropic", "region": "us-east-1"},
                "maintenance": True,
            },
        )
        debug_info = error.get_debug_info()
        assert debug_info["debug_context"]["provider_details"]["service"] == "anthropic"
        assert debug_info["debug_context"]["provider_details"]["region"] == "us-east-1"
        assert debug_info["debug_context"]["maintenance"] is True


class TestConfigurationError:
    """Test configuration-specific errors."""

    def test_configuration_error_creation(self) -> None:
        """Test ConfigurationError creation."""
        error = ConfigurationError("Invalid configuration file")
        assert str(error) == "Invalid configuration file"
        assert isinstance(error, UUTELError)

    def test_configuration_error_with_config_key(self) -> None:
        """Test ConfigurationError with config key information."""
        error = ConfigurationError(
            "Missing required field",
            provider="claude-code",
            config_key="api_key",
            config_section="authentication",
        )
        debug_info = error.get_debug_info()
        assert debug_info["debug_context"]["config_key"] == "api_key"
        assert debug_info["debug_context"]["config_section"] == "authentication"

    def test_configuration_error_with_suggestions(self) -> None:
        """Test ConfigurationError with suggested fix."""
        error = ConfigurationError(
            "Invalid auth type",
            provider="gemini-cli",
            debug_context={
                "config_field": "auth_type",
                "invalid_value": "invalid",
                "suggested_fix": "Use 'api-key', 'oauth', or 'vertex-ai'",
            },
        )
        debug_info = error.get_debug_info()
        assert (
            debug_info["debug_context"]["suggested_fix"]
            == "Use 'api-key', 'oauth', or 'vertex-ai'"
        )


class TestToolCallError:
    """Test tool calling errors."""

    def test_tool_call_error_creation(self) -> None:
        """Test ToolCallError creation."""
        error = ToolCallError("Function call failed")
        assert str(error) == "Function call failed"
        assert isinstance(error, UUTELError)

    def test_tool_call_error_with_tool_details(self) -> None:
        """Test ToolCallError with tool information."""
        error = ToolCallError(
            "Invalid parameters",
            provider="claude-code",
            tool_name="search_files",
            tool_call_id="call_123",
            tool_parameters={"query": "test"},
        )
        debug_info = error.get_debug_info()
        assert debug_info["debug_context"]["tool_name"] == "search_files"
        assert debug_info["debug_context"]["tool_call_id"] == "call_123"
        assert debug_info["debug_context"]["tool_parameters"] == {"query": "test"}

    def test_tool_call_error_with_validation_failure(self) -> None:
        """Test ToolCallError with parameter validation failure."""
        error = ToolCallError(
            "Parameter validation failed",
            provider="gemini-cli",
            tool_name="analyze_code",
            execution_stage="validation",
            debug_context={
                "validation_error": "Missing required parameter 'code'",
                "provided_params": {"language": "python"},
            },
        )
        debug_info = error.get_debug_info()
        assert debug_info["debug_context"]["execution_stage"] == "validation"
        assert (
            "Missing required parameter"
            in debug_info["debug_context"]["validation_error"]
        )


class TestStreamingError:
    """Test streaming response errors."""

    def test_streaming_error_creation(self) -> None:
        """Test StreamingError creation."""
        error = StreamingError("Stream interrupted")
        assert str(error) == "Stream interrupted"
        assert isinstance(error, UUTELError)

    def test_streaming_error_with_stream_details(self) -> None:
        """Test StreamingError with streaming information."""
        error = StreamingError(
            "Connection lost during stream",
            provider="codex",
            connection_id="stream_456",
            bytes_received=1024,
            debug_context={"last_chunk_time": "2024-01-01T12:00:00Z"},
        )
        debug_info = error.get_debug_info()
        assert debug_info["debug_context"]["connection_id"] == "stream_456"
        assert debug_info["debug_context"]["bytes_received"] == 1024

    def test_streaming_error_with_recovery_info(self) -> None:
        """Test StreamingError with recovery information."""
        error = StreamingError(
            "Stream timeout",
            provider="cloud-code",
            debug_context={
                "can_retry": True,
                "retry_delay": 5,
                "recovery_suggestion": "Reduce response timeout and retry",
            },
        )
        debug_info = error.get_debug_info()
        assert debug_info["debug_context"]["can_retry"] is True
        assert debug_info["debug_context"]["retry_delay"] == 5


class TestTimeoutError:
    """Test timeout errors."""

    def test_timeout_error_creation(self) -> None:
        """Test TimeoutError creation."""
        error = TimeoutError("Request timed out")
        assert str(error) == "Request timed out"
        assert isinstance(error, UUTELError)

    def test_timeout_error_with_duration(self) -> None:
        """Test TimeoutError with timeout duration."""
        error = TimeoutError(
            "Operation timed out",
            provider="gemini-cli",
            timeout_duration=30.0,
            operation_type="completion",
        )
        debug_info = error.get_debug_info()
        assert debug_info["debug_context"]["timeout_duration"] == 30.0
        assert debug_info["debug_context"]["operation_type"] == "completion"

    def test_timeout_error_with_retry_suggestion(self) -> None:
        """Test TimeoutError with retry suggestion."""
        error = TimeoutError(
            "Model response timeout",
            provider="claude-code",
            debug_context={
                "timeout_duration": 60.0,
                "can_retry": True,
                "suggested_timeout": 120.0,
                "retry_strategy": "exponential_backoff",
            },
        )
        debug_info = error.get_debug_info()
        assert debug_info["debug_context"]["suggested_timeout"] == 120.0
        assert debug_info["debug_context"]["retry_strategy"] == "exponential_backoff"


class TestQuotaExceededError:
    """Test quota exceeded errors."""

    def test_quota_exceeded_error_creation(self) -> None:
        """Test QuotaExceededError creation."""
        error = QuotaExceededError("Daily quota exceeded")
        assert str(error) == "Daily quota exceeded"
        assert isinstance(error, RateLimitError)
        assert isinstance(error, UUTELError)

    def test_quota_exceeded_error_with_quota_details(self) -> None:
        """Test QuotaExceededError with quota information."""
        error = QuotaExceededError(
            "Token quota exceeded",
            provider="gemini-cli",
            reset_time="2024-02-01T00:00:00Z",  # Passed to RateLimitError
            quota_type="tokens",
            quota_limit=1000000,
            quota_used=1000000,
        )
        debug_info = error.get_debug_info()
        assert debug_info["debug_context"]["quota_type"] == "tokens"
        assert debug_info["debug_context"]["quota_limit"] == 1000000
        assert debug_info["debug_context"]["quota_used"] == 1000000
        assert debug_info["debug_context"]["reset_time"] == "2024-02-01T00:00:00Z"

    def test_quota_exceeded_error_with_reset_time(self) -> None:
        """Test QuotaExceededError with reset information."""
        error = QuotaExceededError(
            "Monthly quota exceeded",
            provider="cloud-code",
            debug_context={
                "quota_type": "requests",
                "reset_time": "2024-02-01T00:00:00Z",
                "upgrade_suggestion": "Consider upgrading to a higher tier",
            },
        )
        debug_info = error.get_debug_info()
        assert debug_info["debug_context"]["reset_time"] == "2024-02-01T00:00:00Z"
        assert "upgrading" in debug_info["debug_context"]["upgrade_suggestion"]


class TestModelNotFoundError:
    """Test model not found errors."""

    def test_model_not_found_error_creation(self) -> None:
        """Test ModelNotFoundError creation."""
        error = ModelNotFoundError("Model not available")
        assert str(error) == "Model not available"
        assert isinstance(error, ModelError)
        assert isinstance(error, UUTELError)

    def test_model_not_found_error_with_model_details(self) -> None:
        """Test ModelNotFoundError with model information."""
        error = ModelNotFoundError(
            "Model not found",
            provider="claude-code",
            model_name="claude-3-5-sonnet-invalid",  # From ModelError
            suggested_model="claude-3-5-sonnet-20241022",
        )
        assert error.model_name == "claude-3-5-sonnet-invalid"
        debug_info = error.get_debug_info()
        assert (
            debug_info["debug_context"]["suggested_model"]
            == "claude-3-5-sonnet-20241022"
        )

    def test_model_not_found_error_with_available_models(self) -> None:
        """Test ModelNotFoundError with available models list."""
        available_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ]
        error = ModelNotFoundError(
            "Invalid model name",
            provider="claude-code",
            model_name="claude-4",
            debug_context={
                "available_models": available_models,
                "suggestion": "Use one of the available Claude models",
            },
        )
        debug_info = error.get_debug_info()
        assert debug_info["debug_context"]["available_models"] == available_models
        assert "available Claude models" in debug_info["debug_context"]["suggestion"]


class TestTokenLimitError:
    """Test token limit errors."""

    def test_token_limit_error_creation(self) -> None:
        """Test TokenLimitError creation."""
        error = TokenLimitError("Token limit exceeded")
        assert str(error) == "Token limit exceeded"
        assert isinstance(error, ModelError)
        assert isinstance(error, UUTELError)

    def test_token_limit_error_with_token_details(self) -> None:
        """Test TokenLimitError with token information."""
        error = TokenLimitError(
            "Input too long",
            provider="gemini-cli",
            model_name="gemini-2.0-flash",  # From ModelError
            token_count=150000,
            token_limit=128000,
            debug_context={"input_tokens": 130000, "output_tokens": 20000},
        )
        assert error.model_name == "gemini-2.0-flash"
        debug_info = error.get_debug_info()
        assert debug_info["debug_context"]["token_count"] == 150000
        assert debug_info["debug_context"]["token_limit"] == 128000
        assert debug_info["debug_context"]["input_tokens"] == 130000

    def test_token_limit_error_with_reduction_suggestion(self) -> None:
        """Test TokenLimitError with reduction suggestions."""
        error = TokenLimitError(
            "Context too large",
            provider="codex",
            model_name="gpt-4o",
            debug_context={
                "token_count": 200000,
                "token_limit": 128000,
                "reduction_suggestion": "Consider summarizing or chunking the input",
                "alternative_models": ["gpt-4o-32k", "claude-3-opus"],
            },
        )
        debug_info = error.get_debug_info()
        assert "summarizing" in debug_info["debug_context"]["reduction_suggestion"]
        assert "gpt-4o-32k" in debug_info["debug_context"]["alternative_models"]


class TestHelperFunctions:
    """Test exception helper functions."""

    def test_create_configuration_error(self) -> None:
        """Test create_configuration_error helper function."""
        error = create_configuration_error(
            "Missing API key",
            provider="claude-code",
            config_key="api_key",
            suggested_fix="Set your API key in the configuration file",
            config_file="~/.claude/config.json",
        )

        assert isinstance(error, ConfigurationError)
        assert (
            str(error)
            == "Missing API key. Suggestion: Set your API key in the configuration file | Provider: claude-code"
        )
        assert error.provider == "claude-code"

        debug_info = error.get_debug_info()
        assert debug_info["debug_context"]["config_key"] == "api_key"
        assert (
            debug_info["debug_context"]["suggested_fix"]
            == "Set your API key in the configuration file"
        )
        assert debug_info["debug_context"]["config_file"] == "~/.claude/config.json"

    def test_create_model_not_found_error_with_suggestions(self) -> None:
        """Test create_model_not_found_error with model suggestions."""
        available_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ]

        error = create_model_not_found_error(
            "claude-sonnet",  # Should suggest claude-3-5-sonnet-20241022
            provider="claude-code",
            available_models=available_models,
        )

        assert isinstance(error, ModelNotFoundError)
        assert "claude-sonnet" in str(error)
        assert error.provider == "claude-code"

        error.get_debug_info()
        # The helper function sets model_name in debug context, not requested_model
        assert "claude-sonnet" in str(error)
        assert error.available_models == available_models
        # Should suggest claude-3-5-sonnet-20241022 based on similarity
        assert error.suggested_model == "claude-3-5-sonnet-20241022"

    def test_create_model_not_found_error_no_suggestions(self) -> None:
        """Test create_model_not_found_error with no available models."""
        error = create_model_not_found_error(
            "unknown-model",
            provider="test-provider",
        )

        assert isinstance(error, ModelNotFoundError)
        assert "unknown-model" in str(error)

        error.get_debug_info()
        assert "unknown-model" in str(error)
        assert error.available_models is None
        assert error.suggested_model is None

    def test_create_token_limit_error(self) -> None:
        """Test create_token_limit_error helper function."""
        error = create_token_limit_error(
            model_name="gpt-4o",
            provider="codex",
            token_count=150000,
            token_limit=128000,
            input_tokens=130000,
            output_tokens=20000,
        )

        assert isinstance(error, TokenLimitError)
        assert "150000" in str(error)  # Numbers without commas
        assert "128000" in str(error)
        assert error.model_name == "gpt-4o"

        debug_info = error.get_debug_info()
        assert debug_info["debug_context"]["token_count"] == 150000
        assert debug_info["debug_context"]["token_limit"] == 128000
        assert debug_info["debug_context"]["input_tokens"] == 130000
        assert debug_info["debug_context"]["output_tokens"] == 20000
        assert debug_info["debug_context"]["suggested_action"]

    def test_create_network_error_with_retry_info(self) -> None:
        """Test create_network_error_with_retry_info helper function."""
        error = create_network_error_with_retry_info(
            "Connection timeout",
            provider="gemini-cli",
            status_code=503,
            retry_after=60,
            max_retries=3,
            retry_count=1,
        )

        assert isinstance(error, NetworkError)
        assert "Connection timeout" in str(error)
        assert error.provider == "gemini-cli"
        assert error.status_code == 503

        debug_info = error.get_debug_info()
        assert debug_info["debug_context"]["retry_after"] == 60
        assert debug_info["debug_context"]["max_retries"] == 3
        assert debug_info["debug_context"]["retry_count"] == 1
        # Check that the message was enhanced with retry info
        assert "503" in str(error)
        assert "retry" in str(error).lower()

    def test_create_network_error_max_retries_reached(self) -> None:
        """Test create_network_error_with_retry_info when max retries reached."""
        error = create_network_error_with_retry_info(
            "Service unavailable",
            provider="cloud-code",
            status_code=503,
            retry_count=3,
            max_retries=3,
        )

        debug_info = error.get_debug_info()
        assert debug_info["debug_context"]["retry_count"] == 3
        assert debug_info["debug_context"]["max_retries"] == 3
        # Should still enhance message based on status code
        assert "503" in str(error)
