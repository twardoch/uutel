# this_file: tests/test_exceptions.py
"""Tests for UUTEL exception framework."""

from __future__ import annotations

from uutel.core.exceptions import (
    AuthenticationError,
    ModelError,
    NetworkError,
    ProviderError,
    RateLimitError,
    UUTELError,
    ValidationError,
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
        assert str(error) == "Test error"
        assert error.provider == "test-provider"

    def test_uutel_error_with_error_code(self) -> None:
        """Test UUTELError with error code."""
        error = UUTELError("Test error", error_code="TEST_001")
        assert str(error) == "Test error"
        assert error.error_code == "TEST_001"

    def test_uutel_error_inheritance(self) -> None:
        """Test that UUTELError inherits from Exception."""
        error = UUTELError("Test error")
        assert isinstance(error, Exception)


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
