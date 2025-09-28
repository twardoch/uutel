# this_file: src/uutel/core/exceptions.py
"""UUTEL exception framework.

This module provides standardized exception classes for consistent error
handling across all UUTEL providers. All exceptions inherit from UUTELError
and include provider context for better debugging.
"""

from __future__ import annotations

from typing import Any


class UUTELError(Exception):
    """Base exception class for all UUTEL errors.

    This class serves as the foundation for all UUTEL-specific exceptions
    and provides common attributes for error context and debugging.

    Attributes:
        message: Human-readable error message
        provider: Name of the provider where the error occurred
        error_code: Provider-specific or UUTEL error code
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        error_code: str | None = None,
    ) -> None:
        """Initialize UUTELError.

        Args:
            message: Human-readable error message
            provider: Name of the provider where error occurred
            error_code: Error code for categorization
        """
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.error_code = error_code

    def __str__(self) -> str:
        """Return string representation of the error."""
        return self.message


class AuthenticationError(UUTELError):
    """Authentication-related errors.

    Raised when authentication fails, tokens expire, or credentials
    are invalid. Includes context for debugging auth flows.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        error_code: str | None = None,
    ) -> None:
        """Initialize AuthenticationError.

        Args:
            message: Human-readable error message
            provider: Name of the provider where auth failed
            error_code: Authentication-specific error code
        """
        super().__init__(message, provider=provider, error_code=error_code)


class RateLimitError(UUTELError):
    """Rate limiting and quota errors.

    Raised when API rate limits are exceeded or quotas are exhausted.
    Includes retry timing information when available.

    Attributes:
        retry_after: Seconds to wait before retrying (if provided by API)
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        error_code: str | None = None,
        retry_after: int | None = None,
    ) -> None:
        """Initialize RateLimitError.

        Args:
            message: Human-readable error message
            provider: Name of the provider where rate limit hit
            error_code: Rate limit specific error code
            retry_after: Seconds to wait before retry
        """
        super().__init__(message, provider=provider, error_code=error_code)
        self.retry_after = retry_after


class ModelError(UUTELError):
    """Model-specific errors.

    Raised when model is not found, overloaded, or encounters internal
    errors. Includes model context for debugging.

    Attributes:
        model_name: Name of the model that caused the error
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        error_code: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize ModelError.

        Args:
            message: Human-readable error message
            provider: Name of the provider
            error_code: Model-specific error code
            model_name: Name of the model that failed
        """
        super().__init__(message, provider=provider, error_code=error_code)
        self.model_name = model_name


class NetworkError(UUTELError):
    """Network and HTTP-related errors.

    Raised when network requests fail, timeouts occur, or HTTP errors
    are encountered. Includes HTTP status information.

    Attributes:
        status_code: HTTP status code (if applicable)
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        error_code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        """Initialize NetworkError.

        Args:
            message: Human-readable error message
            provider: Name of the provider
            error_code: Network-specific error code
            status_code: HTTP status code
        """
        super().__init__(message, provider=provider, error_code=error_code)
        self.status_code = status_code


class ValidationError(UUTELError):
    """Input validation and format errors.

    Raised when input data is invalid, malformed, or doesn't meet
    provider requirements. Includes field context for debugging.

    Attributes:
        field_name: Name of the field that failed validation
        field_value: Value that caused the validation error
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        error_code: str | None = None,
        field_name: str | None = None,
        field_value: Any = None,
    ) -> None:
        """Initialize ValidationError.

        Args:
            message: Human-readable error message
            provider: Name of the provider
            error_code: Validation-specific error code
            field_name: Name of the field that failed
            field_value: Value that caused validation failure
        """
        super().__init__(message, provider=provider, error_code=error_code)
        self.field_name = field_name
        self.field_value = field_value


class ProviderError(UUTELError):
    """Provider-specific internal errors.

    Raised when provider APIs return unexpected errors or when
    provider-specific issues occur. Includes original error context.

    Attributes:
        original_error: Original error message from the provider
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        error_code: str | None = None,
        original_error: Any = None,
    ) -> None:
        """Initialize ProviderError.

        Args:
            message: Human-readable error message
            provider: Name of the provider
            error_code: Provider-specific error code
            original_error: Original error from provider API
        """
        super().__init__(message, provider=provider, error_code=error_code)
        self.original_error = original_error
