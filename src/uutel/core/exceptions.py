# this_file: src/uutel/core/exceptions.py
"""UUTEL exception framework.

This module provides standardized exception classes for consistent error
handling across all UUTEL providers. All exceptions inherit from UUTELError
and include provider context for better debugging.

Example usage:
    Basic error handling:
        try:
            result = provider.completion(model="test", messages=[])
        except UUTELError as e:
            print(f"Provider error: {e}")
            print(f"Provider: {e.provider}")
            print(f"Error code: {e.error_code}")

    Specific error types:
        try:
            provider.authenticate()
        except AuthenticationError as e:
            print(f"Auth failed: {e}")
        except NetworkError as e:
            print(f"Network issue: {e}")
        except RateLimitError as e:
            print(f"Rate limited: {e}")

    Creating custom errors:
        raise ProviderError(
            "Model not found",
            provider="claude-code",
            error_code="MODEL_NOT_FOUND",
            debug_context={"requested_model": "invalid-model"}
        )
"""

from __future__ import annotations

# Standard library imports
import traceback
from datetime import datetime
from typing import Any


class UUTELError(Exception):
    """Base exception class for all UUTEL errors.

    This class serves as the foundation for all UUTEL-specific exceptions
    and provides enhanced context and debugging information.

    Attributes:
        message: Human-readable error message
        provider: Name of the provider where the error occurred
        error_code: Provider-specific or UUTEL error code
        timestamp: When the error occurred
        request_id: Unique identifier for the request (if available)
        debug_context: Additional debugging information
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        debug_context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize UUTELError with enhanced debugging context.

        Args:
            message: Human-readable error message
            provider: Name of the provider where error occurred
            error_code: Error code for categorization
            request_id: Unique identifier for the request
            debug_context: Additional debugging information
        """
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.error_code = error_code
        self.request_id = request_id
        self.debug_context = debug_context or {}
        self.timestamp = datetime.now().isoformat()

    def __str__(self) -> str:
        """Return enhanced string representation with context."""
        parts = [self.message]

        if self.provider:
            parts.append(f"Provider: {self.provider}")

        if self.error_code:
            parts.append(f"Code: {self.error_code}")

        if self.request_id:
            parts.append(f"Request: {self.request_id}")

        return " | ".join(parts)

    def get_debug_info(self) -> dict[str, Any]:
        """Get comprehensive debugging information.

        Returns:
            Dictionary containing all error context for debugging
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "provider": self.provider,
            "error_code": self.error_code,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "debug_context": self.debug_context,
            "traceback": (
                traceback.format_exc()
                if traceback.format_exc().strip() != "NoneType: None"
                else None
            ),
        }

    def add_context(self, key: str, value: Any) -> None:
        """Add debugging context to the error.

        Args:
            key: Context key name
            value: Context value
        """
        self.debug_context[key] = value


class AuthenticationError(UUTELError):
    """Authentication-related errors.

    Raised when authentication fails, tokens expire, or credentials
    are invalid. Includes enhanced context for debugging auth flows.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        auth_method: str | None = None,
        debug_context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize AuthenticationError with enhanced context.

        Args:
            message: Human-readable error message
            provider: Name of the provider where auth failed
            error_code: Authentication-specific error code
            request_id: Unique identifier for the request
            auth_method: Authentication method that failed
            debug_context: Additional debugging information
        """
        debug_context = debug_context or {}
        if auth_method:
            debug_context["auth_method"] = auth_method

        super().__init__(
            message,
            provider=provider,
            error_code=error_code,
            request_id=request_id,
            debug_context=debug_context,
        )


class RateLimitError(UUTELError):
    """Rate limiting and quota errors.

    Raised when API rate limits are exceeded or quotas are exhausted.
    Includes enhanced retry timing information and quota context.

    Attributes:
        retry_after: Seconds to wait before retrying (if provided by API)
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        retry_after: int | None = None,
        current_limit: int | None = None,
        reset_time: str | None = None,
        debug_context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize RateLimitError with enhanced rate limit context.

        Args:
            message: Human-readable error message
            provider: Name of the provider where rate limit hit
            error_code: Rate limit specific error code
            request_id: Unique identifier for the request
            retry_after: Seconds to wait before retry
            current_limit: Current rate limit threshold
            reset_time: When the rate limit resets
            debug_context: Additional debugging information
        """
        debug_context = debug_context or {}
        if current_limit is not None:
            debug_context["current_limit"] = current_limit
        if reset_time:
            debug_context["reset_time"] = reset_time

        super().__init__(
            message,
            provider=provider,
            error_code=error_code,
            request_id=request_id,
            debug_context=debug_context,
        )
        self.retry_after = retry_after


class ModelError(UUTELError):
    """Model-specific errors.

    Raised when model is not found, overloaded, or encounters internal
    errors. Includes enhanced model context for debugging.

    Attributes:
        model_name: Name of the model that caused the error
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        model_name: str | None = None,
        model_parameters: dict[str, Any] | None = None,
        debug_context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ModelError with enhanced model context.

        Args:
            message: Human-readable error message
            provider: Name of the provider
            error_code: Model-specific error code
            request_id: Unique identifier for the request
            model_name: Name of the model that failed
            model_parameters: Parameters passed to the model
            debug_context: Additional debugging information
        """
        debug_context = debug_context or {}
        if model_name:
            debug_context["model_name"] = model_name
        if model_parameters:
            debug_context["model_parameters"] = model_parameters

        super().__init__(
            message,
            provider=provider,
            error_code=error_code,
            request_id=request_id,
            debug_context=debug_context,
        )
        self.model_name = model_name


class NetworkError(UUTELError):
    """Network and HTTP-related errors.

    Raised when network requests fail, timeouts occur, or HTTP errors
    are encountered. Includes enhanced HTTP and network context.

    Attributes:
        status_code: HTTP status code (if applicable)
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        status_code: int | None = None,
        url: str | None = None,
        response_headers: dict[str, str] | None = None,
        debug_context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize NetworkError with enhanced network context.

        Args:
            message: Human-readable error message
            provider: Name of the provider
            error_code: Network-specific error code
            request_id: Unique identifier for the request
            status_code: HTTP status code
            url: URL that caused the error
            response_headers: HTTP response headers
            debug_context: Additional debugging information
        """
        debug_context = debug_context or {}
        if status_code is not None:
            debug_context["status_code"] = status_code
        if url:
            debug_context["url"] = url
        if response_headers:
            debug_context["response_headers"] = response_headers

        super().__init__(
            message,
            provider=provider,
            error_code=error_code,
            request_id=request_id,
            debug_context=debug_context,
        )
        self.status_code = status_code


class ValidationError(UUTELError):
    """Input validation and format errors.

    Raised when input data is invalid, malformed, or doesn't meet
    provider requirements. Includes enhanced validation context.

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
        request_id: str | None = None,
        field_name: str | None = None,
        field_value: Any = None,
        expected_format: str | None = None,
        validation_rules: list[str] | None = None,
        debug_context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ValidationError with enhanced validation context.

        Args:
            message: Human-readable error message
            provider: Name of the provider
            error_code: Validation-specific error code
            request_id: Unique identifier for the request
            field_name: Name of the field that failed
            field_value: Value that caused validation failure
            expected_format: Expected format or type
            validation_rules: List of validation rules that failed
            debug_context: Additional debugging information
        """
        debug_context = debug_context or {}
        if field_name:
            debug_context["field_name"] = field_name
        if field_value is not None:
            # Convert to string for safety
            debug_context["field_value"] = str(field_value)
        if expected_format:
            debug_context["expected_format"] = expected_format
        if validation_rules:
            debug_context["validation_rules"] = validation_rules

        super().__init__(
            message,
            provider=provider,
            error_code=error_code,
            request_id=request_id,
            debug_context=debug_context,
        )
        self.field_name = field_name
        self.field_value = field_value


class ProviderError(UUTELError):
    """Provider-specific internal errors.

    Raised when provider APIs return unexpected errors or when
    provider-specific issues occur. Includes enhanced original error context.

    Attributes:
        original_error: Original error message from the provider
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        original_error: Any = None,
        api_endpoint: str | None = None,
        response_data: dict[str, Any] | None = None,
        debug_context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ProviderError with enhanced provider context.

        Args:
            message: Human-readable error message
            provider: Name of the provider
            error_code: Provider-specific error code
            request_id: Unique identifier for the request
            original_error: Original error from provider API
            api_endpoint: API endpoint that caused the error
            response_data: Raw response data from provider
            debug_context: Additional debugging information
        """
        debug_context = debug_context or {}
        if original_error is not None:
            debug_context["original_error"] = str(original_error)
        if api_endpoint:
            debug_context["api_endpoint"] = api_endpoint
        if response_data:
            debug_context["response_data"] = response_data

        super().__init__(
            message,
            provider=provider,
            error_code=error_code,
            request_id=request_id,
            debug_context=debug_context,
        )
        self.original_error = original_error


class ConfigurationError(UUTELError):
    """Configuration and setup errors.

    Raised when provider configurations are invalid, missing required
    settings, or have conflicting values. Includes enhanced config context.

    Attributes:
        config_key: Configuration key that caused the error
        config_section: Configuration section (if applicable)
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        config_key: str | None = None,
        config_section: str | None = None,
        suggested_fix: str | None = None,
        debug_context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ConfigurationError with enhanced config context.

        Args:
            message: Human-readable error message
            provider: Name of the provider
            error_code: Configuration-specific error code
            request_id: Unique identifier for the request
            config_key: Configuration key that failed
            config_section: Configuration section name
            suggested_fix: Suggested solution for the configuration issue
            debug_context: Additional debugging information
        """
        debug_context = debug_context or {}
        if config_key:
            debug_context["config_key"] = config_key
        if config_section:
            debug_context["config_section"] = config_section
        if suggested_fix:
            debug_context["suggested_fix"] = suggested_fix

        super().__init__(
            message,
            provider=provider,
            error_code=error_code,
            request_id=request_id,
            debug_context=debug_context,
        )
        self.config_key = config_key
        self.config_section = config_section


class ToolCallError(UUTELError):
    """Function/tool calling errors.

    Raised when tool schema validation fails, function execution errors,
    or tool response parsing issues occur.

    Attributes:
        tool_name: Name of the tool that caused the error
        tool_call_id: Unique identifier for the tool call
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        tool_parameters: dict[str, Any] | None = None,
        execution_stage: str | None = None,
        debug_context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ToolCallError with enhanced tool context.

        Args:
            message: Human-readable error message
            provider: Name of the provider
            error_code: Tool-specific error code
            request_id: Unique identifier for the request
            tool_name: Name of the tool that failed
            tool_call_id: Unique identifier for the tool call
            tool_parameters: Parameters passed to the tool
            execution_stage: Stage where the error occurred (validation, execution, etc.)
            debug_context: Additional debugging information
        """
        debug_context = debug_context or {}
        if tool_name:
            debug_context["tool_name"] = tool_name
        if tool_call_id:
            debug_context["tool_call_id"] = tool_call_id
        if tool_parameters:
            debug_context["tool_parameters"] = tool_parameters
        if execution_stage:
            debug_context["execution_stage"] = execution_stage

        super().__init__(
            message,
            provider=provider,
            error_code=error_code,
            request_id=request_id,
            debug_context=debug_context,
        )
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id


class StreamingError(UUTELError):
    """Streaming response errors.

    Raised when streaming connections fail, are interrupted, or encounter
    parsing errors. Includes enhanced streaming context.

    Attributes:
        stream_state: Current state of the stream when error occurred
        bytes_received: Number of bytes received before error
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        stream_state: str | None = None,
        bytes_received: int | None = None,
        connection_id: str | None = None,
        chunk_index: int | None = None,
        debug_context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize StreamingError with enhanced streaming context.

        Args:
            message: Human-readable error message
            provider: Name of the provider
            error_code: Streaming-specific error code
            request_id: Unique identifier for the request
            stream_state: State of stream when error occurred
            bytes_received: Number of bytes received before error
            connection_id: Streaming connection identifier
            chunk_index: Index of chunk where error occurred
            debug_context: Additional debugging information
        """
        debug_context = debug_context or {}
        if stream_state:
            debug_context["stream_state"] = stream_state
        if bytes_received is not None:
            debug_context["bytes_received"] = bytes_received
        if connection_id:
            debug_context["connection_id"] = connection_id
        if chunk_index is not None:
            debug_context["chunk_index"] = chunk_index

        super().__init__(
            message,
            provider=provider,
            error_code=error_code,
            request_id=request_id,
            debug_context=debug_context,
        )
        self.stream_state = stream_state
        self.bytes_received = bytes_received


class TimeoutError(UUTELError):
    """Request timeout errors.

    Raised when requests exceed timeout limits or when specific
    operations take too long. Includes enhanced timeout context.

    Attributes:
        timeout_duration: Duration that was exceeded (in seconds)
        operation_type: Type of operation that timed out
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        timeout_duration: float | None = None,
        operation_type: str | None = None,
        elapsed_time: float | None = None,
        suggested_timeout: float | None = None,
        debug_context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize TimeoutError with enhanced timeout context.

        Args:
            message: Human-readable error message
            provider: Name of the provider
            error_code: Timeout-specific error code
            request_id: Unique identifier for the request
            timeout_duration: Timeout that was exceeded
            operation_type: Type of operation (request, stream, auth, etc.)
            elapsed_time: Time elapsed before timeout
            suggested_timeout: Suggested timeout for retry
            debug_context: Additional debugging information
        """
        debug_context = debug_context or {}
        if timeout_duration is not None:
            debug_context["timeout_duration"] = timeout_duration
        if operation_type:
            debug_context["operation_type"] = operation_type
        if elapsed_time is not None:
            debug_context["elapsed_time"] = elapsed_time
        if suggested_timeout is not None:
            debug_context["suggested_timeout"] = suggested_timeout

        super().__init__(
            message,
            provider=provider,
            error_code=error_code,
            request_id=request_id,
            debug_context=debug_context,
        )
        self.timeout_duration = timeout_duration
        self.operation_type = operation_type


class QuotaExceededError(RateLimitError):
    """Quota and usage limit errors.

    Specific type of rate limit error for quota exhaustion.
    Inherits from RateLimitError but adds quota-specific context.

    Attributes:
        quota_type: Type of quota exceeded (daily, monthly, tokens, etc.)
        quota_limit: Maximum quota limit
        quota_used: Amount of quota already used
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        quota_type: str | None = None,
        quota_limit: int | None = None,
        quota_used: int | None = None,
        reset_time: str | None = None,
        debug_context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize QuotaExceededError with enhanced quota context.

        Args:
            message: Human-readable error message
            provider: Name of the provider
            error_code: Quota-specific error code
            request_id: Unique identifier for the request
            quota_type: Type of quota (daily, monthly, tokens, etc.)
            quota_limit: Maximum quota limit
            quota_used: Amount of quota used
            reset_time: When quota resets
            debug_context: Additional debugging information
        """
        debug_context = debug_context or {}
        if quota_type:
            debug_context["quota_type"] = quota_type
        if quota_limit is not None:
            debug_context["quota_limit"] = quota_limit
        if quota_used is not None:
            debug_context["quota_used"] = quota_used

        super().__init__(
            message,
            provider=provider,
            error_code=error_code,
            request_id=request_id,
            reset_time=reset_time,
            debug_context=debug_context,
        )
        self.quota_type = quota_type
        self.quota_limit = quota_limit
        self.quota_used = quota_used


class ModelNotFoundError(ModelError):
    """Model not found or unavailable errors.

    Specific type of model error when the requested model doesn't exist
    or is not available. Inherits from ModelError with enhanced context.

    Attributes:
        available_models: List of available models (if provided by API)
        suggested_model: Alternative model suggestion
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        model_name: str | None = None,
        available_models: list[str] | None = None,
        suggested_model: str | None = None,
        debug_context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ModelNotFoundError with enhanced model context.

        Args:
            message: Human-readable error message
            provider: Name of the provider
            error_code: Model-specific error code
            request_id: Unique identifier for the request
            model_name: Name of the model that wasn't found
            available_models: List of available models
            suggested_model: Alternative model suggestion
            debug_context: Additional debugging information
        """
        debug_context = debug_context or {}
        if available_models:
            debug_context["available_models"] = available_models
        if suggested_model:
            debug_context["suggested_model"] = suggested_model

        super().__init__(
            message,
            provider=provider,
            error_code=error_code,
            request_id=request_id,
            model_name=model_name,
            debug_context=debug_context,
        )
        self.available_models = available_models
        self.suggested_model = suggested_model


class TokenLimitError(ModelError):
    """Token limit exceeded errors.

    Specific type of model error when input/output tokens exceed limits.
    Inherits from ModelError with token-specific context.

    Attributes:
        token_count: Number of tokens that exceeded the limit
        token_limit: Maximum token limit for the model
        token_type: Type of tokens (input, output, total)
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        model_name: str | None = None,
        token_count: int | None = None,
        token_limit: int | None = None,
        token_type: str | None = None,
        suggested_action: str | None = None,
        debug_context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize TokenLimitError with enhanced token context.

        Args:
            message: Human-readable error message
            provider: Name of the provider
            error_code: Token-specific error code
            request_id: Unique identifier for the request
            model_name: Name of the model
            token_count: Number of tokens that exceeded limit
            token_limit: Maximum token limit
            token_type: Type of tokens (input, output, total)
            suggested_action: Suggested action to resolve the issue
            debug_context: Additional debugging information
        """
        debug_context = debug_context or {}
        if token_count is not None:
            debug_context["token_count"] = token_count
        if token_limit is not None:
            debug_context["token_limit"] = token_limit
        if token_type:
            debug_context["token_type"] = token_type
        if suggested_action:
            debug_context["suggested_action"] = suggested_action

        super().__init__(
            message,
            provider=provider,
            error_code=error_code,
            request_id=request_id,
            model_name=model_name,
            debug_context=debug_context,
        )
        self.token_count = token_count
        self.token_limit = token_limit
        self.token_type = token_type


# Helper functions for creating enhanced error messages


def create_configuration_error(
    message: str,
    provider: str | None = None,
    config_key: str | None = None,
    suggested_fix: str | None = None,
    **kwargs,
) -> ConfigurationError:
    """Create a ConfigurationError with enhanced context and helpful suggestions.

    Args:
        message: Base error message
        provider: Provider name
        config_key: Configuration key that failed
        suggested_fix: Suggested solution
        **kwargs: Additional context

    Returns:
        ConfigurationError with enhanced context
    """
    # Enhance message with specific configuration guidance
    enhanced_message = message
    if config_key and not suggested_fix:
        # Auto-generate helpful suggestions for common config issues
        if "api_key" in config_key.lower():
            suggested_fix = (
                f"Set the {config_key} environment variable or pass it explicitly"
            )
        elif "url" in config_key.lower() or "endpoint" in config_key.lower():
            suggested_fix = f"Verify the {config_key} is a valid URL and accessible"
        elif "timeout" in config_key.lower():
            suggested_fix = f"Increase the {config_key} value or use default timeout"

    if suggested_fix:
        enhanced_message = f"{message}. Suggestion: {suggested_fix}"

    return ConfigurationError(
        enhanced_message,
        provider=provider,
        config_key=config_key,
        suggested_fix=suggested_fix,
        debug_context=kwargs,
    )


def create_model_not_found_error(
    model_name: str,
    provider: str | None = None,
    available_models: list[str] | None = None,
    **kwargs,
) -> ModelNotFoundError:
    """Create a ModelNotFoundError with helpful model suggestions.

    Args:
        model_name: Name of the model that wasn't found
        provider: Provider name
        available_models: List of available models
        **kwargs: Additional context

    Returns:
        ModelNotFoundError with enhanced context
    """
    # Create helpful error message with model suggestions
    message = f"Model '{model_name}' not found"

    # Auto-suggest similar models if available
    suggested_model = None
    if available_models:
        # Simple similarity matching for suggestions
        model_lower = model_name.lower()
        for available in available_models:
            if (
                available.lower().startswith(model_lower[:3])
                or model_lower in available.lower()
                or available.lower() in model_lower
            ):
                suggested_model = available
                break

        if suggested_model:
            message = f"{message}. Did you mean '{suggested_model}'?"
        elif len(available_models) <= 10:
            models_list = "', '".join(available_models)
            message = f"{message}. Available models: ['{models_list}']"
        else:
            message = f"{message}. {len(available_models)} models available"

    return ModelNotFoundError(
        message,
        provider=provider,
        model_name=model_name,
        available_models=available_models,
        suggested_model=suggested_model,
        debug_context=kwargs,
    )


def create_token_limit_error(
    model_name: str,
    token_count: int,
    token_limit: int,
    token_type: str = "input",
    **kwargs,
) -> TokenLimitError:
    """Create a TokenLimitError with helpful reduction suggestions.

    Args:
        model_name: Name of the model
        token_count: Number of tokens that exceeded limit
        token_limit: Maximum token limit
        token_type: Type of tokens (input, output, total)
        **kwargs: Additional context

    Returns:
        TokenLimitError with enhanced context
    """
    # Calculate how much to reduce
    excess_tokens = token_count - token_limit
    reduction_percentage = (excess_tokens / token_count) * 100

    # Create helpful error message with reduction suggestions
    message = (
        f"Token limit exceeded for model '{model_name}': "
        f"{token_count} {token_type} tokens > {token_limit} limit"
    )

    # Auto-generate helpful suggestions
    if token_type in ("input", "total"):
        if reduction_percentage <= 10:
            suggested_action = "Reduce input by removing non-essential details"
        elif reduction_percentage <= 25:
            suggested_action = (
                "Significantly shorten the input or split into multiple requests"
            )
        else:
            suggested_action = "Consider using a model with higher token limits or split into multiple smaller requests"
    else:  # output tokens
        suggested_action = (
            "Reduce max_tokens parameter or use streaming for long responses"
        )

    message = f"{message}. {suggested_action}"

    return TokenLimitError(
        message,
        model_name=model_name,
        token_count=token_count,
        token_limit=token_limit,
        token_type=token_type,
        suggested_action=suggested_action,
        debug_context=kwargs,
    )


def create_network_error_with_retry_info(
    message: str,
    status_code: int | None = None,
    provider: str | None = None,
    url: str | None = None,
    **kwargs,
) -> NetworkError:
    """Create a NetworkError with retry and resolution suggestions.

    Args:
        message: Base error message
        status_code: HTTP status code
        provider: Provider name
        url: URL that failed
        **kwargs: Additional context

    Returns:
        NetworkError with enhanced context
    """
    # Enhance message based on status code
    enhanced_message = message
    suggested_action = None

    if status_code:
        if status_code == 429:
            suggested_action = "Wait and retry with exponential backoff"
        elif status_code in (500, 502, 503, 504):
            suggested_action = "Temporary server issue - retry after a short delay"
        elif status_code == 401:
            suggested_action = "Check authentication credentials and permissions"
        elif status_code == 403:
            suggested_action = "Verify API access permissions and subscription status"
        elif status_code == 404:
            suggested_action = "Verify the API endpoint URL is correct"
        elif status_code >= 400:
            suggested_action = "Check request parameters and format"

        if suggested_action:
            enhanced_message = f"{message} (HTTP {status_code}). {suggested_action}"

    return NetworkError(
        enhanced_message,
        provider=provider,
        status_code=status_code,
        url=url,
        debug_context=kwargs,
    )


# Exception type mapping for easier imports
EXCEPTION_TYPES = {
    "base": UUTELError,
    "auth": AuthenticationError,
    "rate_limit": RateLimitError,
    "quota": QuotaExceededError,
    "model": ModelError,
    "model_not_found": ModelNotFoundError,
    "token_limit": TokenLimitError,
    "network": NetworkError,
    "validation": ValidationError,
    "provider": ProviderError,
    "config": ConfigurationError,
    "tool": ToolCallError,
    "streaming": StreamingError,
    "timeout": TimeoutError,
}


def get_exception_type(error_type: str) -> type[UUTELError]:
    """Get exception class by type name.

    Args:
        error_type: Exception type name

    Returns:
        Exception class

    Raises:
        ValueError: If error_type is not recognized
    """
    if error_type not in EXCEPTION_TYPES:
        raise ValueError(f"Unknown exception type: {error_type}")
    return EXCEPTION_TYPES[error_type]
