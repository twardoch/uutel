# this_file: src/uutel/core/__init__.py
"""UUTEL core module - base classes and common utilities."""

from __future__ import annotations

# Local imports - simplified for core functionality only
from uutel.core.auth import AuthResult, BaseAuth
from uutel.core.base import BaseUU
from uutel.core.exceptions import (
    AuthenticationError,
    ModelError,
    NetworkError,
    ProviderError,
    RateLimitError,
    UUTELError,
    ValidationError,
)
from uutel.core.logging_config import (
    get_logger,
)
from uutel.core.utils import (
    create_http_client,
    create_tool_call_response,
    extract_provider_from_model,
    extract_tool_calls_from_response,
    format_error_message,
    get_error_debug_info,
    transform_openai_to_provider,
    transform_openai_tools_to_provider,
    transform_provider_to_openai,
    transform_provider_tools_to_openai,
    validate_model_name,
    validate_tool_schema,
)

__all__ = [
    # Core authentication
    "AuthResult",
    # Exceptions
    "AuthenticationError",
    "BaseAuth",
    # Base provider class
    "BaseUU",
    "ModelError",
    "NetworkError",
    "ProviderError",
    "RateLimitError",
    "UUTELError",
    "ValidationError",
    # Utilities
    "create_http_client",
    "create_tool_call_response",
    "extract_provider_from_model",
    "extract_tool_calls_from_response",
    "format_error_message",
    "get_error_debug_info",
    # Logging
    "get_logger",
    "transform_openai_to_provider",
    "transform_openai_tools_to_provider",
    "transform_provider_to_openai",
    "transform_provider_tools_to_openai",
    "validate_model_name",
    "validate_tool_schema",
]
