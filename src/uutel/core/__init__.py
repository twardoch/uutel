# this_file: src/uutel/core/__init__.py
"""UUTEL core module - simplified essential functionality."""

from __future__ import annotations

# Core authentication
from uutel.core.auth import (
    AuthResult,
    BaseAuth,
    get_api_key_from_env,
    load_cli_credentials,
)

# Base provider class
from uutel.core.base import BaseUU

# Configuration
from uutel.core.config import (
    UUTELConfig,
    create_default_config,
    load_config,
    save_config,
    validate_config,
)

# Exceptions
from uutel.core.exceptions import (
    AuthenticationError,
    ModelError,
    NetworkError,
    ProviderError,
    RateLimitError,
    UUTELError,
    ValidationError,
)

# Logging
from uutel.core.logging_config import get_logger

# Subprocess utilities
from uutel.core.runners import (
    SubprocessResult,
    astream_subprocess_lines,
    run_subprocess,
    stream_subprocess_lines,
)

# Basic utilities
from uutel.core.utils import (
    RetryConfig,
    create_http_client,
    create_text_chunk,
    create_tool_call_response,
    create_tool_chunk,
    extract_provider_from_model,
    extract_tool_calls_from_response,
    format_error_message,
    get_error_debug_info,
    merge_usage_stats,
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
    "RetryConfig",
    # Subprocess utilities
    "SubprocessResult",
    "UUTELConfig",
    "UUTELError",
    "ValidationError",
    "astream_subprocess_lines",
    # Configuration
    "create_default_config",
    # Basic utilities
    "create_http_client",
    "create_text_chunk",
    "create_tool_call_response",
    "create_tool_chunk",
    "extract_provider_from_model",
    "extract_tool_calls_from_response",
    "format_error_message",
    "get_api_key_from_env",
    "get_error_debug_info",
    # Logging
    "get_logger",
    "load_cli_credentials",
    "load_config",
    "merge_usage_stats",
    "run_subprocess",
    "save_config",
    "stream_subprocess_lines",
    "transform_openai_to_provider",
    "transform_openai_tools_to_provider",
    "transform_provider_to_openai",
    "transform_provider_tools_to_openai",
    "validate_config",
    "validate_model_name",
    "validate_tool_schema",
]
