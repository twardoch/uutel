# this_file: src/uutel/core/__init__.py
"""UUTEL core module - simplified essential functionality."""

from __future__ import annotations

# Core authentication
from uutel.core.auth import AuthResult, BaseAuth

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

# Basic utilities
from uutel.core.utils import (
    create_tool_call_response,
    extract_provider_from_model,
    format_error_message,
    get_error_debug_info,
    transform_openai_to_provider,
    transform_provider_to_openai,
    validate_model_name,
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
    "UUTELConfig",
    "UUTELError",
    "ValidationError",
    # Configuration
    "create_default_config",
    # Basic utilities
    "create_tool_call_response",
    "extract_provider_from_model",
    "format_error_message",
    "get_error_debug_info",
    # Logging
    "get_logger",
    "load_config",
    "save_config",
    "transform_openai_to_provider",
    "transform_provider_to_openai",
    "validate_config",
    "validate_model_name",
]
