# this_file: src/uutel/__init__.py
"""UUTEL: The Universal AI Provider for LiteLLM

Tired of maintaining separate API clients for Claude Code, Gemini CLI, Google Cloud Code, and OpenAI Codex? 
So were we. UUTEL extends LiteLLM's provider ecosystem by implementing the Universal Unit (UU) pattern. 
It wraps these disparate CLI tools and APIs into standard LiteLLM providers. 

Why use this?
- Drop-in compatibility with anything that speaks LiteLLM (or standard OpenAI format).
- Use local, authenticated CLI sessions (like `claude login`) as backend providers.
- Consistent error handling, rate limiting, and tool calling across fundamentally different backends.
"""

from __future__ import annotations

try:
    from uutel._version import __version__
except ImportError:
    # Fallback version when running from source without installation
    __version__ = "0.0.0+unknown"

# Import providers submodule
from uutel import providers

# Import core classes and utilities - simplified for core functionality
from uutel.core import (
    AuthenticationError,
    AuthResult,
    BaseAuth,
    BaseUU,
    ModelError,
    NetworkError,
    ProviderError,
    RateLimitError,
    RetryConfig,
    UUTELError,
    ValidationError,
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
    # Core classes
    "AuthResult",
    # Exceptions
    "AuthenticationError",
    "BaseAuth",
    "BaseUU",
    "ModelError",
    "NetworkError",
    "ProviderError",
    "RateLimitError",
    "RetryConfig",
    "UUTELError",
    "ValidationError",
    # Package metadata
    "__version__",
    # Core utilities
    "create_http_client",
    "create_tool_call_response",
    "extract_provider_from_model",
    "extract_tool_calls_from_response",
    "format_error_message",
    "get_error_debug_info",
    # Submodules
    "providers",
    "transform_openai_to_provider",
    "transform_openai_tools_to_provider",
    "transform_provider_to_openai",
    "transform_provider_tools_to_openai",
    "validate_model_name",
    "validate_tool_schema",
]
