# this_file: src/uutel/core/utils.py
"""UUTEL core utilities and helpers.

This module provides common utilities used across UUTEL providers,
including message transformation, HTTP client creation, retry logic,
and validation functions.
"""

from __future__ import annotations

# Standard library imports
import json
import re
from dataclasses import dataclass, field
from typing import Any

# Third-party imports
import httpx

# Local imports
from .exceptions import UUTELError
from .logging_config import get_logger

logger = get_logger(__name__)

# Pre-compiled regex patterns for performance optimization
_MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")
_INVALID_CHARS_PATTERN = re.compile(r"[\s\n\r\t\0]")

# Performance optimization: Cache for frequently validated models
_MODEL_VALIDATION_CACHE: dict[str, bool] = {}
_PROVIDER_EXTRACTION_CACHE: dict[str, tuple[str, str]] = {}
_CACHE_SIZE_LIMIT = 1000


@dataclass
class RetryConfig:
    """Configuration for retry logic.

    Attributes:
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff multiplier
        retry_on_status: HTTP status codes that should trigger retries
        retry_on_exceptions: Exception types that should trigger retries
    """

    max_retries: int = 3
    backoff_factor: float = 2.0
    retry_on_status: list[int] = field(default_factory=lambda: [429, 502, 503, 504])
    retry_on_exceptions: list[type] = field(
        default_factory=lambda: [ConnectionError, TimeoutError]
    )


def transform_openai_to_provider(
    messages: list[dict[str, Any]], provider_name: str
) -> list[dict[str, Any]]:
    """Transform OpenAI format messages to provider-specific format.

    This function converts messages from OpenAI's standard format to the
    format expected by a specific provider. Base implementation provides
    pass-through behavior; providers can override this for custom formatting.

    Args:
        messages: List of messages in OpenAI format
        provider_name: Name of the target provider

    Returns:
        List of messages in provider-specific format
    """
    # Early return for empty messages (most common optimization)
    if not messages:
        return []

    logger.debug(f"Transforming {len(messages)} messages for provider: {provider_name}")

    # Performance optimization: Pre-allocate list and use list comprehension
    # Base implementation: pass-through with filtering
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
        if isinstance(msg, dict) and "role" in msg and "content" in msg
    ]


def transform_provider_to_openai(
    messages: list[dict[str, Any]], provider_name: str
) -> list[dict[str, Any]]:
    """Transform provider-specific messages to OpenAI format.

    This function converts messages from a provider's format back to
    OpenAI's standard format. Base implementation provides pass-through
    behavior; providers can override this for custom formatting.

    Args:
        messages: List of messages in provider-specific format
        provider_name: Name of the source provider

    Returns:
        List of messages in OpenAI format
    """
    # Early return for empty messages (most common optimization)
    if not messages:
        return []

    logger.debug(
        f"Transforming {len(messages)} messages from provider: {provider_name}"
    )

    # Performance optimization: Use list comprehension instead of loop
    # Base implementation: pass-through with filtering
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
        if isinstance(msg, dict) and "role" in msg and "content" in msg
    ]


def create_http_client(
    async_client: bool = False,
    timeout: float | None = None,
    retry_config: RetryConfig | None = None,
) -> httpx.Client | httpx.AsyncClient:
    """Create HTTP client with retry configuration.

    Creates either a synchronous or asynchronous HTTP client with
    appropriate timeout and retry settings.

    Args:
        async_client: Whether to create an async client
        timeout: Request timeout in seconds (default: 30.0)
        retry_config: Retry configuration (default: RetryConfig())

    Returns:
        Configured HTTP client (sync or async)
    """
    if timeout is None:
        timeout = 30.0

    if retry_config is None:
        retry_config = RetryConfig()

    # Create client with appropriate settings
    if async_client:
        logger.debug(f"Creating async HTTP client with timeout: {timeout}s")
        return httpx.AsyncClient(timeout=timeout, follow_redirects=True)
    else:
        logger.debug(f"Creating sync HTTP client with timeout: {timeout}s")
        return httpx.Client(timeout=timeout, follow_redirects=True)


def validate_model_name(model: Any) -> bool:
    """Validate a model name with performance optimizations.

    Checks if a model name is valid according to UUTEL conventions.
    Valid model names:
    - Must be non-empty strings
    - Can contain letters, numbers, dots, hyphens, underscores
    - Can have provider prefixes (e.g., "uutel/claude-code/model-name")
    - Cannot contain spaces or special characters

    Args:
        model: Model name to validate

    Returns:
        True if model name is valid, False otherwise
    """
    # Handle None and empty cases (fastest checks first)
    if model is None or model == "":
        return False

    # Ensure it's a string (fast type check)
    if not isinstance(model, str):
        return False

    # Performance optimization: Check cache first for repeated validations
    if model in _MODEL_VALIDATION_CACHE:
        return _MODEL_VALIDATION_CACHE[model]

    try:
        # Prevent excessively long names (early optimization)
        if len(model) > 200:
            result = False
        # Use pre-compiled regex for invalid characters (faster than char iteration)
        elif _INVALID_CHARS_PATTERN.search(model):
            result = False
        # Handle provider prefixes with optimized string operations
        elif "/" in model:
            parts = model.split("/", 2)  # Limit splits for efficiency
            # Must have at least 3 parts for valid UUTEL prefix: "uutel/provider/model"
            if len(parts) < 3 or parts[0] != "uutel":
                result = False
            else:
                # Validate each part using pre-compiled pattern (optimized loop)
                result = all(part and _MODEL_NAME_PATTERN.match(part) for part in parts)
        else:
            # Validate simple model name using pre-compiled pattern
            result = bool(_MODEL_NAME_PATTERN.match(model))

        # Cache the result to avoid repeated validation
        if len(_MODEL_VALIDATION_CACHE) < _CACHE_SIZE_LIMIT:
            _MODEL_VALIDATION_CACHE[model] = result

        return result

    except (AttributeError, TypeError, re.error):
        # Handle any unexpected errors during validation
        return False


def extract_provider_from_model(model: str) -> tuple[str, str]:
    """Extract provider and model from a full model string with caching.

    Parses model strings like "uutel/claude-code/claude-3-5-sonnet" to
    extract the provider name and actual model name.

    Args:
        model: Full model string potentially containing provider prefix

    Returns:
        Tuple of (provider_name, model_name)
    """
    # Handle None and empty cases (fastest checks first)
    if not model:
        return "unknown", ""

    if not isinstance(model, str):
        return "unknown", str(model)

    # Performance optimization: Check cache first for repeated extractions
    if model in _PROVIDER_EXTRACTION_CACHE:
        return _PROVIDER_EXTRACTION_CACHE[model]

    try:
        # Handle simple model name without provider prefix (common case)
        if "/" not in model:
            result = ("unknown", model)
        else:
            # Optimized split with limit for better performance
            parts = model.split("/", 2)

            # Must have at least 3 parts and start with "uutel"
            if len(parts) < 3 or parts[0] != "uutel":
                result = ("unknown", model)
            else:
                # Extract provider and model names
                provider_name = parts[1] if parts[1] else "unknown"
                model_name = parts[2] if parts[2] else ""

                # Ensure extracted values are valid
                if not provider_name or not model_name:
                    result = ("unknown", model)
                else:
                    result = (provider_name, model_name)

        # Cache the result to avoid repeated extraction
        if len(_PROVIDER_EXTRACTION_CACHE) < _CACHE_SIZE_LIMIT:
            _PROVIDER_EXTRACTION_CACHE[model] = result

        return result

    except (AttributeError, IndexError, TypeError):
        # Handle any unexpected errors during extraction
        return "unknown", str(model)


def format_error_message(error: Exception | None, provider: Any) -> str:
    """Format error message for consistent error reporting.

    Creates standardized error messages that include provider context
    and relevant error details. Enhanced to utilize UUTEL error context.

    Args:
        error: The exception that occurred
        provider: Name of the provider where the error occurred

    Returns:
        Formatted error message string
    """
    try:
        # Handle None or invalid error objects
        if error is None:
            return f"[{provider or 'unknown'}] Unknown error occurred"

        # Get error type and message safely
        error_type = (
            type(error).__name__ if hasattr(error, "__class__") else "Exception"
        )

        # Get error message with fallback
        try:
            error_msg = str(error) if error else "No error message available"
        except Exception:
            error_msg = "Error occurred but message could not be retrieved"

        # Sanitize provider name
        provider = provider or "unknown"
        if not isinstance(provider, str):
            provider = str(provider)

        # If it's a UUTEL error, use its enhanced formatting
        if isinstance(error, UUTELError):
            return str(error)  # Uses the enhanced __str__ method

        # For non-UUTEL errors, use enhanced formatting with better context
        if not error_msg or error_msg.strip() == "":
            error_msg = f"Empty {error_type} occurred"

        return f"[{provider}] {error_type}: {error_msg}"

    except Exception:
        # Ultimate fallback if everything goes wrong
        return f"[{provider or 'unknown'}] Critical error in error formatting"


def _create_empty_debug_info() -> dict[str, Any]:
    """Create empty debug info structure."""
    return {
        "error_type": "NoneError",
        "message": "No error provided",
        "provider": None,
        "error_code": None,
        "request_id": None,
        "timestamp": None,
        "debug_context": {},
        "traceback": None,
    }


def _extract_error_attributes(error: Exception) -> dict[str, Any]:
    """Extract attributes from exception object."""
    debug_context: dict[str, Any] = {}
    try:
        if hasattr(error, "args") and error.args:
            debug_context["args"] = list(error.args)
        if hasattr(error, "__dict__"):
            for key, value in error.__dict__.items():
                if not key.startswith("_"):
                    try:
                        debug_context[key] = str(value)
                    except Exception:
                        debug_context[key] = f"<{type(value).__name__}>"
    except Exception as e:
        logger.debug(f"Failed to extract exception attributes: {e}")
    return debug_context


def _create_standard_debug_info(error: Exception) -> dict[str, Any]:
    """Create debug info for standard (non-UUTEL) errors."""
    error_type = type(error).__name__ if hasattr(error, "__class__") else "Exception"

    try:
        message = str(error) if error else "No error message available"
    except Exception:
        message = "Error message could not be retrieved"

    debug_context = _extract_error_attributes(error)

    return {
        "error_type": error_type,
        "message": message,
        "provider": None,
        "error_code": getattr(error, "code", None),
        "request_id": getattr(error, "request_id", None),
        "timestamp": None,
        "debug_context": debug_context,
        "traceback": None,
    }


def get_error_debug_info(error: Exception | None) -> dict[str, Any]:
    """Get comprehensive debugging information from an error."""
    try:
        if error is None:
            return _create_empty_debug_info()

        # Try UUTEL error's built-in debug info first
        if isinstance(error, UUTELError):
            try:
                return error.get_debug_info()
            except Exception as e:
                logger.warning(f"Failed to get debug info from UUTEL error: {e}")

        return _create_standard_debug_info(error)

    except Exception:
        return {
            "error_type": "DebugInfoError",
            "message": "Failed to extract debug information",
            "provider": None,
            "error_code": None,
            "request_id": None,
            "timestamp": None,
            "debug_context": {},
            "traceback": None,
        }


# Tool calling utilities


def validate_tool_schema(tool: Any) -> bool:
    """Validate a tool schema according to OpenAI function calling format.

    Checks if a tool dictionary follows the correct OpenAI tool schema:
    - Must have "type" field set to "function"
    - Must have "function" field containing tool definition
    - Function must have "name" and "description" fields
    - Parameters field is optional but if present must be valid JSON schema

    Args:
        tool: Tool definition to validate

    Returns:
        True if tool schema is valid, False otherwise
    """
    # Performance optimization: Early returns for most common failures
    if not isinstance(tool, dict):
        return False

    # Check required top-level fields (optimized access)
    if tool.get("type") != "function":
        return False

    function = tool.get("function")
    if not isinstance(function, dict):
        return False

    # Check required function fields with optimized lookups
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return False

    description = function.get("description")
    if not isinstance(description, str) or not description:
        return False

    # Parameters are optional, but if present must be valid
    parameters = function.get("parameters")
    if parameters is not None:
        # Fast validation for parameters
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            return False

    return True


def transform_openai_tools_to_provider(
    tools: list[dict[str, Any]] | None, provider_name: str
) -> list[dict[str, Any]]:
    """Transform OpenAI format tools to provider-specific format.

    Base implementation provides pass-through behavior. Providers can
    override this for custom tool format transformation.

    Args:
        tools: List of tools in OpenAI format, can be None
        provider_name: Name of the target provider

    Returns:
        List of tools in provider-specific format
    """
    if not tools:
        return []

    logger.debug(f"Transforming {len(tools)} tools for provider: {provider_name}")

    # Filter out invalid tools and pass through valid ones
    transformed = []
    for tool in tools:
        if validate_tool_schema(tool):
            transformed.append(tool.copy())

    return transformed


def transform_provider_tools_to_openai(
    tools: list[dict[str, Any]] | None, provider_name: str
) -> list[dict[str, Any]]:
    """Transform provider-specific tools to OpenAI format.

    Base implementation provides pass-through behavior. Providers can
    override this for custom tool format transformation.

    Args:
        tools: List of tools in provider-specific format, can be None
        provider_name: Name of the source provider

    Returns:
        List of tools in OpenAI format
    """
    if not tools:
        return []

    logger.debug(f"Transforming {len(tools)} tools from provider: {provider_name}")

    # Filter out invalid tools and pass through valid ones
    transformed = []
    for tool in tools:
        if validate_tool_schema(tool):
            transformed.append(tool.copy())

    return transformed


def create_tool_call_response(
    tool_call_id: str,
    function_name: str,
    function_result: Any | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Create a tool call response message.

    Creates a properly formatted response message for tool/function calls
    following OpenAI's tool response format.

    Args:
        tool_call_id: Unique identifier for the tool call
        function_name: Name of the function that was called
        function_result: Result returned by the function (optional)
        error: Error message if function execution failed (optional)

    Returns:
        Tool response message dictionary
    """
    if error:
        content = f"Error executing {function_name}: {error}"
    else:
        try:
            content = (
                json.dumps(function_result) if function_result is not None else "null"
            )
        except (TypeError, ValueError):
            # Fallback to string representation if not JSON serializable
            content = str(function_result) if function_result is not None else "null"

    return {"tool_call_id": tool_call_id, "role": "tool", "content": content}


def extract_tool_calls_from_response(response: Any) -> list[dict[str, Any]]:
    """Extract tool calls from a provider response.

    Parses provider response to extract any tool/function calls that
    the model wants to make. Follows OpenAI response format.

    Args:
        response: Provider response dictionary

    Returns:
        List of tool call dictionaries, empty list if none found
    """
    if not isinstance(response, dict):
        return []

    choices = response.get("choices", [])
    if not choices:
        return []

    # Get first choice (standard behavior)
    choice = choices[0]
    if not isinstance(choice, dict):
        return []

    message = choice.get("message", {})
    if not isinstance(message, dict):
        return []

    tool_calls = message.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        return []

    return tool_calls


# Environment Detection Utilities for Cross-Platform Test Reliability
@dataclass
class EnvironmentInfo:
    """Comprehensive environment information for test reliability."""

    platform: str = ""
    is_ci: bool = False
    is_github_actions: bool = False
    is_parallel_execution: bool = False
    python_version: str = ""
    is_windows: bool = False
    is_macos: bool = False
    is_linux: bool = False
    is_docker: bool = False
    cpu_count: int = 1
    available_memory_gb: float = 0.0
    is_low_resource: bool = False


def detect_execution_environment() -> EnvironmentInfo:
    """Detect comprehensive execution environment information.

    Returns:
        EnvironmentInfo object with detected environment details
    """
    import os
    import platform
    import sys

    env_info = EnvironmentInfo()

    # Platform detection
    env_info.platform = platform.system()
    env_info.is_windows = env_info.platform == "Windows"
    env_info.is_macos = env_info.platform == "Darwin"
    env_info.is_linux = env_info.platform == "Linux"

    # Python version
    env_info.python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    # CI environment detection
    ci_indicators = [
        "CI",
        "GITHUB_ACTIONS",
        "TRAVIS",
        "CIRCLECI",
        "JENKINS_URL",
        "BUILDKITE",
        "BITBUCKET_BUILD_NUMBER",
        "GITLAB_CI",
        "AZURE_PIPELINES",
    ]
    env_info.is_ci = any(os.getenv(indicator) for indicator in ci_indicators)
    env_info.is_github_actions = bool(os.getenv("GITHUB_ACTIONS"))

    # Parallel execution detection
    env_info.is_parallel_execution = (
        os.getenv("PYTEST_XDIST_WORKER") is not None
        or os.getenv("PYTEST_CURRENT_TEST") is not None
    )

    # Docker detection
    env_info.is_docker = os.path.exists("/.dockerenv") or (
        os.path.exists("/proc/self/cgroup")
        and any(
            "docker" in line
            for line in open("/proc/self/cgroup").readlines()
            if os.path.exists("/proc/self/cgroup")
        )
    )

    # Resource detection
    try:
        import psutil

        env_info.cpu_count = psutil.cpu_count()
        env_info.available_memory_gb = psutil.virtual_memory().total / (1024**3)
        env_info.is_low_resource = (
            env_info.available_memory_gb < 2.0 or env_info.cpu_count < 2
        )
    except ImportError:
        # Fallback without psutil
        env_info.cpu_count = os.cpu_count() or 1
        env_info.is_low_resource = env_info.cpu_count < 2

    return env_info


def get_platform_specific_timeout(
    base_timeout: float, operation_type: str = "default"
) -> float:
    """Get platform and environment-specific timeout values.

    Args:
        base_timeout: Base timeout in seconds
        operation_type: Type of operation ("network", "compute", "io", "default")

    Returns:
        Adjusted timeout in seconds
    """
    env_info = detect_execution_environment()
    timeout = base_timeout

    # CI environment adjustments
    if env_info.is_ci:
        timeout *= 3.0  # CI environments are often resource-constrained

    # Parallel execution adjustments
    if env_info.is_parallel_execution:
        timeout *= 2.0  # Resource contention

    # Platform-specific adjustments
    if env_info.is_windows:
        timeout *= 1.5  # Windows can be slower for some operations
    elif env_info.is_docker:
        timeout *= 1.3  # Docker overhead

    # Operation-specific adjustments
    operation_multipliers = {
        "network": 2.0,  # Network operations can be slower
        "compute": 1.0,  # CPU operations are typically consistent
        "io": 1.5,  # I/O operations can vary
        "default": 1.0,
    }
    timeout *= operation_multipliers.get(operation_type, 1.0)

    # Low resource environment adjustments
    if env_info.is_low_resource:
        timeout *= 2.0

    return timeout


def get_cross_platform_temp_dir() -> str:
    """Get platform-appropriate temporary directory.

    Returns:
        Path to temporary directory
    """
    import tempfile

    return tempfile.gettempdir()


def is_asyncio_event_loop_safe() -> bool:
    """Check if asyncio event loop operations are safe in current environment.

    Returns:
        True if asyncio operations are safe, False otherwise
    """
    try:
        import asyncio

        # Check if we're already in an event loop
        try:
            asyncio.get_running_loop()
            return False  # Already in a loop, creating new ones can be problematic
        except RuntimeError:
            return True  # No running loop, safe to create new ones
    except ImportError:
        return False


def get_environment_diagnostics() -> dict[str, Any]:
    """Get comprehensive environment diagnostics for debugging.

    Returns:
        Dictionary containing environment diagnostics
    """
    import os
    import platform
    import sys

    env_info = detect_execution_environment()

    diagnostics = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "version_info": {
                "major": sys.version_info.major,
                "minor": sys.version_info.minor,
                "micro": sys.version_info.micro,
            },
        },
        "environment": {
            "is_ci": env_info.is_ci,
            "is_github_actions": env_info.is_github_actions,
            "is_parallel_execution": env_info.is_parallel_execution,
            "is_docker": env_info.is_docker,
            "is_low_resource": env_info.is_low_resource,
        },
        "resources": {
            "cpu_count": env_info.cpu_count,
            "available_memory_gb": env_info.available_memory_gb,
        },
        "ci_variables": {
            key: value
            for key, value in os.environ.items()
            if any(
                indicator in key.upper()
                for indicator in [
                    "CI",
                    "GITHUB",
                    "TRAVIS",
                    "CIRCLE",
                    "JENKINS",
                    "BUILD",
                ]
            )
        },
    }

    return diagnostics
