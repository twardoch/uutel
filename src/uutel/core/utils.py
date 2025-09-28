# this_file: src/uutel/core/utils.py
"""UUTEL core utilities and helpers.

This module provides common utilities used across UUTEL providers,
including message transformation, HTTP client creation, retry logic,
and validation functions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger


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
    if not messages:
        return []

    logger.debug(f"Transforming {len(messages)} messages for provider: {provider_name}")

    # Base implementation: pass-through
    # Providers can extend this for custom transformation
    transformed = []
    for message in messages:
        if isinstance(message, dict) and "role" in message and "content" in message:
            transformed.append({"role": message["role"], "content": message["content"]})

    return transformed


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
    if not messages:
        return []

    logger.debug(
        f"Transforming {len(messages)} messages from provider: {provider_name}"
    )

    # Base implementation: pass-through
    # Providers can extend this for custom transformation
    transformed = []
    for message in messages:
        if isinstance(message, dict) and "role" in message and "content" in message:
            transformed.append({"role": message["role"], "content": message["content"]})

    return transformed


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
        logger.debug("Creating async HTTP client")
        return httpx.AsyncClient(timeout=timeout, follow_redirects=True)
    else:
        logger.debug("Creating sync HTTP client")
        return httpx.Client(timeout=timeout, follow_redirects=True)


def validate_model_name(model: Any) -> bool:
    """Validate a model name.

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
    if not isinstance(model, str) or not model:
        return False

    # Check for spaces (always invalid)
    if " " in model:
        return False

    # Allow provider prefixes like "uutel/claude-code/model-name"
    if "/" in model:
        parts = model.split("/")
        # Must have at least 3 parts for valid UUTEL prefix: "uutel/provider/model"
        if len(parts) < 3 or parts[0] != "uutel":
            return False
        # Validate each part
        for part in parts:
            if not part or not re.match(r"^[a-zA-Z0-9._-]+$", part):
                return False
        return True

    # Validate simple model name
    return bool(re.match(r"^[a-zA-Z0-9._-]+$", model))


def extract_provider_from_model(model: str) -> tuple[str, str]:
    """Extract provider and model from a full model string.

    Parses model strings like "uutel/claude-code/claude-3-5-sonnet" to
    extract the provider name and actual model name.

    Args:
        model: Full model string potentially containing provider prefix

    Returns:
        Tuple of (provider_name, model_name)
    """
    if "/" not in model:
        return "unknown", model

    parts = model.split("/")
    if len(parts) < 3 or parts[0] != "uutel":
        return "unknown", model

    provider_name = parts[1]
    model_name = "/".join(parts[2:])  # Handle nested model names

    return provider_name, model_name


def format_error_message(error: Exception, provider: str) -> str:
    """Format error message for consistent error reporting.

    Creates standardized error messages that include provider context
    and relevant error details.

    Args:
        error: The exception that occurred
        provider: Name of the provider where the error occurred

    Returns:
        Formatted error message string
    """
    error_type = type(error).__name__
    error_msg = str(error)

    return f"[{provider}] {error_type}: {error_msg}"


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
    if not isinstance(tool, dict):
        return False

    # Check required top-level fields
    if tool.get("type") != "function":
        return False

    function = tool.get("function")
    if not isinstance(function, dict):
        return False

    # Check required function fields
    if not isinstance(function.get("name"), str) or not function["name"]:
        return False

    if not isinstance(function.get("description"), str) or not function["description"]:
        return False

    # Parameters are optional, but if present must be valid
    parameters = function.get("parameters")
    if parameters is not None:
        if not isinstance(parameters, dict):
            return False
        # Basic JSON schema validation - must have type field
        if parameters.get("type") != "object":
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


def extract_tool_calls_from_response(response: dict[str, Any]) -> list[dict[str, Any]]:
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
