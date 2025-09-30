# this_file: src/uutel/core/utils.py
"""UUTEL core utilities - simplified message transformation only."""

from __future__ import annotations

# Standard library imports
import json
import re
from typing import Any

# Local imports
from .exceptions import UUTELError
from .logging_config import get_logger

logger = get_logger(__name__)

# Pre-compiled regex patterns
_MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")
_INVALID_CHARS_PATTERN = re.compile(r"[\s\n\r\t\0]")

# Performance optimization: Cache for frequently validated models
_MODEL_VALIDATION_CACHE: dict[str, bool] = {}
_PROVIDER_EXTRACTION_CACHE: dict[str, tuple[str, str]] = {}
_CACHE_SIZE_LIMIT = 1000


def transform_openai_to_provider(
    messages: list[dict[str, Any]], provider_name: str
) -> list[dict[str, Any]]:
    """Transform OpenAI format messages to provider-specific format.

    Args:
        messages: List of messages in OpenAI format
        provider_name: Name of the target provider

    Returns:
        List of messages in provider-specific format
    """
    if not messages:
        return []

    logger.debug(f"Transforming {len(messages)} messages for provider: {provider_name}")

    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
        if isinstance(msg, dict) and "role" in msg and "content" in msg
    ]


def transform_provider_to_openai(
    messages: list[dict[str, Any]], provider_name: str
) -> list[dict[str, Any]]:
    """Transform provider-specific messages to OpenAI format.

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

    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
        if isinstance(msg, dict) and "role" in msg and "content" in msg
    ]


def validate_model_name(model: Any) -> bool:
    """Validate a model name.

    Args:
        model: Model name to validate

    Returns:
        True if model name is valid, False otherwise
    """
    if model is None or model == "":
        return False

    if not isinstance(model, str):
        return False

    if model in _MODEL_VALIDATION_CACHE:
        return _MODEL_VALIDATION_CACHE[model]

    try:
        if len(model) > 200:
            result = False
        elif _INVALID_CHARS_PATTERN.search(model):
            result = False
        elif "/" in model:
            parts = model.split("/", 2)
            if len(parts) < 3 or parts[0] != "uutel":
                result = False
            else:
                result = all(part and _MODEL_NAME_PATTERN.match(part) for part in parts)
        else:
            result = bool(_MODEL_NAME_PATTERN.match(model))

        if len(_MODEL_VALIDATION_CACHE) < _CACHE_SIZE_LIMIT:
            _MODEL_VALIDATION_CACHE[model] = result

        return result

    except (AttributeError, TypeError, re.error):
        return False


def extract_provider_from_model(model: str) -> tuple[str, str]:
    """Extract provider and model from a full model string.

    Args:
        model: Full model string potentially containing provider prefix

    Returns:
        Tuple of (provider_name, model_name)
    """
    if not model:
        return "unknown", ""

    if not isinstance(model, str):
        return "unknown", str(model)

    if model in _PROVIDER_EXTRACTION_CACHE:
        return _PROVIDER_EXTRACTION_CACHE[model]

    try:
        if "/" not in model:
            result = ("unknown", model)
        else:
            parts = model.split("/", 2)

            if len(parts) < 3 or parts[0] != "uutel":
                result = ("unknown", model)
            else:
                provider_name = parts[1] if parts[1] else "unknown"
                model_name = parts[2] if parts[2] else ""

                if not provider_name or not model_name:
                    result = ("unknown", model)
                else:
                    result = (provider_name, model_name)

        if len(_PROVIDER_EXTRACTION_CACHE) < _CACHE_SIZE_LIMIT:
            _PROVIDER_EXTRACTION_CACHE[model] = result

        return result

    except (AttributeError, IndexError, TypeError):
        return "unknown", str(model)


def format_error_message(
    error: Exception | None, provider: Any, *, cli_format: bool = False
) -> str:
    """Format error message for consistent error reporting.

    Args:
        error: The exception that occurred
        provider: Name of the provider where the error occurred
        cli_format: If True, use CLI-friendly format with emoji indicators

    Returns:
        Formatted error message string
    """
    try:
        if error is None:
            if cli_format:
                return f"❌ Unknown error occurred in {provider or 'unknown'} provider"
            return f"[{provider or 'unknown'}] Unknown error occurred"

        error_type = (
            type(error).__name__ if hasattr(error, "__class__") else "Exception"
        )

        try:
            error_msg = str(error) if error else "No error message available"
        except Exception:
            error_msg = "Error occurred but message could not be retrieved"

        provider = provider or "unknown"
        if not isinstance(provider, str):
            provider = str(provider)

        if isinstance(error, UUTELError):
            base_msg = str(error)
            if cli_format and not base_msg.startswith("❌"):
                return f"❌ {base_msg}"
            return base_msg

        if not error_msg or error_msg.strip() == "":
            error_msg = f"Empty {error_type} occurred"

        if cli_format:
            return f"❌ Error in {provider}: {error_msg}"
        else:
            return f"[{provider}] {error_type}: {error_msg}"

    except Exception:
        if cli_format:
            return f"❌ Critical error in {provider or 'unknown'} provider"
        return f"[{provider or 'unknown'}] Critical error in error formatting"


def get_error_debug_info(error: Exception | None) -> dict[str, Any]:
    """Get comprehensive debugging information from an error.

    Args:
        error: Exception to get debug info from

    Returns:
        Dictionary with debug information
    """
    try:
        if error is None:
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

        if isinstance(error, UUTELError):
            try:
                return error.get_debug_info()
            except Exception as e:
                logger.warning(f"Failed to get debug info from UUTEL error: {e}")

        error_type = (
            type(error).__name__ if hasattr(error, "__class__") else "Exception"
        )

        try:
            message = str(error) if error else "No error message available"
        except Exception:
            message = "Error message could not be retrieved"

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


def create_tool_call_response(
    tool_call_id: str,
    function_name: str,
    function_result: Any | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Create a tool call response message.

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
            content = str(function_result) if function_result is not None else "null"

    return {"tool_call_id": tool_call_id, "role": "tool", "content": content}
