# this_file: src/uutel/core/utils.py
"""Core utilities supporting HTTP access, tool schemas, and message handling."""

from __future__ import annotations

# Standard library imports
import asyncio
import copy
import json
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from litellm.types.utils import GenericStreamingChunk

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
_DEFAULT_TIMEOUT_SECONDS = 60.0


def _default_retry_exceptions() -> tuple[type[BaseException], ...]:
    """Return default retryable exceptions.

    The defaults include generic connection issues and httpx transport
    exceptions when available.
    """

    exceptions: list[type[BaseException]] = [TimeoutError, ConnectionError]
    try:
        import httpx

        exceptions.extend([httpx.TimeoutException, httpx.TransportError])
    except Exception:
        # httpx may not be available at import time during certain test setups
        pass

    return tuple(exceptions)


@dataclass(slots=True)
class RetryConfig:
    """Configuration for HTTP retry behaviour."""

    max_retries: int = 3
    backoff_factor: float = 2.0
    retry_on_status: Sequence[int] = field(
        default_factory=lambda: (408, 409, 425, 429, 500, 502, 503, 504)
    )
    retry_on_exceptions: Sequence[type[BaseException]] = field(
        default_factory=_default_retry_exceptions
    )

    def __post_init__(self) -> None:
        """Normalise configuration values after initialisation."""

        self.max_retries = max(0, int(self.max_retries))
        self.backoff_factor = max(0.0, float(self.backoff_factor))

        # Ensure statuses are stored as a mutable list of ints for easier introspection
        self.retry_on_status = [int(status) for status in self.retry_on_status]

        filtered: list[type[BaseException]] = []
        for exc in self.retry_on_exceptions:
            if isinstance(exc, type) and issubclass(exc, BaseException):
                filtered.append(exc)
        self.retry_on_exceptions = filtered

    def should_retry_status(self, status: int) -> bool:
        """Return True when the HTTP status code is retryable."""

        return status in self.retry_on_status

    def should_retry_exception(self, error: BaseException) -> bool:
        """Return True when the exception type is retryable."""

        return any(isinstance(error, exc) for exc in self.retry_on_exceptions)

    def get_backoff_seconds(self, attempt_number: int) -> float:
        """Compute delay before the next retry attempt.

        Args:
            attempt_number: 1-based index of the upcoming retry attempt.
        """

        if self.backoff_factor <= 0 or attempt_number <= 0:
            return 0.0
        return self.backoff_factor * (2 ** (attempt_number - 1))


class _SyncRetryClient:
    """Synchronous HTTP client wrapper adding basic retry behaviour."""

    def __init__(self, client, config: RetryConfig) -> None:
        self._client = client
        self._config = config

    def request(self, method: str, url: str, **kwargs) -> Any:
        """Execute a request with retry semantics."""

        last_error: BaseException | None = None

        for attempt in range(self._config.max_retries + 1):
            try:
                response = self._client.request(method, url, **kwargs)
            except BaseException as exc:  # pragma: no cover - defensive
                last_error = exc
                if (
                    not self._config.should_retry_exception(exc)
                    or attempt == self._config.max_retries
                ):
                    raise

                delay = self._config.get_backoff_seconds(attempt + 1)
                if delay:
                    time.sleep(delay)
                continue

            if (
                self._config.should_retry_status(response.status_code)
                and attempt < self._config.max_retries
            ):
                delay = self._config.get_backoff_seconds(attempt + 1)
                if delay:
                    time.sleep(delay)
                continue

            return response

        if last_error is not None:  # pragma: no cover - defensive
            raise last_error

        raise UUTELError("HTTP request failed without response", provider="http")

    def get(self, url: str, **kwargs) -> Any:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> Any:
        return self.request("POST", url, **kwargs)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> _SyncRetryClient:
        self._client.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._client.__exit__(exc_type, exc, tb)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._client, item)


class _AsyncRetryClient:
    """Asynchronous HTTP client wrapper adding basic retry behaviour."""

    def __init__(self, client, config: RetryConfig) -> None:
        self._client = client
        self._config = config

    async def request(self, method: str, url: str, **kwargs) -> Any:
        """Execute an async request with retry semantics."""

        last_error: BaseException | None = None

        for attempt in range(self._config.max_retries + 1):
            try:
                response = await self._client.request(method, url, **kwargs)
            except BaseException as exc:  # pragma: no cover - defensive
                last_error = exc
                if (
                    not self._config.should_retry_exception(exc)
                    or attempt == self._config.max_retries
                ):
                    raise

                delay = self._config.get_backoff_seconds(attempt + 1)
                if delay:
                    await asyncio.sleep(delay)
                continue

            if (
                self._config.should_retry_status(response.status_code)
                and attempt < self._config.max_retries
            ):
                delay = self._config.get_backoff_seconds(attempt + 1)
                if delay:
                    await asyncio.sleep(delay)
                continue

            return response

        if last_error is not None:  # pragma: no cover - defensive
            raise last_error

        raise UUTELError("Async HTTP request failed without response", provider="http")

    async def get(self, url: str, **kwargs) -> Any:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> Any:
        return await self.request("POST", url, **kwargs)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> _AsyncRetryClient:
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._client.__aexit__(exc_type, exc, tb)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._client, item)


def create_http_client(
    *,
    async_client: bool = False,
    timeout: float | None = None,
    retry_config: RetryConfig | None = None,
):
    """Create an HTTP client with optional retry support.

    Args:
        async_client: When True, return an asynchronous client.
        timeout: Per-request timeout in seconds. Defaults to 60 seconds.
        retry_config: Optional retry configuration. Defaults to :class:`RetryConfig`.
    """

    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - environment guard
        raise UUTELError(
            "httpx is required to create HTTP clients", provider="http"
        ) from exc

    timeout_value = timeout if timeout is not None else _DEFAULT_TIMEOUT_SECONDS
    timeout_config = httpx.Timeout(timeout_value)

    retry = retry_config or RetryConfig()

    if async_client:
        client = httpx.AsyncClient(timeout=timeout_config)
        if retry.max_retries == 0 or (
            not retry.retry_on_status and not retry.retry_on_exceptions
        ):
            return client
        return _AsyncRetryClient(client, retry)

    client = httpx.Client(timeout=timeout_config)
    if retry.max_retries == 0 or (
        not retry.retry_on_status and not retry.retry_on_exceptions
    ):
        return client
    return _SyncRetryClient(client, retry)


def validate_tool_schema(tool: Any) -> bool:
    """Validate an OpenAI-style tool schema."""

    return _normalise_tool_schema(tool) is not None


def transform_openai_tools_to_provider(
    tools: Iterable[Any] | None, provider_name: str
) -> list[dict[str, Any]]:
    """Convert OpenAI tool definitions into provider friendly structures."""

    if not tools:
        return []

    transformed: list[dict[str, Any]] = []
    for tool in tools:
        normalised = _normalise_tool_schema(tool)
        if normalised is None:
            logger.debug("Skipping invalid tool schema for provider %s", provider_name)
            continue
        transformed.append(normalised)
    return transformed


def transform_provider_tools_to_openai(
    tools: Iterable[Any] | None, provider_name: str
) -> list[dict[str, Any]]:
    """Convert provider tool schemas back into OpenAI format."""

    if not tools:
        return []

    transformed: list[dict[str, Any]] = []
    for tool in tools:
        normalised = _normalise_tool_schema(tool)
        if normalised is None:
            logger.debug("Skipping invalid provider tool for %s", provider_name)
            continue
        transformed.append(normalised)
    return transformed


def extract_tool_calls_from_response(response: Any) -> list[dict[str, Any]]:
    """Extract tool call payloads from an OpenAI response shape."""

    if not isinstance(response, dict):
        return []

    choices = response.get("choices")
    if not isinstance(choices, list):
        return []

    extracted: list[dict[str, Any]] = []

    for choice in choices:
        if not isinstance(choice, dict):
            continue

        message = choice.get("message")
        if not isinstance(message, dict):
            continue

        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue

        for raw_call in tool_calls:
            if not isinstance(raw_call, dict):
                continue

            call_type = raw_call.get("type", "function")
            function_payload = raw_call.get("function")

            if call_type != "function" or not isinstance(function_payload, dict):
                continue

            function_name = function_payload.get("name")
            if not isinstance(function_name, str) or not function_name.strip():
                continue

            arguments = function_payload.get("arguments")
            parsed_arguments = _parse_tool_arguments(arguments)

            extracted.append(
                {
                    "id": raw_call.get("id"),
                    "type": "function",
                    "function": {
                        "name": function_name.strip(),
                        "arguments": parsed_arguments,
                    },
                }
            )

    return extracted


def _normalise_tool_schema(tool: Any) -> dict[str, Any] | None:
    """Return a cleaned tool schema or None when invalid."""

    if not isinstance(tool, dict):
        return None

    tool_type = tool.get("type")
    if tool_type != "function":
        return None

    function_payload = tool.get("function")
    if not isinstance(function_payload, dict):
        return None

    name = function_payload.get("name")
    description = function_payload.get("description")

    if not isinstance(name, str) or not name.strip():
        return None

    if not isinstance(description, str) or not description.strip():
        return None

    normalised: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": name.strip(),
            "description": description.strip(),
        },
    }

    parameters = function_payload.get("parameters", None)
    if parameters is not None:
        if not isinstance(parameters, dict):
            return None

        parameter_type = parameters.get("type")
        if parameter_type != "object":
            return None

        properties = parameters.get("properties")
        if properties is not None and not isinstance(properties, dict):
            return None

        required_fields = parameters.get("required")
        if required_fields is not None and not isinstance(
            required_fields, list | tuple
        ):
            return None

        normalised["function"]["parameters"] = copy.deepcopy(parameters)

    if "strict" in function_payload and isinstance(function_payload["strict"], bool):
        normalised["function"]["strict"] = function_payload["strict"]

    return normalised


def _parse_tool_arguments(arguments: Any) -> Any:
    """Attempt to decode tool arguments from JSON when possible."""

    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments

    if isinstance(arguments, dict | list | tuple) or arguments is None:
        return arguments

    return arguments


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
            if len(parts) == 2:
                prefix, suffix = parts
                if prefix.startswith("uutel-") or prefix == "my-custom-llm":
                    result = bool(suffix) and bool(_MODEL_NAME_PATTERN.match(suffix))
                else:
                    result = False
            elif len(parts) == 3:
                root, provider, remainder = parts
                if root == "uutel":
                    result = all(
                        part and _MODEL_NAME_PATTERN.match(part)
                        for part in (provider, remainder)
                    )
                else:
                    result = False
            else:
                result = False
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


def merge_usage_stats(
    existing: dict[str, int] | None,
    delta: dict[str, int] | None,
) -> dict[str, int] | None:
    """Merge usage statistics while normalising token totals."""

    if not existing and not delta:
        return None

    merged: dict[str, int] = {}
    for source in (existing, delta):
        if not source:
            continue
        for key, value in source.items():
            if isinstance(value, int | float):
                merged[key] = merged.get(key, 0) + int(value)

    merged.setdefault(
        "total_tokens",
        merged.get("prompt_tokens", 0) + merged.get("completion_tokens", 0),
    )
    return merged


def create_text_chunk(
    text: str,
    *,
    index: int = 0,
    finished: bool = False,
    usage: dict[str, int] | None = None,
    finish_reason: str | None = None,
) -> GenericStreamingChunk:
    """Create a GenericStreamingChunk carrying assistant text."""

    chunk = GenericStreamingChunk()
    chunk["index"] = index
    chunk["text"] = text
    resolved_finish = (
        finish_reason if finish_reason is not None else ("stop" if finished else None)
    )
    chunk["finish_reason"] = resolved_finish
    chunk["is_finished"] = bool(resolved_finish)
    chunk["tool_use"] = None
    chunk["usage"] = merge_usage_stats(None, usage)
    return chunk


def create_tool_chunk(
    *,
    name: str,
    arguments: str,
    tool_call_id: str | None = None,
    index: int = 0,
    finished: bool = False,
    finish_reason: str | None = None,
) -> GenericStreamingChunk:
    """Create a GenericStreamingChunk describing a tool invocation."""

    chunk = GenericStreamingChunk()
    chunk["index"] = index
    chunk["text"] = ""
    resolved_finish = (
        finish_reason if finish_reason is not None else ("stop" if finished else None)
    )
    chunk["finish_reason"] = resolved_finish
    chunk["is_finished"] = bool(resolved_finish)
    chunk["tool_use"] = {
        "name": name,
        "arguments": arguments,
    }
    if tool_call_id:
        chunk["tool_use"]["id"] = tool_call_id
    chunk["usage"] = None
    return chunk
