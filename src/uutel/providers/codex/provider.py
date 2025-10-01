# this_file: src/uutel/providers/codex/provider.py
"""Codex provider implementation for UUTEL.

This module implements the CodexUU provider class for integrating with
OpenAI Codex via session token management and ChatGPT backend.
"""

from __future__ import annotations

# Standard library imports
import json
import math
import uuid
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Callable,
    Iterable,
    Iterator,
    Mapping,
)
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

# Third-party imports
import httpx
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import GenericStreamingChunk, ModelResponse

# Local imports
from uutel.core.base import BaseUU
from uutel.core.exceptions import UUTELError
from uutel.core.logging_config import get_logger
from uutel.core.utils import create_http_client, create_text_chunk, create_tool_chunk

logger = get_logger(__name__)


class CodexUU(BaseUU):
    """Codex provider for UUTEL.

    Implements integration with OpenAI Codex via session token management
    and ChatGPT backend integration. Falls back to OpenAI API key if available.
    """

    def __init__(self) -> None:
        """Initialize Codex provider."""
        super().__init__()
        self.provider_name = "codex"
        self.supported_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "o1-preview",
            "o1-mini",
        ]

    def _load_codex_auth(self) -> tuple[str, str]:
        """Load Codex authentication from ~/.codex/auth.json.

        Returns:
            Tuple of (access_token, account_id)

        Raises:
            UUTELError: If auth file not found or invalid
        """
        import json
        from pathlib import Path

        auth_path = Path.home() / ".codex" / "auth.json"

        if not auth_path.exists():
            raise UUTELError(
                f"Codex auth file not found at {auth_path}. "
                "Please run 'codex login' first to authenticate.",
                provider="codex",
            )

        try:
            with open(auth_path, encoding="utf-8") as handle:
                auth_data = json.load(handle)

            if not isinstance(auth_data, dict):
                raise UUTELError(
                    "Unexpected Codex auth payload; rerun 'codex login' to refresh credentials.",
                    provider="codex",
                )

            tokens_section = auth_data.get("tokens")
            if not isinstance(tokens_section, dict):
                tokens_section = {}

            access_token = tokens_section.get("access_token") or tokens_section.get(
                "session_token"
            )
            if not access_token:
                access_token = auth_data.get("access_token") or auth_data.get(
                    "session_token"
                )

            if not access_token:
                raise UUTELError(
                    "No access token found in Codex auth.json. Run 'codex login' or set OPENAI_API_KEY.",
                    provider="codex",
                )

            account_candidates: list[str | None] = [
                tokens_section.get("account_id"),
                tokens_section.get("workspace_id"),
                tokens_section.get("team_id"),
                auth_data.get("account_id"),
                auth_data.get("workspace_id"),
                auth_data.get("team_id"),
            ]

            workspace = auth_data.get("workspace")
            if isinstance(workspace, dict):
                account_candidates.append(workspace.get("id"))
            user_info = auth_data.get("user")
            if isinstance(user_info, dict):
                account_candidates.append(user_info.get("team_id"))

            account_id = next((value for value in account_candidates if value), None)

            if not account_id:
                raise UUTELError(
                    "No account or workspace id found in Codex auth.json. Re-run 'codex login' to refresh tokens.",
                    provider="codex",
                )

            return str(access_token), str(account_id)

        except json.JSONDecodeError as exc:
            raise UUTELError(
                f"Invalid JSON in auth file: {exc}",
                provider="codex",
            ) from exc

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict | None = None,
        timeout: float | None = None,
        client: HTTPHandler | None = None,
    ) -> ModelResponse:
        """Synchronous completion for Codex provider.

        Integrates with ChatGPT Codex API using authentication from ~/.codex/auth.json
        Falls back to OpenAI API if api_key is provided.

        Args:
            model: Model name to use (gpt-4o, gpt-5, etc.)
            messages: Conversation messages in OpenAI format
            api_base: API base URL (default: https://chatgpt.com/backend-api)
            custom_prompt_dict: Custom prompt formatting
            model_response: Response object to populate
            print_verbose: Verbose printing function
            encoding: Text encoding
            api_key: API key for OpenAI fallback
            logging_obj: Logging object
            optional_params: Additional parameters
            acompletion: Async completion function
            litellm_params: LiteLLM parameters
            logger_fn: Custom logger function
            headers: HTTP headers
            timeout: Request timeout
            client: HTTP client instance

        Returns:
            Populated ModelResponse object

        Raises:
            UUTELError: If completion fails
        """
        try:
            logger.debug(f"Codex completion request for model: {model}")

            # Basic validation
            if not model:
                raise UUTELError("Model name is required", provider="codex")

            if not messages:
                raise UUTELError("Messages are required", provider="codex")

            # Determine if using Codex auth or OpenAI API key
            use_codex_auth = not api_key

            if use_codex_auth:
                # Load Codex authentication
                access_token, account_id = self._load_codex_auth()

                # Use Codex backend API
                endpoint = api_base or "https://chatgpt.com/backend-api"
                url = f"{endpoint}/codex/responses"

                # Build Codex-specific headers
                request_headers = {
                    "Authorization": f"Bearer {access_token}",
                    "chatgpt-account-id": account_id,
                    "Content-Type": "application/json",
                    "version": "0.28.0",
                    "openai-beta": "responses=experimental",
                    "originator": "codex_cli_rs",
                    "user-agent": "codex_cli_rs/0.28.0",
                    "accept": "text/event-stream",
                }

                # Convert messages to Codex format (input instead of messages)
                request_body = self._build_codex_payload(
                    model=model,
                    messages=messages,
                    optional_params=optional_params,
                )
            else:
                # Fall back to OpenAI API
                endpoint = api_base or "https://api.openai.com/v1"
                url = f"{endpoint}/chat/completions"

                request_headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }

                request_body = self._build_openai_payload(
                    model=model,
                    messages=messages,
                    optional_params=optional_params,
                )

            # Merge with custom headers
            if headers:
                request_headers.update(headers)

            http_client = client
            close_client = False
            if http_client is None:
                http_client = create_http_client(async_client=False, timeout=timeout)
                close_client = True

            try:
                response = http_client.post(
                    url,
                    headers=request_headers,
                    json=request_body,
                )
                response.raise_for_status()

                result = response.json()
            except httpx.HTTPStatusError as http_error:
                self._handle_http_status_error(http_error)
            finally:
                if close_client and hasattr(http_client, "close"):
                    try:
                        http_client.close()
                    except Exception as close_error:  # pragma: no cover - defensive
                        logger.debug(
                            f"Failed to close Codex HTTP client: {close_error}"
                        )

            self._apply_completion_result(
                result=result,
                model=model,
                model_response=model_response,
            )

            logger.debug("Codex completion completed successfully")
            return model_response

        except httpx.HTTPStatusError as http_error:
            self._handle_http_status_error(http_error)
        except Exception as e:
            logger.error(f"Codex completion failed: {e}")
            if isinstance(e, UUTELError):
                raise
            raise UUTELError(f"Codex completion failed: {e}", provider="codex") from e

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict | None = None,
        timeout: float | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> ModelResponse:
        """Asynchronous completion for Codex provider.

        Args:
            model: Model name to use
            messages: Conversation messages
            api_base: API base URL
            custom_prompt_dict: Custom prompt formatting
            model_response: Response object to populate
            print_verbose: Verbose printing function
            encoding: Text encoding
            api_key: API key for authentication
            logging_obj: Logging object
            optional_params: Additional parameters
            acompletion: Async completion function
            litellm_params: LiteLLM parameters
            logger_fn: Custom logger function
            headers: HTTP headers
            timeout: Request timeout
            client: Async HTTP client instance

        Returns:
            Populated ModelResponse object
        """
        try:
            logger.debug(f"Codex async completion request for model: {model}")

            if not model:
                raise UUTELError("Model name is required", provider="codex")

            if not messages:
                raise UUTELError("Messages are required", provider="codex")

            use_codex_auth = not api_key

            if use_codex_auth:
                access_token, account_id = self._load_codex_auth()
                endpoint = api_base or "https://chatgpt.com/backend-api"
                url = f"{endpoint}/codex/responses"
                request_headers = {
                    "Authorization": f"Bearer {access_token}",
                    "chatgpt-account-id": account_id,
                    "Content-Type": "application/json",
                    "version": "0.28.0",
                    "openai-beta": "responses=experimental",
                    "originator": "codex_cli_rs",
                    "user-agent": "codex_cli_rs/0.28.0",
                    "accept": "text/event-stream",
                }

                request_body = self._build_codex_payload(
                    model=model,
                    messages=messages,
                    optional_params=optional_params,
                )
            else:
                endpoint = api_base or "https://api.openai.com/v1"
                url = f"{endpoint}/chat/completions"
                request_headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                request_body = self._build_openai_payload(
                    model=model,
                    messages=messages,
                    optional_params=optional_params,
                )

            if headers:
                request_headers.update(headers)

            http_client = client
            close_client = False
            if http_client is None:
                http_client = create_http_client(async_client=True, timeout=timeout)
                close_client = True

            try:
                response = await http_client.post(
                    url,
                    headers=request_headers,
                    json=request_body,
                )
                if hasattr(response, "raise_for_status"):
                    response.raise_for_status()

                result = response.json()
            except httpx.HTTPStatusError as http_error:
                self._handle_http_status_error(http_error)
            finally:
                if close_client:
                    close_coro = getattr(http_client, "aclose", None)
                    if callable(close_coro):
                        await close_coro()
                    else:  # pragma: no cover - defensive
                        close_fn = getattr(http_client, "close", None)
                        if callable(close_fn):
                            close_fn()

            self._apply_completion_result(
                result=result,
                model=model,
                model_response=model_response,
            )

            logger.debug("Codex async completion completed successfully")
            return model_response

        except (
            httpx.HTTPStatusError
        ) as http_error:  # pragma: no cover - handled via helper
            self._handle_http_status_error(http_error)
        except Exception as e:  # pragma: no cover - parity with sync path
            logger.error(f"Codex async completion failed: {e}")
            if isinstance(e, UUTELError):
                raise
            raise UUTELError(
                f"Codex async completion failed: {e}", provider="codex"
            ) from e

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict | None = None,
        timeout: float | None = None,
        client: HTTPHandler | None = None,
    ) -> Iterator[GenericStreamingChunk]:
        """Synchronous streaming for Codex provider."""

        use_codex_auth = api_key is None
        stream_params = dict(optional_params)
        stream_params["stream"] = True

        if use_codex_auth:
            access_token, account_id = self._load_codex_auth()
            endpoint = api_base or "https://chatgpt.com/backend-api"
            url = f"{endpoint.rstrip('/')}/codex/responses"
            request_headers = {
                "Authorization": f"Bearer {access_token}",
                "chatgpt-account-id": account_id,
                "Content-Type": "application/json",
                "version": "0.28.0",
                "openai-beta": "responses=experimental",
                "originator": "codex_cli_rs",
                "user-agent": "codex_cli_rs/0.28.0",
                "accept": "text/event-stream",
            }
            payload = self._build_codex_payload(
                model=model,
                messages=messages,
                optional_params=stream_params,
            )
        else:
            endpoint = api_base or "https://api.openai.com/v1"
            url = f"{endpoint.rstrip('/')}/chat/completions"
            request_headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "accept": "text/event-stream",
            }
            payload = self._build_openai_payload(
                model=model,
                messages=messages,
                optional_params=stream_params,
            )

        if headers:
            request_headers.update(headers)

        yield from self._stream_sync(
            url=url,
            headers=request_headers,
            payload=payload,
            timeout=timeout,
            client=client,
        )

    async def astreaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict | None = None,
        timeout: float | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        """Asynchronous streaming for Codex provider.

        Args:
            model: Model name to use
            messages: Conversation messages
            api_base: API base URL
            custom_prompt_dict: Custom prompt formatting
            model_response: Response object to populate
            print_verbose: Verbose printing function
            encoding: Text encoding
            api_key: API key for authentication
            logging_obj: Logging object
            optional_params: Additional parameters
            acompletion: Async completion function
            litellm_params: LiteLLM parameters
            logger_fn: Custom logger function
            headers: HTTP headers
            timeout: Request timeout
            client: Async HTTP client instance

        Yields:
            GenericStreamingChunk objects
        """
        use_codex_auth = api_key is None
        stream_params = dict(optional_params)
        stream_params["stream"] = True

        if use_codex_auth:
            access_token, account_id = self._load_codex_auth()
            endpoint = api_base or "https://chatgpt.com/backend-api"
            url = f"{endpoint.rstrip('/')}/codex/responses"
            request_headers = {
                "Authorization": f"Bearer {access_token}",
                "chatgpt-account-id": account_id,
                "Content-Type": "application/json",
                "version": "0.28.0",
                "openai-beta": "responses=experimental",
                "originator": "codex_cli_rs",
                "user-agent": "codex_cli_rs/0.28.0",
                "accept": "text/event-stream",
            }
            payload = self._build_codex_payload(
                model=model,
                messages=messages,
                optional_params=stream_params,
            )
        else:
            endpoint = api_base or "https://api.openai.com/v1"
            url = f"{endpoint.rstrip('/')}/chat/completions"
            request_headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "accept": "text/event-stream",
            }
            payload = self._build_openai_payload(
                model=model,
                messages=messages,
                optional_params=stream_params,
            )

        if headers:
            request_headers.update(headers)

        async for chunk in self._stream_async(
            url=url,
            headers=request_headers,
            payload=payload,
            timeout=timeout,
            client=client,
        ):
            yield chunk

    def _handle_http_status_error(self, error: httpx.HTTPStatusError) -> None:
        """Raise a UUTELError with guidance for specific HTTP failures."""

        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        headers = getattr(response, "headers", None)

        message = self._format_status_guidance(
            status=status,
            headers=headers,
            fallback=str(error),
        )

        raise UUTELError(message, provider="codex") from error

    def _apply_completion_result(
        self,
        *,
        result: Mapping[str, Any],
        model: str,
        model_response: ModelResponse,
    ) -> None:
        choices = result.get("choices", []) if isinstance(result, Mapping) else []
        if not isinstance(choices, list) or not choices:
            raise UUTELError("No choices in Codex response", provider="codex")

        choice = choices[0] if isinstance(choices[0], Mapping) else {}
        message = choice.get("message", {}) if isinstance(choice, Mapping) else {}
        content = self._extract_message_content(message)
        finish_reason = (
            choice.get("finish_reason", "stop")
            if isinstance(choice, Mapping)
            else "stop"
        )

        usage = result.get("usage", {}) if isinstance(result, Mapping) else {}
        tool_calls = self._extract_tool_calls(choice, result)

        model_response.model = model
        model_response.choices[0].message.content = content
        model_response.choices[0].finish_reason = finish_reason

        if tool_calls:
            model_response.choices[0].message.tool_calls = tool_calls
            if not content:
                model_response.choices[0].message.content = ""

        if usage:
            model_response.usage = usage

    def _stream_sync(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float | None,
        client: HTTPHandler | None,
    ) -> Iterator[GenericStreamingChunk]:
        http_client = client or create_http_client(async_client=False, timeout=timeout)
        close_client = client is None

        try:
            stream_kwargs = {"headers": headers, "json": payload}
            if timeout is not None:
                stream_kwargs["timeout"] = timeout

            with http_client.stream("POST", url, **stream_kwargs) as response:
                status = getattr(response, "status_code", 200)
                if status and status >= 400:
                    message = self._format_status_guidance(
                        status=status,
                        headers=getattr(response, "headers", None),
                        fallback=f"status {status}",
                    )
                    raise UUTELError(message, provider="codex")

                state = {"tool_info": {}, "function_args": {}}
                for sse_payload in _iter_sse_json(response.iter_lines()):
                    yield from _convert_stream_payload(sse_payload, state)
        except UUTELError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise UUTELError(
                f"Codex streaming failed: {exc}", provider="codex"
            ) from exc
        finally:
            if close_client and hasattr(http_client, "close"):
                http_client.close()

    async def _stream_async(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float | None,
        client: AsyncHTTPHandler | None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        http_client = client or create_http_client(async_client=True, timeout=timeout)
        close_client = client is None

        try:
            stream_kwargs = {"headers": headers, "json": payload}
            if timeout is not None:
                stream_kwargs["timeout"] = timeout

            async with http_client.stream("POST", url, **stream_kwargs) as response:
                status = getattr(response, "status_code", 200)
                if status and status >= 400:
                    message = self._format_status_guidance(
                        status=status,
                        headers=getattr(response, "headers", None),
                        fallback=f"status {status}",
                    )
                    raise UUTELError(message, provider="codex")

                state = {"tool_info": {}, "function_args": {}}
                async for sse_payload in _aiter_sse_json(response.aiter_lines()):
                    for chunk in _convert_stream_payload(sse_payload, state):
                        yield chunk
        except UUTELError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise UUTELError(
                f"Codex streaming failed: {exc}", provider="codex"
            ) from exc
        finally:
            if close_client:
                if hasattr(http_client, "aclose"):
                    await http_client.aclose()
                elif hasattr(http_client, "close"):
                    http_client.close()

    def _format_status_guidance(
        self,
        *,
        status: int | None,
        headers: Mapping[str, Any] | None,
        fallback: str,
    ) -> str:
        """Translate HTTP status codes into actionable guidance."""

        if status == 401:
            return "Codex credentials rejected (HTTP 401). Run 'codex login' or set OPENAI_API_KEY."

        if status == 403:
            return "Codex request forbidden (HTTP 403). Run 'codex login' to refresh your session token."

        if status == 429:
            retry_after = self._parse_retry_after(headers)
            retry_hint = (
                f" Retry after {retry_after}s." if retry_after is not None else ""
            )
            return (
                "Codex rate limit reached (HTTP 429)."
                f"{retry_hint} Reduce request volume or wait before retrying."
            )

        if status is not None and 500 <= status < 600:
            return f"Codex service unavailable (HTTP {status}). Try again in a few seconds."

        if status is not None:
            return f"Codex request failed with HTTP {status}: {fallback}"

        return f"Codex request failed: {fallback}"

    def _parse_retry_after(self, headers: Mapping[str, Any] | None) -> int | None:
        """Extract Retry-After header as seconds when available."""

        if not headers:
            return None

        try:
            header_dict = {
                str(key).lower(): str(value) for key, value in headers.items()
            }
        except AttributeError:  # pragma: no cover - defensive
            header_dict = {}

        retry_after = header_dict.get("retry-after")
        if retry_after is None:
            return None

        try:
            candidate = retry_after.strip()
        except AttributeError:
            candidate = str(retry_after)

        if not candidate:
            return None

        try:
            if "." in candidate:
                return int(float(candidate))
            return int(candidate)
        except ValueError:
            try:
                parsed = parsedate_to_datetime(candidate)
            except (TypeError, ValueError, OverflowError):
                return None

            if parsed is None:
                return None

            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            delta_seconds = (parsed - now).total_seconds()
            if delta_seconds <= 0:
                return 0
            return math.ceil(delta_seconds)

    def _build_openai_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        optional_params: dict,
    ) -> dict[str, Any]:
        """Create Chat Completions payload for OpenAI-compatible endpoints."""

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": bool(optional_params.get("stream", False)),
        }

        self._apply_sampling_params(payload, optional_params)
        return payload

    def _build_codex_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        optional_params: dict,
    ) -> dict[str, Any]:
        """Create payload specific to the Codex backend."""

        payload: dict[str, Any] = {
            "model": model,
            "input": messages,
            "stream": bool(optional_params.get("stream", False)),
            "store": False,
            "reasoning": {
                "effort": optional_params.get("reasoning_effort", "medium"),
                "summary": optional_params.get("reasoning_summary", "auto"),
            },
            "tool_choice": optional_params.get("tool_choice", "auto"),
            "parallel_tool_calls": bool(
                optional_params.get("parallel_tool_calls", False)
            ),
        }

        self._apply_sampling_params(payload, optional_params)
        return payload

    def _apply_sampling_params(
        self, payload: dict[str, Any], optional_params: dict
    ) -> None:
        """Copy supported sampling parameters into the request payload."""

        if "temperature" in optional_params:
            payload["temperature"] = optional_params["temperature"]
        if "max_tokens" in optional_params:
            payload["max_tokens"] = optional_params["max_tokens"]
        if "top_p" in optional_params:
            payload["top_p"] = optional_params["top_p"]
        if "frequency_penalty" in optional_params:
            payload["frequency_penalty"] = optional_params["frequency_penalty"]
        if "presence_penalty" in optional_params:
            payload["presence_penalty"] = optional_params["presence_penalty"]

    def _extract_message_content(self, message: dict[str, Any]) -> str:
        """Normalise assistant message content from the API response."""

        if not isinstance(message, dict):
            return ""

        content = message.get("content", "")
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)

        return str(content)

    def _extract_tool_calls(
        self,
        choice: Mapping[str, Any] | Any,
        result: Mapping[str, Any] | Any,
    ) -> list[dict[str, Any]]:
        """Extract and normalise tool calls from Codex/OpenAI responses."""

        tool_calls: list[dict[str, Any]] = []
        raw_calls: list[Any] = []

        if isinstance(choice, Mapping):
            message = choice.get("message")
            if isinstance(message, Mapping):
                message_calls = message.get("tool_calls")
                if isinstance(message_calls, list):
                    raw_calls.extend(message_calls)

            choice_calls = choice.get("tool_calls")
            if isinstance(choice_calls, list):
                raw_calls.extend(choice_calls)

        if isinstance(result, Mapping):
            output_items = result.get("output") or result.get("outputs")
            if isinstance(output_items, list):
                for item in output_items:
                    if isinstance(item, Mapping) and item.get("type") in {
                        "function_call",
                        "tool_call",
                    }:
                        raw_calls.append(item)

        for raw_call in raw_calls:
            normalised = _normalise_tool_call(raw_call)
            if normalised is not None:
                tool_calls.append(normalised)

        return tool_calls


def _iter_sse_json(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Convert Server-Sent Event lines into JSON payloads."""

    buffer: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if not buffer:
                continue
            data = "".join(buffer)
            buffer.clear()
            if data == "[DONE]":
                break
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                logger.debug("Failed to decode Codex SSE payload: %s", data)
            continue

        if not line.startswith("data:"):
            continue

        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            break
        buffer.append(payload)

    if buffer:
        data = "".join(buffer)
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            logger.debug("Failed to decode trailing Codex SSE payload: %s", data)


async def _aiter_sse_json(lines: AsyncIterable[str]) -> AsyncIterator[dict[str, Any]]:
    """Async variant of :func:`_iter_sse_json`."""

    buffer: list[str] = []
    async for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if not buffer:
                continue
            data = "".join(buffer)
            buffer.clear()
            if data == "[DONE]":
                break
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                logger.debug("Failed to decode Codex SSE payload: %s", data)
            continue

        if not line.startswith("data:"):
            continue

        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            break
        buffer.append(payload)

    if buffer:
        data = "".join(buffer)
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            logger.debug("Failed to decode trailing Codex SSE payload: %s", data)


def _convert_stream_payload(
    payload: Mapping[str, Any], state: dict[str, Any]
) -> list[GenericStreamingChunk]:
    """Convert a Codex/OpenAI streaming payload into chunks."""

    if not isinstance(payload, Mapping):
        return []

    event_type = payload.get("type")
    if event_type:
        return _handle_codex_event(payload, state)

    if payload.get("choices"):
        return _handle_openai_event(payload)

    return []


def _handle_codex_event(
    payload: Mapping[str, Any], state: dict[str, Any]
) -> list[GenericStreamingChunk]:
    chunks: list[GenericStreamingChunk] = []
    event_type = payload.get("type")

    if event_type == "response.output_text.delta":
        delta = payload.get("delta")
        if isinstance(delta, str) and delta:
            chunks.append(create_text_chunk(delta))
    elif event_type == "response.reasoning_summary_text.delta":
        delta = payload.get("delta")
        if isinstance(delta, str) and delta:
            chunks.append(create_text_chunk(delta))
    elif event_type == "response.output_item.added":
        item = payload.get("item")
        if isinstance(item, Mapping) and item.get("type") == "function_call":
            item_id = item.get("id") or payload.get("item_id")
            if item_id:
                state.setdefault("tool_info", {})[item_id] = {
                    "name": item.get("name", "unknown"),
                    "id": item_id,
                }
    elif event_type == "response.function_call_name.delta":
        item_id = payload.get("item_id")
        delta = payload.get("delta")
        if item_id and isinstance(delta, str):
            info = state.setdefault("tool_info", {}).setdefault(item_id, {})
            info.setdefault("id", item_id)
            existing = info.get("name")
            base = "" if existing in {None, "unknown"} else str(existing)
            info["name"] = (base + delta).strip() or "unknown"
    elif event_type in {
        "response.function_call_arguments.delta",
        "response.tool_call_arguments.delta",
    }:
        item_id = payload.get("item_id")
        if item_id:
            existing = state.setdefault("function_args", {}).get(item_id, "")
            delta = payload.get("delta")
            if isinstance(delta, str):
                state["function_args"][item_id] = existing + delta
    elif event_type in {
        "response.function_call_arguments.done",
        "response.tool_call_arguments.done",
    }:
        item_id = payload.get("item_id")
        tool_info = (
            state.setdefault("tool_info", {}).get(item_id, {}) if item_id else {}
        )
        accumulated = ""
        if item_id:
            accumulated = state.setdefault("function_args", {}).pop(item_id, "")
        arguments = payload.get("arguments")
        if isinstance(arguments, Mapping):
            arg_string = json.dumps(arguments)
        elif isinstance(arguments, str) and arguments:
            arg_string = arguments
        else:
            arg_string = accumulated or ""
        if accumulated and not arguments:
            arg_string = accumulated
        if (
            arg_string
            and not arg_string.startswith("{")
            and not arg_string.endswith("}")
        ):
            arg_string = arg_string.strip()
        if arg_string:
            chunks.append(
                create_tool_chunk(
                    name=tool_info.get("name", payload.get("name", "unknown")),
                    arguments=arg_string,
                    tool_call_id=payload.get("call_id")
                    or tool_info.get("id")
                    or item_id,
                )
            )
    elif event_type == "response.completed":
        usage = _normalise_usage(payload.get("response", {}).get("usage"))
        finish_reason = _map_finish_reason(payload.get("response", {}).get("status"))
        chunks.append(
            create_text_chunk(
                "", finished=True, usage=usage, finish_reason=finish_reason
            )
        )

    return chunks


def _handle_openai_event(payload: Mapping[str, Any]) -> list[GenericStreamingChunk]:
    chunks: list[GenericStreamingChunk] = []

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        delta = choices[0].get("delta")
        if isinstance(delta, Mapping):
            content = delta.get("content")
            if isinstance(content, str) and content:
                chunks.append(create_text_chunk(content))
        finish_reason = choices[0].get("finish_reason")
        if finish_reason:
            usage = _normalise_usage(payload.get("usage"))
            chunks.append(
                create_text_chunk(
                    "",
                    finished=True,
                    usage=usage,
                    finish_reason=_map_finish_reason(finish_reason),
                )
            )

    return chunks


def _normalise_usage(raw_usage: Any) -> dict[str, int] | None:
    if not isinstance(raw_usage, Mapping):
        return None

    usage: dict[str, int] = {}
    if "input_tokens" in raw_usage:
        usage["input_tokens"] = int(raw_usage.get("input_tokens", 0))
    if "output_tokens" in raw_usage:
        usage["output_tokens"] = int(raw_usage.get("output_tokens", 0))
    if "prompt_tokens" in raw_usage:
        usage["input_tokens"] = int(raw_usage.get("prompt_tokens", 0))
    if "completion_tokens" in raw_usage:
        usage["output_tokens"] = int(raw_usage.get("completion_tokens", 0))

    total = raw_usage.get("total_tokens")
    if isinstance(total, int | float):
        usage["total_tokens"] = int(total)
    elif usage:
        usage["total_tokens"] = usage.get("input_tokens", 0) + usage.get(
            "output_tokens", 0
        )

    return usage or None


def _map_finish_reason(reason: Any) -> str:
    mapping = {
        "stop": "stop",
        "length": "length",
        "tool_calls": "tool_calls",
        "content_filter": "content_filter",
    }
    return mapping.get(str(reason).lower() if reason else "", "stop")


def _normalise_tool_call(tool_call: Any) -> dict[str, Any] | None:
    """Normalise tool call payloads to OpenAI-compatible format."""

    if not isinstance(tool_call, Mapping):
        return None

    function_section = tool_call.get("function")
    name = tool_call.get("name") or (
        function_section.get("name") if isinstance(function_section, Mapping) else None
    )

    arguments = tool_call.get("arguments")
    if arguments is None and isinstance(function_section, Mapping):
        arguments = function_section.get("arguments")

    if isinstance(arguments, Mapping):
        arguments_str = json.dumps(arguments)
    elif isinstance(arguments, str):
        arguments_str = arguments
    elif arguments is None:
        arguments_str = ""
    else:
        arguments_str = json.dumps(arguments)

    call_id = (
        tool_call.get("id") or tool_call.get("call_id") or tool_call.get("tool_call_id")
    )
    if not call_id:
        call_id = uuid.uuid4().hex

    name = str(name) if name is not None else "unknown"

    return {
        "id": str(call_id),
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments_str,
        },
    }
