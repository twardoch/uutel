# this_file: src/uutel/providers/codex/provider.py
"""Codex provider implementation for UUTEL.

This module implements the CodexUU provider class for integrating with
OpenAI Codex via session token management and ChatGPT backend.
"""

from __future__ import annotations

# Standard library imports
from collections.abc import AsyncIterator, Callable, Iterator

# Third-party imports
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import GenericStreamingChunk, ModelResponse

# Local imports
from uutel.core.base import BaseUU
from uutel.core.exceptions import UUTELError
from uutel.core.logging_config import get_logger

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
            with open(auth_path) as f:
                auth_data = json.load(f)

            # Extract tokens
            tokens = auth_data.get("tokens", {})
            access_token = tokens.get("access_token")
            account_id = tokens.get("account_id")

            if not access_token:
                raise UUTELError("No access token found in auth.json", provider="codex")

            if not account_id:
                raise UUTELError("No account ID found in auth.json", provider="codex")

            return access_token, account_id

        except json.JSONDecodeError as e:
            raise UUTELError(f"Invalid JSON in auth file: {e}", provider="codex") from e

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
            import httpx

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
                request_body = {
                    "model": model,
                    "input": messages,  # Codex uses 'input' field
                    "stream": False,
                    "store": False,
                    "temperature": optional_params.get("temperature", 0.7),
                    "max_tokens": optional_params.get("max_tokens", 1000),
                    "reasoning": {
                        "effort": "medium",
                        "summary": "auto",
                    },
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                }
            else:
                # Fall back to OpenAI API
                endpoint = api_base or "https://api.openai.com/v1"
                url = f"{endpoint}/chat/completions"

                request_headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }

                request_body = {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "temperature": optional_params.get("temperature", 0.7),
                    "max_tokens": optional_params.get("max_tokens", 1000),
                }

            # Merge with custom headers
            if headers:
                request_headers.update(headers)

            # Make HTTP request
            with httpx.Client(timeout=timeout or 60.0) as http_client:
                response = http_client.post(
                    url,
                    headers=request_headers,
                    json=request_body,
                )
                response.raise_for_status()

                result = response.json()

                # Extract response content
                if use_codex_auth:
                    # Codex format
                    choices = result.get("choices", [])
                    if not choices:
                        raise UUTELError(
                            "No choices in Codex response", provider="codex"
                        )

                    choice = choices[0]
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    finish_reason = choice.get("finish_reason", "stop")

                    # Extract usage
                    usage = result.get("usage", {})

                else:
                    # OpenAI format
                    choices = result.get("choices", [])
                    if not choices:
                        raise UUTELError(
                            "No choices in OpenAI response", provider="codex"
                        )

                    choice = choices[0]
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    finish_reason = choice.get("finish_reason", "stop")

                    usage = result.get("usage", {})

                # Populate model response
                model_response.model = model
                model_response.choices[0].message.content = content
                model_response.choices[0].finish_reason = finish_reason

                # Add usage information
                if usage:
                    model_response.usage = usage

                logger.debug("Codex completion completed successfully")
                return model_response

        except httpx.HTTPStatusError as e:
            logger.error(f"Codex HTTP error: {e}")
            raise UUTELError(
                f"Codex HTTP error: {e.response.status_code}", provider="codex"
            ) from e
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
        # For proof of concept, use sync implementation
        # In real implementation, this would be properly async
        return self.completion(
            model=model,
            messages=messages,
            api_base=api_base,
            custom_prompt_dict=custom_prompt_dict,
            model_response=model_response,
            print_verbose=print_verbose,
            encoding=encoding,
            api_key=api_key,
            logging_obj=logging_obj,
            optional_params=optional_params,
            acompletion=acompletion,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            headers=headers,
            timeout=timeout,
        )

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
        """Synchronous streaming for Codex provider.

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
            client: HTTP client instance

        Yields:
            GenericStreamingChunk objects
        """
        # Basic mock streaming implementation
        mock_response = f"Streaming response from Codex {model}"
        words = mock_response.split()

        for i, word in enumerate(words):
            # Create GenericStreamingChunk format
            chunk = GenericStreamingChunk()
            chunk["finish_reason"] = "stop" if i == len(words) - 1 else None
            chunk["index"] = 0
            chunk["is_finished"] = i == len(words) - 1
            chunk["text"] = word + " "
            chunk["tool_use"] = None
            chunk["usage"] = {
                "completion_tokens": 1,
                "prompt_tokens": 0,
                "total_tokens": 1,
            }

            yield chunk

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
        # Mock async streaming - in real implementation would be properly async
        for chunk in self.streaming(
            model=model,
            messages=messages,
            api_base=api_base,
            custom_prompt_dict=custom_prompt_dict,
            model_response=model_response,
            print_verbose=print_verbose,
            encoding=encoding,
            api_key=api_key,
            logging_obj=logging_obj,
            optional_params=optional_params,
            acompletion=acompletion,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            headers=headers,
            timeout=timeout,
        ):
            yield chunk
