# this_file: src/uutel/providers/cloud_code/provider.py
"""Google Cloud Code AI provider implementation for UUTEL.

This module implements the CloudCodeUU provider class for integrating with
Google's Cloud Code AI via OAuth authentication and Code Assist API.
"""

from __future__ import annotations

# Standard library imports
import json
import os
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path

# Third-party imports
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import GenericStreamingChunk, ModelResponse

# Local imports
from uutel.core.base import BaseUU
from uutel.core.exceptions import UUTELError
from uutel.core.logging_config import get_logger

logger = get_logger(__name__)


class CloudCodeUU(BaseUU):
    """Google Cloud Code AI provider for UUTEL.

    Implements integration with Google's Cloud Code AI using OAuth authentication.
    Reads credentials from ~/.gemini/oauth_creds.json (from Gemini CLI authentication).
    """

    def __init__(self) -> None:
        """Initialize Cloud Code provider."""
        super().__init__()
        self.provider_name = "cloud_code"
        self.supported_models = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-pro",
            "gemini-flash",
        ]

    def _load_oauth_credentials(self) -> str:
        """Load OAuth access token from Gemini CLI credentials.

        Returns:
            OAuth access token

        Raises:
            UUTELError: If credentials not found or invalid
        """
        # Try multiple possible credential locations
        possible_paths = [
            Path.home() / ".gemini" / "oauth_creds.json",
            Path.home() / ".config" / "gemini" / "oauth_creds.json",
            Path.home() / ".google-cloud-code" / "credentials.json",
        ]

        for creds_path in possible_paths:
            if creds_path.exists():
                try:
                    with open(creds_path) as f:
                        creds = json.load(f)

                    access_token = creds.get("access_token")
                    if access_token:
                        return access_token

                    # Alternative structure
                    tokens = creds.get("tokens", {})
                    access_token = tokens.get("access_token")
                    if access_token:
                        return access_token

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in {creds_path}: {e}")
                    continue

        raise UUTELError(
            "No OAuth credentials found. Please authenticate with: "
            "gemini login (Gemini CLI) or use GOOGLE_API_KEY environment variable",
            provider="cloud_code",
        )

    def _get_api_key(self) -> str | None:
        """Get API key from environment.

        Returns:
            API key if available, None otherwise
        """
        return (
            os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_GENAI_API_KEY")
        )

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
        """Synchronous completion for Cloud Code provider.

        Uses OAuth authentication or API key.

        Args:
            model: Model name to use (gemini-2.5-flash, etc.)
            messages: Conversation messages
            api_base: API base URL (defaults to generativelanguage.googleapis.com)
            custom_prompt_dict: Custom prompt formatting
            model_response: Response object to populate
            print_verbose: Verbose printing function
            encoding: Text encoding
            api_key: API key (or from env, or OAuth)
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

            logger.debug(f"Cloud Code completion request for model: {model}")

            # Basic validation
            if not model:
                raise UUTELError("Model name is required", provider="cloud_code")

            if not messages:
                raise UUTELError("Messages are required", provider="cloud_code")

            # Determine authentication method
            actual_api_key = api_key or self._get_api_key()
            use_oauth = not actual_api_key

            if use_oauth:
                # Try OAuth authentication
                try:
                    access_token = self._load_oauth_credentials()
                    auth_header = f"Bearer {access_token}"
                    logger.debug("Using OAuth authentication")
                except UUTELError:
                    raise UUTELError(
                        "No authentication available. Set GOOGLE_API_KEY or run: gemini login",
                        provider="cloud_code",
                    )
            else:
                # Use API key
                auth_header = None
                logger.debug("Using API key authentication")

            # Build contents from messages
            contents = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                # Map roles to Gemini format
                if role == "system":
                    # Prepend to first user message
                    if contents:
                        contents[0]["parts"][0]["text"] = (
                            f"{content}\n\n{contents[0]['parts'][0]['text']}"
                        )
                    else:
                        contents.append({"role": "user", "parts": [{"text": content}]})
                elif role == "user":
                    contents.append({"role": "user", "parts": [{"text": content}]})
                elif role == "assistant" or role == "model":
                    contents.append({"role": "model", "parts": [{"text": content}]})

            # Build request body
            request_body = {
                "contents": contents,
                "generationConfig": {
                    "temperature": optional_params.get("temperature", 0.7),
                    "maxOutputTokens": optional_params.get("max_tokens", 1000),
                },
            }

            # Determine endpoint
            endpoint = api_base or "https://generativelanguage.googleapis.com"
            url = f"{endpoint}/v1beta/models/{model}:generateContent"

            # Build headers
            request_headers = {"Content-Type": "application/json"}

            if use_oauth:
                request_headers["Authorization"] = auth_header
            else:
                request_headers["x-goog-api-key"] = actual_api_key

            if headers:
                request_headers.update(headers)

            # Make API request
            with httpx.Client(timeout=timeout or 120.0) as http_client:
                response = http_client.post(
                    url,
                    headers=request_headers,
                    json=request_body,
                )
                response.raise_for_status()

                result = response.json()

                # Extract response
                candidates = result.get("candidates", [])
                if not candidates:
                    raise UUTELError("No candidates in response", provider="cloud_code")

                candidate = candidates[0]
                content_parts = candidate.get("content", {}).get("parts", [])

                # Combine text parts
                text = " ".join(part.get("text", "") for part in content_parts)

                finish_reason = candidate.get("finishReason", "STOP").lower()

                # Populate model response
                model_response.model = model
                model_response.choices[0].message.content = text
                model_response.choices[0].finish_reason = finish_reason

                # Add usage information if available
                usage_metadata = result.get("usageMetadata", {})
                if usage_metadata:
                    model_response.usage = {
                        "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
                        "completion_tokens": usage_metadata.get(
                            "candidatesTokenCount", 0
                        ),
                        "total_tokens": usage_metadata.get("totalTokenCount", 0),
                    }

                logger.debug("Cloud Code completion completed successfully")
                return model_response

        except httpx.HTTPStatusError as e:
            error_detail = e.response.text if hasattr(e.response, "text") else str(e)
            logger.error(
                f"Cloud Code HTTP error: {e.response.status_code} - {error_detail}"
            )
            raise UUTELError(
                f"Cloud Code API error: {e.response.status_code} - {error_detail}",
                provider="cloud_code",
            ) from e
        except Exception as e:
            logger.error(f"Cloud Code completion failed: {e}")
            if isinstance(e, UUTELError):
                raise
            raise UUTELError(
                f"Cloud Code completion failed: {e}", provider="cloud_code"
            ) from e

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
        """Asynchronous completion for Cloud Code provider.

        Falls back to sync implementation.

        Args:
            model: Model name to use
            messages: Conversation messages
            api_base: API base URL
            custom_prompt_dict: Custom prompt formatting
            model_response: Response object to populate
            print_verbose: Verbose printing function
            encoding: Text encoding
            api_key: API key
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
        """Synchronous streaming for Cloud Code provider.

        Returns complete response as single chunk.

        Args:
            model: Model name to use
            messages: Conversation messages
            api_base: API base URL
            custom_prompt_dict: Custom prompt formatting
            model_response: Response object to populate
            print_verbose: Verbose printing function
            encoding: Text encoding
            api_key: API key
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
        # Get complete response
        response = self.completion(
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
            client=client,
        )

        # Yield as single chunk
        content = response.choices[0].message.content
        chunk = GenericStreamingChunk()
        chunk["finish_reason"] = "stop"
        chunk["index"] = 0
        chunk["is_finished"] = True
        chunk["text"] = content
        chunk["tool_use"] = None
        chunk["usage"] = getattr(response, "usage", {})

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
        """Asynchronous streaming for Cloud Code provider.

        Args:
            model: Model name to use
            messages: Conversation messages
            api_base: API base URL
            custom_prompt_dict: Custom prompt formatting
            model_response: Response object to populate
            print_verbose: Verbose printing function
            encoding: Text encoding
            api_key: API key
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
