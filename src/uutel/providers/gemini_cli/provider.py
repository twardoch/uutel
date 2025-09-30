# this_file: src/uutel/providers/gemini_cli/provider.py
"""Gemini CLI provider implementation for UUTEL.

This module implements the GeminiCLIUU provider class for integrating with
Google's Gemini models via the gemini CLI tool or direct API access.
"""

from __future__ import annotations

# Standard library imports
import os
import subprocess
from collections.abc import AsyncIterator, Callable, Iterator

# Third-party imports
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import GenericStreamingChunk, ModelResponse

# Local imports
from uutel.core.base import BaseUU
from uutel.core.exceptions import UUTELError
from uutel.core.logging_config import get_logger

logger = get_logger(__name__)


class GeminiCLIUU(BaseUU):
    """Gemini CLI provider for UUTEL.

    Implements integration with Google Gemini via CLI tool or direct API.
    Supports multiple authentication methods:
    - API Key (GOOGLE_API_KEY env var)
    - OAuth (via gemini CLI authentication)
    - Vertex AI credentials
    """

    def __init__(self) -> None:
        """Initialize Gemini CLI provider."""
        super().__init__()
        self.provider_name = "gemini_cli"
        self.supported_models = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-pro",
            "gemini-flash",
        ]

    def _check_gemini_cli(self) -> bool:
        """Check if gemini CLI is installed and accessible.

        Returns:
            True if CLI is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["gemini", "--version"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _get_api_key(self) -> str | None:
        """Get Gemini API key from environment.

        Returns:
            API key if available, None otherwise
        """
        # Check multiple possible env var names
        return (
            os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_GENAI_API_KEY")
        )

    def _call_via_api(
        self,
        messages: list,
        model: str,
        api_key: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> dict:
        """Call Gemini directly via API using httpx.

        Args:
            messages: Conversation messages
            model: Model name
            api_key: API key
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Response dictionary

        Raises:
            UUTELError: If API call fails
        """
        import httpx

        # Build contents from messages
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Map roles to Gemini format
            if role == "system":
                # Gemini doesn't have system role, prepend to first user message
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
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        # Make API request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    url,
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": api_key,
                    },
                    json=request_body,
                )
                response.raise_for_status()

                result = response.json()

                # Extract response
                candidates = result.get("candidates", [])
                if not candidates:
                    raise UUTELError(
                        "No candidates in Gemini response", provider="gemini_cli"
                    )

                candidate = candidates[0]
                content_parts = candidate.get("content", {}).get("parts", [])

                # Combine text parts
                text = " ".join(part.get("text", "") for part in content_parts)

                finish_reason = candidate.get("finishReason", "STOP").lower()

                return {
                    "content": text,
                    "model": model,
                    "finish_reason": finish_reason,
                    "usage": result.get("usageMetadata", {}),
                }

        except httpx.HTTPStatusError as e:
            error_detail = e.response.text if hasattr(e.response, "text") else str(e)
            raise UUTELError(
                f"Gemini API error: {e.response.status_code} - {error_detail}",
                provider="gemini_cli",
            ) from e

    def _call_via_cli(
        self,
        messages: list,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> dict:
        """Call Gemini via CLI tool.

        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Response dictionary

        Raises:
            UUTELError: If CLI call fails
        """
        # Build prompt from messages
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt = "\n\n".join(prompt_parts)

        # Build CLI command
        cmd = [
            "gemini",
            "ask",
            "--model",
            model,
            "--temperature",
            str(temperature),
            "--max-tokens",
            str(max_tokens),
            prompt,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120.0,
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                raise UUTELError(
                    f"Gemini CLI failed: {error_msg}", provider="gemini_cli"
                )

            # Return stdout as content
            return {
                "content": result.stdout.strip(),
                "model": model,
                "finish_reason": "stop",
            }

        except subprocess.TimeoutExpired:
            raise UUTELError(
                "Gemini CLI timeout (120s exceeded)", provider="gemini_cli"
            )
        except FileNotFoundError:
            raise UUTELError(
                "Gemini CLI not found. Please install: npm install -g @google/gemini-cli",
                provider="gemini_cli",
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
        """Synchronous completion for Gemini CLI provider.

        Tries API key first, falls back to CLI if available.

        Args:
            model: Model name to use (gemini-2.5-flash, etc.)
            messages: Conversation messages
            api_base: API base URL (optional)
            custom_prompt_dict: Custom prompt formatting
            model_response: Response object to populate
            print_verbose: Verbose printing function
            encoding: Text encoding
            api_key: API key (or from env)
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
            logger.debug(f"Gemini CLI completion request for model: {model}")

            # Basic validation
            if not model:
                raise UUTELError("Model name is required", provider="gemini_cli")

            if not messages:
                raise UUTELError("Messages are required", provider="gemini_cli")

            # Try API key approach first
            actual_api_key = api_key or self._get_api_key()

            if actual_api_key:
                logger.debug("Using Gemini API with API key")
                response_data = self._call_via_api(
                    messages=messages,
                    model=model,
                    api_key=actual_api_key,
                    temperature=optional_params.get("temperature", 0.7),
                    max_tokens=optional_params.get("max_tokens", 1000),
                )
            elif self._check_gemini_cli():
                logger.debug("Using Gemini CLI")
                response_data = self._call_via_cli(
                    messages=messages,
                    model=model,
                    temperature=optional_params.get("temperature", 0.7),
                    max_tokens=optional_params.get("max_tokens", 1000),
                )
            else:
                raise UUTELError(
                    "No Gemini authentication available. "
                    "Set GOOGLE_API_KEY environment variable or install Gemini CLI: "
                    "npm install -g @google/gemini-cli",
                    provider="gemini_cli",
                )

            # Extract response content
            content = response_data.get("content", "")
            finish_reason = response_data.get("finish_reason", "stop")

            # Populate model response
            model_response.model = model
            model_response.choices[0].message.content = content
            model_response.choices[0].finish_reason = finish_reason

            # Add usage information if available
            if "usage" in response_data:
                model_response.usage = response_data["usage"]

            logger.debug("Gemini CLI completion completed successfully")
            return model_response

        except Exception as e:
            logger.error(f"Gemini CLI completion failed: {e}")
            if isinstance(e, UUTELError):
                raise
            raise UUTELError(
                f"Gemini CLI completion failed: {e}", provider="gemini_cli"
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
        """Asynchronous completion for Gemini CLI provider.

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
        """Synchronous streaming for Gemini CLI provider.

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
        """Asynchronous streaming for Gemini CLI provider.

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
