# this_file: src/uutel/providers/claude_code/provider.py
"""Claude Code provider implementation for UUTEL.

This module implements the ClaudeCodeUU provider class for integrating with
Anthropic's Claude Code CLI, which provides access to Claude models via subprocess.
"""

from __future__ import annotations

# Standard library imports
import json
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


class ClaudeCodeUU(BaseUU):
    """Claude Code provider for UUTEL.

    Implements integration with Anthropic's Claude Code CLI for local
    Claude model access via subprocess execution.
    """

    def __init__(self) -> None:
        """Initialize Claude Code provider."""
        super().__init__()
        self.provider_name = "claude_code"
        self.supported_models = [
            "claude-sonnet-4",
            "claude-opus-4",
            "sonnet",
            "opus",
        ]

    def _check_claude_code_cli(self) -> bool:
        """Check if claude-code CLI is installed and accessible.

        Returns:
            True if CLI is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["claude-code", "--version"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _call_claude_code_cli(
        self,
        messages: list,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> dict:
        """Call Claude Code CLI to generate completion.

        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Response dictionary with content and metadata

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
            "claude-code",
            "ask",
            "--model",
            model,
            "--temperature",
            str(temperature),
            "--max-tokens",
            str(max_tokens),
            "--json",  # Request JSON output
            prompt,
        ]

        try:
            # Execute CLI
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120.0,  # 2 minute timeout
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                raise UUTELError(
                    f"Claude Code CLI failed: {error_msg}", provider="claude_code"
                )

            # Parse JSON response
            try:
                response_data = json.loads(result.stdout)
                return response_data
            except json.JSONDecodeError as e:
                # Fallback: treat stdout as plain text response
                logger.warning(f"Failed to parse JSON response: {e}")
                return {
                    "content": result.stdout,
                    "model": model,
                    "finish_reason": "stop",
                }

        except subprocess.TimeoutExpired:
            raise UUTELError(
                "Claude Code CLI timeout (120s exceeded)", provider="claude_code"
            )
        except FileNotFoundError:
            raise UUTELError(
                "Claude Code CLI not found. Please install: npm install -g @anthropic-ai/claude-code",
                provider="claude_code",
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
        """Synchronous completion for Claude Code provider.

        Integrates with Claude Code CLI via subprocess execution.

        Args:
            model: Model name to use (sonnet, opus, etc.)
            messages: Conversation messages
            api_base: API base URL (unused for CLI)
            custom_prompt_dict: Custom prompt formatting
            model_response: Response object to populate
            print_verbose: Verbose printing function
            encoding: Text encoding
            api_key: API key (unused for CLI)
            logging_obj: Logging object
            optional_params: Additional parameters
            acompletion: Async completion function
            litellm_params: LiteLLM parameters
            logger_fn: Custom logger function
            headers: HTTP headers (unused for CLI)
            timeout: Request timeout
            client: HTTP client instance (unused for CLI)

        Returns:
            Populated ModelResponse object

        Raises:
            UUTELError: If completion fails
        """
        try:
            logger.debug(f"Claude Code completion request for model: {model}")

            # Basic validation
            if not model:
                raise UUTELError("Model name is required", provider="claude_code")

            if not messages:
                raise UUTELError("Messages are required", provider="claude_code")

            # Check CLI availability
            if not self._check_claude_code_cli():
                raise UUTELError(
                    "Claude Code CLI not available. Install with: npm install -g @anthropic-ai/claude-code",
                    provider="claude_code",
                )

            # Call CLI
            response_data = self._call_claude_code_cli(
                messages=messages,
                model=model,
                temperature=optional_params.get("temperature", 0.7),
                max_tokens=optional_params.get("max_tokens", 1000),
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

            logger.debug("Claude Code completion completed successfully")
            return model_response

        except Exception as e:
            logger.error(f"Claude Code completion failed: {e}")
            if isinstance(e, UUTELError):
                raise
            raise UUTELError(
                f"Claude Code completion failed: {e}", provider="claude_code"
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
        """Asynchronous completion for Claude Code provider.

        Falls back to sync implementation as Claude Code CLI is synchronous.

        Args:
            model: Model name to use
            messages: Conversation messages
            api_base: API base URL (unused)
            custom_prompt_dict: Custom prompt formatting
            model_response: Response object to populate
            print_verbose: Verbose printing function
            encoding: Text encoding
            api_key: API key (unused)
            logging_obj: Logging object
            optional_params: Additional parameters
            acompletion: Async completion function
            litellm_params: LiteLLM parameters
            logger_fn: Custom logger function
            headers: HTTP headers (unused)
            timeout: Request timeout
            client: Async HTTP client instance (unused)

        Returns:
            Populated ModelResponse object
        """
        # Claude Code CLI is synchronous, use sync implementation
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
        """Synchronous streaming for Claude Code provider.

        Note: Claude Code CLI doesn't support true streaming, so we return
        the complete response as a single chunk.

        Args:
            model: Model name to use
            messages: Conversation messages
            api_base: API base URL (unused)
            custom_prompt_dict: Custom prompt formatting
            model_response: Response object to populate
            print_verbose: Verbose printing function
            encoding: Text encoding
            api_key: API key (unused)
            logging_obj: Logging object
            optional_params: Additional parameters
            acompletion: Async completion function
            litellm_params: LiteLLM parameters
            logger_fn: Custom logger function
            headers: HTTP headers (unused)
            timeout: Request timeout
            client: HTTP client instance (unused)

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

        # Simulate streaming by yielding complete response as single chunk
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
        """Asynchronous streaming for Claude Code provider.

        Args:
            model: Model name to use
            messages: Conversation messages
            api_base: API base URL (unused)
            custom_prompt_dict: Custom prompt formatting
            model_response: Response object to populate
            print_verbose: Verbose printing function
            encoding: Text encoding
            api_key: API key (unused)
            logging_obj: Logging object
            optional_params: Additional parameters
            acompletion: Async completion function
            litellm_params: LiteLLM parameters
            logger_fn: Custom logger function
            headers: HTTP headers (unused)
            timeout: Request timeout
            client: Async HTTP client instance (unused)

        Yields:
            GenericStreamingChunk objects
        """
        # Use sync streaming implementation
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
