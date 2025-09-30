# this_file: src/uutel/providers/codex/custom_llm.py
"""LiteLLM CustomLLM implementation for CodexUU provider.

This module provides a thin adapter that bridges the gap between LiteLLM's
CustomLLM interface and UUTEL's provider implementations.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

# Standard library imports
from typing import Any

# Third-party imports
import litellm
from litellm import CustomLLM

# Local imports
from uutel.core.logging_config import get_logger

logger = get_logger(__name__)


class CodexCustomLLM(CustomLLM):
    """LiteLLM CustomLLM adapter for Codex provider.

    This class implements the simple CustomLLM interface expected by LiteLLM
    and provides a bridge to UUTEL's more comprehensive provider implementation.
    """

    def __init__(self) -> None:
        """Initialize Codex CustomLLM adapter."""
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

    def completion(self, *args: Any, **kwargs: Any) -> litellm.ModelResponse:
        """Handle completion requests through LiteLLM's CustomLLM interface.

        Args:
            *args: Positional arguments from LiteLLM
            **kwargs: Keyword arguments from LiteLLM including:
                     model, messages, model_response, optional_params, etc.

        Returns:
            litellm.ModelResponse: Standardized response object
        """
        try:
            # Extract parameters from kwargs (LiteLLM passes these)
            model = kwargs.get("model", "")
            messages = kwargs.get("messages", [])
            model_response = kwargs.get("model_response", litellm.ModelResponse())
            kwargs.get("optional_params", {})

            logger.debug(f"Codex CustomLLM completion request for model: {model}")
            logger.debug(f"Available kwargs: {list(kwargs.keys())}")

            # Basic validation
            if not model:
                raise litellm.BadRequestError("Model name is required")

            if not messages:
                raise litellm.BadRequestError("Messages are required")

            # Use the model_response passed by LiteLLM and populate it
            model_response.model = model

            # Create choices if not already present
            if not hasattr(model_response, "choices") or not model_response.choices:
                choice = litellm.utils.Choices()
                choice.message = litellm.utils.Message()
                model_response.choices = [choice]
            else:
                choice = model_response.choices[0]
                if not hasattr(choice, "message") or not choice.message:
                    choice.message = litellm.utils.Message()

            # Set response content
            choice.message.content = (
                f"This is a mock response from Codex provider for model {model}. "
                f"Received {len(messages)} messages. "
                "In a real implementation, this would call the actual Codex API."
            )
            choice.message.role = "assistant"
            choice.finish_reason = "stop"
            choice.index = 0

            # Add usage information if not present
            if not hasattr(model_response, "usage") or not model_response.usage:
                model_response.usage = litellm.utils.Usage()

            model_response.usage.prompt_tokens = (
                sum(len(str(msg.get("content", ""))) for msg in messages) // 4
            )
            model_response.usage.completion_tokens = len(choice.message.content) // 4
            model_response.usage.total_tokens = (
                model_response.usage.prompt_tokens
                + model_response.usage.completion_tokens
            )

            logger.debug("Codex CustomLLM completion completed successfully")
            return model_response

        except Exception as e:
            logger.error(f"Codex CustomLLM completion failed: {e}")
            if isinstance(e, litellm.BadRequestError | litellm.APIConnectionError):
                raise e from None
            raise litellm.APIConnectionError(f"Codex completion failed: {e}") from e

    async def acompletion(self, *args: Any, **kwargs: Any) -> litellm.ModelResponse:
        """Handle async completion requests.

        Args:
            *args: Positional arguments from LiteLLM
            **kwargs: Keyword arguments including model, messages, etc.

        Returns:
            litellm.ModelResponse: Standardized response object
        """
        # For proof of concept, use sync implementation
        # In real implementation, this would be properly async
        return self.completion(*args, **kwargs)

    def streaming(self, *args: Any, **kwargs: Any) -> Iterator[dict[str, Any]]:
        """Handle streaming completion requests.

        Args:
            *args: Positional arguments from LiteLLM
            **kwargs: Keyword arguments including model, messages, etc.

        Yields:
            dict: GenericStreamingChunk objects
        """
        try:
            model = kwargs.get("model", "")

            # Create mock streaming response
            mock_response = f"Streaming response from Codex {model}"
            words = mock_response.split()

            for i, word in enumerate(words):
                # Create GenericStreamingChunk format
                chunk = {
                    "finish_reason": "stop" if i == len(words) - 1 else None,
                    "index": 0,
                    "is_finished": i == len(words) - 1,
                    "text": word + " ",
                    "tool_use": None,
                    "usage": {
                        "completion_tokens": 1,
                        "prompt_tokens": 0,
                        "total_tokens": 1,
                    },
                }

                yield chunk

        except Exception as e:
            logger.error(f"Codex CustomLLM streaming failed: {e}")
            raise litellm.APIConnectionError(f"Codex streaming failed: {e}") from e

    async def astreaming(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        """Handle async streaming completion requests.

        Args:
            *args: Positional arguments from LiteLLM
            **kwargs: Keyword arguments including model, messages, etc.

        Yields:
            dict: GenericStreamingChunk objects
        """
        # For proof of concept, use sync implementation
        # In real implementation, this would be properly async
        for chunk in self.streaming(*args, **kwargs):
            yield chunk
