# this_file: src/uutel/core/base.py
"""UUTEL base classes and interfaces.

This module provides the BaseUU class that serves as the foundation for all
UUTEL provider implementations. All provider classes follow the naming
convention {ProviderName}UU.

Example usage:
    Creating a custom provider:
        from uutel.core.base import BaseUU

        class MyProviderUU(BaseUU):
            def __init__(self):
                super().__init__()
                self.provider_name = "my-provider"
                self.supported_models = ["my-model-1.0", "my-model-2.0"]

            def completion(self, model, messages, **kwargs):
                # Implement your provider's completion logic
                return {"choices": [{"message": {"role": "assistant", "content": "Response"}}]}

            def streaming(self, model, messages, **kwargs):
                # Implement streaming logic
                for chunk in self._generate_chunks():
                    yield chunk

    Using the provider:
        provider = MyProviderUU()
        result = provider.completion("my-model-1.0", [{"role": "user", "content": "Hello"}])

    Provider registration with LiteLLM:
        import litellm
        from my_provider import MyProviderUU

        # Register your provider
        litellm.custom_provider_map["my-provider"] = MyProviderUU
"""

from __future__ import annotations

# Standard library imports
from collections.abc import AsyncIterator, Callable, Iterator

# Third-party imports
from litellm import CustomLLM  # type: ignore[attr-defined]
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import GenericStreamingChunk, ModelResponse
from litellm.utils import CustomStreamWrapper  # type: ignore[attr-defined]


class BaseUU(CustomLLM):
    """Base class for all UUTEL provider implementations.

    This class extends LiteLLM's CustomLLM and provides the foundation for
    implementing custom AI providers. All UUTEL providers should inherit from
    this class and implement the required methods.

    Attributes:
        provider_name: Name of the provider (e.g., "claude-code", "gemini-cli")
        supported_models: List of model names supported by this provider
    """

    def __init__(self) -> None:
        """Initialize BaseUU instance."""
        super().__init__()
        self.provider_name: str = "base"
        self.supported_models: list[str] = []

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
    ) -> ModelResponse | CustomStreamWrapper:
        """Synchronous completion method.

        This method should be implemented by each provider to handle
        synchronous completion requests.

        Args:
            model: The model name to use for completion
            messages: List of messages in the conversation
            api_base: Base URL for the API
            custom_prompt_dict: Custom prompt formatting dictionary
            model_response: The response object to populate
            print_verbose: Verbose printing function
            encoding: Text encoding to use
            api_key: API key for authentication
            logging_obj: Logging object for tracking requests
            optional_params: Additional parameters for the request
            acompletion: Async completion function (if applicable)
            litellm_params: LiteLLM-specific parameters
            logger_fn: Custom logging function
            headers: HTTP headers to include
            timeout: Request timeout in seconds
            client: HTTP client instance

        Returns:
            ModelResponse or CustomStreamWrapper for streaming responses

        Raises:
            NotImplementedError: This base method must be overridden
        """
        raise NotImplementedError(
            f"completion method must be implemented by {self.__class__.__name__}"
        )

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
    ) -> ModelResponse | CustomStreamWrapper:
        """Asynchronous completion method.

        This method should be implemented by each provider to handle
        asynchronous completion requests.

        Args:
            model: The model name to use for completion
            messages: List of messages in the conversation
            api_base: Base URL for the API
            custom_prompt_dict: Custom prompt formatting dictionary
            model_response: The response object to populate
            print_verbose: Verbose printing function
            encoding: Text encoding to use
            api_key: API key for authentication
            logging_obj: Logging object for tracking requests
            optional_params: Additional parameters for the request
            acompletion: Async completion function (if applicable)
            litellm_params: LiteLLM-specific parameters
            logger_fn: Custom logging function
            headers: HTTP headers to include
            timeout: Request timeout in seconds
            client: Async HTTP client instance

        Returns:
            ModelResponse or CustomStreamWrapper for streaming responses

        Raises:
            NotImplementedError: This base method must be overridden
        """
        raise NotImplementedError(
            f"acompletion method must be implemented by {self.__class__.__name__}"
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
        """Synchronous streaming method.

        This method should be implemented by each provider to handle
        synchronous streaming responses.

        Args:
            model: The model name to use for completion
            messages: List of messages in the conversation
            api_base: Base URL for the API
            custom_prompt_dict: Custom prompt formatting dictionary
            model_response: The response object to populate
            print_verbose: Verbose printing function
            encoding: Text encoding to use
            api_key: API key for authentication
            logging_obj: Logging object for tracking requests
            optional_params: Additional parameters for the request
            acompletion: Async completion function (if applicable)
            litellm_params: LiteLLM-specific parameters
            logger_fn: Custom logging function
            headers: HTTP headers to include
            timeout: Request timeout in seconds
            client: HTTP client instance

        Yields:
            GenericStreamingChunk: Streaming response chunks

        Raises:
            NotImplementedError: This base method must be overridden
        """
        raise NotImplementedError(
            f"streaming method must be implemented by {self.__class__.__name__}"
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
        """Asynchronous streaming method.

        This method should be implemented by each provider to handle
        asynchronous streaming responses.

        Args:
            model: The model name to use for completion
            messages: List of messages in the conversation
            api_base: Base URL for the API
            custom_prompt_dict: Custom prompt formatting dictionary
            model_response: The response object to populate
            print_verbose: Verbose printing function
            encoding: Text encoding to use
            api_key: API key for authentication
            logging_obj: Logging object for tracking requests
            optional_params: Additional parameters for the request
            acompletion: Async completion function (if applicable)
            litellm_params: LiteLLM-specific parameters
            logger_fn: Custom logging function
            headers: HTTP headers to include
            timeout: Request timeout in seconds
            client: Async HTTP client instance

        Yields:
            GenericStreamingChunk: Streaming response chunks

        Raises:
            NotImplementedError: This base method must be overridden
        """
        raise NotImplementedError(
            f"astreaming method must be implemented by {self.__class__.__name__}"
        )
