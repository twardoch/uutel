#!/usr/bin/env python3
# this_file: examples/basic_usage.py
"""Basic usage example for UUTEL (Universal AI Provider for LiteLLM).

This example demonstrates how to use UUTEL's core functionality:
- Creating BaseUU provider instances
- Using authentication framework
- Message transformation utilities
- Error handling with custom exceptions
- HTTP client creation and configuration

Run this example to see UUTEL's core functionality in action.
"""

from __future__ import annotations

import asyncio
from typing import Any

from uutel import (
    AuthenticationError,
    BaseAuth,
    BaseUU,
    UUTELError,
    create_http_client,
    extract_provider_from_model,
    format_error_message,
    transform_openai_to_provider,
    transform_provider_to_openai,
    validate_model_name,
)


class ExampleAuth(BaseAuth):
    """Example authentication class for demonstration."""

    def __init__(self, api_key: str = "demo-key") -> None:
        """Initialize example auth with demo credentials."""
        super().__init__()
        self.provider_name = "example"
        self.auth_type = "api-key"
        self.api_key = api_key

    def authenticate(self, **kwargs: Any) -> dict[str, Any]:
        """Simulate authentication process."""
        print(f"🔑 Authenticating with {self.provider_name} using {self.auth_type}")

        if not self.api_key or self.api_key == "invalid":
            raise AuthenticationError(
                "Invalid API key", provider=self.provider_name, error_code="AUTH_001"
            )

        return {
            "success": True,
            "token": f"token-{self.api_key}",
            "expires_at": None,
            "error": None,
        }

    def get_headers(self) -> dict[str, str]:
        """Get authentication headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def refresh_token(self) -> dict[str, Any]:
        """Simulate token refresh."""
        print("🔄 Refreshing authentication token")
        return self.authenticate()

    def is_valid(self) -> bool:
        """Check if authentication is valid."""
        return self.api_key != "invalid"


class ExampleProvider(BaseUU):
    """Example provider class for demonstration."""

    def __init__(self, api_key: str = "demo-key") -> None:
        """Initialize example provider."""
        super().__init__()
        self.provider_name = "example"
        self.supported_models = ["example-model-1.0", "example-model-2.0"]
        self.auth = ExampleAuth(api_key)

    def completion(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        """Simulate completion request (not actually implemented in base)."""
        # This would normally call the actual provider API
        print(f"🤖 Making completion request to {model}")
        print(f"📝 Messages: {len(messages)} messages")

        # Transform messages to provider format (for demonstration)
        transform_openai_to_provider(messages, self.provider_name)

        # Simulate API response
        response = {
            "id": "example-123",
            "model": model,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"Hello! This is a simulated response from {model}.",
                    },
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
        }

        return response


def demonstrate_core_functionality():
    """Demonstrate UUTEL's core functionality."""
    print("🚀 UUTEL Basic Usage Example")
    print("=" * 50)

    # 1. Model name validation
    print("\n1️⃣ Model Name Validation")
    model_names = [
        "example-model-1.0",
        "uutel/claude-code/claude-3-5-sonnet",
        "invalid/model",
        "model with spaces",
    ]

    for model in model_names:
        is_valid = validate_model_name(model)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"   {model}: {status}")

    # 2. Provider and model extraction
    print("\n2️⃣ Provider/Model Extraction")
    full_models = [
        "uutel/claude-code/claude-3-5-sonnet",
        "uutel/gemini-cli/gemini-2.0-flash",
        "simple-model",
    ]

    for full_model in full_models:
        provider, model = extract_provider_from_model(full_model)
        print(f"   {full_model} → Provider: {provider}, Model: {model}")

    # 3. Message transformation
    print("\n3️⃣ Message Transformation")
    sample_messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello, world!"},
    ]

    print(f"   Original messages: {len(sample_messages)}")

    # Transform to provider format and back
    provider_format = transform_openai_to_provider(sample_messages, "example")
    back_to_openai = transform_provider_to_openai(provider_format, "example")

    print(f"   After round-trip: {len(back_to_openai)}")
    print(f"   Content preserved: {sample_messages == back_to_openai}")

    # 4. HTTP client creation
    print("\n4️⃣ HTTP Client Creation")

    sync_client = create_http_client(async_client=False, timeout=10.0)
    async_client = create_http_client(async_client=True, timeout=10.0)

    print(f"   Sync client type: {type(sync_client).__name__}")
    print(f"   Async client type: {type(async_client).__name__}")

    # 5. Authentication demonstration
    print("\n5️⃣ Authentication Framework")

    try:
        # Valid authentication
        auth = ExampleAuth("valid-key")
        result = auth.authenticate()
        print(f"   ✅ Auth successful: {result['success']}")

        headers = auth.get_headers()
        print(f"   🔑 Headers: {len(headers)} headers")

    except UUTELError as e:
        print(f"   ❌ Auth failed: {e}")

    # 6. Provider demonstration
    print("\n6️⃣ Provider Usage")

    try:
        provider = ExampleProvider("demo-key")
        print(f"   📡 Provider: {provider.provider_name}")
        print(f"   🎯 Supported models: {len(provider.supported_models)}")

        # Simulate completion
        response = provider.completion(
            model="example-model-1.0", messages=sample_messages
        )
        print(f"   💬 Response ID: {response['id']}")
        print(f"   📊 Token usage: {response['usage']['total_tokens']}")

    except UUTELError as e:
        error_msg = format_error_message(e, "example")
        print(f"   ❌ Provider error: {error_msg}")

    # 7. Error handling demonstration
    print("\n7️⃣ Error Handling")

    try:
        # This will raise an authentication error
        bad_auth = ExampleAuth("invalid")
        bad_auth.authenticate()

    except AuthenticationError as e:
        print(f"   🚫 Caught AuthenticationError: {e}")
        print(f"   📍 Provider: {e.provider}")
        print(f"   🔢 Error code: {e.error_code}")

    except UUTELError as e:
        print(f"   ❌ Caught UUTELError: {e}")

    print("\n✨ Example completed successfully!")


async def demonstrate_async_functionality():
    """Demonstrate async functionality (placeholder)."""
    print("\n🔄 Async Functionality Demo")

    # Create async HTTP client
    async_client = create_http_client(async_client=True)
    print(f"   Created async client: {type(async_client).__name__}")

    # Simulate async operation
    await asyncio.sleep(0.1)
    print("   ✅ Async operation completed")


def main():
    """Main example function."""
    try:
        # Run synchronous examples
        demonstrate_core_functionality()

        # Run async examples
        asyncio.run(demonstrate_async_functionality())

    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
