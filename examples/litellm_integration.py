#!/usr/bin/env python3
# this_file: examples/litellm_integration.py
"""LiteLLM integration example for UUTEL.

This example demonstrates how to use UUTEL providers with LiteLLM:
- Registering UUTEL providers with LiteLLM
- Using providers via LiteLLM's standard completion() function
- Streaming responses
- Model routing with uutel/ prefixes
- Error handling and provider fallbacks

This is the primary use case for UUTEL - extending LiteLLM's provider ecosystem.
"""

from __future__ import annotations

import asyncio
import sys

# Import LiteLLM for the integration
try:
    import litellm
except ImportError:
    print("❌ LiteLLM not installed. Run: pip install litellm")
    sys.exit(1)

# Import UUTEL components
from uutel.providers.codex.custom_llm import CodexCustomLLM


def setup_litellm_providers():
    """Register UUTEL providers with LiteLLM."""
    print("🔧 Setting up LiteLLM with UUTEL providers...")

    # Enable debug mode to see what's happening (commented out for cleaner output)
    # litellm._turn_on_debug()

    # Register CodexCustomLLM provider
    # This allows model format: my-custom-llm/model-name
    litellm.custom_provider_map = [
        {"provider": "my-custom-llm", "custom_handler": CodexCustomLLM()},
    ]

    print("✅ UUTEL providers registered with LiteLLM")
    print(
        f"   - Registered providers: {[p['provider'] for p in litellm.custom_provider_map]}"
    )
    print("   - Model format: my-custom-llm/model-name")

    # Test registration immediately
    print("🧪 Testing registration immediately...")
    try:
        test_response = litellm.completion(
            model="my-custom-llm/test-model",
            messages=[{"role": "user", "content": "Test"}],
        )
        print(
            f"   ✅ Immediate test successful: {test_response.choices[0].message.content[:50]}..."
        )
    except Exception as e:
        print(f"   ❌ Immediate test failed: {e}")


def demonstrate_basic_completion():
    """Demonstrate basic completion using UUTEL provider via LiteLLM."""
    print("\n1️⃣ Basic Completion via LiteLLM")
    print("-" * 40)

    try:
        # Use UUTEL provider through LiteLLM's standard interface
        response = litellm.completion(
            model="my-custom-llm/codex-large",  # UUTEL model format: my-custom-llm/model
            messages=[
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "Explain what a Python decorator is."},
            ],
            max_tokens=150,
            temperature=0.7,
        )

        print("✅ Completion successful!")
        print(f"   Model used: {response.model}")
        print(f"   Response preview: {response.choices[0].message.content[:100]}...")
        print(f"   Finish reason: {response.choices[0].finish_reason}")

    except Exception as e:
        print(f"❌ Completion failed: {e}")


def demonstrate_streaming():
    """Demonstrate streaming responses via LiteLLM."""
    print("\n2️⃣ Streaming Completion via LiteLLM")
    print("-" * 40)

    try:
        # Stream response using UUTEL provider
        response = litellm.completion(
            model="my-custom-llm/codex-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Count from 1 to 10 with brief explanations.",
                }
            ],
            stream=True,
            max_tokens=200,
        )

        print("📡 Streaming response:")
        print("   ", end="", flush=True)

        full_content = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_content += content
                print(content, end="", flush=True)

        print(f"\n✅ Streaming completed! Total length: {len(full_content)} chars")

    except Exception as e:
        print(f"❌ Streaming failed: {e}")


async def demonstrate_async_completion():
    """Demonstrate async completion via LiteLLM."""
    print("\n3️⃣ Async Completion via LiteLLM")
    print("-" * 40)

    try:
        # Async completion using UUTEL provider
        response = await litellm.acompletion(
            model="my-custom-llm/codex-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "What's the difference between sync and async programming?",
                }
            ],
            max_tokens=100,
        )

        print("✅ Async completion successful!")
        print(f"   Model: {response.model}")
        print(f"   Response: {response.choices[0].message.content[:100]}...")

    except Exception as e:
        print(f"❌ Async completion failed: {e}")


async def demonstrate_async_streaming():
    """Demonstrate async streaming via LiteLLM."""
    print("\n4️⃣ Async Streaming via LiteLLM")
    print("-" * 40)

    try:
        response = await litellm.acompletion(
            model="my-custom-llm/codex-fast",
            messages=[
                {"role": "user", "content": "List 5 benefits of async programming."}
            ],
            stream=True,
            max_tokens=150,
        )

        print("📡 Async streaming response:")
        print("   ", end="", flush=True)

        chunk_count = 0
        async for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                chunk_count += 1

        print(f"\n✅ Async streaming completed! Processed {chunk_count} chunks")

    except Exception as e:
        print(f"❌ Async streaming failed: {e}")


def demonstrate_model_routing():
    """Demonstrate model routing with different UUTEL providers."""
    print("\n5️⃣ Model Routing")
    print("-" * 40)

    # Different model formats supported by UUTEL
    models_to_test = [
        "my-custom-llm/codex-large",
        "my-custom-llm/codex-mini",
        "my-custom-llm/codex-preview",
        # Note: Other providers not yet implemented
        # "claude-code/claude-3-5-sonnet",
        # "gemini-cli/gemini-2.0-flash",
    ]

    for model in models_to_test:
        try:
            print(f"   Testing {model}...")
            litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "Hello!"}],
                max_tokens=10,
            )
            print(f"   ✅ {model} → Response received")

        except Exception as e:
            print(f"   ❌ {model} → Failed: {e}")


def demonstrate_error_handling():
    """Demonstrate error handling with UUTEL providers."""
    print("\n6️⃣ Error Handling")
    print("-" * 40)

    # Test with invalid model
    try:
        print("   Testing with invalid model...")
        litellm.completion(
            model="nonexistent/invalid-model",
            messages=[{"role": "user", "content": "Test"}],
        )

    except Exception as e:
        print(f"   ✅ Properly caught error: {type(e).__name__}")
        print(f"      Error message: {str(e)[:100]}...")

    # Test with empty messages
    try:
        print("   Testing with empty messages...")
        litellm.completion(
            model="my-custom-llm/codex-large",
            messages=[],  # Empty messages should fail
        )

    except Exception as e:
        print(f"   ✅ Properly caught error: {type(e).__name__}")
        print(f"      Error message: {str(e)[:100]}...")


def demonstrate_provider_capabilities():
    """Show capabilities and limitations of current UUTEL providers."""
    print("\n7️⃣ Provider Capabilities")
    print("-" * 40)

    # Get CodexCustomLLM instance to show its capabilities
    codex_provider = CodexCustomLLM()

    print(f"   📡 Provider: {codex_provider.provider_name}")
    print(f"   🎯 Supported models: {len(codex_provider.supported_models)}")
    for model in codex_provider.supported_models:
        print(f"      - {model}")

    print("   ⚙️  Capabilities:")
    print("      - Completion: ✅")
    print("      - Async completion: ✅")
    print("      - Streaming: ✅")
    print("      - Async streaming: ✅")
    print("      - Tool calling: ⏳ (planned)")

    print("   i  Note: This is a demo implementation")
    print("      Real integration would connect to actual Codex API")


async def main():
    """Main example function demonstrating UUTEL + LiteLLM integration."""
    print("🚀 UUTEL + LiteLLM Integration Example")
    print("=" * 50)
    print("This example shows how to use UUTEL providers with LiteLLM")
    print("for unified AI model access across multiple providers.\n")

    try:
        # Setup
        setup_litellm_providers()

        # Sync examples
        demonstrate_basic_completion()
        demonstrate_streaming()
        demonstrate_model_routing()
        demonstrate_error_handling()
        demonstrate_provider_capabilities()

        # Async examples
        await demonstrate_async_completion()
        await demonstrate_async_streaming()

        print("\n✨ Integration example completed successfully!")
        print("\n💡 Key takeaways:")
        print("   • UUTEL extends LiteLLM with custom providers")
        print("   • Use model format: provider/model-name")
        print("   • All LiteLLM features work: streaming, async, etc.")
        print("   • Providers handle auth, transformation, and API calls")
        print("   • Unified interface across different AI providers")

    except KeyboardInterrupt:
        print("\n⏹️  Example interrupted by user")

    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
