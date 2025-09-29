#!/usr/bin/env python3
# this_file: examples/streaming_example.py
"""Streaming responses example for UUTEL (Universal AI Provider for LiteLLM).

This example demonstrates UUTEL's streaming capabilities:
- Simulated streaming response handling
- Async and sync streaming patterns
- Stream chunk processing and aggregation
- Error handling in streaming scenarios
- Real-time response display

The example shows how UUTEL providers would handle streaming
responses while maintaining compatibility with LiteLLM's interface.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, Generator
from typing import Any

from uutel import BaseUU, format_error_message


class StreamingExampleProvider(BaseUU):
    """Example provider that demonstrates streaming capabilities."""

    def __init__(self) -> None:
        """Initialize the streaming example provider."""
        super().__init__()
        self.provider_name = "streaming-example"
        self.supported_models = ["stream-model-1.0", "stream-model-fast"]

    def completion(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        """Non-streaming completion for comparison."""
        # Simulate processing time
        time.sleep(0.5)

        full_response = self._generate_full_response(messages)
        return {
            "id": "completion_123",
            "model": model,
            "choices": [
                {
                    "message": {"role": "assistant", "content": full_response},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 50, "total_tokens": 70},
        }

    def _generate_full_response(self, messages: list[dict[str, Any]]) -> str:
        """Generate a complete response based on the conversation."""
        last_message = messages[-1] if messages else {"content": ""}
        user_content = last_message.get("content", "")

        if "weather" in user_content.lower():
            return (
                "The weather is sunny with a temperature of 22°C. "
                "Perfect for outdoor activities!"
            )
        elif "joke" in user_content.lower():
            return "Why don't scientists trust atoms? Because they make up everything!"
        elif "math" in user_content.lower() or any(
            op in user_content for op in ["+", "-", "*", "/", "="]
        ):
            return (
                "I can help with mathematical calculations. "
                "The result of your expression would be computed here."
            )
        else:
            return (
                f"Thank you for your message: '{user_content}'. "
                "This is a streaming response demonstration showing how UUTEL "
                "handles real-time communication with AI providers."
            )

    def simulate_streaming_chunks(self, full_response: str) -> Generator[dict]:
        """Simulate streaming response by breaking full response into chunks."""
        words = full_response.split()
        chunk_id = "chunk_123"

        # Send chunks with realistic timing
        for i, word in enumerate(words):
            # Add space before word (except first)
            content = word if i == 0 else f" {word}"

            chunk = {
                "id": chunk_id,
                "model": "stream-model-1.0",
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": content},
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
            }

            yield chunk
            time.sleep(0.1)  # Simulate network delay

        # Send final chunk
        final_chunk = {
            "id": chunk_id,
            "model": "stream-model-1.0",
            "choices": [
                {"delta": {}, "index": 0, "finish_reason": "stop"}  # Empty delta
            ],
        }
        yield final_chunk

    async def simulate_async_streaming_chunks(
        self, full_response: str
    ) -> AsyncGenerator[dict]:
        """Simulate async streaming response."""
        words = full_response.split()
        chunk_id = "async_chunk_456"

        for i, word in enumerate(words):
            content = word if i == 0 else f" {word}"

            chunk = {
                "id": chunk_id,
                "model": "stream-model-fast",
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": content},
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
            }

            yield chunk
            await asyncio.sleep(0.05)  # Faster async streaming

        # Send final chunk
        final_chunk = {
            "id": chunk_id,
            "model": "stream-model-fast",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        }
        yield final_chunk


def demonstrate_sync_streaming():
    """Demonstrate synchronous streaming response handling."""
    print("🌊 Synchronous Streaming Response")
    print("=" * 50)

    provider = StreamingExampleProvider()
    messages = [{"role": "user", "content": "Tell me about the weather today please"}]

    print("📤 User message:", messages[-1]["content"])
    print("🤖 Assistant response (streaming):")
    print("   ", end="", flush=True)

    # Generate full response for chunking
    full_response = provider._generate_full_response(messages)

    # Process streaming chunks
    accumulated_content = ""
    chunk_count = 0

    for chunk in provider.simulate_streaming_chunks(full_response):
        chunk_count += 1
        choices = chunk.get("choices", [])

        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")

            if content:
                print(content, end="", flush=True)
                accumulated_content += content

            finish_reason = choices[0].get("finish_reason")
            if finish_reason:
                print("\n\n✅ Streaming completed!")
                print(f"   Chunks processed: {chunk_count}")
                print(f"   Total content length: {len(accumulated_content)} characters")
                print(f"   Finish reason: {finish_reason}")
                break


async def demonstrate_async_streaming():
    """Demonstrate asynchronous streaming response handling."""
    print("\n🚀 Asynchronous Streaming Response")
    print("=" * 50)

    provider = StreamingExampleProvider()
    messages = [{"role": "user", "content": "Tell me a programming joke"}]

    print("📤 User message:", messages[-1]["content"])
    print("🤖 Assistant response (async streaming):")
    print("   ", end="", flush=True)

    # Generate full response for chunking
    full_response = provider._generate_full_response(messages)

    # Process async streaming chunks
    accumulated_content = ""
    chunk_count = 0

    async for chunk in provider.simulate_async_streaming_chunks(full_response):
        chunk_count += 1
        choices = chunk.get("choices", [])

        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")

            if content:
                print(content, end="", flush=True)
                accumulated_content += content

            finish_reason = choices[0].get("finish_reason")
            if finish_reason:
                print("\n\n✅ Async streaming completed!")
                print(f"   Chunks processed: {chunk_count}")
                print(f"   Total content length: {len(accumulated_content)} characters")
                print(f"   Finish reason: {finish_reason}")
                break


def demonstrate_stream_aggregation():
    """Demonstrate how to aggregate streaming chunks into a complete response."""
    print("\n📊 Stream Aggregation and Processing")
    print("=" * 50)

    provider = StreamingExampleProvider()
    messages = [{"role": "user", "content": "Explain 2 + 2 * 3 math calculation"}]

    print("📤 User message:", messages[-1]["content"])
    print("🔄 Processing streaming chunks...")

    full_response = provider._generate_full_response(messages)

    # Aggregate chunks
    aggregated_response = {
        "id": None,
        "model": None,
        "choices": [{"message": {"role": "assistant", "content": ""}, "index": 0}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    chunk_count = 0
    start_time = time.time()

    for chunk in provider.simulate_streaming_chunks(full_response):
        chunk_count += 1

        # Update aggregated response
        if aggregated_response["id"] is None:
            aggregated_response["id"] = chunk.get("id")
            aggregated_response["model"] = chunk.get("model")

        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")

            if content:
                aggregated_response["choices"][0]["message"]["content"] += content

            finish_reason = choices[0].get("finish_reason")
            if finish_reason:
                aggregated_response["choices"][0]["finish_reason"] = finish_reason
                break

    end_time = time.time()

    print("\n📈 Aggregation complete:")
    print(f"   Streaming time: {end_time - start_time:.2f} seconds")
    print(f"   Chunks processed: {chunk_count}")
    print(f"   Response ID: {aggregated_response['id']}")
    print(f"   Model: {aggregated_response['model']}")
    print(
        f"   Content length: "
        f"{len(aggregated_response['choices'][0]['message']['content'])} chars"
    )

    # Display final aggregated content
    print("\n💬 Final aggregated response:")
    print(f"   {aggregated_response['choices'][0]['message']['content']}")


def demonstrate_streaming_error_handling():
    """Demonstrate error handling in streaming scenarios."""
    print("\n🚨 Streaming Error Handling")
    print("=" * 50)

    # Simulate various streaming error scenarios
    error_scenarios = [
        {
            "name": "Network interruption",
            "chunks": [
                {"choices": [{"delta": {"content": "Hello"}}]},
                {"choices": [{"delta": {"content": " world"}}]},
                # Simulate network error - no more chunks
            ],
            "should_timeout": True,
        },
        {
            "name": "Malformed chunk",
            "chunks": [
                {"choices": [{"delta": {"content": "Start"}}]},
                {"invalid": "chunk"},  # Malformed chunk
                {"choices": [{"delta": {"content": " end"}}]},
            ],
            "should_timeout": False,
        },
        {
            "name": "Empty response",
            "chunks": [],
            "should_timeout": False,
        },
    ]

    for scenario in error_scenarios:
        print(f"\n🔍 Testing: {scenario['name']}")

        accumulated_content = ""
        chunk_count = 0
        error_count = 0

        try:
            for chunk in scenario["chunks"]:
                chunk_count += 1

                # Validate chunk structure
                if not isinstance(chunk, dict) or "choices" not in chunk:
                    error_count += 1
                    print(f"   ⚠️  Malformed chunk {chunk_count}: {chunk}")
                    continue

                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")

                    if content:
                        accumulated_content += content
                        print(f"   ✅ Chunk {chunk_count}: '{content}'")

            # Simulate timeout for network interruption
            if scenario["should_timeout"] and not accumulated_content.endswith("world"):
                raise TimeoutError("Streaming connection timeout")

            print(
                f"   📊 Result: {len(accumulated_content)} chars, {error_count} errors"
            )

        except Exception as e:
            error_msg = format_error_message(e, "streaming-provider")
            print(f"   ❌ Error: {error_msg}")

            # In real implementation, you might:
            # - Retry the request
            # - Switch to non-streaming mode
            # - Return partial content with error flag
            print(
                f"   🔄 Fallback: Return partial content "
                f"({len(accumulated_content)} chars)"
            )


async def demonstrate_concurrent_streaming():
    """Demonstrate handling multiple concurrent streaming requests."""
    print("\n⚡ Concurrent Streaming Requests")
    print("=" * 50)

    provider = StreamingExampleProvider()

    # Multiple concurrent requests
    requests = [
        {"id": "req_1", "content": "What's the weather like?"},
        {"id": "req_2", "content": "Tell me a joke"},
        {"id": "req_3", "content": "Calculate 10 + 5"},
    ]

    print(f"🚀 Starting {len(requests)} concurrent streaming requests...")

    async def process_streaming_request(request: dict[str, str]) -> dict[str, Any]:
        """Process a single streaming request."""
        messages = [{"role": "user", "content": request["content"]}]
        full_response = provider._generate_full_response(messages)

        accumulated_content = ""
        chunk_count = 0

        async for chunk in provider.simulate_async_streaming_chunks(full_response):
            chunk_count += 1
            choices = chunk.get("choices", [])

            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")

                if content:
                    accumulated_content += content

                finish_reason = choices[0].get("finish_reason")
                if finish_reason:
                    break

        return {
            "request_id": request["id"],
            "content": accumulated_content,
            "chunks": chunk_count,
        }

    # Execute all requests concurrently
    start_time = time.time()
    results = await asyncio.gather(
        *[process_streaming_request(req) for req in requests]
    )
    end_time = time.time()

    print("\n📊 Concurrent processing complete:")
    print(f"   Total time: {end_time - start_time:.2f} seconds")
    print(f"   Requests processed: {len(results)}")

    for result in results:
        print(f"\n   📤 {result['request_id']}:")
        print(f"      Chunks: {result['chunks']}")
        print(f"      Content: {result['content'][:50]}...")


def demonstrate_streaming_with_tools():
    """Demonstrate streaming responses that include tool calls."""
    print("\n🛠️  Streaming with Tool Calls")
    print("=" * 50)

    print("🔄 Simulating streaming response with tool calls...")

    # Simulate streaming chunks that build up to a tool call
    streaming_chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "content": (
                            "I'll help you with the weather. Let me check that for you."
                        ),
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_weather_123",
                                "type": "function",
                                "function": {"name": "get_weather"},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": json.dumps(
                                        {"location": "San Francisco", "unit": "celsius"}
                                    )
                                },
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {"delta": {}, "finish_reason": "tool_calls"}  # Tool call complete
            ]
        },
    ]

    accumulated_content = ""
    tool_calls: list[dict[str, Any]] = []

    for i, chunk in enumerate(streaming_chunks):
        print(f"📦 Processing chunk {i + 1}/{len(streaming_chunks)}")

        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})

            # Handle content
            content = delta.get("content", "")
            if content:
                accumulated_content += content
                print(f"   💬 Content: '{content}'")

            # Handle tool calls
            delta_tool_calls = delta.get("tool_calls", [])
            if delta_tool_calls:
                print(f"   🛠️  Tool call delta: {delta_tool_calls}")
                # In real implementation, you'd accumulate tool call data
                tool_calls.extend(delta_tool_calls)

            finish_reason = choices[0].get("finish_reason")
            if finish_reason:
                print(f"   🏁 Finished: {finish_reason}")
                break

    print("\n📊 Streaming with tools complete:")
    print(f"   Content: '{accumulated_content}'")
    print(f"   Tool calls initiated: {len(tool_calls)}")
    if tool_calls:
        print(
            f"   First tool: {tool_calls[0].get('function', {}).get('name', 'unknown')}"
        )


async def main():
    """Main function demonstrating all streaming capabilities."""
    print("🌊 UUTEL Streaming Response Example")
    print("=" * 60)
    print("This example demonstrates how UUTEL handles streaming responses")
    print("=" * 60)

    try:
        # Run all streaming demonstrations
        demonstrate_sync_streaming()
        await demonstrate_async_streaming()
        demonstrate_stream_aggregation()
        demonstrate_streaming_error_handling()
        await demonstrate_concurrent_streaming()
        demonstrate_streaming_with_tools()

        print("\n" + "=" * 60)
        print("🎉 Streaming example completed successfully!")
        print("=" * 60)
        print("\nKey capabilities demonstrated:")
        print("✅ Synchronous streaming response handling")
        print("✅ Asynchronous streaming response handling")
        print("✅ Stream chunk aggregation and processing")
        print("✅ Error handling and recovery in streaming")
        print("✅ Concurrent streaming request processing")
        print("✅ Streaming responses with tool calls")
        print(
            "\nUUTEL provides robust streaming support for real-time AI interactions!"
        )

    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
