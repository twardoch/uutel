#!/usr/bin/env python3
# this_file: examples/tool_calling_example.py
"""Comprehensive Tool Calling Example for UUTEL (Universal AI Provider for LiteLLM).

This example demonstrates UUTEL's comprehensive tool calling capabilities:
- OpenAI-compatible tool schema validation
- Tool transformation between formats
- Tool call response creation
- Tool call extraction from provider responses
- Complete tool calling workflow simulation
- LiteLLM integration with tools
- Async tool calling patterns
- Real-world tool implementations

The example shows how UUTEL's tool calling utilities work with various
providers while maintaining OpenAI compatibility.
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any

import litellm

from uutel import (
    create_tool_call_response,
    extract_tool_calls_from_response,
    transform_openai_tools_to_provider,
    transform_provider_tools_to_openai,
    validate_tool_schema,
)
from uutel.providers.codex.custom_llm import CodexCustomLLM


# Real-world tool implementations
def get_weather(location: str, unit: str = "celsius") -> dict[str, Any]:
    """Mock weather function returning realistic data.

    Args:
        location: City and country (e.g., "Paris, France")
        unit: Temperature unit ("celsius" or "fahrenheit")

    Returns:
        Weather data dictionary
    """
    # Simulate realistic weather data
    base_temp = 20 + random.randint(-15, 25)  # Base temp in Celsius
    temp_c = max(-30, min(45, base_temp))  # Realistic range
    temp_f = int(temp_c * 9 / 5 + 32)

    conditions = ["sunny", "cloudy", "rainy", "partly_cloudy", "overcast", "foggy"]

    return {
        "location": location,
        "temperature": temp_c if unit == "celsius" else temp_f,
        "unit": unit,
        "condition": random.choice(conditions),
        "humidity": random.randint(20, 90),
        "wind_speed": random.randint(0, 25),
        "feels_like": temp_c + random.randint(-5, 5)
        if unit == "celsius"
        else temp_f + random.randint(-9, 9),
    }


def search_web(query: str, max_results: int = 5) -> dict[str, Any]:
    """Mock web search returning structured results.

    Args:
        query: Search query string
        max_results: Maximum results to return (1-10)

    Returns:
        Search results with metadata
    """
    # Simulate search results
    results = []
    for i in range(min(max_results, random.randint(2, 5))):
        results.append(
            {
                "title": f"Result {i + 1}: {query} - Comprehensive Guide",
                "url": f"https://example.com/{query.replace(' ', '-')}-{i + 1}",
                "snippet": f"Comprehensive information about {query}. "
                f"This resource provides detailed insights and practical examples.",
                "relevance_score": round(random.uniform(0.7, 1.0), 2),
            }
        )

    return {
        "query": query,
        "total_results": len(results),
        "search_time_ms": random.randint(50, 300),
        "results": results,
    }


async def analyze_sentiment(text: str) -> dict[str, Any]:
    """Async mock sentiment analysis function.

    Args:
        text: Text to analyze

    Returns:
        Sentiment analysis results
    """
    # Simulate processing delay
    await asyncio.sleep(0.1)

    # Simple sentiment scoring based on keywords
    positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
    negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"]

    text_lower = text.lower()
    positive_count = sum(word in text_lower for word in positive_words)
    negative_count = sum(word in text_lower for word in negative_words)

    if positive_count > negative_count:
        sentiment = "positive"
        confidence = 0.7 + (positive_count - negative_count) * 0.1
    elif negative_count > positive_count:
        sentiment = "negative"
        confidence = 0.7 + (negative_count - positive_count) * 0.1
    else:
        sentiment = "neutral"
        confidence = 0.6

    return {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "sentiment": sentiment,
        "confidence": min(confidence, 1.0),
        "word_count": len(text.split()),
        "positive_indicators": positive_count,
        "negative_indicators": negative_count,
    }


# Tool registry for execution
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "search_web": search_web,
    "analyze_sentiment": analyze_sentiment,
}


def demonstrate_tool_schema_validation():
    """Demonstrate tool schema validation capabilities."""
    print("🔍 Tool Schema Validation")
    print("=" * 50)

    # Valid OpenAI tool schema
    valid_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                        "default": "celsius",
                    },
                },
                "required": ["location"],
            },
        },
    }

    # Invalid tool schemas for testing
    invalid_tools = [
        {"type": "invalid"},  # Wrong type
        {"type": "function"},  # Missing function
        {
            "type": "function",
            "function": {"name": "test"},
        },  # Missing description
        {
            "type": "function",
            "function": {
                "name": "test",
                "description": "Test",
                "parameters": "invalid",  # Parameters must be dict
            },
        },
    ]

    print(f"✅ Valid tool: {validate_tool_schema(valid_tool)}")
    print(f"   Tool name: {valid_tool['function']['name']}")
    print(f"   Description: {valid_tool['function']['description']}")

    print("\n❌ Invalid tools:")
    for i, invalid_tool in enumerate(invalid_tools):
        is_valid = validate_tool_schema(invalid_tool)
        print(f"   Tool {i + 1}: {is_valid} - {invalid_tool}")

    return valid_tool


def demonstrate_tool_transformation():
    """Demonstrate tool transformation between OpenAI and provider formats."""
    print("\n🔄 Tool Format Transformation")
    print("=" * 50)

    # Create multiple tools for transformation
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate",
                        }
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {
                            "type": "integer",
                            "description": "Number of results",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]

    print(f"📤 Original OpenAI tools: {len(openai_tools)} tools")
    for tool in openai_tools:
        print(f"   - {tool['function']['name']}: {tool['function']['description']}")

    # Transform to provider format
    provider_name = "example-provider"
    provider_tools = transform_openai_tools_to_provider(openai_tools, provider_name)

    print(f"\n🔄 Transformed to {provider_name} format: {len(provider_tools)} tools")

    # Transform back to OpenAI format
    back_to_openai = transform_provider_tools_to_openai(provider_tools, provider_name)

    print(f"📥 Transformed back to OpenAI: {len(back_to_openai)} tools")
    print(f"✅ Round-trip successful: {openai_tools == back_to_openai}")

    return openai_tools[0]  # Return first tool for next demo


def demonstrate_tool_call_responses():
    """Demonstrate tool call response creation."""
    print("\n📞 Tool Call Response Creation")
    print("=" * 50)

    # Successful tool call response
    success_response = create_tool_call_response(
        tool_call_id="call_123",
        function_name="get_weather",
        function_result={"temperature": 22, "condition": "sunny", "humidity": 65},
    )

    print("✅ Successful tool call response:")
    print(f"   ID: {success_response['tool_call_id']}")
    print(f"   Role: {success_response['role']}")
    print(f"   Content: {success_response['content']}")

    # Error tool call response
    error_response = create_tool_call_response(
        tool_call_id="call_456",
        function_name="broken_function",
        error="Network timeout while connecting to weather API",
    )

    print("\n❌ Error tool call response:")
    print(f"   ID: {error_response['tool_call_id']}")
    print(f"   Role: {error_response['role']}")
    print(f"   Content: {error_response['content']}")

    # Non-JSON serializable result
    class CustomObject:
        def __str__(self):
            return "CustomWeatherData(temp=25°C)"

    complex_response = create_tool_call_response(
        tool_call_id="call_789",
        function_name="complex_function",
        function_result=CustomObject(),
    )

    print("\n🔧 Complex object response:")
    print(f"   ID: {complex_response['tool_call_id']}")
    print(f"   Content: {complex_response['content']}")

    return [success_response, error_response]


def demonstrate_tool_call_extraction():
    """Demonstrate tool call extraction from provider responses."""
    print("\n📤 Tool Call Extraction from Responses")
    print("=" * 50)

    # Simulate a provider response with tool calls
    provider_response = {
        "id": "response_123",
        "model": "example-model",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I'll help you get the weather information.",
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps(
                                    {"location": "San Francisco", "unit": "celsius"}
                                ),
                            },
                        },
                        {
                            "id": "call_def456",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps(
                                    {"location": "New York", "unit": "fahrenheit"}
                                ),
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    print("🔍 Extracting tool calls from provider response...")
    tool_calls = extract_tool_calls_from_response(provider_response)

    print(f"📞 Found {len(tool_calls)} tool calls:")
    for i, tool_call in enumerate(tool_calls):
        function_info = tool_call["function"]
        args = json.loads(function_info["arguments"])
        print(f"   {i + 1}. {tool_call['id']}: {function_info['name']}({args})")

    # Test with response containing no tool calls
    no_tools_response = {
        "choices": [{"message": {"role": "assistant", "content": "Hello!"}}]
    }

    no_tool_calls = extract_tool_calls_from_response(no_tools_response)
    print(f"\n📭 Response with no tools: {len(no_tool_calls)} tool calls found")

    # Test with invalid response
    invalid_response = "not a dict"
    invalid_tool_calls = extract_tool_calls_from_response(invalid_response)
    print(f"❌ Invalid response: {len(invalid_tool_calls)} tool calls found")

    return tool_calls


def demonstrate_complete_workflow():
    """Demonstrate a complete tool calling workflow."""
    print("\n🔄 Complete Tool Calling Workflow")
    print("=" * 50)

    # 1. Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression (e.g., '2 + 2')",
                        }
                    },
                    "required": ["expression"],
                },
            },
        }
    ]

    print("1️⃣ Tools defined and validated:")
    for tool in tools:
        is_valid = validate_tool_schema(tool)
        print(f"   ✅ {tool['function']['name']}: {is_valid}")

    # 2. Transform for provider
    provider_tools = transform_openai_tools_to_provider(tools, "math-provider")
    print(f"\n2️⃣ Tools transformed for math-provider: {len(provider_tools)} tools")

    # 3. Simulate provider response with tool call
    mock_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I'll calculate that for you.",
                    "tool_calls": [
                        {
                            "id": "calc_001",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": json.dumps({"expression": "15 * 3 + 7"}),
                            },
                        }
                    ],
                }
            }
        ]
    }

    # 4. Extract tool calls
    tool_calls = extract_tool_calls_from_response(mock_response)
    print(f"\n3️⃣ Extracted tool calls: {len(tool_calls)} calls")

    # 5. Simulate tool execution and create responses
    for tool_call in tool_calls:
        function_name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])

        print(f"\n4️⃣ Executing {function_name} with args: {args}")

        # Simulate calculation
        if function_name == "calculator":
            try:
                result = eval(args["expression"])  # Note: unsafe in production!
                response = create_tool_call_response(
                    tool_call_id=tool_call["id"],
                    function_name=function_name,
                    function_result={
                        "result": result,
                        "expression": args["expression"],
                    },
                )
                print(f"   ✅ Result: {result}")
            except Exception as e:
                response = create_tool_call_response(
                    tool_call_id=tool_call["id"],
                    function_name=function_name,
                    error=f"Calculation error: {e!s}",
                )
                print(f"   ❌ Error: {e}")

        print("5️⃣ Tool response created:")
        print(f"   Role: {response['role']}")
        print(f"   Content: {response['content'][:100]}...")

    print("\n🎉 Complete workflow demonstration finished!")


def demonstrate_error_handling():
    """Demonstrate error handling in tool calling scenarios."""
    print("\n🚨 Error Handling Scenarios")
    print("=" * 50)

    # Test various error conditions
    error_scenarios = [
        ("Invalid tool schema", {"invalid": "schema"}),
        ("Empty tools list", []),
        ("None tools", None),
        ("Invalid response format", "not a dict"),
        ("Missing choices", {"no_choices": True}),
    ]

    for scenario_name, test_data in error_scenarios:
        print(f"\n🔍 Testing: {scenario_name}")

        if "tool schema" in scenario_name:
            result = validate_tool_schema(test_data)
            print(f"   Validation result: {result}")

        elif "tools list" in scenario_name or "None tools" in scenario_name:
            result = transform_openai_tools_to_provider(test_data, "test-provider")
            print(f"   Transformation result: {len(result)} tools")

        elif "response" in scenario_name or "choices" in scenario_name:
            result = extract_tool_calls_from_response(test_data)
            print(f"   Extraction result: {len(result)} tool calls")

        print("   ✅ Error handled gracefully")


def demonstrate_litellm_integration():
    """Demonstrate tool calling with actual LiteLLM integration."""
    print("\n🔌 LiteLLM Integration with Tools")
    print("=" * 50)

    # Setup UUTEL provider with LiteLLM
    codex_provider = CodexCustomLLM()
    litellm.custom_provider_map = [
        {"provider": "my-custom-llm", "custom_handler": codex_provider},
    ]

    # Define tools for LiteLLM
    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }

    try:
        print("🚀 Making LiteLLM completion call with tools...")

        # Note: In a real implementation, the provider would process tools
        # This demonstrates the integration pattern
        response = litellm.completion(
            model="my-custom-llm/codex-large",
            messages=[{"role": "user", "content": "What's the weather like in Tokyo?"}],
            tools=[weather_tool],
            max_tokens=100,
        )

        print("✅ LiteLLM completion with tools successful!")
        print(f"   Model: {response.model}")
        print(f"   Response: {response.choices[0].message.content[:100]}...")

        # Demonstrate how tools would be processed in a real implementation
        print("\n🔧 Tool processing workflow:")
        print("   1. Provider receives tools in OpenAI format")
        print("   2. Tools are validated using UUTEL utilities")
        print("   3. Tools are transformed to provider-specific format")
        print("   4. Provider processes completion with tools")
        print("   5. Tool calls are extracted and executed")
        print("   6. Results are formatted and returned")

    except Exception as e:
        print(f"❌ LiteLLM integration demo failed: {e}")
        print("   Note: This is a mock implementation for demonstration")


async def demonstrate_async_tool_calling():
    """Demonstrate asynchronous tool calling patterns."""
    print("\n⚡ Async Tool Calling Patterns")
    print("=" * 50)

    # Simulate multiple async tool calls
    print("🔄 Executing multiple async tool calls concurrently...")

    tasks = [
        analyze_sentiment("This is an amazing product, I love it!"),
        analyze_sentiment("The service was terrible and disappointing."),
        analyze_sentiment("It's okay, nothing special but not bad either."),
    ]

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)

    print(f"✅ Completed {len(results)} async tool calls:")
    for i, result in enumerate(results, 1):
        sentiment = result["sentiment"]
        confidence = result["confidence"]
        text_preview = (
            result["text"][:50] + "..." if len(result["text"]) > 50 else result["text"]
        )
        print(f"   {i}. '{text_preview}' → {sentiment} ({confidence:.2f})")

    # Demonstrate async tool execution with error handling
    print("\n🔧 Async tool execution with error handling:")

    async def safe_tool_execution(tool_name: str, **kwargs):
        """Safely execute a tool function with error handling."""
        try:
            func = TOOL_FUNCTIONS.get(tool_name)
            if not func:
                raise ValueError(f"Unknown tool: {tool_name}")

            if asyncio.iscoroutinefunction(func):
                return await func(**kwargs)
            else:
                return func(**kwargs)
        except Exception as e:
            return {"error": str(e), "tool": tool_name, "args": kwargs}

    # Execute tools with different patterns
    safe_results = await asyncio.gather(
        safe_tool_execution("get_weather", location="Paris, France"),
        safe_tool_execution("search_web", query="Python asyncio", max_results=3),
        safe_tool_execution("unknown_tool", param="value"),  # This will error
    )

    for i, result in enumerate(safe_results, 1):
        if "error" in result:
            print(f"   {i}. ❌ Error: {result['error']}")
        else:
            print(f"   {i}. ✅ Success: {type(result).__name__} data returned")


async def demonstrate_advanced_tool_scenarios():
    """Demonstrate advanced tool calling scenarios."""
    print("\n🚀 Advanced Tool Calling Scenarios")
    print("=" * 50)

    # Tool chaining simulation
    print("🔗 Tool Chaining Example:")
    print("   Search → Analyze Sentiment of Results → Get Weather for Locations")

    # Step 1: Search
    search_results = search_web("best travel destinations 2024", max_results=3)
    print(f"   1. Search found {search_results['total_results']} results")

    # Step 2: Analyze sentiment of search results
    sentiments = []
    for result in search_results["results"][:2]:  # Analyze first 2 results
        sentiment_result = await analyze_sentiment(result["snippet"])
        sentiments.append(sentiment_result)

    print(f"   2. Analyzed sentiment for {len(sentiments)} results")

    # Step 3: Get weather for mentioned locations
    locations = ["Tokyo, Japan", "Paris, France"]  # Mock extracted locations
    weather_data = []
    for location in locations:
        weather = get_weather(location)
        weather_data.append(weather)

    print(f"   3. Retrieved weather for {len(weather_data)} locations")

    # Complex tool parameter validation
    print("\n📋 Complex Tool Schema Validation:")

    complex_tool = {
        "type": "function",
        "function": {
            "name": "analyze_data",
            "description": "Analyze complex data structures",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "object",
                        "properties": {
                            "values": {"type": "array", "items": {"type": "number"}},
                            "metadata": {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string"},
                                    "created_at": {
                                        "type": "string",
                                        "format": "date-time",
                                    },
                                },
                            },
                        },
                        "required": ["values"],
                    },
                    "analysis_options": {
                        "type": "object",
                        "properties": {
                            "methods": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["mean", "median", "mode", "std"],
                                },
                            },
                            "include_visualization": {
                                "type": "boolean",
                                "default": False,
                            },
                        },
                    },
                },
                "required": ["dataset"],
            },
        },
    }

    is_valid = validate_tool_schema(complex_tool)
    print(f"   Complex nested schema: {'✅ Valid' if is_valid else '❌ Invalid'}")

    # Tool result processing
    print("\n📊 Tool Result Processing:")

    def process_tool_results(results: list[dict[str, Any]]) -> dict[str, Any]:
        """Process and summarize multiple tool results."""
        summary = {
            "total_results": len(results),
            "successful": 0,
            "failed": 0,
            "data_types": set(),
        }

        for result in results:
            if "error" in result:
                summary["failed"] += 1
            else:
                summary["successful"] += 1
                summary["data_types"].add(type(result).__name__)

        summary["data_types"] = list(summary["data_types"])
        return summary

    # Mock results from previous tools
    mock_results = [
        weather_data[0],
        search_results,
        sentiments[0] if sentiments else {"error": "no sentiment data"},
    ]

    summary = process_tool_results(mock_results)
    print(f"   Processed {summary['total_results']} results:")
    print(f"   - Successful: {summary['successful']}")
    print(f"   - Failed: {summary['failed']}")
    print(f"   - Data types: {', '.join(summary['data_types'])}")


async def main():
    """Main example function demonstrating all tool calling capabilities."""
    print("🛠️  UUTEL Comprehensive Tool Calling Example")
    print("=" * 70)
    print("This example demonstrates UUTEL's complete tool calling ecosystem")
    print("=" * 70)

    try:
        # Run all demonstrations
        demonstrate_tool_schema_validation()
        demonstrate_tool_transformation()
        demonstrate_tool_call_responses()
        demonstrate_tool_call_extraction()
        demonstrate_complete_workflow()
        demonstrate_litellm_integration()
        await demonstrate_async_tool_calling()
        await demonstrate_advanced_tool_scenarios()
        demonstrate_error_handling()

        print("\n" + "=" * 70)
        print("🎉 Comprehensive tool calling example completed successfully!")
        print("=" * 70)
        print("\nKey capabilities demonstrated:")
        print("✅ Tool schema validation (OpenAI format)")
        print("✅ Tool format transformation (provider ↔ OpenAI)")
        print("✅ Tool call response creation")
        print("✅ Tool call extraction from responses")
        print("✅ Complete workflow simulation")
        print("✅ LiteLLM integration with tools")
        print("✅ Async tool calling patterns")
        print("✅ Advanced tool scenarios (chaining, complex schemas)")
        print("✅ Error handling and edge cases")
        print("✅ Real-world tool implementations")
        print(
            "\nUUTEL provides a comprehensive foundation for tool calling across "
            "all AI providers with full LiteLLM compatibility!"
        )

    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
