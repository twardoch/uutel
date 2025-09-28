# this_file: tests/test_tool_calling.py
"""Tests for UUTEL tool calling utilities."""

from __future__ import annotations

from uutel.core.utils import (
    create_tool_call_response,
    extract_tool_calls_from_response,
    transform_openai_tools_to_provider,
    transform_provider_tools_to_openai,
    validate_tool_schema,
)


class TestToolSchemaValidation:
    """Test tool schema validation utilities."""

    def test_validate_tool_schema_valid_openai_format(self) -> None:
        """Test validation of valid OpenAI tool schema."""
        valid_tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }

        assert validate_tool_schema(valid_tool) is True

    def test_validate_tool_schema_invalid_format(self) -> None:
        """Test validation of invalid tool schemas."""
        invalid_tools = [
            {},  # Empty
            {"type": "invalid"},  # Wrong type
            {"type": "function"},  # Missing function
            {"type": "function", "function": {}},  # Empty function
            {"type": "function", "function": {"name": "test"}},  # Missing description
        ]

        for invalid_tool in invalid_tools:
            assert validate_tool_schema(invalid_tool) is False

    def test_validate_tool_schema_missing_parameters(self) -> None:
        """Test validation when parameters are missing."""
        tool_without_params = {
            "type": "function",
            "function": {
                "name": "simple_tool",
                "description": "A simple tool",
                # Missing parameters
            },
        }

        # Should still be valid - parameters are optional
        assert validate_tool_schema(tool_without_params) is True

    def test_validate_tool_schema_invalid_parameters(self) -> None:
        """Test validation with invalid parameter schemas."""
        tool_invalid_params = {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "Test tool",
                "parameters": "invalid",  # Should be object
            },
        }

        assert validate_tool_schema(tool_invalid_params) is False


class TestToolTransformation:
    """Test tool transformation utilities."""

    def test_transform_openai_tools_to_provider_basic(self) -> None:
        """Test basic OpenAI to provider tool transformation."""
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform calculation",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }
        ]

        transformed = transform_openai_tools_to_provider(openai_tools, "test-provider")

        assert isinstance(transformed, list)
        assert len(transformed) == 1
        assert transformed[0]["type"] == "function"
        assert transformed[0]["function"]["name"] == "calculate"

    def test_transform_openai_tools_to_provider_empty(self) -> None:
        """Test transformation of empty tool list."""
        assert transform_openai_tools_to_provider([], "test") == []
        assert transform_openai_tools_to_provider(None, "test") == []

    def test_transform_provider_tools_to_openai_basic(self) -> None:
        """Test basic provider to OpenAI tool transformation."""
        provider_tools = [
            {
                "type": "function",
                "function": {"name": "search", "description": "Search for information"},
            }
        ]

        transformed = transform_provider_tools_to_openai(
            provider_tools, "test-provider"
        )

        assert isinstance(transformed, list)
        assert len(transformed) == 1
        assert transformed[0]["type"] == "function"
        assert transformed[0]["function"]["name"] == "search"

    def test_transform_tools_round_trip(self) -> None:
        """Test round-trip transformation preserves tool structure."""
        original_tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_function",
                    "description": "Test function for round-trip",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "param1": {"type": "string"},
                            "param2": {"type": "number"},
                        },
                        "required": ["param1"],
                    },
                },
            }
        ]

        # Transform to provider format and back
        provider_format = transform_openai_tools_to_provider(original_tools, "test")
        back_to_openai = transform_provider_tools_to_openai(provider_format, "test")

        assert original_tools == back_to_openai

    def test_transform_tools_with_invalid_tools(self) -> None:
        """Test transformation handles invalid tools gracefully."""
        invalid_tools = [
            {"invalid": "format"},
            None,
            "not a dict",
            {"type": "function"},  # Missing function definition
        ]

        # Should filter out invalid tools
        result = transform_openai_tools_to_provider(invalid_tools, "test")
        assert len(result) == 0

        result = transform_provider_tools_to_openai(invalid_tools, "test")
        assert len(result) == 0


class TestToolCallHandling:
    """Test tool call creation and extraction utilities."""

    def test_create_tool_call_response_basic(self) -> None:
        """Test creating tool call response."""
        response = create_tool_call_response(
            tool_call_id="call_123",
            function_name="get_weather",
            function_result={"temperature": 72, "condition": "sunny"},
        )

        assert response["tool_call_id"] == "call_123"
        assert response["role"] == "tool"
        assert "temperature" in response["content"]

    def test_create_tool_call_response_with_error(self) -> None:
        """Test creating tool call response with error."""
        response = create_tool_call_response(
            tool_call_id="call_456",
            function_name="failing_function",
            function_result=None,
            error="Function execution failed",
        )

        assert response["tool_call_id"] == "call_456"
        assert response["role"] == "tool"
        assert "error" in response["content"].lower()

    def test_extract_tool_calls_from_response_basic(self) -> None:
        """Test extracting tool calls from provider response."""
        response_with_tools = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I'll help you with that.",
                        "tool_calls": [
                            {
                                "id": "call_789",
                                "type": "function",
                                "function": {
                                    "name": "search_web",
                                    "arguments": '{"query": "Python tutorials"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        tool_calls = extract_tool_calls_from_response(response_with_tools)

        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "call_789"
        assert tool_calls[0]["function"]["name"] == "search_web"

    def test_extract_tool_calls_from_response_no_tools(self) -> None:
        """Test extracting tool calls when none exist."""
        response_without_tools = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Here's my response without tools.",
                    }
                }
            ]
        }

        tool_calls = extract_tool_calls_from_response(response_without_tools)
        assert tool_calls == []

    def test_extract_tool_calls_from_response_invalid_format(self) -> None:
        """Test extracting tool calls from invalid response format."""
        invalid_responses = [
            {},  # Empty
            {"choices": []},  # No choices
            {"choices": [{}]},  # No message
            {"choices": [{"message": {}}]},  # No tool_calls
        ]

        for invalid_response in invalid_responses:
            tool_calls = extract_tool_calls_from_response(invalid_response)
            assert tool_calls == []


class TestToolUtilitiesIntegration:
    """Test integration of tool utilities."""

    def test_complete_tool_workflow(self) -> None:
        """Test complete tool calling workflow."""
        # 1. Start with OpenAI format tools
        openai_tools = [
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
                                "description": "Math expression",
                            }
                        },
                        "required": ["expression"],
                    },
                },
            }
        ]

        # 2. Validate tools
        for tool in openai_tools:
            assert validate_tool_schema(tool) is True

        # 3. Transform to provider format
        provider_tools = transform_openai_tools_to_provider(
            openai_tools, "calculator-provider"
        )
        assert len(provider_tools) == 1

        # 4. Simulate provider response with tool call
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
                                    "arguments": '{"expression": "2 + 2"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        # 5. Extract tool calls
        tool_calls = extract_tool_calls_from_response(mock_response)
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "calculator"

        # 6. Create tool response
        tool_response = create_tool_call_response(
            tool_call_id="calc_001",
            function_name="calculator",
            function_result={"result": 4},
        )
        assert tool_response["tool_call_id"] == "calc_001"
        assert "4" in tool_response["content"]

    def test_error_handling_in_tool_workflow(self) -> None:
        """Test error handling throughout tool workflow."""
        # Test with invalid tool schema
        invalid_tool = {"type": "invalid"}
        assert validate_tool_schema(invalid_tool) is False

        # Test with malformed response
        malformed_response = {"invalid": "format"}
        tool_calls = extract_tool_calls_from_response(malformed_response)
        assert tool_calls == []

        # Test tool response with error
        error_response = create_tool_call_response(
            tool_call_id="error_001",
            function_name="failing_function",
            function_result=None,
            error="Network timeout",
        )
        assert "error" in error_response["content"].lower()
        assert "timeout" in error_response["content"].lower()
