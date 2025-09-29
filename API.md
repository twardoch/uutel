# UUTEL API Reference

**Universal Units for AI Telegraphy (UUTEL)** - Comprehensive API Documentation

UUTEL extends LiteLLM with custom AI providers including Claude Code, Gemini CLI, Google Cloud Code, and OpenAI Codex, providing a unified interface for AI model inference with advanced features like tool calling, streaming, and authentication.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [Provider Integration](#provider-integration)
- [Authentication](#authentication)
- [Tool Calling](#tool-calling)
- [Message Transformation](#message-transformation)
- [Error Handling](#error-handling)
- [Advanced Usage](#advanced-usage)
- [Configuration](#configuration)

---

## Installation

### Basic Installation

```bash
# Core UUTEL functionality
pip install uutel

# With specific provider support
pip install uutel[codex]
pip install uutel[claude-code]
pip install uutel[gemini-cli]
pip install uutel[cloud-code]

# All providers
pip install uutel[providers]

# Full development environment
pip install uutel[full]
```

### Using uv (Recommended)

```bash
uv add uutel
uv add uutel[providers]
```

---

## Quick Start

### Basic LiteLLM Integration

```python
import litellm
from uutel.providers.codex.custom_llm import CodexCustomLLM

# Register UUTEL provider
litellm.custom_provider_map = [
    {"provider": "my-custom-llm", "custom_handler": CodexCustomLLM()},
]

# Use with LiteLLM
response = litellm.completion(
    model="my-custom-llm/codex-large",
    messages=[{"role": "user", "content": "Hello, world!"}]
)

print(response.choices[0].message.content)
```

### Streaming Example

```python
response = litellm.completion(
    model="my-custom-llm/codex-large",
    messages=[{"role": "user", "content": "Count from 1 to 5"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

---

## Core Components

### BaseUU Class

The foundation class for all UUTEL providers.

```python
from uutel.core.base import BaseUU

class CustomProviderUU(BaseUU):
    def __init__(self):
        super().__init__()
        self.provider_name = "custom-provider"
        self.supported_models = ["model-1", "model-2"]

    def completion(self, *args, **kwargs):
        # Implement completion logic
        pass
```

**Key Methods:**
- `completion(*args, **kwargs) -> ModelResponse`
- `acompletion(*args, **kwargs) -> ModelResponse`
- `streaming(*args, **kwargs) -> Iterator[GenericStreamingChunk]`
- `astreaming(*args, **kwargs) -> AsyncIterator[GenericStreamingChunk]`

### Authentication Classes

#### BaseAuth

Abstract base class for authentication handlers.

```python
from uutel.core.auth import BaseAuth, AuthResult

class CustomAuth(BaseAuth):
    async def authenticate(self) -> AuthResult:
        # Implement authentication logic
        return AuthResult(success=True, token="token", expires_at=None)

    async def get_headers(self) -> dict[str, str]:
        return {"Authorization": "Bearer token"}
```

#### OAuthAuth

OAuth 2.0 authentication handler.

```python
from uutel.core.auth import OAuthAuth

oauth_auth = OAuthAuth(
    client_id="your-client-id",
    client_secret="your-client-secret",
    authorization_url="https://provider.com/oauth/authorize",
    token_url="https://provider.com/oauth/token",
    scopes=["read", "write"]
)

# Authenticate
result = await oauth_auth.authenticate()
if result.success:
    headers = await oauth_auth.get_headers()
```

#### ApiKeyAuth

Simple API key authentication.

```python
from uutel.core.auth import ApiKeyAuth

api_auth = ApiKeyAuth(
    api_key="your-api-key",
    header_name="X-API-Key"  # Default: "Authorization"
)

headers = await api_auth.get_headers()
# Returns: {"X-API-Key": "your-api-key"}
```

#### ServiceAccountAuth

Google Cloud service account authentication.

```python
from uutel.core.auth import ServiceAccountAuth

service_auth = ServiceAccountAuth(
    service_account_path="/path/to/service-account.json",
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

result = await service_auth.authenticate()
```

---

## Provider Integration

### CodexUU Provider

Integration with OpenAI Codex via session token management.

```python
from uutel.providers.codex.provider import CodexUU
from uutel.providers.codex.custom_llm import CodexCustomLLM

# Direct usage
codex = CodexUU()
response = codex.completion(
    model="codex-large",
    messages=[{"role": "user", "content": "Write a Python function"}],
    api_base="https://api.openai.com/v1",
    custom_prompt_dict={},
    model_response=ModelResponse(),
    # ... other parameters
)

# LiteLLM integration
codex_llm = CodexCustomLLM()
litellm.custom_provider_map = [
    {"provider": "codex", "custom_handler": codex_llm}
]
```

**Supported Models:**
- `codex-large` - High capability model
- `codex-mini` - Fast, lightweight model
- `codex-turbo` - Balanced performance
- `codex-fast` - Ultra-fast responses
- `codex-preview` - Latest features

### Future Providers

#### ClaudeCodeUU (Planned)

```python
# Future implementation
from uutel.providers.claude_code import ClaudeCodeUU

claude = ClaudeCodeUU()
# OAuth browser-based authentication
# MCP tool integration
# Advanced conversation management
```

#### GeminiCLIUU (Planned)

```python
# Future implementation
from uutel.providers.gemini_cli import GeminiCLIUU

gemini = GeminiCLIUU()
# Multi-auth support (API key, Vertex AI, OAuth)
# Advanced tool calling capabilities
# Vertex AI integration
```

---

## Tool Calling

UUTEL provides comprehensive tool calling utilities compatible with OpenAI's format.

### Tool Schema Validation

```python
from uutel import validate_tool_schema

tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}

is_valid = validate_tool_schema(tool)  # True
```

### Tool Format Transformation

```python
from uutel import (
    transform_openai_tools_to_provider,
    transform_provider_tools_to_openai
)

# Transform for provider
provider_tools = transform_openai_tools_to_provider([tool], "codex")

# Transform back to OpenAI format
openai_tools = transform_provider_tools_to_openai(provider_tools, "codex")
```

### Tool Call Response Creation

```python
from uutel import create_tool_call_response

# Success response
response = create_tool_call_response(
    tool_call_id="call_123",
    function_name="get_weather",
    function_result={"temperature": 22, "condition": "sunny"}
)

# Error response
error_response = create_tool_call_response(
    tool_call_id="call_123",
    function_name="get_weather",
    error="API connection failed"
)
```

### Tool Call Extraction

```python
from uutel import extract_tool_calls_from_response

# Extract tool calls from provider response
provider_response = {
    "choices": [{
        "message": {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}'
                    }
                }
            ]
        }
    }]
}

tool_calls = extract_tool_calls_from_response(provider_response)
```

### Complete Tool Calling Workflow

```python
import json
import litellm
from uutel import *

# Define tools
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}

# Tool function registry
def get_weather(location: str) -> dict:
    return {"location": location, "temperature": 22, "condition": "sunny"}

TOOLS = {"get_weather": get_weather}

# Make completion with tools
response = litellm.completion(
    model="my-custom-llm/codex-large",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[weather_tool]
)

# In a real implementation, extract and execute tool calls
# tool_calls = extract_tool_calls_from_response(response)
# for tool_call in tool_calls:
#     func_name = tool_call["function"]["name"]
#     args = json.loads(tool_call["function"]["arguments"])
#     result = TOOLS[func_name](**args)
#     tool_response = create_tool_call_response(
#         tool_call["id"], func_name, result
#     )
```

---

## Message Transformation

### Transform Messages Between Formats

```python
from uutel import (
    transform_openai_to_provider,
    transform_provider_to_openai
)

# OpenAI format messages
openai_messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]

# Transform to provider format
provider_messages = transform_openai_to_provider(
    openai_messages,
    "codex"
)

# Transform back
back_to_openai = transform_provider_to_openai(
    provider_messages,
    "codex"
)
```

### Advanced Message Handling

```python
# Handle different message types
complex_messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Analyze this image", "images": ["data:image/png;base64,..."]}
]

# Transform with context preservation
transformed = transform_openai_to_provider(
    complex_messages,
    "multimodal-provider",
    preserve_metadata=True
)
```

---

## Error Handling

UUTEL provides comprehensive error handling with detailed context information.

### Exception Hierarchy

```python
from uutel.core.exceptions import *

# Base exception
UUTELError  # Base class for all UUTEL errors

# Specific exception types
AuthenticationError  # Authentication failures
RateLimitError      # Rate limiting issues
ModelError          # Model-related errors
NetworkError        # Network connectivity issues
ValidationError     # Input validation errors
ProviderError       # Provider-specific errors
ConfigurationError  # Configuration issues
ToolCallError       # Tool calling errors
StreamingError      # Streaming-related errors
TimeoutError        # Request timeouts
QuotaExceededError  # Quota/usage limits
ModelNotFoundError  # Model not available
TokenLimitError     # Token limit exceeded
```

### Error Handling Examples

```python
from uutel.core.exceptions import *

try:
    response = litellm.completion(
        model="my-custom-llm/codex-large",
        messages=messages
    )
except AuthenticationError as e:
    print(f"Auth failed: {e.message}")
    print(f"Provider: {e.provider}")
    print(f"Auth method: {e.auth_method}")

except RateLimitError as e:
    print(f"Rate limited: {e.message}")
    print(f"Retry after: {e.retry_after} seconds")
    print(f"Quota type: {e.quota_type}")

except ModelError as e:
    print(f"Model error: {e.message}")
    print(f"Model: {e.model_name}")
    available = e.get_debug_info().get("available_models", [])
    print(f"Available models: {available}")

except NetworkError as e:
    print(f"Network error: {e.message}")
    if e.status_code:
        print(f"HTTP status: {e.status_code}")
    if e.should_retry():
        print("Retrying recommended")
```

### Error Context and Debugging

```python
# All UUTEL exceptions provide rich context
try:
    # Some operation
    pass
except UUTELError as e:
    # Basic info
    print(f"Error: {e.message}")
    print(f"Provider: {e.provider}")
    print(f"Error code: {e.error_code}")

    # Debug information
    debug_info = e.get_debug_info()
    print(f"Request ID: {debug_info.get('request_id')}")
    print(f"Timestamp: {debug_info.get('timestamp')}")

    # Add additional context
    e.add_context("operation", "completion_request")
    e.add_context("model", "codex-large")
```

### Helper Functions for Error Creation

```python
from uutel import (
    create_configuration_error,
    create_model_not_found_error,
    create_token_limit_error,
    create_network_error
)

# Create specific error types with context
config_error = create_configuration_error(
    "API key not found",
    config_key="CODEX_API_KEY",
    suggestions=["Set CODEX_API_KEY environment variable"]
)

model_error = create_model_not_found_error(
    "codex-ultra",
    available_models=["codex-large", "codex-mini"]
)

token_error = create_token_limit_error(
    used_tokens=8192,
    max_tokens=8191,
    suggestion="Reduce input length or use a model with higher token limit"
)
```

---

## Advanced Usage

### Async Operations

```python
import asyncio
from uutel.providers.codex.provider import CodexUU

async def async_completion_example():
    codex = CodexUU()

    # Async completion
    response = await codex.acompletion(
        model="codex-large",
        messages=[{"role": "user", "content": "Hello"}],
        # ... other parameters
    )

    # Async streaming
    async for chunk in codex.astreaming(
        model="codex-large",
        messages=[{"role": "user", "content": "Count to 10"}],
        # ... other parameters
    ):
        if chunk.get("text"):
            print(chunk["text"], end="")

# Run async operations
asyncio.run(async_completion_example())
```

### Custom HTTP Client Configuration

```python
from uutel import create_http_client, RetryConfig

# Create custom HTTP client
retry_config = RetryConfig(
    max_retries=5,
    base_delay=1.0,
    max_delay=60.0,
    backoff_factor=2.0,
    retry_on_status_codes=[429, 502, 503, 504]
)

http_client = create_http_client(
    timeout=30.0,
    retry_config=retry_config,
    is_async=True
)

# Use with provider
codex = CodexUU(http_client=http_client)
```

### Model Validation

```python
from uutel import validate_model_name, extract_provider_from_model

# Validate model names
is_valid = validate_model_name("codex-large")  # True
is_valid = validate_model_name("invalid$model")  # False

# Extract provider information
provider = extract_provider_from_model("uutel/codex/gpt-4")  # "codex"
provider = extract_provider_from_model("codex-large")  # None (no prefix)
```

### Environment Detection

```python
from uutel.core.utils import EnvironmentInfo

env_info = EnvironmentInfo.detect()
print(f"Platform: {env_info.platform}")
print(f"Python version: {env_info.python_version}")
print(f"In CI: {env_info.is_ci}")
print(f"Has Docker: {env_info.has_docker}")
print(f"Architecture: {env_info.arch}")
```

---

## Configuration

### Environment Variables

```bash
# Core configuration
UUTEL_LOG_LEVEL=INFO
UUTEL_DEBUG=false
UUTEL_TIMEOUT=30

# Provider-specific
CODEX_API_KEY=your-api-key
CODEX_API_BASE=https://api.openai.com/v1
CLAUDE_CODE_SESSION_TOKEN=session-token
GEMINI_CLI_API_KEY=api-key
CLOUD_CODE_PROJECT_ID=project-id

# Authentication
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

### Configuration Classes

```python
from uutel.core.config import Config

# Create configuration
config = Config(
    provider="codex",
    model="codex-large",
    api_key="your-key",
    timeout=30.0,
    max_retries=3,
    debug=True
)

# Access configuration
print(config.provider)  # "codex"
print(config.timeout)   # 30.0
```

### Logging Configuration

```python
from uutel.core.logging_config import get_logger

# Get provider-specific logger
logger = get_logger("uutel.providers.codex")

# Use in your code
logger.info("Starting completion request")
logger.debug("Request parameters: %s", params)
logger.error("Request failed: %s", error)
```

---

## API Reference Summary

### Core Modules

| Module | Description |
|--------|-------------|
| `uutel.core.base` | BaseUU class and core abstractions |
| `uutel.core.auth` | Authentication classes and utilities |
| `uutel.core.utils` | Utility functions for transformation and validation |
| `uutel.core.exceptions` | Exception classes and error handling |
| `uutel.core.config` | Configuration management |
| `uutel.core.logging_config` | Logging setup and utilities |

### Provider Modules

| Module | Description | Status |
|--------|-------------|---------|
| `uutel.providers.codex` | OpenAI Codex integration | ✅ Available |
| `uutel.providers.claude_code` | Anthropic Claude Code integration | 🚧 Planned |
| `uutel.providers.gemini_cli` | Google Gemini CLI integration | 🚧 Planned |
| `uutel.providers.cloud_code` | Google Cloud Code integration | 🚧 Planned |

### Key Functions

| Function | Module | Description |
|----------|---------|-------------|
| `validate_tool_schema()` | `uutel.core.utils` | Validate OpenAI tool schemas |
| `transform_openai_tools_to_provider()` | `uutel.core.utils` | Transform tools to provider format |
| `create_tool_call_response()` | `uutel.core.utils` | Create tool call response messages |
| `extract_tool_calls_from_response()` | `uutel.core.utils` | Extract tool calls from responses |
| `transform_openai_to_provider()` | `uutel.core.utils` | Transform messages to provider format |
| `create_http_client()` | `uutel.core.utils` | Create configured HTTP client |
| `validate_model_name()` | `uutel.core.utils` | Validate model name format |
| `get_logger()` | `uutel.core.logging_config` | Get configured logger instance |

---

## Version Compatibility

- **Python**: 3.10+ (tested on 3.10, 3.11, 3.12)
- **LiteLLM**: 1.74.0+
- **Async Support**: Full asyncio compatibility
- **Type Hints**: Complete type annotations with mypy support

## Performance Considerations

- **Provider Initialization**: < 100ms target
- **Message Transformation**: < 10ms target
- **Memory Usage**: Optimized for minimal overhead
- **Concurrent Requests**: Full async support for high throughput
- **Streaming**: Low-latency chunk processing
- **Tool Calling**: Efficient schema validation and transformation

For detailed examples and advanced usage patterns, see the [examples directory](examples/) in the repository.