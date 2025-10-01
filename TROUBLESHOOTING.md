# UUTEL Troubleshooting Guide

This guide helps you diagnose and resolve common issues when using UUTEL (Universal Units for AI Telegraphy).

## Table of Contents

- [Quick Diagnostic Steps](#quick-diagnostic-steps)
- [Installation Issues](#installation-issues)
- [Provider Registration Problems](#provider-registration-problems)
- [Authentication Failures](#authentication-failures)
- [Model and Completion Issues](#model-and-completion-issues)
- [Streaming Problems](#streaming-problems)
- [Tool Calling Issues](#tool-calling-issues)
- [Performance Issues](#performance-issues)
- [Network and Connectivity](#network-and-connectivity)
- [Development and Testing](#development-and-testing)
- [Logging and Debugging](#logging-and-debugging)
- [Common Error Messages](#common-error-messages)
- [Getting Help](#getting-help)

---

## Quick Diagnostic Steps

### 1. Verify Installation

```bash
python -c "import uutel; print(uutel.__version__)"
```

**Expected Output:** Version number (e.g., `1.0.5`)

**If it fails:**
- Install UUTEL: `pip install uutel` or `uv add uutel`
- Check Python version: `python --version` (requires 3.10+)

### 2. Test Basic Functionality

```python
from uutel.providers.codex.custom_llm import CodexCustomLLM
import litellm

# Basic test
codex = CodexCustomLLM()
print(f"Provider: {codex.provider_name}")
print(f"Models: {len(codex.supported_models)}")
```

**Expected Output:**
```
Provider: codex
Models: 6
```

### 3. Test LiteLLM Integration

```python
import litellm
from uutel.providers.codex.custom_llm import CodexCustomLLM

litellm.custom_provider_map = [
    {"provider": "test-provider", "custom_handler": CodexCustomLLM()}
]

response = litellm.completion(
    model="test-provider/codex-large",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=10
)

print("✅ Integration working!")
```

---

## Installation Issues

### Problem: ImportError or ModuleNotFoundError

```python
ImportError: No module named 'uutel'
```

**Solutions:**

1. **Install UUTEL:**
   ```bash
   pip install uutel
   # or with uv
   uv add uutel
   ```

2. **Check Python environment:**
   ```bash
   python -m pip list | grep uutel
   ```

3. **Install with specific provider:**
   ```bash
   pip install uutel[codex]
   pip install uutel[providers]  # All providers
   ```

### Problem: Version Conflicts

```
ERROR: pip's dependency resolver does not currently have a complete picture
```

**Solutions:**

1. **Use uv (recommended):**
   ```bash
   uv add uutel
   ```

2. **Update pip and try again:**
   ```bash
   pip install --upgrade pip
   pip install uutel
   ```

3. **Use virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or venv\Scripts\activate  # Windows
   pip install uutel
   ```

### Problem: Python Version Compatibility

```
Requires-Python: >=3.10
```

**Solution:**
- Upgrade Python to 3.10 or later
- Use pyenv or conda to manage Python versions

---

## Provider Registration Problems

### Problem: Provider Not Recognized

```python
litellm.exceptions.BadRequestError: LLM Provider NOT provided
```

**Check Registration:**
```python
import litellm
print(litellm.custom_provider_map)
```

**Correct Registration:**
```python
from uutel.providers.codex.custom_llm import CodexCustomLLM

litellm.custom_provider_map = [
    {"provider": "my-custom-llm", "custom_handler": CodexCustomLLM()},
]

# Test immediately
response = litellm.completion(
    model="my-custom-llm/codex-large",  # Use registered provider name
    messages=[{"role": "user", "content": "test"}]
)
```

### Problem: Multiple Provider Conflicts

**Solution:**
```python
# Clear existing registrations
litellm.custom_provider_map = []

# Register only what you need
litellm.custom_provider_map = [
    {"provider": "uutel-codex", "custom_handler": CodexCustomLLM()},
]
```

### Problem: Provider Name Validation

```
'my-provider-name' is not a valid LlmProviders
```

**Solutions:**
1. **Use alphanumeric names only:**
   ```python
   {"provider": "uutel", "custom_handler": handler}  # Good
   {"provider": "my-provider", "custom_handler": handler}  # May fail
   ```

2. **Use custom model names:**
   ```python
   # Instead of real model names
   model="uutel/custom-model"  # Good
   # Don't use
   model="uutel/gpt-4"  # May trigger validation errors
   ```

---

## Authentication Failures

### Problem: API Key Issues

```python
AuthenticationError: Invalid API key
```

**Check Environment Variables:**
```bash
echo $CODEX_API_KEY
echo $OPENAI_API_KEY
echo $GOOGLE_APPLICATION_CREDENTIALS
```

**Set Environment Variables:**
```bash
# Linux/Mac
export CODEX_API_KEY="your-api-key"

# Windows
set CODEX_API_KEY=your-api-key
```

**Programmatic Configuration:**
```python
from uutel.core.auth import ApiKeyAuth

auth = ApiKeyAuth(api_key="your-api-key")
headers = await auth.get_headers()
```

### Problem: OAuth Flow Issues

```python
AuthenticationError: OAuth authentication failed
```

**Debugging OAuth:**
```python
from uutel.core.auth import OAuthAuth

oauth = OAuthAuth(
    client_id="your-id",
    client_secret="your-secret",
    authorization_url="https://provider.com/oauth/authorize",
    token_url="https://provider.com/oauth/token"
)

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

result = await oauth.authenticate()
print(f"Success: {result.success}")
if not result.success:
    print(f"Error: {result.error_message}")
```

### Problem: Token Expiry

**Check Token Status:**
```python
auth_result = await auth_handler.authenticate()
if auth_result.expires_at:
    from datetime import datetime
    expires = datetime.fromisoformat(auth_result.expires_at)
    if expires < datetime.now():
        print("Token expired - refreshing...")
        await auth_handler.refresh_token()
```

---

## Model and Completion Issues

### Problem: Model Not Found

```python
ModelNotFoundError: Model 'invalid-model' not found
```

**Check Available Models:**
```python
from uutel.providers.codex.custom_llm import CodexCustomLLM

codex = CodexCustomLLM()
print("Available models:")
for model in codex.supported_models:
    print(f"  - {model}")
```

**Use Correct Model Names:**
```python
# Correct usage
response = litellm.completion(
    model="my-custom-llm/codex-large",  # Use supported model
    messages=messages
)
```

### Problem: Empty or Invalid Responses

**Debug Response Structure:**
```python
response = litellm.completion(
    model="my-custom-llm/codex-large",
    messages=[{"role": "user", "content": "Hello"}]
)

print(f"Response type: {type(response)}")
print(f"Model: {response.model}")
print(f"Choices: {len(response.choices)}")
print(f"Content: {response.choices[0].message.content}")
```

### Problem: Token Limit Errors

```python
TokenLimitError: Token limit exceeded
```

**Solutions:**
1. **Reduce message length:**
   ```python
   # Truncate long messages
   content = long_content[:1000] + "..." if len(long_content) > 1000 else long_content
   ```

2. **Use appropriate max_tokens:**
   ```python
   response = litellm.completion(
       model="my-custom-llm/codex-large",
       messages=messages,
       max_tokens=500  # Adjust based on needs
   )
   ```

3. **Count tokens beforehand:**
   ```python
   import tiktoken

   encoding = tiktoken.encoding_for_model("gpt-4")
   token_count = len(encoding.encode("Your message"))
   print(f"Token count: {token_count}")
   ```

---

## Streaming Problems

### Problem: Streaming Not Working

```python
response = litellm.completion(
    model="my-custom-llm/codex-large",
    messages=messages,
    stream=True
)

# No chunks received or error
```

**Debug Streaming:**
```python
response = litellm.completion(
    model="my-custom-llm/codex-large",
    messages=[{"role": "user", "content": "Count from 1 to 3"}],
    stream=True
)

print(f"Response type: {type(response)}")

chunk_count = 0
for chunk in response:
    chunk_count += 1
    print(f"Chunk {chunk_count}: {type(chunk)}")
    if hasattr(chunk, 'choices') and chunk.choices:
        delta = chunk.choices[0].delta
        if hasattr(delta, 'content') and delta.content:
            print(f"Content: {delta.content}")

print(f"Total chunks: {chunk_count}")
```

### Problem: GenericStreamingChunk Format Issues

```python
AttributeError: 'dict' object has no attribute 'choices'
```

**Understanding Chunk Format:**

UUTEL uses `GenericStreamingChunk` format:
```python
{
    "text": "content here",
    "finish_reason": None,
    "index": 0,
    "is_finished": False,
    "tool_use": None,
    "usage": {"completion_tokens": 1, "prompt_tokens": 0, "total_tokens": 1}
}
```

**Correct Streaming Handling:**
```python
for chunk in response:
    if isinstance(chunk, dict):
        # GenericStreamingChunk format
        if "text" in chunk and chunk["text"]:
            print(chunk["text"], end="")
    else:
        # OpenAI format (if using different provider)
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
```

### Problem: Async Streaming Issues

**Correct Async Pattern:**
```python
import asyncio

async def stream_example():
    response = await litellm.acompletion(
        model="my-custom-llm/codex-large",
        messages=messages,
        stream=True
    )

    async for chunk in response:
        if isinstance(chunk, dict) and "text" in chunk:
            print(chunk["text"], end="")

asyncio.run(stream_example())
```

---

## Tool Calling Issues

### Problem: Tool Schema Validation Fails

```python
ValidationError: Invalid tool schema
```

**Validate Tool Schema:**
```python
from uutel import validate_tool_schema

tool = {
    "type": "function",  # Required
    "function": {
        "name": "get_weather",  # Required
        "description": "Get weather info",  # Required
        "parameters": {  # Optional but recommended
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}

is_valid = validate_tool_schema(tool)
print(f"Tool valid: {is_valid}")

if not is_valid:
    print("Check required fields: type, function.name, function.description")
```

### Problem: Tool Calls Not Extracted

```python
tool_calls = extract_tool_calls_from_response(response)
print(f"Found {len(tool_calls)} tool calls")  # Returns 0
```

**Debug Tool Call Extraction:**
```python
from uutel import extract_tool_calls_from_response

# Check response structure
print("Response structure:")
print(f"Type: {type(response)}")
if hasattr(response, 'choices'):
    print(f"Choices: {len(response.choices)}")
    choice = response.choices[0]
    print(f"Message: {choice.message}")
    if hasattr(choice.message, 'tool_calls'):
        print(f"Tool calls: {choice.message.tool_calls}")

# Create example output for testing
mock_response = {
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

tool_calls = extract_tool_calls_from_response(mock_response)
print(f"Mock tool calls: {len(tool_calls)}")
```

### Problem: Tool Function Execution Errors

**Safe Tool Execution:**
```python
import json
from uutel import create_tool_call_response

def safe_execute_tool(tool_call, tool_functions):
    """Safely execute a tool call with error handling."""
    try:
        func_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])

        if func_name not in tool_functions:
            return create_tool_call_response(
                tool_call["id"],
                func_name,
                error=f"Unknown function: {func_name}"
            )

        result = tool_functions[func_name](**arguments)
        return create_tool_call_response(
            tool_call["id"],
            func_name,
            function_result=result
        )

    except json.JSONDecodeError as e:
        return create_tool_call_response(
            tool_call["id"],
            tool_call["function"]["name"],
            error=f"Invalid JSON arguments: {e}"
        )
    except Exception as e:
        return create_tool_call_response(
            tool_call["id"],
            tool_call["function"]["name"],
            error=f"Function execution failed: {e}"
        )

# Usage
tool_functions = {
    "get_weather": lambda location: {"temp": 22, "location": location}
}

for tool_call in tool_calls:
    response = safe_execute_tool(tool_call, tool_functions)
    print(f"Tool response: {response}")
```

---

## Performance Issues

### Problem: Slow Response Times

**Profile Performance:**
```python
import time
from uutel.providers.codex.custom_llm import CodexCustomLLM

start_time = time.time()

codex = CodexCustomLLM()
response = codex.completion(
    model="codex-large",
    messages=[{"role": "user", "content": "Hello"}]
)

end_time = time.time()
print(f"Response time: {end_time - start_time:.2f} seconds")
```

**Optimize with Async:**
```python
import asyncio
import time

async def async_completions():
    start = time.time()

    tasks = [
        codex.acompletion(
            model="codex-large",
            messages=[{"role": "user", "content": f"Hello {i}"}]
        )
        for i in range(5)
    ]

    responses = await asyncio.gather(*tasks)
    end = time.time()

    print(f"5 async requests: {end - start:.2f} seconds")
    return responses

asyncio.run(async_completions())
```

### Problem: Memory Usage

**Monitor Memory:**
```python
import psutil
import os

process = psutil.Process(os.getpid())

print(f"Memory before: {process.memory_info().rss / 1024 / 1024:.1f} MB")

# Your UUTEL operations here
response = litellm.completion(
    model="my-custom-llm/codex-large",
    messages=[{"role": "user", "content": "Hello"}]
)

print(f"Memory after: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

### Problem: High CPU Usage

**Optimize Tool Validation:**
```python
from uutel import validate_tool_schema

# Cache validation results
validation_cache = {}

def cached_validate_tool(tool):
    tool_hash = hash(str(tool))
    if tool_hash not in validation_cache:
        validation_cache[tool_hash] = validate_tool_schema(tool)
    return validation_cache[tool_hash]
```

---

## Network and Connectivity

### Problem: Connection Timeouts

```python
NetworkError: Request timeout
```

**Configure Timeout:**
```python
from uutel import create_http_client, RetryConfig

# Custom timeout and retry
http_client = create_http_client(
    timeout=60.0,  # 60 second timeout
    retry_config=RetryConfig(
        max_retries=3,
        base_delay=1.0,
        backoff_factor=2.0
    )
)
```

**Environment Configuration:**
```bash
export UUTEL_TIMEOUT=60
export UUTEL_MAX_RETRIES=5
```

### Problem: SSL Certificate Issues

```python
SSL: CERTIFICATE_VERIFY_FAILED
```

**Solutions:**
```python
import ssl
import httpx

# For development only - NOT for production
client = httpx.AsyncClient(
    verify=False  # Disable SSL verification
)

# Or specify custom CA bundle
client = httpx.AsyncClient(
    verify="/path/to/cacert.pem"
)
```

### Problem: Proxy Configuration

**Configure Proxy:**
```python
import httpx

proxies = {
    "http://": "http://proxy.company.com:8080",
    "https://": "http://proxy.company.com:8080"
}

client = httpx.AsyncClient(proxies=proxies)
```

**Environment Variables:**
```bash
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
```

---

## Development and Testing

### Problem: Tests Failing

**Run Specific Tests:**
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_codex_provider.py

# Run with verbose output
uv run pytest -v tests/

# Run with coverage
uv run pytest --cov=src/uutel tests/
```

**Debug Test Failures:**
```python
import pytest

# Run single test with debugging
pytest.main(["-v", "-s", "tests/test_codex_provider.py::TestCodexUU::test_basic"])
```

### Problem: Import Errors in Tests

```python
ImportError: No module named 'uutel'
```

**Install in Development Mode:**
```bash
# With uv
uv pip install -e .

# With pip
pip install -e .

# Or with all extras
uv pip install -e .[full]
```

### Problem: Async Test Issues

```python
RuntimeError: asyncio.run() cannot be called from a running event loop
```

**Use Proper Async Testing:**
```python
import asyncio
import pytest

# Method 1: pytest-asyncio
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None

# Method 2: Manual asyncio
def test_async_function_manual():
    async def run_test():
        return await some_async_function()

    result = asyncio.run(run_test())
    assert result is not None
```

---

## Logging and Debugging

### Enable Debug Logging

```python
import logging
from uutel.core.logging_config import get_logger

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Get UUTEL logger
logger = get_logger("uutel")
logger.setLevel(logging.DEBUG)

# Test logging
logger.debug("Debug message")
logger.info("Info message")
```

**Environment Variable:**
```bash
export UUTEL_LOG_LEVEL=DEBUG
```

### Problem: No Log Output

**Check Logger Configuration:**
```python
from uutel.core.logging_config import get_logger

logger = get_logger("uutel.providers.codex")
print(f"Logger level: {logger.level}")
print(f"Handlers: {logger.handlers}")

# Force a test message
logger.error("Test error message - should appear")
```

### Problem: Too Verbose Logging

**Filter Logs:**
```python
import logging

# Suppress specific loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

# Only show UUTEL logs
logging.getLogger("uutel").setLevel(logging.INFO)
```

---

## Common Error Messages

### "LiteLLM Provider NOT provided"

**Cause:** Provider not registered or incorrect model format
**Solution:** Check provider registration and model name format

### "CustomLLM completion failed"

**Cause:** Error in provider implementation
**Solution:** Check provider logs and implementation

### "GenericStreamingChunk AttributeError"

**Cause:** Treating dict as object
**Solution:** Use dict-style access: `chunk["text"]` not `chunk.text`

### "Token limit exceeded"

**Cause:** Input too long for model
**Solution:** Reduce input length or increase max_tokens

### "Authentication failed"

**Cause:** Invalid credentials or expired tokens
**Solution:** Check API keys and token expiry

### "Codex request forbidden (HTTP 403)"

**Cause:** Codex session token expired or missing required permissions
**Solution:** Run `codex login` to refresh the CLI session or set `OPENAI_API_KEY`

### "Codex rate limit reached (HTTP 429)"

**Cause:** Too many Codex requests in a short period
**Solution:** Wait for the reported `Retry-After` interval (printed in the error message) before retrying, or reduce request concurrency

### "Codex service unavailable (HTTP 5xx)"

**Cause:** Temporary outage or maintenance on the Codex backend
**Solution:** Retry after a short delay; use `uutel diagnostics` to confirm local configuration before retrying

### "Model not found"

**Cause:** Unsupported model name
**Solution:** Use supported model names from provider

---

## Getting Help

### Check Documentation

1. **API Reference:** [API.md](API.md)
2. **Examples:** [examples/](examples/)
3. **Source Code:** Browse `src/uutel/` for implementation details

### Enable Comprehensive Debug Logging

```python
import logging
import sys

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Your UUTEL code here with full debug output
```

### Create Minimal Reproduction Case

```python
"""Minimal example demonstrating the issue"""
import litellm
from uutel.providers.codex.custom_llm import CodexCustomLLM

# Setup
litellm.custom_provider_map = [
    {"provider": "test", "custom_handler": CodexCustomLLM()}
]

# The problematic code
try:
    response = litellm.completion(
        model="test/codex-large",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

### Report Issues

When reporting issues, please include:

1. **UUTEL version:** `python -c "import uutel; print(uutel.__version__)"`
2. **Python version:** `python --version`
3. **LiteLLM version:** `python -c "import litellm; print(litellm.__version__)"`
4. **Operating system:** `uname -a` (Linux/Mac) or `systeminfo` (Windows)
5. **Complete error traceback**
6. **Minimal reproduction code**
7. **Expected vs actual behavior**

### Community Support

- **GitHub Issues:** [https://github.com/twardoch/uutel/issues](https://github.com/twardoch/uutel/issues)
- **Documentation:** [README.md](README.md)
- **Examples:** [examples/](examples/)

Remember: The more specific information you provide, the faster we can help resolve your issue!
