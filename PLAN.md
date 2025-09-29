# UUTEL: Universal AI Provider for LiteLLM - Implementation Plan

## Project Overview

**UUTEL** (Universal Units for AI Telegraphy) is a Python package that extends LiteLLM's provider ecosystem by implementing custom providers for Claude Code, Gemini CLI, Google Cloud Code, and OpenAI Codex. It enables unified LLM inferencing through LiteLLM's standardized interface while leveraging the unique capabilities of each AI provider.

## Architecture Analysis

### AI SDK Provider Pattern Study

From analyzing the external AI SDK provider implementations, we identified these key patterns:

1. **Vercel AI SDK v5 Pattern**:
   - Provider factory functions (`createProvider()`)
   - Language model classes implementing `LanguageModelV2`
   - Transformation utilities for message/tool format conversion
   - Streaming support with `ReadableStream<LanguageModelV2StreamPart>`
   - Tool calling support with schema validation
   - Error handling with provider-specific error types

2. **LiteLLM Integration Pattern**:
   - Custom providers inherit from `BaseLLM` or use `CustomLLM`
   - Provider registration via `custom_provider_map`
   - Request/response transformation methods
   - Async/streaming support through specialized wrappers
   - Model routing based on provider prefixes

### Target Provider Functionality

Each provider will be ported from the corresponding AI SDK implementation:

1. **Claude Code** (`ai-sdk-provider-claude-code`):
   - OAuth authentication with MCP tool integration
   - Browser-based auth flow
   - Tool calling support

2. **Gemini CLI** (`ai-sdk-provider-gemini-cli`):
   - Multi-auth support (API key, Vertex AI, OAuth)
   - Gemini CLI Core integration
   - Advanced tool calling

3. **Cloud Code** (`cloud-code-ai-provider`):
   - Google Cloud Code API integration
   - Service account authentication
   - OAuth flow with Code Assist API

4. **Codex** (`codex-ai-provider`):
   - ChatGPT backend integration
   - Session token management
   - Codex CLI compatibility

## Implementation Plan

### Phase 1: Core Infrastructure (Days 1-2)

#### 1.1 Package Structure Setup
```
uutel/
├── __init__.py                 # Main exports and provider registration
├── core/
│   ├── __init__.py
│   ├── base.py                 # BaseUU class (inherits from LiteLLM BaseLLM)
│   ├── auth.py                 # Common authentication utilities
│   ├── exceptions.py           # Custom exception classes
│   └── utils.py                # Message transformation utilities
└── providers/                  # Provider implementations (Phase 2)
```

#### 1.2 Base Provider Class (`core/base.py`)
```python
from litellm.llms.base import BaseLLM
from typing import Optional, Dict, Any, AsyncIterator, Iterator
from litellm.types.utils import ModelResponse, GenericStreamingChunk

class BaseUU(BaseLLM):
    """Base class for all UUTEL providers following LiteLLM patterns."""

    def __init__(self, provider_name: str):
        super().__init__()
        self.provider_name = provider_name

    # Abstract methods to be implemented by each provider
    def completion(self, *args, **kwargs) -> ModelResponse:
        raise NotImplementedError

    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        raise NotImplementedError

    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        raise NotImplementedError

    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        raise NotImplementedError
```

#### 1.3 Authentication Framework (`core/auth.py`)
```python
from abc import ABC, abstractmethod
from typing import Dict, Optional

class BaseAuth(ABC):
    """Base authentication interface for all providers."""

    @abstractmethod
    async def get_headers(self) -> Dict[str, str]:
        pass

    @abstractmethod
    async def refresh_if_needed(self) -> bool:
        pass

class OAuthAuth(BaseAuth):
    """OAuth 2.0 authentication implementation."""
    pass

class ApiKeyAuth(BaseAuth):
    """API key authentication implementation."""
    pass

class ServiceAccountAuth(BaseAuth):
    """Service account authentication implementation."""
    pass
```

#### 1.4 Message Transformation (`core/utils.py`)
```python
from typing import List, Dict, Any, Optional
from litellm.types.utils import ModelResponse, GenericStreamingChunk

def transform_openai_to_provider(messages: List[Dict], provider_format: str) -> Any:
    """Transform OpenAI-style messages to provider-specific format."""
    pass

def transform_provider_to_openai(response: Any, provider_format: str) -> ModelResponse:
    """Transform provider response to OpenAI-compatible format."""
    pass

def handle_tool_calls(tools: Optional[List[Dict]], provider_format: str) -> Any:
    """Transform tool definitions to provider-specific format."""
    pass
```

### Phase 2: Provider Implementations (Days 3-6)

#### 2.1 Claude Code Provider (`providers/claude_code/`)
```
claude_code/
├── __init__.py
├── provider.py         # ClaudeCodeUU class
├── auth.py            # ClaudeCodeAuth class
├── transform.py       # Message transformation
└── types.py           # Provider-specific types
```

**Key Implementation Points**:
- Port OAuth authentication from `ai-sdk-provider-claude-code/src/client.ts`
- Implement MCP tool calling support
- Handle browser-based auth flow
- Transform messages using patterns from `ai-sdk-provider-claude-code/src/message-mapper.ts`

#### 2.2 Gemini CLI Provider (`providers/gemini_cli/`)
```
gemini_cli/
├── __init__.py
├── provider.py         # GeminiCLIUU class
├── auth.py            # Multi-auth support (API key, Vertex AI, OAuth)
├── transform.py       # Message/tool transformation
└── client.py          # Gemini CLI Core integration
```

**Key Implementation Points**:
- Port multi-auth from `ai-sdk-provider-gemini-cli/src/client.ts`
- Integrate with `@google/gemini-cli-core`
- Implement advanced tool calling from `ai-sdk-provider-gemini-cli/src/tool-mapper.ts`
- Handle different auth types (API key, Vertex AI, OAuth)

#### 2.3 Cloud Code Provider (`providers/cloud_code/`)
```
cloud_code/
├── __init__.py
├── provider.py         # CloudCodeUU class
├── auth.py            # OAuth + Service Account auth
├── transform.py       # Message transformation
└── api.py             # Code Assist API integration
```

**Key Implementation Points**:
- Port authentication from `cloud-code-ai-provider/src/google-cloud-code-auth.ts`
- Implement Code Assist API integration
- Handle OAuth flow with project setup
- Transform messages using patterns from `cloud-code-ai-provider/src/convert-to-cloud-code-messages.ts`

#### 2.4 Codex Provider (`providers/codex/`)
```
codex/
├── __init__.py
├── provider.py         # CodexUU class
├── auth.py            # Session token management
├── transform.py       # Message transformation
└── streaming.py       # Codex-specific streaming
```

**Key Implementation Points**:
- Port authentication from `codex-ai-provider/src/codex-auth.ts`
- Implement session token management and refresh
- Handle Codex CLI compatibility
- Implement streaming from `codex-ai-provider/src/codex-language-model.ts`

### Phase 3: LiteLLM Integration (Day 7)

#### 3.1 Provider Registration (`__init__.py`)
```python
import litellm
from .providers.claude_code import ClaudeCodeUU
from .providers.gemini_cli import GeminiCLIUU
from .providers.cloud_code import CloudCodeUU
from .providers.codex import CodexUU

# Register providers with LiteLLM
litellm.custom_provider_map = [
    {"provider": "claude-code", "custom_handler": ClaudeCodeUU()},
    {"provider": "gemini-cli", "custom_handler": GeminiCLIUU()},
    {"provider": "cloud-code", "custom_handler": CloudCodeUU()},
    {"provider": "codex", "custom_handler": CodexUU()},
]

# Model routing configuration
def setup_model_routing():
    """Configure model routing for UUTEL providers."""
    pass
```

#### 3.2 Provider Factory Functions
```python
def create_claude_code_provider(auth_config: Optional[Dict] = None) -> ClaudeCodeUU:
    """Create Claude Code provider instance."""
    pass

def create_gemini_cli_provider(auth_type: str = "oauth", **kwargs) -> GeminiCLIUU:
    """Create Gemini CLI provider instance."""
    pass

def create_cloud_code_provider(project_id: Optional[str] = None) -> CloudCodeUU:
    """Create Cloud Code provider instance."""
    pass

def create_codex_provider() -> CodexUU:
    """Create Codex provider instance."""
    pass
```

### Phase 4: Examples and Documentation (Day 8)

#### 4.1 Usage Examples (`examples/`)
```python
# examples/basic_usage.py
import litellm
from uutel import setup_providers

# Setup UUTEL providers
setup_providers()

# Use Claude Code
response = litellm.completion(
    model="uutel/claude-code/claude-3-5-sonnet",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Use Gemini CLI
response = litellm.completion(
    model="uutel/gemini-cli/gemini-2.0-flash-exp",
    messages=[{"role": "user", "content": "Generate code"}],
    stream=True
)

# Use Cloud Code
response = litellm.completion(
    model="uutel/cloud-code/gemini-2.5-pro",
    messages=[{"role": "user", "content": "Review this PR"}]
)

# Use Codex
response = litellm.completion(
    model="uutel/codex/gpt-4o",
    messages=[{"role": "user", "content": "Explain this algorithm"}],
    max_tokens=4000
)
```

#### 4.2 Streaming Example (`examples/streaming_example.py`)
```python
import asyncio
from uutel import create_claude_code_provider

async def streaming_example():
    provider = create_claude_code_provider()

    async for chunk in provider.astreaming(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Write a story"}]
    ):
        print(chunk.delta, end="")

asyncio.run(streaming_example())
```

#### 4.3 Tool Calling Example (`examples/tool_calling_example.py`)
```python
import litellm
from uutel import setup_providers

setup_providers()

def get_weather(location: str) -> str:
    return f"Weather in {location}: Sunny, 72°F"

response = litellm.completion(
    model="uutel/gemini-cli/gemini-2.0-flash-exp",
    messages=[{"role": "user", "content": "What's the weather in SF?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }],
    tool_choice="auto"
)
```

### Phase 5: Testing and Validation (Day 9)

#### 5.1 Core Tests (`tests/`)
```
tests/
├── conftest.py
├── test_base.py           # Test BaseUU functionality
├── test_auth.py           # Test authentication classes
├── test_utils.py          # Test transformation utilities
├── test_exceptions.py     # Test error handling
├── test_providers_init.py # Test provider initialization
├── test_claude_code.py    # Claude Code provider tests
├── test_gemini_cli.py     # Gemini CLI provider tests
├── test_cloud_code.py     # Cloud Code provider tests
├── test_codex.py          # Codex provider tests
└── test_integration.py    # LiteLLM integration tests
```

#### 5.2 Test Strategy
- **Unit Tests**: Test each provider class individually
- **Integration Tests**: Test LiteLLM integration and routing
- **Authentication Tests**: Test auth flows (mocked)
- **Transformation Tests**: Test message/tool format conversion
- **Streaming Tests**: Test async streaming functionality
- **Tool Calling Tests**: Test function calling capabilities

### Phase 6: Package Distribution (Day 10)

#### 6.1 Package Configuration
- **pyproject.toml**: Modern Python packaging
- **Dependencies**: Minimal required dependencies
- **Optional Dependencies**: Provider-specific extras
- **Entry Points**: CLI utilities if needed

#### 6.2 Documentation
- **README.md**: Installation and basic usage
- **DEPENDENCIES.md**: Rationale for each dependency
- **API Documentation**: Provider-specific configuration
- **Migration Guide**: From existing provider implementations

## Technical Specifications

### Dependencies Strategy
```toml
[project]
dependencies = [
    "litellm>=1.44.0",
    "httpx>=0.25.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
claude-code = ["browser-cookie3>=0.19.1"]
gemini-cli = ["google-auth>=2.23.0", "google-auth-oauthlib>=1.1.0"]
cloud-code = ["google-cloud-core>=2.3.0"]
codex = ["cryptography>=41.0.0"]
all = ["uutel[claude-code,gemini-cli,cloud-code,codex]"]
dev = ["pytest>=7.0.0", "pytest-asyncio>=0.21.0", "pytest-cov>=4.1.0"]
```

### Model Naming Convention
- **Claude Code**: `uutel/claude-code/{model-name}`
- **Gemini CLI**: `uutel/gemini-cli/{model-name}`
- **Cloud Code**: `uutel/cloud-code/{model-name}`
- **Codex**: `uutel/codex/{model-name}`

### Authentication Configuration
```python
# Environment variables
CLAUDE_CODE_AUTH_DIR=~/.claude
GEMINI_CLI_AUTH_TYPE=oauth  # or api-key, vertex-ai
GEMINI_API_KEY=your-key
GOOGLE_CLOUD_PROJECT=your-project
CODEX_AUTH_FILE=~/.codex/auth.json
OPENAI_API_KEY=fallback-key
```

## Success Criteria

### Functional Requirements
1. ✅ All four providers successfully registered with LiteLLM
2. ✅ Completion and streaming work for each provider
3. ✅ Tool calling functions correctly where supported
4. ✅ Authentication flows work (OAuth, API key, service account)
5. ✅ Error handling provides clear, actionable messages
6. ✅ Message transformation maintains fidelity across providers

### Performance Requirements
1. ✅ Provider initialization < 100ms
2. ✅ Request transformation < 10ms
3. ✅ No memory leaks in streaming scenarios
4. ✅ Graceful handling of network timeouts

### Code Quality Requirements
1. ✅ 100% type coverage with mypy
2. ✅ >90% test coverage
3. ✅ All functions < 20 lines
4. ✅ All files < 200 lines
5. ✅ No enterprise patterns or abstractions
6. ✅ Clear, readable code with minimal complexity

### Integration Requirements
1. ✅ Works with standard LiteLLM Router
2. ✅ Compatible with LiteLLM proxy server
3. ✅ Supports LiteLLM's logging and monitoring hooks
4. ✅ Handles LiteLLM's error retry mechanisms
5. ✅ Works with LiteLLM's cost tracking features

## Development Guidelines

### Code Principles
1. **Simplicity First**: Choose simple solutions over complex ones
2. **Port, Don't Reinvent**: Base implementations on existing AI SDK providers
3. **LiteLLM Patterns**: Follow established LiteLLM provider patterns
4. **Minimal Dependencies**: Only add dependencies when absolutely necessary
5. **Test Coverage**: Every public method must have tests
6. **Documentation**: Every provider needs usage examples

### Anti-Patterns to Avoid
1. ❌ Complex configuration systems
2. ❌ Enterprise monitoring or health checks
3. ❌ Performance benchmarking or profiling
4. ❌ Security hardening beyond basic input validation
5. ❌ Abstract factories or dependency injection
6. ❌ Sophisticated logging frameworks
7. ❌ Caching layers or optimization systems

This plan provides a clear, step-by-step approach to building UUTEL as a focused, simple LiteLLM provider package that ports functionality from existing AI SDK providers while maintaining the simplicity and usability that users expect.