# UUTEL: Universal AI Provider for LiteLLM

## Project Overview

**UUTEL** is a Python package that extends LiteLLM's provider ecosystem by implementing custom providers for Claude Code, Gemini CLI, Google Cloud Code, and OpenAI Codex. It enables unified LLM inferencing through LiteLLM's standardized interface while leveraging the unique capabilities of each AI provider.

## Technical Architecture

### Core Design Principles

1. **LiteLLM Compatibility**: Full adherence to LiteLLM's provider interface patterns
2. **Unified API**: Consistent OpenAI-compatible interface across all providers
3. **Authentication Management**: Secure handling of different auth mechanisms (OAuth, API keys, service accounts)
4. **Streaming Support**: Real-time response streaming for all providers
5. **Tool Calling**: Function calling capabilities where supported
6. **Error Handling**: Robust error mapping and fallback mechanisms

### Naming Convention

**IMPORTANT**: All provider classes follow the pattern `{ProviderName}UU` where `UU` stands for "Universal Unit":

- **Base class**: `BaseUU` (extends LiteLLM's `BaseLLM`)
- **Claude Code provider**: `ClaudeCodeUU`
- **Gemini CLI provider**: `GeminiCLIUU`
- **Cloud Code provider**: `CloudCodeUU`
- **Codex provider**: `CodexUU`

This naming convention:
- Keeps class names concise and memorable
- Maintains consistency across all providers
- Clearly identifies UUTEL providers in the codebase
- Avoids conflicts with existing provider names

### Implementation Guidelines

**File Structure Pattern**:
```
providers/{provider_name}/
├── __init__.py              # Exports: {ProviderName}UU, {ProviderName}Auth
├── provider.py              # Main class: {ProviderName}UU(BaseUU)
├── auth.py                  # Auth class: {ProviderName}Auth
├── transforms.py            # Transform class: {ProviderName}Transform
└── models.py                # Model classes: {ProviderName}Request, {ProviderName}Response
```

**Class Naming Examples**:
```python
# Base infrastructure
class BaseUU(BaseLLM):                    # core/base.py

# Claude Code provider
class ClaudeCodeUU(BaseUU):               # providers/claude_code/provider.py
class ClaudeCodeAuth:                     # providers/claude_code/auth.py
class ClaudeCodeTransform:                # providers/claude_code/transforms.py
class ClaudeCodeRequest:                  # providers/claude_code/models.py
class ClaudeCodeResponse:                 # providers/claude_code/models.py

# Gemini CLI provider
class GeminiCLIUU(BaseUU):                # providers/gemini_cli/provider.py
class GeminiCLIAuth:                      # providers/gemini_cli/auth.py
class GeminiCLITransform:                 # providers/gemini_cli/transforms.py

# Cloud Code provider
class CloudCodeUU(BaseUU):                # providers/cloud_code/provider.py
class CloudCodeAuth:                      # providers/cloud_code/auth.py

# Codex provider
class CodexUU(BaseUU):                    # providers/codex/provider.py
class CodexAuth:                          # providers/codex/auth.py
```

**Import Pattern**:
```python
# Main package exports
from uutel.providers.claude_code import ClaudeCodeUU
from uutel.providers.gemini_cli import GeminiCLIUU
from uutel.providers.cloud_code import CloudCodeUU
from uutel.providers.codex import CodexUU

# Usage in LiteLLM registration
litellm.custom_provider_map = [
    {"provider": "claude-code", "custom_handler": ClaudeCodeUU()},
    {"provider": "gemini-cli", "custom_handler": GeminiCLIUU()},
    {"provider": "cloud-code", "custom_handler": CloudCodeUU()},
    {"provider": "codex", "custom_handler": CodexUU()},
]
```

### Package Structure

```
uutel/
├── __init__.py                 # Main exports and provider registration
├── core/
│   ├── __init__.py
│   ├── base.py                 # Base provider classes and interfaces
│   ├── auth.py                 # Common authentication utilities
│   ├── exceptions.py           # Custom exception classes
│   └── utils.py                # Common utilities and helpers
├── providers/
│   ├── __init__.py
│   ├── claude_code/
│   │   ├── __init__.py
│   │   ├── provider.py         # Claude Code provider implementation
│   │   ├── auth.py             # Claude Code authentication
│   │   ├── models.py           # Response/request models
│   │   └── transforms.py       # Message transformation logic
│   ├── gemini_cli/
│   │   ├── __init__.py
│   │   ├── provider.py         # Gemini CLI provider implementation
│   │   ├── auth.py             # Gemini CLI authentication
│   │   ├── models.py           # Response/request models
│   │   └── transforms.py       # Message transformation logic
│   ├── cloud_code/
│   │   ├── __init__.py
│   │   ├── provider.py         # Google Cloud Code provider implementation
│   │   ├── auth.py             # Cloud Code authentication
│   │   ├── models.py           # Response/request models
│   │   └── transforms.py       # Message transformation logic
│   └── codex/
│       ├── __init__.py
│       ├── provider.py         # OpenAI Codex provider implementation
│       ├── auth.py             # Codex authentication
│       ├── models.py           # Response/request models
│       └── transforms.py       # Message transformation logic
├── tests/
│   ├── __init__.py
│   ├── test_claude_code.py
│   ├── test_gemini_cli.py
│   ├── test_cloud_code.py
│   ├── test_codex.py
│   └── conftest.py             # Pytest configuration
└── examples/
    ├── basic_usage.py
    ├── streaming_example.py
    ├── tool_calling_example.py
    └── auth_examples.py
```

## Implementation Strategy

### Phase 1: Core Infrastructure (Foundation)

#### 1.1 Base Classes and Interfaces
- Implement `BaseUU` extending LiteLLM's `BaseLLM`
- Define common interfaces for authentication, request/response transformation
- Create standardized error handling and logging framework
- Implement utility functions for message format conversion

#### 1.2 Authentication Framework
- OAuth 2.0 handler for Claude Code and Cloud Code
- API key management for Gemini CLI
- Token refresh and caching mechanisms
- Secure credential storage and retrieval

#### 1.3 Core Utilities
- HTTP client wrapper with retry logic
- Response streaming utilities
- Message format transformation helpers
- Tool/function calling adapters

### Phase 2: Provider Implementations

#### 2.1 Claude Code Provider
**Based on**: AI SDK provider pattern from `ai-sdk-provider-claude-code`

**Key Features**:
- MCP (Model Context Protocol) tool integration
- OAuth authentication with browser-based flow
- Streaming chat completions
- Support for Claude 3.5 Sonnet and Haiku models

**Implementation Details**:
```python
class ClaudeCodeUU(BaseUU):
    provider_name = "claude-code"
    supported_models = ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]

    def __init__(self, api_key=None, **kwargs):
        self.auth_manager = ClaudeCodeAuth(api_key=api_key)
        super().__init__(**kwargs)

    async def completion(self, model, messages, **kwargs):
        # Transform messages to Claude Code format
        # Handle streaming and non-streaming requests
        # Map responses back to LiteLLM format
        pass
```

#### 2.2 Gemini CLI Provider
**Based on**: AI SDK provider pattern from `ai-sdk-provider-gemini-cli`

**Key Features**:
- Multiple authentication methods (API key, Vertex AI, OAuth)
- Function calling with tool use
- Streaming responses with proper chunk handling
- Support for Gemini 2.0 Flash and Pro models

**Implementation Details**:
```python
class GeminiCLIUU(BaseUU):
    provider_name = "gemini-cli"
    supported_models = ["gemini-2.0-flash-exp", "gemini-1.5-pro-latest"]

    def __init__(self, auth_type="api-key", **kwargs):
        self.auth_manager = GeminiCLIAuth(auth_type=auth_type, **kwargs)
        super().__init__(**kwargs)

    async def completion(self, model, messages, **kwargs):
        # Use @google/gemini-cli-core patterns
        # Handle multiple auth types
        # Process tool calling and streaming
        pass
```

#### 2.3 Google Cloud Code Provider
**Based on**: Cloud Code AI provider implementation

**Key Features**:
- Google Cloud authentication with service accounts
- Code Assist API integration
- Project-based model access
- Safety settings and content filtering

**Implementation Details**:
```python
class CloudCodeUU(BaseUU):
    provider_name = "cloud-code"
    supported_models = ["gemini-2.5-flash", "gemini-2.5-pro"]

    def __init__(self, project_id=None, **kwargs):
        self.auth_manager = CloudCodeAuth(project_id=project_id)
        super().__init__(**kwargs)

    async def completion(self, model, messages, **kwargs):
        # Use Cloud Code Assist API patterns
        # Handle project-based routing
        # Implement safety settings
        pass
```

#### 2.4 OpenAI Codex Provider
**Based on**: Codex AI provider implementation patterns

**Key Features**:
- ChatGPT backend API integration
- Advanced reasoning capabilities (o1-style models)
- Tool calling with function definitions
- Authentication via Codex CLI token

**Implementation Details**:
```python
class CodexUU(BaseUU):
    provider_name = "codex"
    supported_models = ["gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini"]

    def __init__(self, **kwargs):
        self.auth_manager = CodexAuth()
        super().__init__(**kwargs)

    async def completion(self, model, messages, **kwargs):
        # Use ChatGPT backend API
        # Handle reasoning mode for o1 models
        # Process tool calling and streaming
        pass
```

### Phase 3: LiteLLM Integration

#### 3.1 Provider Registration
- Register all providers with LiteLLM's provider system
- Implement model name mapping (e.g., `uutel/claude-code/claude-3-5-sonnet`)
- Set up routing logic for different model prefixes
- Configure default settings and parameters

#### 3.2 Configuration Management
- Environment variable support for authentication
- Configuration file support (YAML/JSON)
- Runtime provider configuration
- Model-specific parameter handling

#### 3.3 Testing and Validation
- Unit tests for each provider
- Integration tests with actual APIs
- Performance benchmarking
- Error handling validation

### Phase 4: Advanced Features

#### 4.1 Tool Calling Standardization
- Unified function calling interface across providers
- Tool schema validation and conversion
- Streaming tool responses
- Error handling for tool failures

#### 4.2 Caching and Performance
- Response caching with TTL
- Request deduplication
- Connection pooling
- Rate limiting and backoff

#### 4.3 Monitoring and Observability
- Request/response logging
- Performance metrics
- Cost tracking integration
- Health check endpoints

## Package Dependencies

### Core Dependencies
```python
# Core LiteLLM integration
litellm >= 1.70.0

# HTTP and async support
httpx >= 0.25.0
aiohttp >= 3.8.0

# Authentication and OAuth
google-auth >= 2.15.0
google-auth-oauthlib >= 1.0.0

# Data validation and parsing
pydantic >= 2.0.0
pydantic-settings >= 2.0.0

# CLI and configuration
typer >= 0.9.0
rich >= 13.0.0

# Logging and monitoring
loguru >= 0.7.0

# Testing
pytest >= 7.0.0
pytest-asyncio >= 0.21.0
pytest-mock >= 3.10.0
```

### Optional Dependencies
```python
# Development tools
black >= 23.0.0
ruff >= 0.1.0
mypy >= 1.0.0

# Documentation
mkdocs >= 1.5.0
mkdocs-material >= 9.0.0
```

## Authentication Strategies

### 1. Claude Code Authentication
- **Method**: OAuth 2.0 with PKCE
- **Tokens**: Access and refresh tokens
- **Storage**: Local file or environment variables
- **Flow**: Browser-based authentication

### 2. Gemini CLI Authentication
- **Methods**: Multiple (API key, Vertex AI, OAuth)
- **API Key**: `GEMINI_API_KEY` environment variable
- **Vertex AI**: Service account JSON or ADC
- **OAuth**: Personal Google account

### 3. Google Cloud Code Authentication
- **Method**: Google Cloud service accounts
- **Credentials**: Service account JSON or ADC
- **Project**: Google Cloud project ID required
- **Scopes**: Cloud platform and Code Assist APIs

### 4. OpenAI Codex Authentication
- **Method**: Session tokens from Codex CLI
- **Storage**: `~/.codex/auth.json`
- **Refresh**: Automatic token refresh
- **Fallback**: OpenAI API key for compatibility

## Message Format Transformations

### Input Transformation (LiteLLM → Provider)
Each provider requires specific message format conversion:

1. **OpenAI Format** (LiteLLM input) → **Provider Format**
2. **Role Mapping**: system/user/assistant → provider-specific roles
3. **Content Processing**: text, images, files → provider content types
4. **Tool Definitions**: OpenAI function schema → provider tool schema

### Output Transformation (Provider → LiteLLM)
Standardize responses to OpenAI format:

1. **Response Structure**: Provider response → OpenAI ChatCompletion
2. **Content Extraction**: Provider content → OpenAI message content
3. **Usage Statistics**: Provider tokens → OpenAI usage format
4. **Streaming Chunks**: Provider deltas → OpenAI delta format

## Error Handling and Retry Logic

### Error Categories
1. **Authentication Errors**: Invalid tokens, expired credentials
2. **Rate Limiting**: Request throttling, quota exceeded
3. **Model Errors**: Invalid model, model overloaded
4. **Network Errors**: Connection timeouts, DNS failures
5. **Validation Errors**: Invalid input format, parameter errors

### Retry Strategy
```python
class RetryConfig:
    max_retries: int = 3
    backoff_factor: float = 2.0
    retry_on_status: List[int] = [429, 502, 503, 504]
    retry_on_exceptions: List[Exception] = [ConnectionError, TimeoutError]
```

## Testing Strategy

### Unit Tests
- Test each provider in isolation
- Mock external API calls
- Validate message transformations
- Test error handling paths

### Integration Tests
- Test with real API endpoints
- Validate authentication flows
- Test streaming responses
- Validate tool calling

### Performance Tests
- Benchmark response times
- Test concurrent requests
- Memory usage profiling
- Rate limiting validation

## Documentation Plan

### API Documentation
- Provider-specific configuration
- Authentication setup guides
- Usage examples for each provider
- Tool calling examples

### Integration Guides
- LiteLLM integration steps
- Environment setup
- Configuration file examples
- Troubleshooting guides

### Developer Documentation
- Contributing guidelines
- Code style standards
- Testing procedures
- Release process

## Deployment and Distribution

### Package Distribution
- PyPI package publication
- Semantic versioning
- GitHub releases with changelog
- Docker image for containerized usage

### CI/CD Pipeline
- GitHub Actions for testing
- Automated testing on multiple Python versions
- Code quality checks (black, ruff, mypy)
- Security scanning

### Installation Methods
```bash
# Standard installation
pip install uutel

# With all optional dependencies
pip install uutel[all]

# Development installation
pip install -e .[dev]
```

## Success Criteria

### Technical Requirements
1. ✅ Full LiteLLM provider compatibility
2. ✅ Support for all 4 target providers
3. ✅ Streaming response handling
4. ✅ Tool calling functionality
5. ✅ Comprehensive error handling
6. ✅ >90% test coverage

### Performance Requirements
1. ✅ <200ms overhead per request
2. ✅ Support for 100+ concurrent requests
3. ✅ Proper memory management
4. ✅ Efficient connection pooling

### Documentation Requirements
1. ✅ Complete API documentation
2. ✅ Setup and configuration guides
3. ✅ Working examples for each provider
4. ✅ Troubleshooting documentation

## Risk Assessment and Mitigation

### Technical Risks
1. **API Changes**: Provider APIs may change → Version pinning and adapters
2. **Authentication Issues**: Complex auth flows → Comprehensive testing
3. **Rate Limiting**: Provider-specific limits → Intelligent retry logic
4. **Performance**: Network latency → Async operations and caching

### Operational Risks
1. **Maintenance Burden**: Multiple providers → Automated testing and monitoring
2. **Security**: Credential handling → Secure storage and transmission
3. **Compatibility**: LiteLLM updates → Continuous integration testing

## Future Enhancements

### Potential Extensions
1. **Additional Providers**: Perplexity, Anthropic direct, Azure OpenAI
2. **Advanced Features**: Response caching, load balancing, failover
3. **Monitoring**: Detailed metrics, cost tracking, usage analytics
4. **Enterprise Features**: Team management, audit logging, compliance

### Community Contributions
1. **Open Source**: MIT license for community contributions
2. **Plugin System**: Allow third-party provider extensions
3. **Documentation**: Community-driven examples and guides
4. **Testing**: Community testing with different configurations

This plan provides a comprehensive roadmap for implementing UUTEL as a robust, production-ready Python package that extends LiteLLM's capabilities while maintaining compatibility and performance standards.