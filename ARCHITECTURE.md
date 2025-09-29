# UUTEL Architecture Documentation

## Overview

UUTEL (Universal AI Provider for LiteLLM) implements a comprehensive architecture for extending LiteLLM's provider ecosystem with custom AI providers. The architecture follows modern Python design patterns with emphasis on simplicity, testability, and extensibility.

## Core Design Principles

### 1. Universal Unit (UU) Pattern

The cornerstone of UUTEL's architecture is the **Universal Unit (UU)** naming convention:

```python
# Base class
class BaseUU(CustomLLM):  # Extends LiteLLM's CustomLLM
    ...

# Provider implementations
class ClaudeCodeUU(BaseUU): ...
class GeminiCLIUU(BaseUU): ...
class CloudCodeUU(BaseUU): ...
class CodexUU(BaseUU): ...
```

**Benefits:**
- Consistent naming across all providers
- Clear inheritance hierarchy
- Easy to identify UUTEL components
- Predictable code organization

### 2. Composition Over Inheritance

Each provider follows a compositional architecture:

```python
class ProviderUU(BaseUU):
    def __init__(self):
        self.auth = ProviderAuth()           # Authentication handling
        self.transform = ProviderTransform() # Message transformation
        self.models = ProviderModels()       # Model management
```

### 3. Interface Segregation

Clean, focused interfaces for each concern:
- `BaseAuth`: Authentication management
- `BaseUU`: Core provider functionality
- Tool calling utilities: Modular function-based design

## Architecture Layers

### Layer 1: Core Infrastructure

Located in `src/uutel/core/`, this layer provides:

#### BaseUU Class (`core/base.py`)
```python
class BaseUU(CustomLLM):
    """Universal base class for all UUTEL providers."""

    # Core LiteLLM interface methods
    def completion(self, model: str, messages: List[Dict], **kwargs) -> Dict
    def acompletion(self, model: str, messages: List[Dict], **kwargs) -> Dict
    def streaming(self, model: str, messages: List[Dict], **kwargs) -> Iterator
    def astreaming(self, model: str, messages: List[Dict], **kwargs) -> AsyncIterator
```

#### Authentication Framework (`core/auth.py`)
```python
@dataclass
class AuthResult:
    """Standardized authentication result."""
    success: bool
    token: Optional[str] = None
    expires_at: Optional[datetime] = None
    error: Optional[str] = None

class BaseAuth:
    """Base authentication interface."""
    def authenticate(self, **kwargs) -> AuthResult: ...
    def get_headers(self) -> Dict[str, str]: ...
    def refresh_token(self) -> AuthResult: ...
    def is_valid(self) -> bool: ...
```

#### Exception Hierarchy (`core/exceptions.py`)
```
UUTELError (base)
├── AuthenticationError
├── RateLimitError
├── ModelError
├── NetworkError
├── ValidationError
└── ProviderError
```

#### Utilities (`core/utils.py`)
- Message transformation functions
- HTTP client creation with retry logic
- Model validation and provider extraction
- Tool calling utilities (5 functions)
- Error formatting helpers

### Layer 2: Provider Implementations

Located in `src/uutel/providers/`, each provider follows this structure:

```
providers/
├── claude_code/
│   ├── __init__.py
│   ├── auth.py          # ClaudeCodeAuth
│   ├── models.py        # Model definitions
│   ├── provider.py      # ClaudeCodeUU
│   └── transforms.py    # Message transformation
├── gemini_cli/
├── cloud_code/
└── codex/
```

#### Provider Implementation Pattern
```python
# provider.py
class ProviderUU(BaseUU):
    def __init__(self):
        super().__init__()
        self.provider_name = "provider-name"
        self.supported_models = ["model-1", "model-2"]
        self.auth = ProviderAuth()

    def completion(self, model: str, messages: List[Dict], **kwargs) -> Dict:
        # 1. Authenticate
        auth_result = self.auth.authenticate()

        # 2. Transform messages
        provider_messages = transform_openai_to_provider(messages, self.provider_name)

        # 3. Make API call
        response = self._make_request(provider_messages, **kwargs)

        # 4. Transform response
        return self._transform_response(response)
```

### Layer 3: Testing Architecture

Comprehensive testing strategy with 84% coverage:

#### Test Organization
```
tests/
├── conftest.py          # Pytest configuration and fixtures
├── test_auth.py         # Authentication framework tests
├── test_base.py         # BaseUU class tests
├── test_exceptions.py   # Exception hierarchy tests
├── test_tool_calling.py # Tool calling utilities tests
├── test_utils.py        # Core utilities tests
└── test_package.py      # Package integration tests
```

#### Testing Patterns
- **Test-Driven Development**: Tests written before implementation
- **Fixtures**: Reusable test components in `conftest.py`
- **Parametrized Tests**: Testing multiple scenarios efficiently
- **Mock-based Testing**: Isolating units under test

#### Example Test Structure
```python
class TestToolSchemaValidation:
    """Test tool schema validation functionality."""

    def test_validate_tool_schema_valid_openai_format(self):
        """Test validation of valid OpenAI tool schema."""
        # Arrange
        valid_tool = {...}

        # Act
        result = validate_tool_schema(valid_tool)

        # Assert
        assert result is True
```

### Layer 4: Quality Assurance

#### Code Quality Tools
- **Ruff**: Fast linting and formatting
- **MyPy**: Static type checking
- **Bandit**: Security scanning
- **Pre-commit**: Automated quality checks

#### CI/CD Pipeline (`.github/workflows/ci.yml`)
```yaml
jobs:
  test:      # Multi-platform testing (Ubuntu, macOS, Windows)
  lint:      # Code quality checks
  security:  # Security scanning
  examples:  # Validate all examples
  build:     # Package building
```

## Design Patterns

### 1. Template Method Pattern (BaseUU)
```python
class BaseUU:
    def completion(self, model, messages, **kwargs):
        # Template method with common workflow
        self._validate_input(model, messages)
        auth = self._authenticate()
        transformed = self._transform_messages(messages)
        response = self._make_request(transformed, **kwargs)
        return self._transform_response(response)

    # Abstract methods for providers to implement
    def _make_request(self, messages, **kwargs): raise NotImplementedError
```

### 2. Strategy Pattern (Authentication)
```python
# Different authentication strategies
class APIKeyAuth(BaseAuth): ...
class OAuthAuth(BaseAuth): ...
class ServiceAccountAuth(BaseAuth): ...

# Provider selects appropriate strategy
class ProviderUU(BaseUU):
    def __init__(self, auth_type="api-key"):
        self.auth = self._create_auth_strategy(auth_type)
```

### 3. Factory Pattern (HTTP Clients)
```python
def create_http_client(async_client=False, timeout=None, retry_config=None):
    """Factory for creating configured HTTP clients."""
    if async_client:
        return httpx.AsyncClient(timeout=timeout, ...)
    return httpx.Client(timeout=timeout, ...)
```

### 4. Facade Pattern (Tool Calling)
```python
# Simple interface hiding complex tool calling logic
def validate_tool_schema(tool): ...
def transform_openai_tools_to_provider(tools, provider): ...
def create_tool_call_response(tool_call_id, function_name, result): ...
def extract_tool_calls_from_response(response): ...
```

## Data Flow Architecture

### 1. Request Flow
```
User Request → BaseUU.completion() → Provider Auth → Message Transform → API Call → Response Transform → User
```

### 2. Streaming Flow
```
User Request → BaseUU.streaming() → Provider Auth → Message Transform → Stream API → Chunk Processing → User
```

### 3. Tool Calling Flow
```
User Request → Tool Validation → Provider Transform → API Call → Tool Extraction → Response Creation → User
```

## Extension Patterns

### Adding a New Provider

1. **Create Provider Directory**
   ```
   src/uutel/providers/new_provider/
   ├── __init__.py
   ├── auth.py
   ├── models.py
   ├── provider.py
   └── transforms.py
   ```

2. **Implement Provider Class**
   ```python
   class NewProviderUU(BaseUU):
       def __init__(self):
           super().__init__()
           self.provider_name = "new-provider"
           self.supported_models = ["model-1"]
   ```

3. **Add Authentication**
   ```python
   class NewProviderAuth(BaseAuth):
       def authenticate(self, **kwargs) -> AuthResult: ...
   ```

4. **Write Tests**
   ```python
   # tests/test_new_provider.py
   class TestNewProviderUU:
       def test_completion(self): ...
   ```

5. **Update Package Exports**
   ```python
   # src/uutel/__init__.py
   from .providers.new_provider import NewProviderUU
   ```

### Extending Tool Calling

```python
def custom_tool_validator(tool: Dict) -> bool:
    """Custom validation logic for specific provider needs."""
    if not validate_tool_schema(tool):  # Use base validation
        return False
    # Add custom validation
    return custom_check(tool)
```

## Performance Considerations

### 1. Connection Pooling
```python
# HTTP clients reuse connections
client = httpx.Client()  # Reused across requests
```

### 2. Async Support
```python
# Full async/await support
async def acompletion(self, ...):
    async with httpx.AsyncClient() as client:
        response = await client.post(...)
```

### 3. Streaming Optimization
```python
# Efficient chunk processing
def streaming(self, ...):
    for chunk in response.iter_lines():
        yield self._process_chunk(chunk)
```

## Security Architecture

### 1. Credential Management
- Environment variable support
- Secure token storage
- Automatic token refresh
- No hardcoded secrets

### 2. Input Validation
```python
def validate_model_name(model: str) -> bool:
    """Prevent injection attacks through model names."""
    return bool(re.match(r"^[a-zA-Z0-9._-]+$", model))
```

### 3. Error Handling
```python
try:
    response = make_api_call()
except Exception as e:
    # Don't expose internal details
    raise ProviderError(f"API call failed") from e
```

## Monitoring and Observability

### 1. Structured Logging
```python
from loguru import logger

logger.info("Making API request", provider=self.provider_name, model=model)
```

### 2. Error Context
```python
def format_error_message(error: Exception, provider: str) -> str:
    """Include provider context in error messages."""
    return f"[{provider}] {type(error).__name__}: {error}"
```

### 3. Health Checks
```python
def is_provider_healthy(self) -> bool:
    """Check if provider is accessible."""
    try:
        self.auth.authenticate()
        return True
    except Exception:
        return False
```

## Future Architecture Considerations

### 1. Plugin System
```python
# Future: Dynamic provider loading
def register_provider(provider_class: Type[BaseUU]):
    """Register custom providers at runtime."""
    providers.register(provider_class.provider_name, provider_class)
```

### 2. Configuration Management
```python
# Future: Centralized configuration
@dataclass
class ProviderConfig:
    api_key: Optional[str]
    base_url: str
    timeout: float
    max_retries: int
```

### 3. Caching Layer
```python
# Future: Response caching
class CachedUU(BaseUU):
    def completion(self, ...):
        cache_key = self._generate_cache_key(...)
        if cached := self.cache.get(cache_key):
            return cached
        result = super().completion(...)
        self.cache.set(cache_key, result)
        return result
```

## Summary

UUTEL's architecture prioritizes:
- **Simplicity**: Clear patterns and minimal complexity
- **Extensibility**: Easy to add new providers
- **Testability**: Comprehensive test coverage and TDD approach
- **Maintainability**: Consistent structure and documentation
- **Performance**: Efficient HTTP handling and streaming
- **Security**: Safe credential management and input validation

The Universal Unit pattern provides a consistent foundation for all AI providers while maintaining the flexibility to adapt to each provider's unique requirements.