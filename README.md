# UUTEL: Universal AI Provider for LiteLLM

[![CI](https://github.com/twardoch/uutel/actions/workflows/ci.yml/badge.svg)](https://github.com/twardoch/uutel/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/twardoch/uutel/branch/main/graph/badge.svg)](https://codecov.io/gh/twardoch/uutel)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**UUTEL** is a comprehensive Python package that provides a robust foundation for extending LiteLLM's provider ecosystem. It implements the **Universal Unit (UU)** pattern and provides core infrastructure for custom AI providers including Claude Code, Gemini CLI, Google Cloud Code, and OpenAI Codex.

## 🚀 Quick CLI Usage

UUTEL includes a powerful command-line interface for instant AI completions:

```bash
# Install and use immediately
pip install uutel

# Single-turn completions
uutel complete --prompt "Explain Python async/await"
uutel complete --prompt "Write a hello world in Rust" --engine my-custom-llm/codex-mini

# List available engines
uutel list_engines

# Test engine connectivity
uutel test --engine my-custom-llm/codex-large

# Get help
uutel help
```

## Current Status: Foundation Complete ✅

UUTEL currently provides a **production-ready foundation** with comprehensive tooling and infrastructure. The core framework is complete and ready for provider implementations.

### What's Built and Working

- **⚡ Command-Line Interface**: Complete Fire CLI with `uutel` command for instant completions
- **🏗️ Core Infrastructure**: Complete `BaseUU` class extending LiteLLM's `CustomLLM`
- **🔐 Authentication Framework**: Flexible `BaseAuth` system with secure credential handling
- **🛠️ Tool Calling**: 5 OpenAI-compatible utilities for function calling workflows
- **📡 Streaming Support**: Async/sync streaming with chunk processing and error handling
- **🚨 Exception Handling**: 7 specialized exception types with provider context
- **🧪 Testing Infrastructure**: 173 tests with comprehensive coverage, including CLI tests
- **⚙️ CI/CD Pipeline**: Multi-platform testing, code quality, security scanning
- **📚 Examples**: Working demonstrations of all capabilities
- **🔧 Developer Experience**: Modern tooling with ruff, mypy, pre-commit ready

### Planned Providers (Phase 2)

The foundation supports these upcoming provider implementations:

- **ClaudeCodeUU**: OAuth-based Claude Code provider with MCP tool integration
- **GeminiCLIUU**: Multi-auth Gemini CLI provider (API keys, Vertex AI, OAuth)
- **CloudCodeUU**: Google Cloud Code provider with service account authentication
- **CodexUU**: OpenAI Codex provider with ChatGPT backend integration

## Key Features

- **⚡ Command-Line Interface**: Ready-to-use `uutel` CLI for instant AI completions
- **🔗 LiteLLM Compatibility**: Full adherence to LiteLLM's provider interface patterns
- **🌐 Unified API**: Consistent OpenAI-compatible interface across all providers
- **🔐 Authentication Management**: Secure handling of OAuth, API keys, and service accounts
- **📡 Streaming Support**: Real-time response streaming with comprehensive error handling
- **🛠️ Tool Calling**: Complete OpenAI-compatible function calling implementation
- **🚨 Error Handling**: Robust error mapping, fallback mechanisms, and detailed context
- **🧪 Test Coverage**: Comprehensive test suite with CLI testing included
- **⚙️ Production Ready**: CI/CD pipeline, security scanning, quality checks

## Installation

```bash
# Standard installation (includes CLI)
pip install uutel

# With all optional dependencies
pip install uutel[all]

# Development installation
pip install -e .[dev]
```

After installation, the `uutel` command is available system-wide:

```bash
# Verify installation
uutel help

# Quick test
uutel complete --prompt "Hello, AI!"
```

### CLI Usage

UUTEL provides a comprehensive command-line interface with four main commands:

#### `uutel complete` - Run AI Completions

```bash
# Basic completion
uutel complete --prompt "Explain machine learning"

# Specify engine
uutel complete --prompt "Write Python code" --engine my-custom-llm/codex-mini

# Enable streaming output
uutel complete --prompt "Tell a story" --stream

# Adjust response parameters
uutel complete --prompt "Summarize this" --max_tokens 500 --temperature 0.7
```

#### `uutel list_engines` - Show Available Engines

```bash
# List all available engines with descriptions
uutel list_engines
```

#### `uutel test` - Test Engine Connectivity

```bash
# Test default engine
uutel test

# Test specific engine
uutel test --engine my-custom-llm/codex-large --verbose
```

#### `uutel help` - Get Help

```bash
# Show general help
uutel help

# Command-specific help
uutel complete --help
```

### Troubleshooting CLI Issues

#### Command Not Found: `uutel: command not found`

If you get "command not found" after installation:

```bash
# 1. Verify installation
pip list | grep uutel

# 2. Check if pip bin directory is in PATH
python -m site --user-base

# 3. Use Python module syntax as fallback
python -m uutel complete --prompt "test"

# 4. Reinstall with user flag
pip install --user uutel
```

#### Engine Errors: Provider Not Available

```bash
# Check available engines
uutel list_engines

# Test connectivity
uutel test --verbose

# Use default engine
uutel complete --prompt "test"  # Omit --engine to use default
```

#### Authentication Issues

Most CLI errors are due to missing authentication. UUTEL currently implements a working Codex provider for demonstration:

```bash
# Default engine works out of the box
uutel complete --prompt "Hello world"

# For future providers, you'll configure authentication separately
# Each provider will have its own auth setup process
```

#### Getting More Help

```bash
# Enable verbose output for debugging
uutel test --verbose

# Check specific command help
uutel complete --help
uutel list_engines --help
uutel test --help
```

## Quick Start

### Using Core Infrastructure

```python
from uutel import BaseUU, BaseAuth, validate_tool_schema, create_tool_call_response

# Example of extending BaseUU for your own provider
class MyProviderUU(BaseUU):
    def __init__(self):
        super().__init__()
        self.provider_name = "my-provider"
        self.supported_models = ["my-model-1.0"]

    def completion(self, model, messages, **kwargs):
        # Your provider implementation
        return {"choices": [{"message": {"role": "assistant", "content": "Hello!"}}]}

# Use authentication framework
auth = BaseAuth()
# Implement your authentication logic

# Use tool calling utilities
tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
    }
}

is_valid = validate_tool_schema(tool)  # True
response = create_tool_call_response("call_123", "get_weather", {"temp": "22°C"})
```

### Tool Calling Capabilities

```python
from uutel import (
    validate_tool_schema,
    transform_openai_tools_to_provider,
    create_tool_call_response,
    extract_tool_calls_from_response
)

# Validate OpenAI tool schemas
tool = {"type": "function", "function": {"name": "calc", "description": "Calculate"}}
is_valid = validate_tool_schema(tool)

# Transform tools between formats
provider_tools = transform_openai_tools_to_provider([tool], "my-provider")

# Create tool responses
response = create_tool_call_response(
    tool_call_id="call_123",
    function_name="calculate",
    function_result={"result": 42}
)

# Extract tool calls from provider responses
tool_calls = extract_tool_calls_from_response(provider_response)
```

### Streaming Support

```python
from uutel import BaseUU
import asyncio

class StreamingProvider(BaseUU):
    def simulate_streaming(self, text):
        """Example streaming implementation"""
        for word in text.split():
            yield {"choices": [{"delta": {"content": f"{word} "}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

# Use streaming (see examples/streaming_example.py for full demo)
provider = StreamingProvider()
for chunk in provider.simulate_streaming("Hello world"):
    content = chunk["choices"][0]["delta"].get("content", "")
    if content:
        print(content, end="")
```

### Authentication Framework

```python
from uutel import BaseAuth, AuthResult
from datetime import datetime

class MyAuth(BaseAuth):
    def authenticate(self, **kwargs):
        # Implement your authentication logic
        return AuthResult(
            success=True,
            token="your-token",
            expires_at=datetime.now(),
            error=None
        )

    def get_headers(self):
        return {"Authorization": f"Bearer {self.get_token()}"}

# Use in your provider
auth = MyAuth()
headers = auth.get_headers()
```

## Package Structure

```
uutel/
├── __init__.py                 # Main exports and provider registration
├── core/
│   ├── base.py                 # BaseUU class and common interfaces
│   ├── auth.py                 # Common authentication utilities
│   ├── exceptions.py           # Custom exception classes
│   └── utils.py                # Common utilities and helpers
├── providers/
│   ├── claude_code/           # Claude Code provider implementation
│   ├── gemini_cli/            # Gemini CLI provider implementation
│   ├── cloud_code/            # Google Cloud Code provider implementation
│   └── codex/                 # OpenAI Codex provider implementation
├── tests/                     # Comprehensive test suite
└── examples/                  # Usage examples and demos
```

## Examples

UUTEL includes comprehensive examples demonstrating all capabilities:

### CLI Usage Examples
```bash
# Quick completion examples
uutel complete --prompt "Explain Python decorators"
uutel complete --prompt "Write a sorting algorithm" --engine my-custom-llm/codex-mini
uutel list_engines
uutel test --verbose
```

### Programmatic API Examples

#### Basic Usage Example
```bash
python examples/basic_usage.py
```
Demonstrates core infrastructure, authentication framework, error handling, and utilities.

#### Tool Calling Example
```bash
python examples/tool_calling_example.py
```
Complete demonstration of OpenAI-compatible tool calling with validation, transformation, and workflow simulation.

#### Streaming Example
```bash
python examples/streaming_example.py
```
Async/sync streaming responses with chunk processing, error handling, and concurrent request management.

## Development

This project uses modern Python tooling for an excellent developer experience:

### Development Tools
- **[Hatch](https://hatch.pypa.io/)**: Project management and virtual environments
- **[Ruff](https://github.com/astral-sh/ruff)**: Fast linting and formatting
- **[MyPy](https://mypy.readthedocs.io/)**: Static type checking
- **[Pytest](https://pytest.org/)**: Testing framework with 173+ tests including CLI coverage
- **[GitHub Actions](https://github.com/features/actions)**: CI/CD pipeline

### Quick Setup

```bash
# Clone repository
git clone https://github.com/twardoch/uutel.git
cd uutel

# Install UV (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run all quality checks
uv run ruff check src/uutel tests
uv run ruff format src/uutel tests
uv run mypy src/uutel
```

### Using Hatch (Alternative)

```bash
# Install hatch
pip install hatch

# Create and activate development environment
hatch shell

# Run tests (RECOMMENDED for all async tests)
hatch run test

# Run tests with coverage
hatch run test-cov

# Note: Always use 'hatch run test' instead of 'hatch test'
# to ensure proper async plugin loading

# Run linting and formatting
hatch run lint
hatch run format

# Type checking
hatch run typecheck
```

### Using Make (Convenience)

```bash
# Install development dependencies
make install-dev

# Run all checks
make check

# Run tests
make test

# Clean build artifacts
make clean
```

## Architecture & Design

### Universal Unit (UU) Pattern

UUTEL implements a consistent **Universal Unit** pattern where all provider classes follow the `{ProviderName}UU` naming convention:

```python
# Base class
class BaseUU(CustomLLM):  # Extends LiteLLM's CustomLLM
    def __init__(self):
        self.provider_name: str = "base"
        self.supported_models: list[str] = []

# Provider implementations (future)
class ClaudeCodeUU(BaseUU): ...
class GeminiCLIUU(BaseUU): ...
class CloudCodeUU(BaseUU): ...
class CodexUU(BaseUU): ...
```

### Core Components

1. **`BaseUU`**: LiteLLM-compatible provider base class
2. **`BaseAuth`**: Flexible authentication framework
3. **Exception Framework**: 7 specialized exception types
4. **Tool Calling**: 5 OpenAI-compatible utilities
5. **Streaming Support**: Async/sync response handling
6. **Utilities**: HTTP clients, validation, transformation

### Quality Assurance

- **Comprehensive Test Coverage**: 173+ tests including CLI functionality
- **CI/CD Pipeline**: Multi-platform testing (Ubuntu, macOS, Windows)
- **Code Quality**: Ruff formatting, MyPy type checking
- **Security Scanning**: Bandit and Safety integration
- **Documentation**: Examples, architecture docs, API reference, CLI troubleshooting

## Roadmap

### Phase 2: Provider Implementations (Upcoming)
- **ClaudeCodeUU**: OAuth authentication, MCP tool integration
- **GeminiCLIUU**: Multi-auth support (API key, Vertex AI, OAuth)
- **CloudCodeUU**: Google Cloud service account authentication
- **CodexUU**: ChatGPT backend integration with session management

### Phase 3: LiteLLM Integration
- Provider registration with LiteLLM
- Model name mapping (`uutel/provider/model`)
- Configuration management and validation
- Production deployment support

### Phase 4: Advanced Features
- Response caching and performance optimization
- Monitoring and observability tools
- Community plugin system
- Enterprise features and team management

## Contributing

We welcome contributions! The project is designed with simplicity and extensibility in mind.

### Getting Started
1. Fork the repository
2. Set up development environment: `uv sync --all-extras`
3. Run tests: `uv run pytest`
4. Make your changes
5. Ensure tests pass and code quality checks pass
6. Submit a pull request

### Development Guidelines
- Follow the **UU naming pattern** (`{ProviderName}UU`)
- Write tests first (TDD approach)
- Maintain 80%+ test coverage
- Use modern Python features (3.10+ type hints)
- Keep functions under 20 lines, files under 200 lines
- Document with clear docstrings

### Current Focus
The project is currently accepting contributions for:
- Provider implementations (Phase 2)
- Documentation improvements
- Example applications
- Performance optimizations
- Bug fixes and quality improvements

## Support

- **Documentation**: [GitHub Wiki](https://github.com/twardoch/uutel/wiki) (coming soon)
- **Issues**: [GitHub Issues](https://github.com/twardoch/uutel/issues)
- **Discussions**: [GitHub Discussions](https://github.com/twardoch/uutel/discussions)

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**UUTEL** provides the universal foundation for AI provider integration with both CLI and programmatic interfaces. Built with modern Python practices, comprehensive testing, and extensibility in mind.
