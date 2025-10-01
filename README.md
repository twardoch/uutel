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

# Single-turn completions (alias-first examples)
uutel complete --prompt "Write a sorter" --engine codex
uutel complete --prompt "Say hello" --engine claude
uutel complete --prompt "Summarise Gemini API" --engine gemini
uutel complete --prompt "Deployment checklist" --engine cloud

# List available engines
uutel list_engines

# Test engine connectivity
uutel test --engine codex

# Get help
uutel help
```

## Current Status: Providers Live ✅

UUTEL now includes live Codex, Claude Code, Gemini CLI, and Cloud Code providers wired through LiteLLM. The universal unit foundation stays in place, but the default CLI commands exercise real vendor APIs once credentials are configured.

### What's Built and Working

- **⚡ Command-Line Interface**: Complete Fire CLI with `uutel` command for instant completions
- **🤖 Provider Coverage**: Live Codex, Claude Code, Gemini CLI, and Cloud Code adapters with sync/async completions and streaming.
- **🏗️ Core Infrastructure**: Complete `BaseUU` class extending LiteLLM's `CustomLLM`
- **🔐 Authentication Framework**: Flexible `BaseAuth` system with secure credential handling
- **🛠️ Tool Calling**: 5 OpenAI-compatible utilities for function calling workflows
- **📡 Streaming Support**: Async/sync streaming with chunk processing and error handling
- **🚨 Exception Handling**: 7 specialized exception types with provider context
- **🧪 Testing Infrastructure**: 173 tests with comprehensive coverage, including CLI tests
- **⚙️ CI/CD Pipeline**: Multi-platform testing, code quality, security scanning
- **📚 Examples**: Working demonstrations of all capabilities
- **🔧 Developer Experience**: Modern tooling with ruff, mypy, pre-commit ready

### Provider Highlights

- **CodexUU**: Calls ChatGPT Codex backend when `codex login` tokens exist, or falls back to OpenAI API keys with tool-call support.
- **ClaudeCodeUU**: Streams JSONL events from `@anthropic-ai/claude-code`, handling cancellation, tool filtering, and credential guidance.
- **GeminiCLIUU**: Adapts Google Generative AI via API keys or the `gemini` CLI with retry, streaming, and tool functions.
- **CloudCodeUU**: Talks to Google Cloud Code endpoints using shared Gemini OAuth credentials and project-aware readiness checks.

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
uutel complete --prompt "Write Python code" --engine codex

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
uutel test --engine codex --verbose
```

#### `uutel diagnostics` - Check Provider Readiness

```bash
# Summarise credential and tooling status for each alias
uutel diagnostics

# Combine with verbose logging for deeper troubleshooting
LITELLM_LOG=DEBUG uutel diagnostics
```

#### `uutel help` - Get Help

```bash
# Show general help
uutel help

# Command-specific help
uutel complete --help
```

### Configuration File

Persist defaults in `~/.uutel.toml` when you want repeatable CLI behaviour:

```toml
# UUTEL Configuration

engine = "my-custom-llm/codex-large"
max_tokens = 500
temperature = 0.7
stream = false
verbose = false
```

Run `uutel config init` to create the file automatically or edit the snippet above.

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
uutel complete --prompt "test"  # Defaults to codex when you omit --engine
```

#### Authentication Setup

Each provider ships with vendor-specific credentials. Configure these before attempting live requests:

- **Codex (ChatGPT backend)**
  - Install CLI: `npm install -g @openai/codex@latest` (installs the `codex` binary).
  - Authenticate: `codex login` launches OpenAI OAuth and writes `~/.codex/auth.json` with `access_token` and `account_id`.
  - Alternative: export `OPENAI_API_KEY` to use standard Chat Completions without the Codex CLI token.
  - Verification: `codex --version` should print ≥0.28; rerun `codex login` if tokens expire.
- **Claude Code (Anthropic)**
  - Install CLI: `npm install -g @anthropic-ai/claude-code` (provides the `claude`/`claude-code` binaries).
  - Authenticate: `claude login` stores refreshed credentials under `~/.claude*/`.
  - Requirements: Node.js 18+, CLI present on `PATH`.
  - Verification: `claude --version` confirms installation; rerun `claude login` if completions fail with auth errors.
- **Gemini CLI Core (Google)**
  - Install CLI: `npm install -g @google/gemini-cli` (installs the `gemini` binary).
  - Authenticate via OAuth: `gemini login` writes `~/.gemini/oauth_creds.json`.
  - Authenticate via API key: export one of `GOOGLE_API_KEY`, `GEMINI_API_KEY`, or `GOOGLE_GENAI_API_KEY`.
  - Verification: `gemini models list` should succeed once credentials are valid.
- **Google Cloud Code AI**
  - Reuses Gemini credentials: prefer `gemini login` (OAuth) or the same Google API key env vars.
  - Credential lookup order: `~/.gemini/oauth_creds.json`, `~/.config/gemini/oauth_creds.json`, `~/.google-cloud-code/credentials.json`.
  - Verification: ensure the chosen Google account has access to Cloud Code; rerun `gemini login` if access tokens lapse.

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
Includes an offline replay of recorded provider fixtures so you can preview responses without installing any CLIs.
Set `UUTEL_LIVE_EXAMPLE=1` to trigger live requests when credentials are available, or point
`UUTEL_LIVE_FIXTURES_DIR` at a directory of JSON payloads to run deterministic stubbed "live" replays.

#### Claude Code Fixture Replay
```bash
python examples/basic_usage.py  # replay runs as part of the script output
```
Shows the deterministic Claude Code fixture output and provides the exact commands required to enable live runs:

1. `npm install -g @anthropic-ai/claude-code`
2. `claude login`
3. `uutel complete --engine claude --stream`

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
