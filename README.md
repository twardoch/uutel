# UUTEL: Universal AI Provider for LiteLLM

**UUTEL** is a Python package that extends LiteLLM's provider ecosystem by implementing custom providers for Claude Code, Gemini CLI, Google Cloud Code, and OpenAI Codex. It enables unified LLM inferencing through LiteLLM's standardized interface while leveraging the unique capabilities of each AI provider.

## Architecture

UUTEL implements the **Universal Unit (UU)** pattern where each provider follows the naming convention `{ProviderName}UU`:

- **ClaudeCodeUU**: OAuth-based Claude Code provider with MCP tool integration
- **GeminiCLIUU**: Multi-auth Gemini CLI provider supporting API keys, Vertex AI, and OAuth
- **CloudCodeUU**: Google Cloud Code provider with service account authentication
- **CodexUU**: OpenAI Codex provider with ChatGPT backend integration

Each provider extends the `BaseUU` class (which inherits from LiteLLM's `BaseLLM`) and includes:
- Authentication management (`{Provider}Auth`)
- Message transformation (`{Provider}Transform`)
- Request/response models (`{Provider}Request`, `{Provider}Response`)

## Features

- **LiteLLM Compatibility**: Full adherence to LiteLLM's provider interface patterns
- **Unified API**: Consistent OpenAI-compatible interface across all providers
- **Authentication Management**: Secure handling of OAuth, API keys, and service accounts
- **Streaming Support**: Real-time response streaming for all providers
- **Tool Calling**: Function calling capabilities where supported
- **Error Handling**: Robust error mapping and fallback mechanisms

## Installation

```bash
pip install uutel

# With all optional dependencies
pip install uutel[all]

# Development installation
pip install -e .[dev]
```

## Basic Usage

```python
import litellm
from uutel import ClaudeCodeUU, GeminiCLIUU, CloudCodeUU, CodexUU

# Register UUTEL providers with LiteLLM
litellm.custom_provider_map = [
    {"provider": "claude-code", "custom_handler": ClaudeCodeUU()},
    {"provider": "gemini-cli", "custom_handler": GeminiCLIUU()},
    {"provider": "cloud-code", "custom_handler": CloudCodeUU()},
    {"provider": "codex", "custom_handler": CodexUU()},
]

# Use via LiteLLM's standard interface
response = litellm.completion(
    model="uutel/claude-code/claude-3-5-sonnet",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Provider-Specific Usage

### Claude Code Provider
```python
from uutel.providers.claude_code import ClaudeCodeUU

# OAuth authentication with browser flow
provider = ClaudeCodeUU()
response = litellm.completion(
    model="uutel/claude-code/claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Analyze this code"}],
    tools=[{"type": "function", "function": {"name": "analyze_code"}}]
)
```

### Gemini CLI Provider
```python
from uutel.providers.gemini_cli import GeminiCLIUU

# API key authentication
provider = GeminiCLIUU(auth_type="api-key", api_key="your-key")

# Vertex AI authentication
provider = GeminiCLIUU(auth_type="vertex-ai", project_id="your-project")

response = litellm.completion(
    model="uutel/gemini-cli/gemini-2.0-flash-exp",
    messages=[{"role": "user", "content": "Generate code"}],
    stream=True
)
```

### Google Cloud Code Provider
```python
from uutel.providers.cloud_code import CloudCodeUU

# Service account authentication
provider = CloudCodeUU(project_id="your-gcp-project")
response = litellm.completion(
    model="uutel/cloud-code/gemini-2.5-pro",
    messages=[{"role": "user", "content": "Review this PR"}]
)
```

### OpenAI Codex Provider
```python
from uutel.providers.codex import CodexUU

# Uses Codex CLI session tokens
provider = CodexUU()
response = litellm.completion(
    model="uutel/codex/gpt-4o",
    messages=[{"role": "user", "content": "Explain this algorithm"}],
    max_tokens=4000
)
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

## Authentication Setup

### Claude Code
- OAuth 2.0 with PKCE flow
- Browser-based authentication
- Automatic token refresh

### Gemini CLI
- Multiple methods: API key, Vertex AI, OAuth
- Environment variables: `GEMINI_API_KEY`
- Google Cloud Application Default Credentials

### Google Cloud Code
- Service account JSON or ADC
- Environment: `GOOGLE_APPLICATION_CREDENTIALS`
- Project ID required

### OpenAI Codex
- Session tokens from `~/.codex/auth.json`
- Automatic token refresh
- Fallback to OpenAI API key

## Development

This project uses [Hatch](https://hatch.pypa.io/) for development workflow management.

### Setup Development Environment

```bash
# Install hatch if you haven't already
pip install hatch

# Create and activate development environment
hatch shell

# Run tests
hatch run test

# Run tests with coverage
hatch run test-cov

# Run linting
hatch run lint

# Format code
hatch run format
```

## License

MIT License 