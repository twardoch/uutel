# UUTEL: Universal AI Provider for LiteLLM

[![CI](https://github.com/twardoch/uutel/actions/workflows/ci.yml/badge.svg)](https://github.com/twardoch/uutel/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/twardoch/uutel/branch/main/graph/badge.svg)](https://codecov.io/gh/twardoch/uutel)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

UUTEL bridges local AI CLI tools (Claude Code, Gemini CLI, Google Cloud Code, OpenAI Codex) to [LiteLLM](https://github.com/BerriAI/litellm). You authenticate with each tool's standard login flow; UUTEL handles error translation, rate-limit catching, and tool-call normalisation so you can call all four through a single OpenAI-compatible Python interface.

## Install

```bash
pip install uutel
```

## Quick start

```bash
# Single-turn completions using short aliases
uutel complete --prompt "Write a sorter" --engine codex
uutel complete --prompt "Say hello" --engine claude
uutel complete --prompt "Summarise Gemini API" --engine gemini
uutel complete --prompt "Deployment checklist" --engine cloud

uutel list_engines          # show all engines and aliases
uutel test --engine codex   # connectivity check
uutel diagnostics           # credential and tooling status per provider
```

## Authentication setup

Each provider authenticates independently via its own CLI token or API key.

| Provider | Install | Authenticate | Verify |
|----------|---------|--------------|--------|
| **Codex** (ChatGPT backend) | `npm install -g @openai/codex@latest` | `codex login` → writes `~/.codex/auth.json` | `codex --version` |
| **Claude Code** (Anthropic) | `npm install -g @anthropic-ai/claude-code` | `claude login` → stores tokens under `~/.claude*/` | `claude --version` |
| **Gemini CLI** (Google) | `npm install -g @google/gemini-cli` | `gemini login` (OAuth) or export `GOOGLE_API_KEY` | `gemini models list` |
| **Cloud Code** (Google) | reuses Gemini credentials | `gemini login` or same API key env vars | ensure account has Cloud Code access |

Alternative for Codex: export `OPENAI_API_KEY` to bypass CLI auth and use OpenAI Chat Completions directly.

## Configuration

Persist defaults in `~/.uutel.toml`:

```toml
# UUTEL Configuration

engine = "my-custom-llm/codex-large"
max_tokens = 500
temperature = 0.7
stream = false
verbose = false
```

Run `uutel config init` to create the file, or `uutel config show` to inspect it.

## Python API

```python
from uutel import BaseUU

class MyProviderUU(BaseUU):
    def __init__(self):
        super().__init__()
        self.provider_name = "my-provider"
        self.supported_models = ["my-model-1.0"]

    def completion(self, model, messages, **kwargs):
        return {"choices": [{"message": {"role": "assistant", "content": "Hello!"}}]}
```

`BaseUU` extends LiteLLM's `CustomLLM`. All provider classes follow the `{ProviderName}UU` naming convention.

### Tool calling

```python
from uutel import validate_tool_schema, create_tool_call_response, extract_tool_calls_from_response

tool = {"type": "function", "function": {"name": "get_weather", "description": "...",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}}

is_valid = validate_tool_schema(tool)
response = create_tool_call_response("call_123", "get_weather", {"temp": "22°C"})
```

## Providers

| Class | Engine alias | Backend |
|-------|-------------|---------|
| `CodexUU` | `codex` | ChatGPT Codex API (`chatgpt.com/backend-api`) + fallback to OpenAI API key |
| `ClaudeCodeUU` | `claude` | `@anthropic-ai/claude-code` JSONL stream |
| `GeminiCLIUU` | `gemini` | Google Generative AI via API key or `gemini` CLI |
| `CloudCodeUU` | `cloud` | Google Cloud Code endpoints with Gemini OAuth |

Engine aliases resolve through `validate_engine`; shorthand variants (`claude-code`, `gemini-cli`, `codex-large`, etc.) map to the canonical engine automatically.

## Package structure

```
src/uutel/
├── core/
│   ├── base.py          # BaseUU — LiteLLM CustomLLM subclass
│   ├── auth.py          # BaseAuth and credential helpers
│   ├── exceptions.py    # 7 specialised exception types
│   └── utils.py         # Shared utilities
├── providers/
│   ├── claude_code/     # ClaudeCodeUU
│   ├── gemini_cli/      # GeminiCLIUU
│   ├── cloud_code/      # CloudCodeUU
│   └── codex/           # CodexUU
├── docs/
│   └── recorded_examples.py  # Fixture metadata for offline demos
└── cli.py               # Fire CLI entry point
```

## Development

```bash
git clone https://github.com/twardoch/uutel.git
cd uutel
uv sync --all-extras
uv run pytest              # 326+ tests
uv run ruff check src tests
uv run mypy src/uutel
```

Or with Hatch: `hatch run test`, `hatch run lint`, `hatch run typecheck`.

## License

MIT
