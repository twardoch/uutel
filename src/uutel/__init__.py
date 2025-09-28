# this_file: src/uutel/__init__.py
"""UUTEL: Universal AI Provider for LiteLLM

This package extends LiteLLM's provider ecosystem by implementing custom providers
for Claude Code, Gemini CLI, Google Cloud Code, and OpenAI Codex.
"""

from __future__ import annotations

try:
    from uutel._version import __version__
except ImportError:
    # Fallback version when running from source without installation
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
