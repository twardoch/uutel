# this_file: src/uutel/providers/codex/__init__.py
"""OpenAI Codex provider for UUTEL.

This module implements the CodexUU provider for integrating with OpenAI Codex
via session token management and ChatGPT backend integration.
"""

from __future__ import annotations

from .provider import CodexUU

__all__ = ["CodexUU"]
