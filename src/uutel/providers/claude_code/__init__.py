# this_file: src/uutel/providers/claude_code/__init__.py
"""Claude Code provider for UUTEL.

This module implements the ClaudeCodeUU provider for integrating with Claude Code
via CLI subprocess execution.
"""

from __future__ import annotations

from uutel.providers.claude_code.provider import ClaudeCodeUU

__all__: list[str] = ["ClaudeCodeUU"]
