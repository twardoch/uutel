# this_file: src/uutel/providers/gemini_cli/__init__.py
"""Gemini CLI provider for UUTEL.

This module implements the GeminiCLIUU provider for integrating with Gemini
via multiple authentication methods (API key, CLI OAuth).
"""

from __future__ import annotations

from uutel.providers.gemini_cli.provider import GeminiCLIUU

__all__: list[str] = ["GeminiCLIUU"]
