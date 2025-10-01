# this_file: tests/test_cli_helpers.py
"""Tests for low-level CLI helper utilities."""

from __future__ import annotations

import errno
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from uutel.__main__ import (
    _extract_provider_metadata,
    _safe_output,
    _scrub_control_sequences,
)


class TestScrubControlSequences:
    """Ensure terminal control sequences are fully stripped before output."""

    def test_removes_device_control_strings_and_preserves_tabs(self) -> None:
        """Device control strings (ESC P ... ST) should be stripped while keeping tabs/newlines."""

        esc = ""
        payload = "".join(
            [
                "Start	",
                esc + "P1;2|ignored payload",
                "Middle",
                esc + 'Ptmux;"ignored"' + esc + "\\",
                "End\r\n",
            ]
        )

        cleaned = _scrub_control_sequences(payload)

        assert cleaned == "Start	MiddleEnd\r\n"

    def test_removes_osc_sequences_and_preserves_whitespace(self) -> None:
        """Operating system command sequences should be removed entirely."""

        payload = "Hello ]0;titleworld![31m![0m\n"

        cleaned = _scrub_control_sequences(payload)

        assert cleaned == "Hello world!!\n"

    def test_removes_c1_control_sequences(self) -> None:
        """8-bit CSI sequences should be stripped while leaving visible text intact."""

        payload = "\x9b31mAlert\x9b0m!"

        cleaned = _scrub_control_sequences(payload)

        assert cleaned == "Alert!"

    def test_removes_direct_c1_string_sequences(self) -> None:
        """8-bit OSC/DCS/APC/PM strings should be removed completely."""

        payload = "".join(
            [
                "\x90ignored payload\x9c",
                "Visible",
                "\x9d1;title\x07",
                " Text",
                "\x9fapp command\x9c",
                "\x9eprivacy message\x9c",
            ]
        )

        cleaned = _scrub_control_sequences(payload)

        assert cleaned == "Visible Text"


class TestSafeOutput:
    """_safe_output should guard against broken pipes and strip control bytes."""

    def test_prints_scrubbed_text_to_stdout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ANSI and OSC control sequences should be removed before printing."""

        captured: dict[str, Any] = {}

        def _mock_print(text: str, *, end: str, file: Any, flush: bool) -> None:
            captured["text"] = text
            captured["end"] = end
            captured["file"] = file
            captured["flush"] = flush

        monkeypatch.setattr("builtins.print", _mock_print)

        message = "Hi\x1b]2;ignored\x07 there\x1b[35m!\x1b[0m"

        _safe_output(message, target="stdout", end="", flush=True)

        assert captured["text"] == "Hi there!"
        assert captured["end"] == ""
        assert captured["file"] is sys.stdout
        assert captured["flush"] is True

    def test_accepts_bytes_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bytes payloads should be decoded, scrubbed, and printed without prefixes."""

        captured: dict[str, Any] = {}

        def _mock_print(text: str, *, end: str, file: Any, flush: bool) -> None:
            captured["text"] = text
            captured["end"] = end
            captured["file"] = file
            captured["flush"] = flush

        monkeypatch.setattr("builtins.print", _mock_print)

        payload = b"Status:\x9b32m ready\x9c!"

        _safe_output(payload, target="stdout", end="", flush=True)

        assert captured["text"] == "Status: ready!"
        assert captured["end"] == ""
        assert captured["file"] is sys.stdout
        assert captured["flush"] is True

    def test_swallow_broken_pipe_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """BrokenPipeError from print should be suppressed."""

        def _raise_broken_pipe(*_args: Any, **_kwargs: Any) -> None:
            raise BrokenPipeError

        monkeypatch.setattr("builtins.print", _raise_broken_pipe)

        _safe_output("payload", target="stderr")

        def _raise_os_error(*_args: Any, **_kwargs: Any) -> None:
            raise OSError(errno.EPIPE, "Bad pipe")

        monkeypatch.setattr("builtins.print", _raise_os_error)

        _safe_output("payload", target="stderr")

    def test_rejects_unknown_target(self) -> None:
        """Unknown target streams should raise ValueError to expose typos."""

        with pytest.raises(ValueError) as exc_info:
            _safe_output("payload", target="stdouterr")

        assert "target" in str(exc_info.value)


class TestExtractProviderMetadata:
    """Metadata extraction should handle diverse LiteLLM error shapes."""

    def test_reads_attributes_directly(self) -> None:
        """Provider/model attributes should be returned when present."""

        error = SimpleNamespace(provider="codex", model="gpt-4o")

        provider, model = _extract_provider_metadata(error)  # type: ignore[arg-type]

        assert provider == "codex"
        assert model == "gpt-4o"

    def test_reads_kwargs_when_present(self) -> None:
        """LiteLLM exceptions often stash metadata inside kwargs."""

        error = SimpleNamespace(
            kwargs={"llm_provider": "claude", "model": "claude-sonnet-4"}
        )

        provider, model = _extract_provider_metadata(error)  # type: ignore[arg-type]

        assert provider == "claude"
        assert model == "claude-sonnet-4"

    def test_falls_back_to_none_when_missing(self) -> None:
        """Absent metadata should render as (None, None)."""

        provider, model = _extract_provider_metadata(Exception("oops"))

        assert provider is None
        assert model is None
