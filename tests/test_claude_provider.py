# this_file: tests/test_claude_provider.py
"""Test suite covering Claude Code provider behaviours."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Event
from typing import Any
from unittest.mock import Mock

import pytest
from litellm.types.utils import ModelResponse

from uutel.core.exceptions import UUTELError
from uutel.core.runners import SubprocessResult
from uutel.providers.claude_code import ClaudeCodeUU

FIXTURE_PATH = (
    Path(__file__).parent / "data" / "providers" / "claude" / "simple_completion.json"
)


def _load_fixture() -> dict[str, Any]:
    """Load deterministic CLI fixture payload."""

    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _make_model_response() -> ModelResponse:
    """Create a reusable ModelResponse stub."""

    response = ModelResponse()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = ""
    response.choices[0].message.tool_calls = None
    response.choices[0].finish_reason = None
    response.usage = None
    return response


@pytest.fixture
def claude_payload() -> dict[str, Any]:
    """Return a cached Claude CLI payload."""

    return _load_fixture()


def test_completion_when_cli_emits_json_then_model_response_populated(
    monkeypatch: pytest.MonkeyPatch,
    claude_payload: dict[str, Any],
) -> None:
    """Completion should parse JSON stdout and populate ModelResponse."""

    provider = ClaudeCodeUU()
    monkeypatch.setattr(provider, "_resolve_cli", lambda: "claude")

    recorded: dict[str, Any] = {}

    def _fake_run_subprocess(*args: Any, **kwargs: Any) -> SubprocessResult:
        command = list(args[0])
        recorded["command"] = command
        recorded["env"] = kwargs.get("env")
        recorded["timeout"] = kwargs.get("timeout")
        return SubprocessResult(
            command=tuple(command),
            returncode=0,
            stdout=json.dumps(claude_payload),
            stderr="",
            duration_seconds=0.42,
        )

    monkeypatch.setattr(
        "uutel.providers.claude_code.provider.run_subprocess",
        _fake_run_subprocess,
    )

    model_response = _make_model_response()
    optional_params = {
        "temperature": 0.35,
        "max_tokens": 256,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_docs",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ],
        "tool_choice": {"type": "tool", "tool_name": "search_docs"},
        "working_dir": "/tmp/project",
        "timeout": 90.0,
        "environment": {"CLAUDE_REGION": "us"},
    }

    result = provider.completion(
        model="claude-sonnet-4",
        messages=[
            {"role": "system", "content": "You are careful."},
            {"role": "user", "content": "Summarise the repo."},
        ],
        api_base="",
        custom_prompt_dict={},
        model_response=model_response,
        print_verbose=Mock(),
        encoding="utf-8",
        api_key=None,
        logging_obj=Mock(),
        optional_params=optional_params,
    )

    assert recorded["command"][:3] == ["claude", "api", "--json"], (
        "CLI invocation prefix should request JSON output"
    )
    assert "--model" in recorded["command"], "Model flag missing"
    assert recorded["timeout"] == pytest.approx(90.0), "Timeout should propagate"
    env = recorded["env"] or {}
    conversation_payload = json.loads(env["CLAUDE_CONVERSATION_JSON"])
    assert conversation_payload["messages"][0]["role"] == "system"
    assert conversation_payload["messages"][1]["content"] == "Summarise the repo."
    assert env["CLAUDE_ALLOWED_TOOLS"] == "search_docs"
    assert env["CLAUDE_WORKING_DIR"] == "/tmp/project"

    assert result.choices[0].message.content.strip() == claude_payload["result"].strip()
    assert result.choices[0].finish_reason == "stop"
    assert result.usage["input_tokens"] == claude_payload["usage"]["input_tokens"]
    assert result.usage["output_tokens"] == claude_payload["usage"]["output_tokens"]
    assert result.usage["total_tokens"] == (
        claude_payload["usage"]["input_tokens"]
        + claude_payload["usage"]["output_tokens"]
    )


def test_streaming_when_jsonl_events_then_chunks_emitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming should convert CLI JSONL events into GenericStreamingChunks."""

    provider = ClaudeCodeUU()
    monkeypatch.setattr(provider, "_resolve_cli", lambda: "claude")

    stream_lines = [
        json.dumps({"type": "event", "subtype": "partial", "result": "chunk one"}),
        json.dumps({"type": "event", "subtype": "partial", "result": "chunk two"}),
        json.dumps(
            {
                "type": "event",
                "subtype": "tool_call",
                "tool": {"name": "search_docs", "input": {"query": "docs"}},
            }
        ),
        json.dumps(
            {
                "type": "result",
                "result": "chunk onechunk two",
                "usage": {"input_tokens": 10, "output_tokens": 20},
                "finish_reason": "stop",
            }
        ),
    ]

    recorded: dict[str, Any] = {}

    def _fake_stream_subprocess_lines(*args: Any, **kwargs: Any):
        recorded["command"] = list(args[0])
        recorded["timeout"] = kwargs.get("timeout")
        recorded["env"] = kwargs.get("env")
        yield from stream_lines

    monkeypatch.setattr(
        "uutel.providers.claude_code.provider.stream_subprocess_lines",
        _fake_stream_subprocess_lines,
    )

    model_response = _make_model_response()
    chunks = list(
        provider.streaming(
            model="claude-opus-4",
            messages=[{"role": "user", "content": "Stream"}],
            api_base="",
            custom_prompt_dict={},
            model_response=model_response,
            print_verbose=Mock(),
            encoding="utf-8",
            api_key=None,
            logging_obj=Mock(),
            optional_params={"timeout": 120.0},
        )
    )

    assert recorded["command"][0] == "claude"
    assert len(chunks) == 4, "Expected text, tool, and final chunks"
    assert [chunk["text"] for chunk in chunks[:2]] == ["chunk one", "chunk two"]
    tool_chunk = chunks[2]
    assert tool_chunk["is_finished"] is False
    assert tool_chunk["tool_use"]["name"] == "search_docs"
    final_chunk = chunks[3]
    assert final_chunk["is_finished"] is True
    assert final_chunk["usage"]["total_tokens"] == 30
    assert final_chunk["text"] == "chunk onechunk two"


def test_streaming_when_cancel_event_set_then_raises_uutelerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation event should abort streaming with a helpful error."""

    provider = ClaudeCodeUU()
    monkeypatch.setattr(provider, "_resolve_cli", lambda: "claude")

    def _fake_stream(*args: Any, **kwargs: Any):
        yield json.dumps({"type": "event", "subtype": "partial", "result": "chunk"})
        yield json.dumps({"type": "event", "subtype": "partial", "result": "chunk 2"})

    monkeypatch.setattr(
        "uutel.providers.claude_code.provider.stream_subprocess_lines",
        _fake_stream,
    )

    cancel_event = Event()
    cancel_event.set()

    model_response = _make_model_response()

    with pytest.raises(UUTELError) as excinfo:
        list(
            provider.streaming(
                model="claude-sonnet-4",
                messages=[{"role": "user", "content": "cancel"}],
                api_base="",
                custom_prompt_dict={},
                model_response=model_response,
                print_verbose=Mock(),
                encoding="utf-8",
                api_key=None,
                logging_obj=Mock(),
                optional_params={"cancel_event": cancel_event},
            )
        )

    assert "cancelled" in str(excinfo.value).lower()


def test_completion_when_cli_missing_then_raises_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider should raise descriptive error when CLI is unavailable."""

    provider = ClaudeCodeUU()
    monkeypatch.setattr(
        provider,
        "_resolve_cli",
        lambda: (_ for _ in ()).throw(UUTELError("missing", provider="claude_code")),
    )

    model_response = _make_model_response()

    with pytest.raises(UUTELError):
        provider.completion(
            model="claude-sonnet-4",
            messages=[{"role": "user", "content": "hello"}],
            api_base="",
            custom_prompt_dict={},
            model_response=model_response,
            print_verbose=Mock(),
            encoding="utf-8",
            api_key=None,
            logging_obj=Mock(),
            optional_params={},
        )
