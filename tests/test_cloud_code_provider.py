# this_file: tests/test_cloud_code_provider.py
"""Tests for Google Cloud Code provider implementation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from litellm.types.utils import ModelResponse

from uutel.core.exceptions import UUTELError
from uutel.providers.cloud_code import CloudCodeUU
from uutel.providers.cloud_code.provider import _API_KEY_ENV_VARS, _PROJECT_ENV_VARS

FIXTURE_PATH = (
    Path(__file__).parent
    / "data"
    / "providers"
    / "cloud_code"
    / "simple_completion.json"
)


def _load_fixture() -> dict[str, Any]:
    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _make_model_response() -> ModelResponse:
    response = ModelResponse()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = ""
    response.choices[0].message.tool_calls = None
    response.choices[0].finish_reason = None
    response.usage = None
    return response


class DummyHTTPResponse:
    """Minimal httpx-style response returning a prepared JSON payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:  # pragma: no cover - no error branch in stub
        return None


class DummyHTTPClient:
    """Collect requests made by the provider for assertions."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.last_request: dict[str, Any] | None = None

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
    ) -> DummyHTTPResponse:
        self.last_request = {"url": url, "headers": headers or {}, "json": json or {}}
        return DummyHTTPResponse(self.payload)

    def close(self) -> None:  # pragma: no cover - included for interface parity
        return None


class DummyStreamResponse:
    """Context manager mirroring httpx stream responses for SSE payloads."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __enter__(self) -> DummyStreamResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - nothing to do
        return None

    def iter_lines(self) -> Iterator[str]:
        yield from self._lines


class DummyStreamingClient:
    """Stub httpx client exposing a stream method for SSE testing."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.last_request: dict[str, Any] | None = None

    def stream(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
    ):
        self.last_request = {
            "method": method,
            "url": url,
            "headers": headers or {},
            "json": json or {},
        }
        return DummyStreamResponse(self.lines)

    def close(self) -> None:  # pragma: no cover - parity with httpx
        return None


@pytest.fixture
def cloud_code_payload() -> dict[str, Any]:
    return _load_fixture()


def test_get_api_key_returns_none_for_blank_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace-only API key env vars should not be treated as configured."""

    provider = CloudCodeUU()
    for env_var in _API_KEY_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
        monkeypatch.setenv(env_var, "   ")

    assert provider._get_api_key() is None, (
        "Blank env vars should be ignored when resolving API key"
    )


def test_get_api_key_strips_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leading/trailing whitespace should be removed from detected API keys."""

    provider = CloudCodeUU()
    for env_var in _API_KEY_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_API_KEY", "  secret-key \n")

    assert provider._get_api_key() == "secret-key", (
        "API key resolution should trim surrounding whitespace"
    )


def test_completion_when_api_key_provided_then_posts_internal_endpoint(
    cloud_code_payload: dict[str, Any],
) -> None:
    provider = CloudCodeUU()
    provider._get_api_key = lambda: "test-key"  # type: ignore[assignment]
    provider._load_oauth_credentials = lambda: "unused"  # type: ignore[assignment]

    client = DummyHTTPClient(cloud_code_payload)
    model_response = _make_model_response()
    optional_params = {
        "temperature": 0.2,
        "max_tokens": 64,
        "project_id": "my-project",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "list_tasks",
                    "description": "List build tasks",
                    "parameters": {
                        "type": "object",
                        "properties": {"status": {"type": "string"}},
                    },
                },
            }
        ],
        "tool_choice": {"type": "tool", "tool_name": "list_tasks"},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "TaskResponse",
                "schema": {
                    "type": "object",
                    "properties": {
                        "tasks": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["tasks"],
                },
            },
        },
    }

    result = provider.completion(
        model="gemini-2.5-pro",
        messages=[{"role": "user", "content": "List deploy steps"}],
        api_base="https://cloudcode-pa.googleapis.com",
        custom_prompt_dict={},
        model_response=model_response,
        print_verbose=Mock(),
        encoding="utf-8",
        api_key=None,
        logging_obj=Mock(),
        optional_params=optional_params,
        client=client,
    )

    assert client.last_request is not None, "Client should record request"
    assert client.last_request["url"].endswith("/v1internal:generateContent"), (
        "Unexpected endpoint"
    )
    sent_headers = client.last_request["headers"]
    assert sent_headers.get("x-goog-api-key") == "test-key", "API key header missing"
    body = client.last_request["json"]
    assert body["project"] == "my-project", "Project id not populated"
    generation = body["request"]["generationConfig"]
    assert generation["temperature"] == pytest.approx(0.2), "Temperature mismatch"
    assert generation["maxOutputTokens"] == 64, "Max tokens mismatch"
    tools = body["request"]["tools"][0]["functionDeclarations"]
    assert tools and tools[0]["name"] == "list_tasks", "Tool declaration missing"
    tool_config = body["request"]["toolConfig"]["functionCallingConfig"]
    assert tool_config["mode"] == "ANY", "Tool config mode mismatch"
    assert tool_config["allowedFunctionNames"] == ["list_tasks"], (
        "Tool choice not enforced"
    )
    assert result.choices[0].message.content.strip(), (
        "Completion should populate content"
    )
    assert result.choices[0].finish_reason == "stop", "Finish reason should be stop"
    assert result.usage["total_tokens"] == 120, "Usage totals should map from payload"


def test_completion_when_oauth_credentials_used_then_authorization_header_set(
    cloud_code_payload: dict[str, Any],
) -> None:
    provider = CloudCodeUU()
    provider._get_api_key = lambda: None  # type: ignore[assignment]
    provider._load_oauth_credentials = lambda: "oauth-token"  # type: ignore[assignment]

    client = DummyHTTPClient(cloud_code_payload)
    model_response = _make_model_response()

    provider.completion(
        model="gemini-2.5-pro",
        messages=[{"role": "user", "content": "Hello"}],
        api_base="",
        custom_prompt_dict={},
        model_response=model_response,
        print_verbose=Mock(),
        encoding="utf-8",
        api_key=None,
        logging_obj=Mock(),
        optional_params={"project_id": "proj-123"},
        client=client,
    )

    assert client.last_request is not None, "OAuth request should be captured"
    headers = client.last_request["headers"]
    assert headers.get("Authorization") == "Bearer oauth-token", "OAuth header missing"


def test_completion_when_function_call_returned_then_tool_calls_populated() -> None:
    provider = CloudCodeUU()
    provider._get_api_key = lambda: "key"  # type: ignore[assignment]

    payload = {
        "response": {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "deploy_service",
                                    "args": {"env": "prod"},
                                }
                            }
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }
    }

    client = DummyHTTPClient(payload)
    model_response = _make_model_response()

    result = provider.completion(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Deploy"}],
        api_base="",
        custom_prompt_dict={},
        model_response=model_response,
        print_verbose=Mock(),
        encoding="utf-8",
        api_key=None,
        logging_obj=Mock(),
        optional_params={"project_id": "proj"},
        client=client,
    )

    tool_calls = result.choices[0].message.tool_calls
    assert tool_calls, "Tool calls should be populated from functionCall"
    assert tool_calls[0]["function"]["name"] == "deploy_service", "Tool name mismatch"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"env": "prod"}, (
        "Arguments not preserved"
    )


def test_streaming_when_sse_chunks_returned_then_text_chunks_emitted() -> None:
    provider = CloudCodeUU()
    provider._get_api_key = lambda: "api"  # type: ignore[assignment]

    payload_lines = [
        'data: {"response": {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}}\n',
        'data: {"response": {"candidates": [{"content": {"parts": [{"text": " world"}]}, "finishReason": "STOP"}], "usageMetadata": {"totalTokenCount": 200}}}\n',
        "data: [DONE]\n",
    ]

    client = DummyStreamingClient(payload_lines)
    model_response = _make_model_response()

    chunks = list(
        provider.streaming(
            model="gemini-2.5-pro",
            messages=[{"role": "user", "content": "Stream"}],
            api_base="",
            custom_prompt_dict={},
            model_response=model_response,
            print_verbose=Mock(),
            encoding="utf-8",
            api_key=None,
            logging_obj=Mock(),
            optional_params={"project_id": "proj"},
            client=client,
        )
    )

    assert client.last_request is not None, "Streaming request not captured"
    assert client.last_request["method"] == "POST", "Streaming should use POST"
    assert len(chunks) == 2, "Streaming should yield two chunks"
    assert chunks[0]["text"] == "hello", "First chunk text mismatch"
    assert not chunks[0]["is_finished"], "First chunk should not be finished"
    assert chunks[1]["text"].strip() == "world", "Second chunk text mismatch"
    assert chunks[1]["is_finished"], "Second chunk should flag completion"
    assert chunks[1]["usage"]["total_tokens"] == 200, "Streaming usage not propagated"


def test_resolve_project_id_trims_optional_parameter() -> None:
    provider = CloudCodeUU()

    project = provider._resolve_project_id({"project_id": "  example-123  "})

    assert project == "example-123"


def test_resolve_project_id_prefers_environment_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = CloudCodeUU()

    for env_var in _PROJECT_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    monkeypatch.setenv("CLOUD_CODE_PROJECT", "  env-project  ")

    project = provider._resolve_project_id({})

    assert project == "env-project"


def test_resolve_project_id_raises_with_guidance_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = CloudCodeUU()

    for env_var in _PROJECT_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    with pytest.raises(UUTELError) as exc:
        provider._resolve_project_id({})

    message = str(exc.value)
    assert "CLOUD_CODE_PROJECT" in message
    assert "project id" in message.lower()
