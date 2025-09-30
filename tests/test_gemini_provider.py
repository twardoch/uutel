# this_file: tests/test_gemini_provider.py
"""Test suite covering Gemini provider API and CLI behaviours."""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from litellm.types.utils import ModelResponse

from uutel.core.exceptions import UUTELError
from uutel.core.runners import SubprocessResult
from uutel.providers.gemini_cli import GeminiCLIUU

FIXTURE_PATH = (
    Path(__file__).parent / "data" / "providers" / "gemini" / "simple_completion.json"
)


def _load_fixture() -> dict[str, Any]:
    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _make_model_response() -> ModelResponse:
    response = ModelResponse()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = ""
    response.choices[0].finish_reason = None
    response.usage = None
    return response


@pytest.fixture
def gemini_payload() -> dict[str, Any]:
    return _load_fixture()


@pytest.fixture
def stub_genai(monkeypatch: pytest.MonkeyPatch, gemini_payload: dict[str, Any]):
    """Provide a stub google.generativeai module for deterministic testing."""

    calls: dict[str, list[Any]] = {"configure": [], "generations": []}

    class StubResponse:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        @property
        def candidates(self) -> list[dict[str, Any]]:
            return self._payload.get("candidates", [])

        @property
        def usage_metadata(self) -> dict[str, Any]:
            return self._payload.get("usageMetadata", {})

        @property
        def text(self) -> str:
            parts = (
                self._payload.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
            )
            return "".join(part.get("text", "") for part in parts)

        def to_dict(self) -> dict[str, Any]:  # pragma: no cover - compatibility shim
            return self._payload

    class StubGenerativeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name
            self.calls: list[dict[str, Any]] = []

        def generate_content(self, contents, **kwargs):
            record = {"contents": contents, **kwargs}
            self.calls.append(record)
            calls["generations"].append(record)
            if kwargs.get("stream"):
                chunk_1 = StubResponse(
                    {
                        "candidates": [
                            {
                                "content": {"parts": [{"text": "chunk one"}]},
                                "finishReason": None,
                            }
                        ]
                    }
                )
                chunk_2 = StubResponse(
                    {
                        "candidates": [
                            {
                                "content": {"parts": [{"text": "chunk two"}]},
                                "finishReason": "STOP",
                            }
                        ]
                    }
                )
                yield chunk_1
                yield chunk_2
                return
            return StubResponse(gemini_payload)

    stub_module = types.SimpleNamespace()
    stub_module.configure = lambda **kwargs: calls["configure"].append(kwargs)
    stub_module.GenerativeModel = StubGenerativeModel
    stub_module.types = types.SimpleNamespace()
    stub_module.types.generation_types = types.SimpleNamespace()
    stub_module.types.generation_types.StopCandidateException = RuntimeError
    monkeypatch.setattr(
        "uutel.providers.gemini_cli.provider.genai", stub_module, raising=False
    )
    return stub_module, calls


def test_completion_with_api_key_uses_google_api(
    monkeypatch: pytest.MonkeyPatch,
    stub_genai,
) -> None:
    module, calls = stub_genai
    provider = GeminiCLIUU()
    monkeypatch.setattr(provider, "_get_api_key", lambda: "test-api-key")
    model_response = _make_model_response()
    optional_params = {
        "temperature": 0.25,
        "max_tokens": 128,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "list_colours",
                    "description": "List colours in a palette",
                    "parameters": {
                        "type": "object",
                        "properties": {"palette": {"type": "string"}},
                        "required": ["palette"],
                    },
                },
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "ColourResponse",
                "schema": {
                    "type": "object",
                    "properties": {
                        "colours": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["colours"],
                },
            },
        },
    }

    result = provider.completion(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "What is Gemini?"}],
        api_base="",
        custom_prompt_dict={},
        model_response=model_response,
        print_verbose=Mock(),
        encoding="utf-8",
        api_key=None,
        logging_obj=Mock(),
        optional_params=optional_params,
    )

    assert result.choices[0].message.content.strip(), (
        "Gemini completion should populate content"
    )
    assert result.choices[0].finish_reason == "stop"
    configure_call = calls["configure"][0]
    assert configure_call["api_key"] == "test-api-key"
    generation_call = calls["generations"][0]
    assert generation_call["generation_config"]["temperature"] == pytest.approx(0.25)
    assert generation_call["generation_config"]["max_output_tokens"] == 128
    response_schema = generation_call["generation_config"]["response_schema"]
    assert response_schema["title"] == "ColourResponse"
    tool_payload = generation_call["tools"][0]["function_declarations"][0]
    assert tool_payload["name"] == "list_colours"
    assert tool_payload["parameters"]["required"] == ["palette"]


def test_streaming_with_api_key_yields_chunks(
    monkeypatch: pytest.MonkeyPatch,
    stub_genai,
) -> None:
    provider = GeminiCLIUU()
    monkeypatch.setattr(provider, "_get_api_key", lambda: "stream-key")
    model_response = _make_model_response()

    chunks = list(
        provider.streaming(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Stream a greeting."}],
            api_base="",
            custom_prompt_dict={},
            model_response=model_response,
            print_verbose=Mock(),
            encoding="utf-8",
            api_key=None,
            logging_obj=Mock(),
            optional_params={},
        )
    )

    assert len(chunks) == 2
    assert chunks[0]["text"] == "chunk one"
    assert not chunks[0]["is_finished"]
    assert chunks[1]["text"] == "chunk two"
    assert chunks[1]["is_finished"]


def test_cli_fallback_uses_refreshed_credentials(
    monkeypatch: pytest.MonkeyPatch,
    gemini_payload: dict[str, Any],
) -> None:
    provider = GeminiCLIUU()
    monkeypatch.setattr(provider, "_get_api_key", lambda: None)
    monkeypatch.setattr(provider, "_check_gemini_cli", lambda: True)

    credential_calls: list[dict[str, Any]] = []

    def fake_load_cli_credentials(**kwargs):
        credential_calls.append(kwargs)
        payload = {
            "access_token": "cli-token",
            "expires_at": "2099-01-01T00:00:00Z",
        }
        return Path("/tmp/creds.json"), payload

    monkeypatch.setattr(
        "uutel.providers.gemini_cli.provider.load_cli_credentials",
        fake_load_cli_credentials,
        raising=False,
    )

    invoked_commands: list[tuple[str, ...]] = []

    def fake_run_subprocess(command, **kwargs):
        invoked_commands.append(tuple(command))
        return SubprocessResult(
            command=tuple(command),
            returncode=0,
            stdout=json.dumps(gemini_payload),
            stderr="",
            duration_seconds=0.05,
        )

    monkeypatch.setattr(
        "uutel.providers.gemini_cli.provider.run_subprocess",
        fake_run_subprocess,
        raising=False,
    )

    model_response = _make_model_response()
    result = provider.completion(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Describe Gemini"}],
        api_base="",
        custom_prompt_dict={},
        model_response=model_response,
        print_verbose=Mock(),
        encoding="utf-8",
        api_key=None,
        logging_obj=Mock(),
        optional_params={},
    )

    assert result.choices[0].message.content.strip(), (
        "CLI fallback should yield content"
    )
    assert credential_calls, "Expected CLI credentials to be loaded"
    refresh_command = tuple(credential_calls[0]["refresh_command"])
    assert refresh_command == ("gemini", "login")
    assert invoked_commands, "Gemini CLI should be invoked when API key missing"


def test_cli_fallback_raises_when_cli_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = GeminiCLIUU()
    monkeypatch.setattr(provider, "_get_api_key", lambda: None)
    monkeypatch.setattr(provider, "_check_gemini_cli", lambda: False)
    model_response = _make_model_response()

    with pytest.raises(UUTELError):
        provider.completion(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hello"}],
            api_base="",
            custom_prompt_dict={},
            model_response=model_response,
            print_verbose=Mock(),
            encoding="utf-8",
            api_key=None,
            logging_obj=Mock(),
            optional_params={},
        )
