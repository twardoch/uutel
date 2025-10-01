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
from uutel.providers.gemini_cli.provider import (
    _DEFAULT_MAX_TOKENS,
    _DEFAULT_TEMPERATURE,
    _GEMINI_ENV_VARS,
)

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


def test_extract_cli_completion_payload_handles_cli_noise() -> None:
    """Trailing CLI control sequences should not break JSON extraction."""

    provider = GeminiCLIUU()
    raw_json = {"gemini": {"text": "Ready for deployment"}}
    stdout = "".join(
        [
            "\x1b[2K\rPreparing...\n",
            "\x1b[?25l",
            "\x1b]0;Gemini CLI\x07",
            '{"gemini": {"text": "Ready for deployment"}}\n',
        ]
    )

    extracted = provider._extract_cli_completion_payload(stdout)

    assert extracted == raw_json, "CLI extraction should return the final JSON object"


def test_parse_cli_stream_lines_handles_prefixed_json() -> None:
    """Streaming lines prefixed with control sequences should still produce chunks."""

    provider = GeminiCLIUU()
    noisy_lines = [
        '\x1b[2K\r\x1b]0;Gemini CLI\x07{"type": "message", "data": {"text": "chunk one"}}',
        '\x1b[2K\r{"type": "message", "data": {"text": "chunk two"}}',
        '\x1b[2K\r{"type": "finish", "data": {"finish_reason": "STOP"}}',
    ]

    cleaned = [provider._strip_ansi_sequences(line) for line in noisy_lines]
    chunks = provider._parse_cli_stream_lines(cleaned)

    assert len(chunks) == 3, "Two text chunks and one finish chunk should be produced"
    assert chunks[0]["text"] == "chunk one"
    assert chunks[-1]["is_finished"], "Last chunk should mark stream completion"


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


def test_cli_fallback_strips_preamble_logs(
    monkeypatch: pytest.MonkeyPatch,
    gemini_payload: dict[str, Any],
) -> None:
    """CLI completion should tolerate banner output before JSON payload."""

    provider = GeminiCLIUU()
    monkeypatch.setattr(provider, "_get_api_key", lambda: None)
    monkeypatch.setattr(provider, "_check_gemini_cli", lambda: True)

    monkeypatch.setattr(
        "uutel.providers.gemini_cli.provider.load_cli_credentials",
        lambda **kwargs: (Path("/tmp/creds.json"), {"access_token": "cli-token"}),
        raising=False,
    )

    banner = "WARNING: Using cached session\nGemini CLI 1.4.2\n"
    trailer = "\nDone in 0.8s\n"
    stdout = banner + json.dumps(gemini_payload) + trailer

    monkeypatch.setattr(
        "uutel.providers.gemini_cli.provider.run_subprocess",
        lambda *args, **kwargs: SubprocessResult(
            command=tuple(args[0]),
            returncode=0,
            stdout=stdout,
            stderr="",
            duration_seconds=0.05,
        ),
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

    assert "Gemini" in result.choices[0].message.content


def test_cli_fallback_raises_structured_error_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structured CLI error JSON should raise a UUTELError with context."""

    provider = GeminiCLIUU()
    monkeypatch.setattr(provider, "_get_api_key", lambda: None)
    monkeypatch.setattr(provider, "_check_gemini_cli", lambda: True)

    monkeypatch.setattr(
        "uutel.providers.gemini_cli.provider.load_cli_credentials",
        lambda **kwargs: (Path("/tmp/creds.json"), {"access_token": "cli-token"}),
        raising=False,
    )

    error_payload = {
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "message": "Quota exceeded for project",
        }
    }

    monkeypatch.setattr(
        "uutel.providers.gemini_cli.provider.run_subprocess",
        lambda *args, **kwargs: SubprocessResult(
            command=tuple(args[0]),
            returncode=0,
            stdout=json.dumps(error_payload),
            stderr="",
            duration_seconds=0.04,
        ),
        raising=False,
    )

    model_response = _make_model_response()

    with pytest.raises(UUTELError) as exc_info:
        provider.completion(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Trigger error"}],
            api_base="",
            custom_prompt_dict={},
            model_response=model_response,
            print_verbose=Mock(),
            encoding="utf-8",
            api_key=None,
            logging_obj=Mock(),
            optional_params={},
        )

    message = str(exc_info.value)
    assert "RESOURCE_EXHAUSTED" in message
    assert "Quota exceeded" in message
    assert "429" in message
    assert exc_info.value.provider == "gemini_cli"


def test_cli_streaming_yields_incremental_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI streaming should emit incremental GenericStreamingChunk objects."""

    provider = GeminiCLIUU()
    monkeypatch.setattr(provider, "_get_api_key", lambda: None)
    monkeypatch.setattr(provider, "_check_gemini_cli", lambda: True)

    monkeypatch.setattr(
        "uutel.providers.gemini_cli.provider.load_cli_credentials",
        lambda **kwargs: (Path("/tmp/creds.json"), {"access_token": "cli-token"}),
        raising=False,
    )

    fake_lines = [
        '{"type": "text-delta", "text": "Hello"}',
        '{"type": "text-delta", "text": " world"}',
        '{"type": "finish", "reason": "STOP"}',
    ]

    def fake_stream_subprocess_lines(command, **kwargs):
        yield from fake_lines

    monkeypatch.setattr(
        "uutel.providers.gemini_cli.provider.stream_subprocess_lines",
        fake_stream_subprocess_lines,
        raising=False,
    )

    model_response = _make_model_response()
    chunks = list(
        provider.streaming(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Stream via CLI"}],
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

    assert [chunk["text"] for chunk in chunks] == ["Hello", " world", ""]
    assert [chunk["is_finished"] for chunk in chunks] == [False, False, True]
    assert chunks[-1]["finish_reason"] == "stop"


def test_cli_streaming_sanitises_mixed_json_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed CLI JSON payloads should strip control bytes and ignore tool events."""

    provider = GeminiCLIUU()
    monkeypatch.setattr(provider, "_get_api_key", lambda: None)
    monkeypatch.setattr(provider, "_check_gemini_cli", lambda: True)

    monkeypatch.setattr(
        "uutel.providers.gemini_cli.provider.load_cli_credentials",
        lambda **kwargs: (Path("/tmp/creds.json"), {"access_token": "cli-token"}),
        raising=False,
    )

    raw_lines = [
        '\x1b[2K{"type": "message", "data": {"type": "text-delta", "text": "Alpha"}}',
        '{"type": "message", "data": {"type": "text-delta", "text": "\x1b[31mBeta\x1b[0m"}}',
        '{"type": "message", "data": {"type": "tool_call", "text": "skip"}}',
        '{"type": "message", "data": {"type": "text-delta", "text": "Gamma"}}',
        '{"type": "finish", "data": {"finish_reason": "STOP"}}',
    ]

    monkeypatch.setattr(
        "uutel.providers.gemini_cli.provider.stream_subprocess_lines",
        lambda *args, **kwargs: (line for line in raw_lines),
        raising=False,
    )

    model_response = _make_model_response()
    chunks = list(
        provider.streaming(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Stream via CLI"}],
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

    texts = [chunk["text"] for chunk in chunks]
    assert texts == ["Alpha", "Beta", "Gamma", ""], (
        "Control sequences should be stripped and tool events ignored"
    )
    assert [chunk["finish_reason"] for chunk in chunks][-1] == "stop"


def test_cli_streaming_plain_text_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plain text CLI output should continue to emit raw chunks."""

    provider = GeminiCLIUU()
    monkeypatch.setattr(provider, "_get_api_key", lambda: None)
    monkeypatch.setattr(provider, "_check_gemini_cli", lambda: True)

    monkeypatch.setattr(
        "uutel.providers.gemini_cli.provider.load_cli_credentials",
        lambda **kwargs: (Path("/tmp/creds.json"), {"access_token": "cli-token"}),
        raising=False,
    )

    plain_lines = ["first", "second"]

    def fake_stream_subprocess_lines(command, **kwargs):
        yield from plain_lines

    monkeypatch.setattr(
        "uutel.providers.gemini_cli.provider.stream_subprocess_lines",
        fake_stream_subprocess_lines,
        raising=False,
    )

    model_response = _make_model_response()
    chunks = list(
        provider.streaming(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Stream via CLI"}],
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

    assert [chunk["text"] for chunk in chunks] == plain_lines
    assert [chunk["is_finished"] for chunk in chunks] == [False, True]


def test_cli_streaming_raises_on_fragmented_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fragmented CLI error output should collapse into a single UUTELError."""

    provider = GeminiCLIUU()
    monkeypatch.setattr(provider, "_get_api_key", lambda: None)
    monkeypatch.setattr(provider, "_check_gemini_cli", lambda: True)

    monkeypatch.setattr(
        "uutel.providers.gemini_cli.provider.load_cli_credentials",
        lambda **kwargs: (Path("/tmp/creds.json"), {"access_token": "cli-token"}),
        raising=False,
    )

    error_lines = [
        "{",
        ' "error": {',
        ' "code": 401,',
        ' "message": "Request had invalid authentication credentials.",',
        ' "status": "UNAUTHENTICATED"',
        " }",
        "}",
    ]

    def fake_stream_subprocess_lines(command, **kwargs):
        yield from error_lines

    monkeypatch.setattr(
        "uutel.providers.gemini_cli.provider.stream_subprocess_lines",
        fake_stream_subprocess_lines,
        raising=False,
    )

    model_response = _make_model_response()

    with pytest.raises(UUTELError) as exc:
        list(
            provider.streaming(
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": "Stream via CLI"}],
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

    message = str(exc.value).lower()
    assert "unauthenticated" in message or "401" in message
    assert exc.value.provider == "gemini_cli"


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


def test_get_api_key_trims_and_prioritises_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace should be stripped and precedence should favour GOOGLE_API_KEY."""

    provider = GeminiCLIUU()
    for env_var in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)

    monkeypatch.setenv("GOOGLE_API_KEY", "  primary-key  ")
    monkeypatch.setenv("GEMINI_API_KEY", "  secondary  ")
    monkeypatch.setenv("GOOGLE_GENAI_API_KEY", " fallback ")

    assert provider._get_api_key() == "primary-key", (
        "GOOGLE_API_KEY should take precedence once trimmed"
    )

    monkeypatch.setenv("GOOGLE_API_KEY", "   ")

    assert provider._get_api_key() == "secondary", (
        "Whitespace-only GOOGLE_API_KEY should fall back to GEMINI_API_KEY"
    )

    monkeypatch.setenv("GEMINI_API_KEY", "   ")

    assert provider._get_api_key() == "fallback", (
        "Fallback env var should be used when higher precedence keys blank"
    )


def test_get_api_key_returns_none_when_all_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace-only environment variables should yield no API key."""

    provider = GeminiCLIUU()
    for env_var in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY"):
        monkeypatch.setenv(env_var, "   ")

    assert provider._get_api_key() is None


def test_build_cli_env_populates_all_known_env_vars() -> None:
    """CLI env injection should populate all recognised variables with trimmed token."""

    provider = GeminiCLIUU()
    env = provider._build_cli_env({"access_token": "  secret-token  "})

    expected = dict.fromkeys(_GEMINI_ENV_VARS, "secret-token")

    assert env == expected, (
        "All Gemini env vars should receive the trimmed access token"
    )


def test_build_cli_env_returns_empty_dict_without_token() -> None:
    """CLI env injection should return empty mapping when token missing or blank."""

    provider = GeminiCLIUU()

    assert provider._build_cli_env({}) == {}, "Missing token should produce empty env"
    assert provider._build_cli_env({"access_token": "   "}) == {}, (
        "Blank token should be ignored"
    )


class TestGenerationConfigDefaults:
    """Ensure generation config handles malformed optional parameters gracefully."""

    def setup_method(self) -> None:
        """Create a fresh provider for each test."""

        self.provider = GeminiCLIUU()

    @pytest.mark.parametrize("value", [None, True, False, float("nan"), 3.5])
    def test_temperature_invalid_values_use_default(self, value: Any) -> None:
        """Invalid temperature inputs should fall back to the documented default."""

        config = self.provider._build_generation_config({"temperature": value})

        assert config["temperature"] == pytest.approx(_DEFAULT_TEMPERATURE)

    @pytest.mark.parametrize("value", [None, True, False, 0, -5, 9999999])
    def test_max_tokens_invalid_values_use_default(self, value: Any) -> None:
        """Invalid max_token inputs should fall back to the documented default."""

        config = self.provider._build_generation_config({"max_tokens": value})

        assert config["max_output_tokens"] == _DEFAULT_MAX_TOKENS

    def test_valid_values_are_preserved(self) -> None:
        """Valid numeric overrides should be preserved in the config payload."""

        config = self.provider._build_generation_config(
            {"temperature": 1.25, "max_tokens": 256}
        )

        assert config["temperature"] == pytest.approx(1.25)
        assert config["max_output_tokens"] == 256


class TestGeminiCLICommandBuilder:
    """Validate CLI command assembly for optional parameter sanitisation."""

    def setup_method(self) -> None:
        """Create a fresh provider for each test."""

        self.provider = GeminiCLIUU()

    def _get_flag_value(self, command: list[str], flag: str) -> str:
        index = command.index(flag)
        return command[index + 1]

    def test_invalid_optional_params_fall_back_to_defaults(self) -> None:
        """None and out-of-range values should not leak into the CLI command."""

        command = self.provider._build_cli_command(
            model="gemini-2.5-pro",
            messages=[{"role": "user", "content": "Hi there"}],
            optional_params={"temperature": None, "max_tokens": 0},
        )

        assert self._get_flag_value(command, "--temperature") == str(
            _DEFAULT_TEMPERATURE
        )
        assert self._get_flag_value(command, "--max-tokens") == str(_DEFAULT_MAX_TOKENS)
        assert "--stream" not in command, "Stream flag should be absent by default"

    def test_stream_flag_included_when_requested(self) -> None:
        """Explicit stream requests should append the --stream flag."""

        command = self.provider._build_cli_command(
            model="gemini-2.5-pro",
            messages=[{"role": "user", "content": "stream please"}],
            optional_params={},
            stream=True,
        )

        assert command[-2] == "--stream"
        assert command[-1].startswith("User:"), (
            "Prompt should remain the final argument"
        )


class TestGeminiContentBuilder:
    """Ensure Gemini request content normalisation handles edge cases."""

    def setup_method(self) -> None:
        self.provider = GeminiCLIUU()

    def test_build_contents_folds_system_prompt_into_first_user_part(self) -> None:
        """System prompts should prepend the first user part instead of emitting separately."""

        messages = [
            {"role": "system", "content": "Stay concise."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarise the release notes."},
                ],
            },
            {"role": "assistant", "content": "Sure thing."},
        ]

        contents = self.provider._build_contents(messages)

        assert contents[0]["role"] == "user"
        first_text = contents[0]["parts"][0]["text"]
        assert first_text.startswith("Stay concise."), (
            "System prompt should prefix the first user part"
        )
        assert "Summarise the release notes." in first_text
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"][0]["text"] == "Sure thing."

    def test_build_contents_skips_tool_and_function_blocks(self) -> None:
        """Tool/function call payloads should be omitted from Gemini parts."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Step one."},
                    {"type": "tool_call", "id": "call-1", "text": "ignored"},
                    {
                        "functionCall": {
                            "name": "search",
                            "args": {"query": "uutel"},
                        }
                    },
                    {"type": "text", "text": "Step two."},
                ],
            }
        ]

        contents = self.provider._build_contents(messages)

        texts = [part["text"] for part in contents[0]["parts"]]
        assert texts == ["Step one.", "Step two."], (
            "Tool/function payloads should be stripped from Gemini request parts"
        )

    def test_convert_message_part_handles_inline_data_uri(self) -> None:
        """Data URI images should become inline_data payloads with extracted mime type."""

        part = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,QUJDRA=="},
        }

        converted = self.provider._convert_message_part(part)

        assert converted == {
            "inline_data": {"mime_type": "image/png", "data": "QUJDRA=="}
        }


class TestGeminiCLIPromptBuilder:
    """Ensure CLI prompt assembly removes empty content and tidies whitespace."""

    def setup_method(self) -> None:
        self.provider = GeminiCLIUU()

    def test_prompt_ignores_empty_messages_and_collapses_whitespace(self) -> None:
        """None or whitespace-only content should be skipped and whitespace normalised."""

        messages = [
            {"role": "system", "content": None},
            {"role": "user", "content": "  hello   world   "},
            {"role": "assistant", "content": "line1\n\nline2\tline3"},
        ]

        prompt = self.provider._build_cli_prompt(messages)

        assert "System:" not in prompt, "Empty system message should be omitted"
        assert prompt == "User: hello world\n\nAssistant: line1 line2 line3"


class TestGeminiResponseNormalisation:
    """Regression tests for Gemini response normalisation helpers."""

    def setup_method(self) -> None:
        """Create a fresh provider instance for each test."""

        self.provider = GeminiCLIUU()

    def test_normalise_response_skips_empty_candidates_and_flattens_parts(self) -> None:
        """First non-empty candidate text should be returned with nested parts flattened."""

        payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "   "},
                            {"functionCall": {"name": "noop", "args": {}}},
                        ]
                    }
                },
                {
                    "content": {
                        "parts": [
                            {"text": "Launch"},
                            {
                                "content": [
                                    {"text": " sequence"},
                                    {"content": [{"text": " initiated"}]},
                                ]
                            },
                        ]
                    },
                    "finishReason": "STOP",
                },
            ]
        }

        normalised = self.provider._normalise_response(payload)

        assert normalised["content"] == "Launch sequence initiated"
        assert normalised["finish_reason"] == "stop"

    def test_normalise_response_extracts_tool_call_parts(self) -> None:
        """functionCall parts should be emitted as LiteLLM-compatible tool calls."""

        payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "lookup_system",
                                    "args": {"query": "Gemini"},
                                }
                            },
                            {"text": "Lookup queued"},
                        ]
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 7,
            },
        }

        normalised = self.provider._normalise_response(payload)

        tool_calls = normalised["tool_calls"]
        assert tool_calls, "Tool calls should be captured from functionCall parts"
        tool_call = tool_calls[0]
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "lookup_system"
        assert json.loads(tool_call["function"]["arguments"]) == {"query": "Gemini"}
        assert normalised["usage"]["total_tokens"] == 12
