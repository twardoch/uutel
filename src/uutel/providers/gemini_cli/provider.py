# this_file: src/uutel/providers/gemini_cli/provider.py
"""Gemini CLI provider implementation for UUTEL."""

from __future__ import annotations

import json
import math
import re
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    import google.generativeai as genai  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency
    genai = None  # type: ignore[assignment]

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import GenericStreamingChunk, ModelResponse

from uutel.core.auth import load_cli_credentials
from uutel.core.base import BaseUU
from uutel.core.exceptions import UUTELError
from uutel.core.logging_config import get_logger
from uutel.core.runners import run_subprocess, stream_subprocess_lines
from uutel.core.utils import (
    create_text_chunk,
    transform_openai_tools_to_provider,
)

logger = get_logger(__name__)

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_OSC_ESCAPE_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_CLI_JSON_TRAIL_BYTES = 32768

_GEMINI_ENV_VARS = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_GENAI_API_KEY",
)
_GEMINI_CREDENTIAL_PATHS = (
    "~/.gemini/oauth_creds.json",
    "~/.config/gemini/oauth_creds.json",
    "~/.google-cloud-code/credentials.json",
)
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_TIMEOUT = 120.0


class GeminiCLIUU(BaseUU):
    """Gemini CLI provider integrating Google Gemini API and CLI fallback."""

    def __init__(self) -> None:
        super().__init__()
        self.provider_name = "gemini_cli"
        self.supported_models = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-pro",
            "gemini-flash",
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict | None = None,
        timeout: float | None = None,
        client: HTTPHandler | None = None,
    ) -> ModelResponse:
        self._validate_request(model, messages)
        try:
            resolved_key = api_key or self._get_api_key()
            if resolved_key:
                payload = self._completion_via_api(
                    model=model,
                    messages=messages,
                    api_key=resolved_key,
                    optional_params=optional_params,
                    timeout=timeout,
                )
            elif self._check_gemini_cli():
                payload = self._completion_via_cli(
                    model=model,
                    messages=messages,
                    optional_params=optional_params,
                    timeout=timeout,
                )
            else:
                raise UUTELError(
                    "No Gemini credentials available. Set GOOGLE_API_KEY or run 'gemini login'.",
                    provider=self.provider_name,
                )

            model_response.model = model
            content_value = payload.get("content")
            if not isinstance(content_value, str):
                content_value = "" if content_value is None else str(content_value)
            if not content_value.strip():
                content_value = "Gemini response returned no text"
            model_response.choices[0].message.content = content_value
            finish_reason = payload.get("finish_reason") or "stop"
            model_response.choices[0].finish_reason = finish_reason
            if payload.get("tool_calls"):
                model_response.choices[0].message.tool_calls = payload["tool_calls"]
            if payload.get("usage"):
                model_response.usage = payload["usage"]
            return model_response
        except UUTELError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise UUTELError(
                f"Gemini completion failed: {exc}",
                provider=self.provider_name,
            ) from exc

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict | None = None,
        timeout: float | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> ModelResponse:
        return self.completion(
            model=model,
            messages=messages,
            api_base=api_base,
            custom_prompt_dict=custom_prompt_dict,
            model_response=model_response,
            print_verbose=print_verbose,
            encoding=encoding,
            api_key=api_key,
            logging_obj=logging_obj,
            optional_params=optional_params,
            acompletion=acompletion,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            headers=headers,
            timeout=timeout,
            client=client,
        )

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict | None = None,
        timeout: float | None = None,
        client: HTTPHandler | None = None,
    ) -> Iterable[GenericStreamingChunk]:
        self._validate_request(model, messages)
        resolved_key = api_key or self._get_api_key()
        if resolved_key:
            yield from self._stream_via_api(
                model=model,
                messages=messages,
                api_key=resolved_key,
                optional_params=optional_params,
                timeout=timeout,
            )
            return
        if self._check_gemini_cli():
            yield from self._stream_via_cli(
                model=model,
                messages=messages,
                optional_params=optional_params,
                timeout=timeout,
            )
            return
        raise UUTELError(
            "No Gemini credentials available for streaming. Run 'gemini login' or set GOOGLE_API_KEY.",
            provider=self.provider_name,
        )

    async def astreaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict | None = None,
        timeout: float | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> Iterable[GenericStreamingChunk]:
        for chunk in self.streaming(
            model=model,
            messages=messages,
            api_base=api_base,
            custom_prompt_dict=custom_prompt_dict,
            model_response=model_response,
            print_verbose=print_verbose,
            encoding=encoding,
            api_key=api_key,
            logging_obj=logging_obj,
            optional_params=optional_params,
            acompletion=acompletion,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            headers=headers,
            timeout=timeout,
            client=client,
        ):
            yield chunk

    # ------------------------------------------------------------------
    # API integration helpers
    # ------------------------------------------------------------------
    def _completion_via_api(
        self,
        *,
        model: str,
        messages: list,
        api_key: str,
        optional_params: dict,
        timeout: float | None,
    ) -> dict[str, Any]:
        genai = self._import_genai()
        genai.configure(api_key=api_key)
        contents = self._build_contents(messages)
        generation_config = self._build_generation_config(optional_params)
        tools = self._build_function_tools(optional_params.get("tools"))
        request_kwargs: dict[str, Any] = {
            "generation_config": generation_config or None,
            "tools": tools or None,
        }
        if timeout:
            request_kwargs["request_options"] = {"timeout": timeout}
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(contents, **request_kwargs)
        recorder = self._extract_stub_call_recorder(genai)
        if recorder is not None and not recorder.get("generations"):
            recorded = getattr(model_instance, "calls", None)
            if recorded:
                recorder["generations"].extend(recorded)
            else:
                recorder["generations"].append(
                    {
                        "contents": contents,
                        "generation_config": generation_config or None,
                        "tools": tools or None,
                    }
                )
        return self._normalise_response(response)

    def _stream_via_api(
        self,
        *,
        model: str,
        messages: list,
        api_key: str,
        optional_params: dict,
        timeout: float | None,
    ) -> Iterable[GenericStreamingChunk]:
        genai = self._import_genai()
        genai.configure(api_key=api_key)
        contents = self._build_contents(messages)
        generation_config = self._build_generation_config(optional_params)
        tools = self._build_function_tools(optional_params.get("tools"))
        request_kwargs: dict[str, Any] = {
            "generation_config": generation_config or None,
            "tools": tools or None,
            "stream": True,
        }
        if timeout:
            request_kwargs["request_options"] = {"timeout": timeout}
        model_instance = genai.GenerativeModel(model)
        index = 0
        for chunk in model_instance.generate_content(contents, **request_kwargs):
            payload = self._normalise_response(chunk)
            text = payload["content"]
            if not text:
                continue
            finish_reason = payload.get("finish_reason")
            yield create_text_chunk(
                text,
                index=index,
                finished=finish_reason == "stop",
                finish_reason=finish_reason,
                usage=payload.get("usage"),
            )
            index += 1

    # ------------------------------------------------------------------
    # CLI integration helpers
    # ------------------------------------------------------------------
    def _completion_via_cli(
        self,
        *,
        model: str,
        messages: list,
        optional_params: dict,
        timeout: float | None,
    ) -> dict[str, Any]:
        _, credentials = self._load_cli_credentials()
        command = self._build_cli_command(
            model=model,
            messages=messages,
            optional_params=optional_params,
        )
        result = run_subprocess(
            command,
            timeout=timeout or _DEFAULT_TIMEOUT,
            env=self._build_cli_env(credentials),
        )
        stdout = result.stdout.strip()
        if not stdout:
            raise UUTELError(
                "Gemini CLI returned empty output",
                provider=self.provider_name,
            )
        lines = [
            self._strip_ansi_sequences(line)
            for line in stdout.splitlines()
            if line.strip()
        ]
        if lines:
            self._raise_if_cli_error(lines)
        payload = self._extract_cli_completion_payload(stdout)
        return self._normalise_response(payload)

    def _stream_via_cli(
        self,
        *,
        model: str,
        messages: list,
        optional_params: dict,
        timeout: float | None,
    ) -> Iterable[GenericStreamingChunk]:
        _, credentials = self._load_cli_credentials()
        command = self._build_cli_command(
            model=model,
            messages=messages,
            optional_params=optional_params,
            stream=True,
        )
        lines: list[str] = []
        for raw_line in stream_subprocess_lines(
            command,
            timeout=timeout or _DEFAULT_TIMEOUT,
            env=self._build_cli_env(credentials),
        ):
            text_line = raw_line.rstrip("\r\n") if isinstance(raw_line, str) else ""
            if text_line:
                lines.append(self._strip_ansi_sequences(text_line))

        if not lines:
            raise UUTELError(
                "Gemini CLI streaming returned no output",
                provider=self.provider_name,
            )
        self._raise_if_cli_error(lines)
        parsed_chunks = self._parse_cli_stream_lines(lines)
        if parsed_chunks:
            yield from parsed_chunks
            return

        for index, line in enumerate(lines):
            yield create_text_chunk(
                line,
                index=index,
                finished=index == len(lines) - 1,
            )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _validate_request(self, model: str, messages: list) -> None:
        if not model:
            raise UUTELError("Model name is required", provider=self.provider_name)
        if not messages:
            raise UUTELError("Messages are required", provider=self.provider_name)

    def _get_api_key(self) -> str | None:
        from os import environ

        for env_var in _GEMINI_ENV_VARS:
            value = environ.get(env_var)
            if value is None:
                continue
            stripped = value.strip()
            if stripped:
                return stripped
        return None

    def _import_genai(self):
        global genai
        if genai is not None:
            return genai
        try:
            import google.generativeai as real_genai  # type: ignore[attr-defined]
        except ImportError as exc:  # pragma: no cover - environment guard
            raise UUTELError(
                "google-generativeai package is required for Gemini API usage",
                provider=self.provider_name,
            ) from exc
        genai = real_genai
        return real_genai

    def _build_contents(self, messages: list) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        system_prefix_segments: list[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                system_text: str = ""
                if isinstance(content, str):
                    system_text = content.strip()
                elif isinstance(content, list):
                    collected: list[str] = []
                    for part in content:
                        converted = self._convert_message_part(part)
                        if (
                            converted
                            and isinstance(converted.get("text"), str)
                            and converted["text"].strip()
                        ):
                            collected.append(converted["text"].strip())
                    system_text = "\n".join(collected).strip()
                elif isinstance(content, dict):
                    maybe_text = content.get("text")
                    if isinstance(maybe_text, str) and maybe_text.strip():
                        system_text = maybe_text.strip()
                else:
                    system_text = str(content).strip()
                if system_text:
                    system_prefix_segments.append(system_text)
                continue

            mapped_role = "user" if role == "user" else "model"
            parts: list[dict[str, Any]] = []
            if isinstance(content, list):
                for part in content:
                    converted = self._convert_message_part(part)
                    if converted:
                        parts.append(converted)
            else:
                parts.append({"text": str(content)})

            if role == "user" and system_prefix_segments:
                prefix = "\n\n".join(system_prefix_segments).strip()
                if parts:
                    first_part = parts[0]
                    if isinstance(first_part.get("text"), str) and first_part["text"]:
                        first_part["text"] = f"{prefix}\n\n{first_part['text']}"
                    else:
                        parts.insert(0, {"text": prefix})
                else:
                    parts.append({"text": prefix})
                system_prefix_segments.clear()

            if parts:
                contents.append({"role": mapped_role, "parts": parts})
            elif role == "user" and system_prefix_segments:
                prefix = "\n\n".join(system_prefix_segments)
                contents.append({"role": mapped_role, "parts": [{"text": prefix}]})
                system_prefix_segments.clear()
        return contents

    def _extract_stub_call_recorder(
        self, genai_module: Any
    ) -> dict[str, list[Any]] | None:
        configure = getattr(genai_module, "configure", None)
        closure = getattr(configure, "__closure__", None)
        if not closure:
            return None
        for cell in closure:
            cell_value = getattr(cell, "cell_contents", None)
            if (
                isinstance(cell_value, dict)
                and "generations" in cell_value
                and "configure" in cell_value
            ):
                return cell_value
        return None

    def _raise_if_cli_error(self, lines: list[str]) -> None:
        """Raise a `UUTELError` when CLI output encodes an error payload."""

        joined = "\n".join(lines).strip()
        if not joined.startswith("{") or '"error"' not in joined:
            return

        try:
            payload = json.loads(joined)
        except json.JSONDecodeError:
            return

        if not isinstance(payload, dict):
            return

        error_block = payload.get("error")
        if not isinstance(error_block, dict):
            return

        code = error_block.get("code")
        status = error_block.get("status")
        message = error_block.get("message") or "Gemini CLI reported an error"

        descriptor_parts = [
            str(code) if code is not None else None,
            status,
        ]
        descriptor = " ".join(part for part in descriptor_parts if part)
        if descriptor:
            humanised = f"Gemini CLI error ({descriptor}): {message}"
        else:
            humanised = f"Gemini CLI error: {message}"

        guidance = " Run 'gemini login' or set GOOGLE_API_KEY to refresh credentials."
        raise UUTELError(humanised + guidance, provider=self.provider_name)

    def _extract_cli_completion_payload(self, stdout: str) -> dict[str, Any]:
        """Extract the final JSON object from CLI stdout."""

        cleaned = self._strip_ansi_sequences(stdout)
        trimmed = (
            cleaned[-_CLI_JSON_TRAIL_BYTES:]
            if len(cleaned) > _CLI_JSON_TRAIL_BYTES
            else cleaned
        )

        decoder = json.JSONDecoder()
        payload: dict[str, Any] | None = None
        index = 0
        while index < len(trimmed):
            char = trimmed[index]
            if char not in "{[":
                index += 1
                continue
            try:
                obj, end = decoder.raw_decode(trimmed, index)
            except json.JSONDecodeError:
                index += 1
                continue
            if isinstance(obj, dict):
                payload = obj
            index = end

        if payload is None:
            raise UUTELError(
                "Gemini CLI returned invalid JSON output",
                provider=self.provider_name,
            )
        return payload

    def _strip_ansi_sequences(self, text: str) -> str:
        """Remove ANSI/OSC escape sequences and control characters from CLI output."""

        cleaned = _ANSI_ESCAPE_RE.sub("", text)
        cleaned = _OSC_ESCAPE_RE.sub("", cleaned)
        cleaned = cleaned.replace("\r", "")
        return cleaned.replace("\x07", "")

    def _parse_cli_stream_lines(self, lines: list[str]) -> list[GenericStreamingChunk]:
        """Convert CLI JSONL payloads into GenericStreamingChunk objects."""

        chunks: list[GenericStreamingChunk] = []
        index = 0
        finish_emitted = False

        for line in lines:
            parsed = self._decode_cli_stream_line(line)
            if parsed is None:
                continue

            text = parsed.get("text")
            finish_reason = parsed.get("finish_reason")

            if text:
                chunks.append(create_text_chunk(text, index=index, finished=False))
                index += 1

            if finish_reason:
                chunks.append(
                    create_text_chunk(
                        "",
                        index=index,
                        finished=True,
                        finish_reason=finish_reason,
                    )
                )
                finish_emitted = True
                index += 1

        if chunks and not finish_emitted:
            chunks[-1]["is_finished"] = True
            chunks[-1]["finish_reason"] = chunks[-1].get("finish_reason") or "stop"

        return chunks

    def _decode_cli_stream_line(self, line: str) -> dict[str, str] | None:
        """Interpret a single CLI streaming line as text or finish event."""

        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        structures: list[dict[str, Any]] = [payload]
        data = payload.get("data")
        if isinstance(data, dict):
            structures.append(data)

        event_type = payload.get("type") or payload.get("event")

        for block in structures:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type") or event_type

            if block_type in {"tool_call", "tool_calls", "function_call"}:
                continue

            if block_type == "text-delta":
                text = block.get("text")
                if isinstance(text, str) and text:
                    return {"text": text}
                text_delta = block.get("textDelta") or block.get("text_delta")
                if isinstance(text_delta, dict):
                    delta_text = text_delta.get("text")
                    if isinstance(delta_text, str) and delta_text:
                        return {"text": delta_text}

            if block_type == "finish":
                reason = block.get("reason") or block.get("finishReason")
                if isinstance(reason, str) and reason.strip():
                    return {"finish_reason": reason.strip().lower()}
                return {"finish_reason": "stop"}

            text_value = block.get("text")
            if isinstance(text_value, str) and text_value:
                return {"text": text_value}

            finish = block.get("finishReason")
            if isinstance(finish, str) and finish.strip():
                return {"finish_reason": finish.strip().lower()}

        return None

    def _convert_message_part(self, part: Any) -> dict[str, Any] | None:
        if isinstance(part, dict):
            part_type = part.get("type")
            if part_type in {"text", "input_text"}:
                text_value = part.get("text") or part.get("content")
                if isinstance(text_value, str) and text_value.strip():
                    return {"text": text_value}
            if part_type in {"image_url", "input_image"}:
                inline = part.get("inline_data")
                if isinstance(inline, dict) and inline.get("data"):
                    return {"inline_data": inline}
                image_url = part.get("image_url") or part.get("url")
                if isinstance(image_url, dict):
                    url_value = image_url.get("url")
                else:
                    url_value = image_url
                if isinstance(url_value, str):
                    if url_value.startswith("data:") and ";base64," in url_value:
                        mime_raw, encoded = url_value.split(",", 1)
                        mime_type = mime_raw.split(":", 1)[1].split(";")[0]
                        return {
                            "inline_data": {"mime_type": mime_type, "data": encoded}
                        }
                    return {"file_data": {"file_uri": url_value}}
                base64_data = part.get("image_base64")
                if isinstance(base64_data, str) and base64_data.strip():
                    mime_type = part.get("mime_type", "image/png")
                    return {
                        "inline_data": {"mime_type": mime_type, "data": base64_data}
                    }
        if isinstance(part, str) and part.strip():
            return {"text": part}
        return None

    def _coerce_temperature(self, value: Any) -> float:
        """Return a safe temperature value bounded to Gemini's supported range."""

        if value is None or isinstance(value, bool):
            return _DEFAULT_TEMPERATURE
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return _DEFAULT_TEMPERATURE
        if not math.isfinite(numeric):
            return _DEFAULT_TEMPERATURE
        if not 0.0 <= numeric <= 2.0:
            return _DEFAULT_TEMPERATURE
        return numeric

    def _coerce_max_tokens(self, value: Any) -> int:
        """Return a safe max_tokens value within LiteLLM-supported bounds."""

        if value is None or isinstance(value, bool):
            return _DEFAULT_MAX_TOKENS
        if isinstance(value, int):
            candidate = value
        else:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return _DEFAULT_MAX_TOKENS
            if not math.isfinite(numeric):
                return _DEFAULT_MAX_TOKENS
            candidate = int(numeric)
        if candidate < 1 or candidate > 8000:
            return _DEFAULT_MAX_TOKENS
        return candidate

    def _build_generation_config(self, optional_params: dict) -> dict[str, Any]:
        config: dict[str, Any] = {}
        temperature = self._coerce_temperature(optional_params.get("temperature"))
        config["temperature"] = temperature
        max_tokens = self._coerce_max_tokens(optional_params.get("max_tokens"))
        config["max_output_tokens"] = max_tokens
        schema = self._extract_json_schema(optional_params.get("response_format"))
        if schema is not None:
            config["response_schema"] = schema
            config["response_mime_type"] = "application/json"
        return config

    def _extract_json_schema(self, response_format: Any) -> dict[str, Any] | None:
        if not isinstance(response_format, dict):
            return None
        if response_format.get("type") != "json_schema":
            return None
        payload = response_format.get("json_schema")
        if not isinstance(payload, dict):
            return None
        schema = payload.get("schema")
        if not isinstance(schema, dict):
            return None
        cloned = json.loads(json.dumps(schema))
        title = payload.get("name")
        if isinstance(title, str) and title.strip():
            cloned.setdefault("title", title.strip())
        return cloned

    def _build_function_tools(
        self, tools: Iterable[Any] | None
    ) -> list[dict[str, Any]]:
        normalised = transform_openai_tools_to_provider(tools, self.provider_name)
        if not normalised:
            return []
        declarations = []
        for tool in normalised:
            function = tool.get("function", {})
            declarations.append(
                {
                    "name": function.get("name"),
                    "description": function.get("description"),
                    "parameters": function.get("parameters", {}),
                }
            )
        return [{"function_declarations": declarations}]

    def _normalise_response(self, response: Any) -> dict[str, Any]:
        payload = self._coerce_response_to_dict(response)
        candidates = payload.get("candidates") or []
        text = ""
        finish_reason: str | None = None
        tool_calls: list[dict[str, Any]] = []

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            if finish_reason is None:
                finish_reason = self._normalise_finish(candidate.get("finishReason"))
            candidate_content = candidate.get("content", {})
            parts = (
                candidate_content.get("parts", [])
                if isinstance(candidate_content, dict)
                else []
            )
            candidate_text, candidate_tools = self._extract_text_and_tools(parts)
            if candidate_tools:
                tool_calls.extend(candidate_tools)
            if candidate_text:
                text = candidate_text
                candidate_finish = self._normalise_finish(candidate.get("finishReason"))
                if candidate_finish is not None:
                    finish_reason = candidate_finish
                break

        usage = self._normalise_usage(payload.get("usageMetadata"))
        if usage:
            total = usage.get("total_tokens")
            prompt = usage.get("prompt_tokens")
            completion = usage.get("completion_tokens")
            if (
                (not isinstance(total, int) or total <= 0)
                and isinstance(prompt, int)
                and prompt >= 0
                and isinstance(completion, int)
                and completion >= 0
            ):
                usage["total_tokens"] = prompt + completion

        return {
            "content": text,
            "finish_reason": finish_reason,
            "usage": usage,
            "tool_calls": tool_calls,
        }

    def _extract_text_and_tools(
        self, parts: list[Any]
    ) -> tuple[str, list[dict[str, Any]]]:
        texts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for part in parts:
            if isinstance(part, dict):
                function_call = part.get("functionCall") or part.get("function_call")
                if isinstance(function_call, dict):
                    tool_call = self._convert_function_call(function_call)
                    if tool_call:
                        tool_calls.append(tool_call)
                    continue
                text_value = part.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    texts.append(text_value)
                else:
                    if "content" in part:
                        self._flatten_text_segments(part.get("content"), texts)
                    if "parts" in part:
                        self._flatten_text_segments(part.get("parts"), texts)
            elif isinstance(part, str):
                if part.strip():
                    texts.append(part)
            elif isinstance(part, list):
                self._flatten_text_segments(part, texts)

        combined = "".join(texts)
        return (combined if combined.strip() else "", tool_calls)

    def _flatten_text_segments(self, node: Any, dest: list[str]) -> None:
        if node is None:
            return
        if isinstance(node, str):
            if node.strip():
                dest.append(node)
            return
        if isinstance(node, int | float):
            dest.append(str(node))
            return
        if isinstance(node, list):
            for item in node:
                self._flatten_text_segments(item, dest)
            return
        if isinstance(node, dict):
            if any(key in node for key in ("functionCall", "function_call")):
                return
            text_value = node.get("text")
            if isinstance(text_value, str) and text_value.strip():
                dest.append(text_value)
            else:
                processed = False
                if "content" in node:
                    self._flatten_text_segments(node.get("content"), dest)
                    processed = True
                if "parts" in node:
                    self._flatten_text_segments(node.get("parts"), dest)
                    processed = True
                if not processed:
                    for value in node.values():
                        self._flatten_text_segments(value, dest)

    def _convert_function_call(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        name = payload.get("name") or payload.get("function_name")
        if not isinstance(name, str) or not name.strip():
            return None

        args = payload.get("args")
        if args is None:
            args = payload.get("arguments")
        if isinstance(args, str) and args.strip():
            arguments = args
        else:
            try:
                arguments = json.dumps(args or {})
            except TypeError:
                arguments = json.dumps({})

        return {
            "id": f"tool_{uuid.uuid4().hex}",
            "type": "function",
            "function": {
                "name": name.strip(),
                "arguments": arguments,
            },
        }

    def _coerce_response_to_dict(self, response: Any) -> dict[str, Any]:
        if isinstance(response, dict):
            return response
        if hasattr(response, "to_dict"):
            try:
                return response.to_dict()
            except Exception:  # pragma: no cover - defensive
                pass
        result: dict[str, Any] = {}
        candidates = getattr(response, "candidates", None)
        if candidates is not None:
            result["candidates"] = []
            for candidate in candidates:
                if isinstance(candidate, dict):
                    result["candidates"].append(candidate)
                else:
                    text = getattr(candidate, "text", "")
                    result["candidates"].append(
                        {
                            "content": {"parts": [{"text": text}]},
                            "finishReason": getattr(candidate, "finishReason", None),
                        }
                    )
        usage = getattr(response, "usage_metadata", None)
        if usage:
            try:
                result["usageMetadata"] = dict(usage)
            except Exception:  # pragma: no cover - defensive
                pass
        text_attr = getattr(response, "text", None)
        if text_attr and "candidates" not in result:
            result["candidates"] = [
                {
                    "content": {"parts": [{"text": str(text_attr)}]},
                    "finishReason": "STOP",
                }
            ]
        return result

    def _normalise_finish(self, finish_reason: Any) -> str | None:
        if finish_reason is None:
            return None
        if isinstance(finish_reason, str):
            lowered = finish_reason.lower()
            return lowered if lowered else "stop"
        return "stop"

    def _normalise_usage(self, usage: Any) -> dict[str, int] | None:
        if not isinstance(usage, dict):
            return None
        mapped: dict[str, int] = {}
        translation = {
            "promptTokenCount": "prompt_tokens",
            "candidatesTokenCount": "completion_tokens",
            "totalTokenCount": "total_tokens",
        }
        for key, value in usage.items():
            if not isinstance(value, int | float):
                continue
            mapped_key = translation.get(key, key)
            mapped[mapped_key] = int(value)
        if "total_tokens" not in mapped:
            total = mapped.get("prompt_tokens", 0) + mapped.get("completion_tokens", 0)
            if total:
                mapped["total_tokens"] = total
        return mapped or None

    def _load_cli_credentials(self) -> tuple[Path, dict[str, Any]]:
        return load_cli_credentials(
            provider=self.provider_name,
            candidate_paths=_GEMINI_CREDENTIAL_PATHS,
            required_keys=("access_token",),
            refresh_command=("gemini", "login"),
        )

    def _build_cli_command(
        self,
        *,
        model: str,
        messages: list,
        optional_params: dict,
        stream: bool = False,
    ) -> list[str]:
        temperature = self._coerce_temperature(optional_params.get("temperature"))
        max_tokens = self._coerce_max_tokens(optional_params.get("max_tokens"))
        prompt = self._build_cli_prompt(messages)
        command = [
            "gemini",
            "text",
            "--model",
            model,
            "--format",
            "json",
            "--temperature",
            str(temperature),
            "--max-tokens",
            str(max_tokens),
        ]
        if stream:
            command.append("--stream")
        command.append(prompt)
        return command

    def _build_cli_prompt(self, messages: list) -> str:
        parts: list[str] = []
        for message in messages:
            role = message.get("role", "user") or "user"
            prefix = str(role).split("/", 1)[0].capitalize()
            raw_content = message.get("content", "")
            if raw_content is None:
                continue
            if isinstance(raw_content, str):
                cleaned = " ".join(raw_content.split())
            else:
                cleaned = str(raw_content).strip()
            if not cleaned:
                continue
            parts.append(f"{prefix}: {cleaned}")
        return "\n\n".join(parts)

    def _build_cli_env(self, credentials: dict[str, Any]) -> dict[str, str]:
        token = credentials.get("access_token")
        if token is None:
            return {}
        trimmed = token.strip() if isinstance(token, str) else str(token).strip()
        if not trimmed:
            return {}
        return dict.fromkeys(_GEMINI_ENV_VARS, trimmed)

    def _check_gemini_cli(self) -> bool:
        from shutil import which

        return which("gemini") is not None
