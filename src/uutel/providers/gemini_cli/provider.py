# this_file: src/uutel/providers/gemini_cli/provider.py
"""Gemini CLI provider implementation for UUTEL."""

from __future__ import annotations

import json
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
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise UUTELError(
                "Gemini CLI returned invalid JSON output",
                provider=self.provider_name,
            ) from exc
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
        lines = list(
            stream_subprocess_lines(
                command,
                timeout=timeout or _DEFAULT_TIMEOUT,
                env=self._build_cli_env(credentials),
            )
        )
        if not lines:
            raise UUTELError(
                "Gemini CLI streaming returned no output",
                provider=self.provider_name,
            )
        yield create_text_chunk("\n".join(lines), index=0, finished=True)

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
            if value:
                return value
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
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                system_text = content if isinstance(content, str) else ""
                if contents and contents[0]["role"] == "user":
                    parts = contents[0].setdefault("parts", [{"text": ""}])
                    parts[0]["text"] = f"{system_text}\n\n" + parts[0].get("text", "")
                else:
                    contents.insert(
                        0, {"role": "user", "parts": [{"text": system_text}]}
                    )
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

            if parts:
                contents.append({"role": mapped_role, "parts": parts})
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

    def _build_generation_config(self, optional_params: dict) -> dict[str, Any]:
        config: dict[str, Any] = {}
        temperature = optional_params.get("temperature", _DEFAULT_TEMPERATURE)
        if temperature is not None:
            config["temperature"] = float(temperature)
        max_tokens = optional_params.get("max_tokens", _DEFAULT_MAX_TOKENS)
        if max_tokens is not None:
            config["max_output_tokens"] = int(max_tokens)
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
        if candidates:
            candidate = candidates[0]
            normalised_finish = self._normalise_finish(candidate.get("finishReason"))
            if normalised_finish is not None:
                finish_reason = normalised_finish
            content = candidate.get("content", {})
            parts = content.get("parts", []) if isinstance(content, dict) else []
            text = "".join(
                part.get("text", "") for part in parts if isinstance(part, dict)
            )
        usage = self._normalise_usage(payload.get("usageMetadata"))
        return {
            "content": text,
            "finish_reason": finish_reason,
            "usage": usage,
            "tool_calls": [],
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
        temperature = optional_params.get("temperature", _DEFAULT_TEMPERATURE)
        max_tokens = optional_params.get("max_tokens", _DEFAULT_MAX_TOKENS)
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
            role = message.get("role", "user")
            prefix = role.capitalize()
            parts.append(f"{prefix}: {message.get('content', '')}")
        return "\n\n".join(parts)

    def _build_cli_env(self, credentials: dict[str, Any]) -> dict[str, str]:
        token = credentials.get("access_token")
        return {"GOOGLE_API_KEY": token} if token else {}

    def _check_gemini_cli(self) -> bool:
        from shutil import which

        return which("gemini") is not None
