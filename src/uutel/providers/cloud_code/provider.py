# this_file: src/uutel/providers/cloud_code/provider.py
"""Google Cloud Code AI provider implementation for UUTEL.

This module implements the CloudCodeUU provider class for integrating with
Google's Cloud Code AI via OAuth authentication and Code Assist API.
"""

from __future__ import annotations

# Standard library imports
import json
import os
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

# Third-party imports
import httpx
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import GenericStreamingChunk, ModelResponse

# Local imports
from uutel.core.base import BaseUU
from uutel.core.exceptions import UUTELError
from uutel.core.logging_config import get_logger
from uutel.core.utils import (
    create_http_client,
    create_text_chunk,
    create_tool_chunk,
    merge_usage_stats,
)

logger = get_logger(__name__)


_DEFAULT_BASE_URL = "https://cloudcode-pa.googleapis.com"
_GENERATE_PATH = "/v1internal:generateContent"
_STREAM_PATH = "/v1internal:streamGenerateContent?alt=sse"
_JSON_MIME = "application/json"
_PROJECT_ENV_VARS = ("CLOUD_CODE_PROJECT", "GOOGLE_CLOUD_PROJECT", "GOOGLE_PROJECT")
_API_KEY_ENV_VARS = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY")


class CloudCodeUU(BaseUU):
    """Google Cloud Code AI provider for UUTEL.

    Implements integration with Google's Cloud Code AI using OAuth authentication.
    Reads credentials from ~/.gemini/oauth_creds.json (from Gemini CLI authentication).
    """

    def __init__(self) -> None:
        """Initialize Cloud Code provider."""
        super().__init__()
        self.provider_name = "cloud_code"
        self.supported_models = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-pro",
            "gemini-flash",
        ]

    def _validate_request(self, model: str, messages: list[Any]) -> None:
        """Validate that model and messages are present."""

        if not model or not isinstance(model, str):
            raise UUTELError("Model name is required", provider=self.provider_name)
        if not messages:
            raise UUTELError("Messages are required", provider=self.provider_name)

    def _credential_paths(self) -> list[Path]:
        """Return possible credential file locations."""

        return [
            Path.home() / ".gemini" / "oauth_creds.json",
            Path.home() / ".config" / "gemini" / "oauth_creds.json",
            Path.home() / ".google-cloud-code" / "credentials.json",
        ]

    def _extract_access_token(self, payload: dict[str, Any]) -> str | None:
        """Extract an OAuth access token from stored credentials."""

        token = payload.get("access_token")
        if isinstance(token, str) and token.strip():
            return token.strip()
        tokens = payload.get("tokens", {}) if isinstance(payload, dict) else {}
        token = tokens.get("access_token") if isinstance(tokens, dict) else None
        if isinstance(token, str) and token.strip():
            return token.strip()
        return None

    def _load_oauth_credentials(self) -> str:
        """Load OAuth access token from stored Gemini credentials."""

        for creds_path in self._credential_paths():
            try:
                data = json.loads(creds_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue
            except json.JSONDecodeError as exc:
                logger.warning("Invalid JSON in %s: %s", creds_path, exc)
                continue
            token = self._extract_access_token(data)
            if token:
                return token
        raise UUTELError(
            "No OAuth credentials found. Run 'gemini login' or provide GOOGLE_API_KEY.",
            provider=self.provider_name,
        )

    def _get_api_key(self) -> str | None:
        """Return the first configured Google API key if available."""

        for env_var in _API_KEY_ENV_VARS:
            value = os.getenv(env_var)
            if value is None:
                continue
            stripped = value.strip()
            if stripped:
                return stripped
        return None

    def _resolve_project_id(self, optional_params: dict[str, Any]) -> str:
        """Resolve Google Cloud project ID from params or environment."""

        for key in ("project_id", "project", "google_project"):
            value = optional_params.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for env_var in _PROJECT_ENV_VARS:
            value = os.getenv(env_var)
            if value is None:
                continue
            stripped = value.strip()
            if stripped:
                return stripped
        raise UUTELError(
            (
                "Google Cloud project ID required. Set project_id parameter or export CLOUD_CODE_PROJECT. "
                "Tip: run 'gcloud config set project <id>' or set CLOUD_CODE_PROJECT before retrying."
            ),
            provider=self.provider_name,
        )

    def _extract_text(self, content: Any) -> str:
        """Extract textual content from message payloads."""

        if isinstance(content, str):
            return content
        if isinstance(content, dict) and isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content, list):
            parts = [
                part.get("text")
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            return " ".join(part.strip() for part in parts if part)
        return "" if content is None else str(content)

    def _convert_user_message(self, content: Any) -> dict[str, Any]:
        """Convert a user message into Cloud Code format."""

        text = self._extract_text(content).strip()
        if not text:
            raise UUTELError(
                "User message cannot be empty", provider=self.provider_name
            )
        return {"role": "user", "parts": [{"text": text}]}

    def _convert_assistant_message(self, content: Any) -> dict[str, Any] | None:
        """Convert an assistant/model message into Cloud Code format."""

        text = self._extract_text(content).strip()
        if not text:
            return None
        return {"role": "model", "parts": [{"text": text}]}

    def _render_json_schema_instruction(
        self, response_format: dict[str, Any] | None
    ) -> str | None:
        """Render instruction text when JSON schema is requested."""

        if not isinstance(response_format, dict):
            return None
        if response_format.get("type") != "json_schema":
            return None
        schema_block = response_format.get("json_schema", {}).get("schema")
        if not schema_block:
            return None
        schema_text = json.dumps(schema_block, indent=2, sort_keys=True)
        name = response_format.get("json_schema", {}).get("name", "JSON payload")
        return (
            f"Respond strictly with JSON matching the schema '{name}':\n{schema_text}"
        )

    def _convert_messages(
        self,
        messages: list[Any],
        response_format: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """Convert OpenAI-style messages into Cloud Code payload."""

        contents: list[dict[str, Any]] = []
        system_parts: list[str] = []

        for message in messages:
            role = message.get("role", "user") if isinstance(message, dict) else "user"
            content = message.get("content") if isinstance(message, dict) else message
            if role == "system":
                text = self._extract_text(content).strip()
                if text:
                    system_parts.append(text)
                continue
            if role in {"assistant", "model"}:
                converted = self._convert_assistant_message(content)
                if converted:
                    contents.append(converted)
                continue
            if role == "user":
                contents.append(self._convert_user_message(content))
                continue
            raise UUTELError(
                f"Unsupported message role: {role}", provider=self.provider_name
            )

        schema_instruction = self._render_json_schema_instruction(response_format)
        if schema_instruction:
            system_parts.append(schema_instruction)

        system_instruction = None
        if system_parts:
            system_instruction = {
                "role": "user",
                "parts": [{"text": "\n\n".join(system_parts)}],
            }
        return contents, system_instruction

    def _build_tool_payload(
        self,
        optional_params: dict[str, Any],
    ) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
        """Build Cloud Code tool declarations and config from optional params."""

        raw_tools = optional_params.get("tools")
        if not isinstance(raw_tools, list) or not raw_tools:
            return None, None

        declarations: list[dict[str, Any]] = []
        for tool in raw_tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                continue
            func = tool.get("function", {})
            name = func.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            parameters = func.get("parameters") if isinstance(func, dict) else None
            if isinstance(parameters, dict) and "$schema" in parameters:
                parameters = {k: v for k, v in parameters.items() if k != "$schema"}
            declarations.append(
                {
                    "name": name.strip(),
                    "description": func.get("description"),
                    "parameters": parameters,
                }
            )

        if not declarations:
            return None, None

        tools_payload = [{"functionDeclarations": declarations}]
        choice = optional_params.get("tool_choice")
        mode = "AUTO"
        allowed: list[str] | None = None
        if isinstance(choice, dict):
            choice_type = choice.get("type")
            if choice_type == "none":
                mode = "NONE"
            elif choice_type == "required":
                mode = "ANY"
            elif choice_type == "tool":
                mode = "ANY"
                tool_name = choice.get("tool_name")
                if isinstance(tool_name, str) and tool_name.strip():
                    allowed = [tool_name.strip()]
        tool_config = {"functionCallingConfig": {"mode": mode}}
        if allowed:
            tool_config["functionCallingConfig"]["allowedFunctionNames"] = allowed
        return tools_payload, tool_config

    def _build_generation_config(
        self, optional_params: dict[str, Any]
    ) -> dict[str, Any]:
        """Map optional params into Cloud Code generation config."""

        config = {
            "temperature": optional_params.get("temperature"),
            "topP": optional_params.get("top_p"),
            "topK": optional_params.get("top_k"),
            "maxOutputTokens": optional_params.get("max_tokens")
            or optional_params.get("max_output_tokens"),
            "stopSequences": optional_params.get("stop")
            if isinstance(optional_params.get("stop"), list)
            else None,
            "thinkingConfig": optional_params.get("thinking_config")
            or {"includeThoughts": False},
        }
        if optional_params.get("response_format", {}).get("type") == "json_schema":
            config["responseMimeType"] = _JSON_MIME
        return {
            key: value for key, value in config.items() if value not in (None, [], {})
        }

    def _map_usage_metadata(self, metadata: dict[str, Any]) -> dict[str, int]:
        """Convert usage metadata into LiteLLM-style usage dict."""

        return {
            "prompt_tokens": int(metadata.get("promptTokenCount", 0) or 0),
            "completion_tokens": int(metadata.get("candidatesTokenCount", 0) or 0),
            "total_tokens": int(metadata.get("totalTokenCount", 0) or 0),
        }

    def _prepare_request(
        self,
        *,
        model: str,
        messages: list[Any],
        api_base: str,
        optional_params: dict[str, Any],
        headers: dict | None,
        api_key,
        timeout: float | None,
        client: HTTPHandler | None,
        stream: bool,
    ) -> tuple[str, dict[str, str], dict[str, Any], HTTPHandler, bool]:
        """Prepare request components for Cloud Code API calls."""

        project_id = self._resolve_project_id(optional_params)
        contents, system_instruction = self._convert_messages(
            messages,
            optional_params.get("response_format"),
        )
        if not contents:
            raise UUTELError(
                "Cloud Code requires at least one user or assistant message",
                provider=self.provider_name,
            )

        tools_payload, tool_config = self._build_tool_payload(optional_params)
        request_body: dict[str, Any] = {
            "model": model,
            "project": project_id,
            "request": {
                "contents": contents,
                "generationConfig": self._build_generation_config(optional_params),
            },
        }
        if system_instruction:
            request_body["request"]["systemInstruction"] = system_instruction
        safety_settings = optional_params.get("safety_settings")
        if isinstance(safety_settings, list) and safety_settings:
            request_body["request"]["safetySettings"] = safety_settings
        if tools_payload:
            request_body["request"]["tools"] = tools_payload
        if tool_config:
            request_body["request"]["toolConfig"] = tool_config

        base_url = (api_base or _DEFAULT_BASE_URL).rstrip("/") or _DEFAULT_BASE_URL
        endpoint = _STREAM_PATH if stream else _GENERATE_PATH
        url = f"{base_url}{endpoint}"

        request_headers: dict[str, str] = {"Content-Type": _JSON_MIME}
        resolved_api_key = api_key or self._get_api_key()
        if resolved_api_key:
            request_headers["x-goog-api-key"] = resolved_api_key
        else:
            token = self._load_oauth_credentials()
            request_headers["Authorization"] = f"Bearer {token}"
        if headers:
            request_headers.update(headers)

        http_client = client or create_http_client(async_client=False, timeout=timeout)
        return url, request_headers, request_body, http_client, client is None

    def _normalise_payload(
        self, payload: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """Return candidates and usage metadata from Cloud Code responses."""

        if not isinstance(payload, dict):
            return [], None
        response = payload.get("response")
        if isinstance(response, dict):
            return response.get("candidates", []) or [], response.get("usageMetadata")
        return payload.get("candidates", []) or [], payload.get("usageMetadata")

    def _extract_candidate_parts(
        self, candidate: dict[str, Any]
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Extract text and function calls from a candidate payload."""

        content = candidate.get("content") if isinstance(candidate, dict) else None
        parts = content.get("parts", []) if isinstance(content, dict) else []
        texts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            text_value = part.get("text")
            if isinstance(text_value, str) and text_value.strip():
                texts.append(text_value)
                continue
            function_call = part.get("functionCall")
            if isinstance(function_call, dict):
                name = function_call.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                arguments = function_call.get("args") or {}
                try:
                    argument_payload = json.dumps(arguments)
                except TypeError:
                    argument_payload = json.dumps({})
                tool_calls.append(
                    {
                        "id": f"tool_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": name.strip(),
                            "arguments": argument_payload,
                        },
                    }
                )
        return texts, tool_calls

    def _normalise_finish_reason(self, finish: Any) -> str | None:
        """Return a normalised finish reason string."""

        if not isinstance(finish, str):
            return None
        value = finish.strip().lower()
        return value or None

    def _iter_candidate_parts(self, candidate: dict[str, Any]):
        """Yield candidate parts as (type, payload) tuples."""

        content = candidate.get("content") if isinstance(candidate, dict) else None
        parts = content.get("parts", []) if isinstance(content, dict) else []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("text"), str):
                yield "text", part["text"]
                continue
            function_call = part.get("functionCall")
            if isinstance(function_call, dict):
                yield "tool", function_call

    def _iter_sse_payloads(self, response) -> Iterator[dict[str, Any]]:
        """Parse Server-Sent Event payloads from the streaming response."""

        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            if isinstance(raw_line, bytes):
                raw_line = raw_line.decode("utf-8")
            line = raw_line.strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                logger.debug("Ignoring malformed SSE payload: %s", data)

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
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
        """Perform a synchronous Cloud Code completion request."""

        optional_params = optional_params or {}
        self._validate_request(model, messages)
        url, req_headers, body, http_client, should_close = self._prepare_request(
            model=model,
            messages=messages,
            api_base=api_base,
            optional_params=optional_params,
            headers=headers,
            api_key=api_key,
            timeout=timeout,
            client=client,
            stream=False,
        )
        try:
            response = http_client.post(url, headers=req_headers, json=body)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text if hasattr(exc.response, "text") else str(exc)
            logger.error("Cloud Code HTTP error: %s", detail)
            raise UUTELError(
                f"Cloud Code API error: {exc.response.status_code} - {detail}",
                provider=self.provider_name,
            ) from exc
        except httpx.HTTPError as exc:
            raise UUTELError(
                f"Cloud Code request failed: {exc}",
                provider=self.provider_name,
            ) from exc
        finally:
            if should_close:
                http_client.close()

        candidates, usage_metadata = self._normalise_payload(payload)
        if not candidates:
            raise UUTELError(
                "No candidates in Cloud Code response", provider=self.provider_name
            )

        texts, tool_calls = self._extract_candidate_parts(candidates[0])
        merged_text = " ".join(part.strip() for part in texts if part).strip()
        if not merged_text:
            merged_text = "Cloud Code response did not include text"

        finish_reason = (
            self._normalise_finish_reason(candidates[0].get("finishReason")) or "stop"
        )

        model_response.model = model
        model_response.choices[0].message.content = merged_text
        model_response.choices[0].message.tool_calls = tool_calls or None
        model_response.choices[0].finish_reason = finish_reason
        if usage_metadata:
            model_response.usage = self._map_usage_metadata(usage_metadata)
        return model_response

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
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
        """Delegate async completion to the synchronous implementation."""

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
        print_verbose: Callable,
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
    ) -> Iterator[GenericStreamingChunk]:
        """Stream Cloud Code responses as GenericStreamingChunk objects."""

        optional_params = optional_params or {}
        self._validate_request(model, messages)
        url, req_headers, body, http_client, should_close = self._prepare_request(
            model=model,
            messages=messages,
            api_base=api_base,
            optional_params=optional_params,
            headers=headers,
            api_key=api_key,
            timeout=timeout,
            client=client,
            stream=True,
        )

        latest_usage: dict[str, int] | None = None
        try:
            with http_client.stream(
                "POST", url, headers=req_headers, json=body
            ) as response:
                status_code = getattr(response, "status_code", 200)
                if status_code and status_code >= 400:
                    detail = getattr(response, "text", "")
                    raise UUTELError(
                        f"Cloud Code streaming error: {status_code} - {detail}",
                        provider=self.provider_name,
                    )
                for payload in self._iter_sse_payloads(response):
                    candidates, usage_metadata = self._normalise_payload(payload)
                    if usage_metadata:
                        latest_usage = self._map_usage_metadata(usage_metadata)
                    for candidate in candidates:
                        parts = list(self._iter_candidate_parts(candidate))
                        if not parts:
                            continue
                        finish_reason = self._normalise_finish_reason(
                            candidate.get("finishReason")
                        )
                        for index, (part_type, value) in enumerate(parts):
                            is_last = index == len(parts) - 1
                            finished = bool(finish_reason) and is_last
                            finish_value = finish_reason if finished else None
                            usage = latest_usage if finished else None
                            if part_type == "text":
                                chunk = create_text_chunk(
                                    value,
                                    finished=finished,
                                    usage=usage,
                                    finish_reason=finish_value,
                                )
                            else:
                                arguments = (
                                    value.get("args") if isinstance(value, dict) else {}
                                )
                                try:
                                    arguments_json = json.dumps(arguments)
                                except TypeError:
                                    arguments_json = json.dumps({})
                                chunk = create_tool_chunk(
                                    name=value.get("name", ""),
                                    arguments=arguments_json,
                                    tool_call_id=f"tool_{uuid.uuid4().hex}",
                                    finished=finished,
                                    finish_reason=finish_value,
                                )
                                if usage:
                                    chunk["usage"] = merge_usage_stats(
                                        chunk.get("usage"), usage
                                    )
                            yield chunk
        finally:
            if should_close:
                http_client.close()

    async def astreaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
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
    ) -> AsyncIterator[GenericStreamingChunk]:
        """Delegate async streaming to the synchronous iterator."""

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
