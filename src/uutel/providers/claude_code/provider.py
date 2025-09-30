# this_file: src/uutel/providers/claude_code/provider.py
"""Claude Code provider implementation for UUTEL."""

from __future__ import annotations

import json
import shutil
from collections.abc import AsyncIterator, Iterator
from typing import Any

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import GenericStreamingChunk, ModelResponse

from uutel.core.base import BaseUU
from uutel.core.exceptions import UUTELError
from uutel.core.logging_config import get_logger
from uutel.core.runners import (
    astream_subprocess_lines,
    run_subprocess,
    stream_subprocess_lines,
)
from uutel.core.utils import create_text_chunk, create_tool_chunk, merge_usage_stats

logger = get_logger(__name__)

_DEFAULT_TIMEOUT = 120.0
_ALLOWED_TOOL_TYPES = {"function"}
_STREAM_RESULT_TYPES = {"result", "completion"}
_STREAM_TOOL_SUBTYPES = {"tool", "tool_use", "tool-call", "tool_call"}
_STREAM_PARTIAL_SUBTYPES = {"partial", "delta", "message", "message_delta"}


class ClaudeCodeUU(BaseUU):
    """LiteLLM-compatible provider backed by the Claude Code CLI."""

    def __init__(self) -> None:
        super().__init__()
        self.provider_name = "claude_code"
        self.supported_models = [
            "claude-sonnet-4",
            "claude-opus-4",
            "sonnet",
            "opus",
        ]

    # ---------------------------------------------------------------------
    # Helpers

    def _resolve_cli(self) -> str:
        """Return the Claude Code CLI binary path, raising when unavailable."""

        for candidate in ("claude", "claude-code"):
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        raise UUTELError(
            "Claude Code CLI not found. Install with 'npm install -g @anthropic-ai/claude-code' "
            "and run 'claude login' before using this provider.",
            provider=self.provider_name,
        )

    def _ensure_messages(self, model: str, messages: list[Any]) -> None:
        """Validate required arguments."""

        if not model or not isinstance(model, str):
            raise UUTELError("Model name is required", provider=self.provider_name)
        if not messages:
            raise UUTELError("Messages are required", provider=self.provider_name)

    def _normalise_content(self, content: Any) -> str:
        """Convert LiteLLM message content into plain text for the CLI."""

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        parts.append(text_value)
            return "\n".join(parts)
        if isinstance(content, dict) and isinstance(content.get("text"), str):
            return content["text"]
        if content is None:
            return ""
        return json.dumps(content, ensure_ascii=False)

    def _serialise_messages(self, messages: list[Any]) -> dict[str, Any]:
        """Return conversation payload for the CLI environment variable."""

        serialised: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "user")
            content = self._normalise_content(message.get("content"))
            if not content.strip():
                continue
            serialised.append({"role": role, "content": content})
        if not serialised:
            raise UUTELError(
                "At least one non-empty message is required for Claude Code",
                provider=self.provider_name,
            )
        return {"messages": serialised}

    def _collect_tools(
        self, optional_params: dict[str, Any]
    ) -> tuple[list[str], str | None]:
        """Extract allowed tools and chosen tool name from OpenAI-style params."""

        allowed: list[str] = []
        tool_choice: str | None = None
        tools = optional_params.get("tools")
        if isinstance(tools, list):
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                if tool.get("type") not in _ALLOWED_TOOL_TYPES:
                    continue
                function_block = tool.get("function")
                if not isinstance(function_block, dict):
                    continue
                name = function_block.get("name")
                if isinstance(name, str) and name.strip():
                    allowed.append(name.strip())
        choice = optional_params.get("tool_choice")
        if isinstance(choice, dict):
            if choice.get("type") == "tool":
                name = choice.get("tool_name")
                if isinstance(name, str) and name.strip():
                    tool_choice = name.strip()
        return allowed, tool_choice

    def _resolve_working_dir(self, optional_params: dict[str, Any]) -> str | None:
        """Return working directory if provided."""

        cwd = optional_params.get("working_dir") or optional_params.get(
            "working_directory"
        )
        if isinstance(cwd, str) and cwd.strip():
            return cwd
        return None

    def _resolve_timeout(self, optional_params: dict[str, Any]) -> float:
        """Return timeout from optional params or fallback default."""

        timeout = optional_params.get("timeout")
        if isinstance(timeout, int | float) and timeout > 0:
            return float(timeout)
        return _DEFAULT_TIMEOUT

    def _build_command(
        self,
        cli_path: str,
        model: str,
        *,
        stream: bool,
        temperature: float,
        max_tokens: int,
    ) -> list[str]:
        """Construct CLI invocation arguments."""

        command = [cli_path, "api", "--json", "--model", model]
        command.extend(["--temperature", f"{float(temperature):.2f}"])
        command.extend(["--max-tokens", str(int(max_tokens))])
        if stream:
            command.append("--stream")
        return command

    def _build_environment(
        self,
        messages: list[Any],
        optional_params: dict[str, Any],
    ) -> dict[str, str]:
        """Prepare environment variables for subprocess execution."""

        payload = self._serialise_messages(messages)
        allowed_tools, tool_choice = self._collect_tools(optional_params)
        env: dict[str, str] = {
            "CLAUDE_CONVERSATION_JSON": json.dumps(payload, ensure_ascii=False),
        }
        if allowed_tools:
            env["CLAUDE_ALLOWED_TOOLS"] = ",".join(sorted(allowed_tools))
        if tool_choice:
            env["CLAUDE_TOOL_CHOICE"] = tool_choice
        working_dir = self._resolve_working_dir(optional_params)
        if working_dir:
            env["CLAUDE_WORKING_DIR"] = working_dir
        user_env = optional_params.get("environment")
        if isinstance(user_env, dict):
            for key, value in user_env.items():
                if isinstance(key, str) and isinstance(value, str | int | float):
                    env[key] = str(value)
        return env

    def _parse_cli_payload(self, text: str) -> dict[str, Any]:
        """Parse JSON payload returned by the CLI."""

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            for line in text.splitlines():
                candidate = line.strip()
                if not candidate:
                    continue
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
        raise UUTELError(
            "Claude Code CLI returned no JSON payload", provider=self.provider_name
        )

    def _extract_usage(self, payload: dict[str, Any]) -> dict[str, int]:
        """Extract token usage from CLI payload."""

        usage_payload = payload.get("usage")
        if not isinstance(usage_payload, dict):
            return {}
        input_tokens = int(usage_payload.get("input_tokens") or 0)
        output_tokens = int(usage_payload.get("output_tokens") or 0)
        total_tokens = int(
            usage_payload.get("total_tokens") or (input_tokens + output_tokens)
        )
        usage: dict[str, int] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        for extra_key in ("cache_creation_input_tokens", "cache_read_input_tokens"):
            value = usage_payload.get(extra_key)
            if isinstance(value, int):
                usage[extra_key] = value
        return usage

    def _extract_text(self, payload: dict[str, Any]) -> str:
        """Return result text from CLI payload."""

        result_text = payload.get("result")
        if isinstance(result_text, str):
            return result_text
        text = payload.get("text")
        if isinstance(text, str):
            return text
        return ""

    # ------------------------------------------------------------------
    # LiteLLM interface

    def completion(
        self,
        model: str,
        messages: list[Any],
        api_base: str,
        custom_prompt_dict: dict[str, Any],
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict[str, Any],
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        client: HTTPHandler | None = None,
    ) -> ModelResponse:
        """Synchronous completion via Claude Code CLI."""

        del api_base, custom_prompt_dict, print_verbose, encoding, api_key, logging_obj
        del acompletion, litellm_params, logger_fn, headers, timeout, client

        self._ensure_messages(model, messages)
        cli_path = self._resolve_cli()
        temperature = float(optional_params.get("temperature", 0.7))
        max_tokens = int(optional_params.get("max_tokens", 1024))
        command = self._build_command(
            cli_path,
            model,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        env = self._build_environment(messages, optional_params)
        resolved_timeout = self._resolve_timeout(optional_params)
        working_dir = self._resolve_working_dir(optional_params)

        logger.debug(
            "Running Claude CLI completion", command=command, env_keys=list(env)
        )
        result = run_subprocess(
            command,
            env=env,
            timeout=resolved_timeout,
            cwd=working_dir,
        )
        payload = self._parse_cli_payload(result.stdout)
        usage = self._extract_usage(payload)
        content = self._extract_text(payload)
        finish_reason = payload.get("finish_reason", "stop")

        model_response.model = model
        model_response.choices[0].message.content = content
        model_response.choices[0].finish_reason = finish_reason
        model_response.choices[0].message.tool_calls = None
        model_response.usage = usage

        logger.debug("Claude CLI completion succeeded", usage=usage)
        return model_response

    async def acompletion(
        self,
        model: str,
        messages: list[Any],
        api_base: str,
        custom_prompt_dict: dict[str, Any],
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict[str, Any],
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> ModelResponse:
        """Async completion delegates to synchronous implementation."""

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
            client=None,
        )

    def streaming(
        self,
        model: str,
        messages: list[Any],
        api_base: str,
        custom_prompt_dict: dict[str, Any],
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict[str, Any],
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        client: HTTPHandler | None = None,
    ) -> Iterator[GenericStreamingChunk]:
        """Synchronous streaming using Claude Code CLI JSONL output."""

        del api_base, custom_prompt_dict, model_response, print_verbose, encoding
        del api_key, logging_obj, acompletion, litellm_params, logger_fn, headers
        del timeout, client

        self._ensure_messages(model, messages)
        cli_path = self._resolve_cli()
        temperature = float(optional_params.get("temperature", 0.7))
        max_tokens = int(optional_params.get("max_tokens", 1024))
        command = self._build_command(
            cli_path,
            model,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        env = self._build_environment(messages, optional_params)
        resolved_timeout = self._resolve_timeout(optional_params)
        working_dir = self._resolve_working_dir(optional_params)
        cancel_event = optional_params.get("cancel_event")
        if (
            cancel_event
            and getattr(cancel_event, "is_set", None)
            and cancel_event.is_set()
        ):
            raise UUTELError(
                "Claude Code streaming cancelled before start",
                provider=self.provider_name,
            )

        logger.debug("Starting Claude CLI streaming", command=command)
        aggregated_usage: dict[str, int] | None = None
        aggregated_text: list[str] = []
        chunk_index = 0

        for raw_line in stream_subprocess_lines(
            command,
            env=env,
            timeout=resolved_timeout,
            cwd=working_dir,
        ):
            if (
                cancel_event
                and getattr(cancel_event, "is_set", None)
                and cancel_event.is_set()
            ):
                raise UUTELError(
                    "Claude Code streaming cancelled by caller",
                    provider=self.provider_name,
                )
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                logger.debug("Ignoring non-JSON CLI line: %s", raw_line)
                continue

            usage = self._extract_usage(payload)
            if usage:
                aggregated_usage = merge_usage_stats(aggregated_usage, usage)

            subtype = str(payload.get("subtype", "")).lower()
            event_type = str(payload.get("type", "")).lower()

            if (
                subtype in _STREAM_PARTIAL_SUBTYPES
                or event_type in _STREAM_PARTIAL_SUBTYPES
            ):
                text = self._extract_text(payload)
                if text:
                    aggregated_text.append(text)
                    yield create_text_chunk(text, index=chunk_index)
                    chunk_index += 1
                continue

            if subtype in _STREAM_TOOL_SUBTYPES:
                tool_payload = payload.get("tool") or {}
                tool_name = tool_payload.get("name", "tool")
                arguments = tool_payload.get("input") or {}
                yield create_tool_chunk(
                    name=str(tool_name),
                    arguments=json.dumps(arguments, ensure_ascii=False),
                    index=chunk_index,
                )
                chunk_index += 1
                continue

            if event_type in _STREAM_RESULT_TYPES or subtype in _STREAM_RESULT_TYPES:
                final_text = self._extract_text(payload) or "".join(aggregated_text)
                finish_reason = payload.get("finish_reason", "stop")
                yield create_text_chunk(
                    final_text,
                    index=chunk_index,
                    finished=True,
                    finish_reason=str(finish_reason),
                    usage=aggregated_usage,
                )
                return

        if aggregated_text:
            final_text = "".join(aggregated_text)
            yield create_text_chunk(
                final_text,
                index=chunk_index,
                finished=True,
                usage=aggregated_usage,
            )

    async def astreaming(
        self,
        model: str,
        messages: list[Any],
        api_base: str,
        custom_prompt_dict: dict[str, Any],
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict[str, Any],
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        """Async streaming using asyncio subprocess helper."""

        del api_base, custom_prompt_dict, model_response, print_verbose, encoding
        del api_key, logging_obj, acompletion, litellm_params, logger_fn, headers
        del timeout, client

        self._ensure_messages(model, messages)
        cli_path = self._resolve_cli()
        temperature = float(optional_params.get("temperature", 0.7))
        max_tokens = int(optional_params.get("max_tokens", 1024))
        command = self._build_command(
            cli_path,
            model,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        env = self._build_environment(messages, optional_params)
        working_dir = self._resolve_working_dir(optional_params)
        cancel_event = optional_params.get("cancel_event")
        if (
            cancel_event
            and getattr(cancel_event, "is_set", None)
            and cancel_event.is_set()
        ):
            raise UUTELError(
                "Claude Code streaming cancelled before start",
                provider=self.provider_name,
            )

        aggregated_usage: dict[str, int] | None = None
        aggregated_text: list[str] = []
        chunk_index = 0

        async for raw_line in astream_subprocess_lines(
            command,
            env=env,
            cwd=working_dir,
        ):
            if (
                cancel_event
                and getattr(cancel_event, "is_set", None)
                and cancel_event.is_set()
            ):
                raise UUTELError(
                    "Claude Code streaming cancelled by caller",
                    provider=self.provider_name,
                )
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                logger.debug("Ignoring non-JSON CLI line: %s", raw_line)
                continue

            usage = self._extract_usage(payload)
            if usage:
                aggregated_usage = merge_usage_stats(aggregated_usage, usage)

            subtype = str(payload.get("subtype", "")).lower()
            event_type = str(payload.get("type", "")).lower()

            if (
                subtype in _STREAM_PARTIAL_SUBTYPES
                or event_type in _STREAM_PARTIAL_SUBTYPES
            ):
                text = self._extract_text(payload)
                if text:
                    aggregated_text.append(text)
                    yield create_text_chunk(text, index=chunk_index)
                    chunk_index += 1
                continue

            if subtype in _STREAM_TOOL_SUBTYPES:
                tool_payload = payload.get("tool") or {}
                tool_name = tool_payload.get("name", "tool")
                arguments = tool_payload.get("input") or {}
                yield create_tool_chunk(
                    name=str(tool_name),
                    arguments=json.dumps(arguments, ensure_ascii=False),
                    index=chunk_index,
                )
                chunk_index += 1
                continue

            if event_type in _STREAM_RESULT_TYPES or subtype in _STREAM_RESULT_TYPES:
                final_text = self._extract_text(payload) or "".join(aggregated_text)
                finish_reason = payload.get("finish_reason", "stop")
                yield create_text_chunk(
                    final_text,
                    index=chunk_index,
                    finished=True,
                    finish_reason=str(finish_reason),
                    usage=aggregated_usage,
                )
                return

        if aggregated_text:
            final_text = "".join(aggregated_text)
            yield create_text_chunk(
                final_text,
                index=chunk_index,
                finished=True,
                usage=aggregated_usage,
            )
