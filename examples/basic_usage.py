#!/usr/bin/env python3
# this_file: examples/basic_usage.py
"""Deterministic walkthrough of UUTEL capabilities using recorded completions."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from uutel import (
    create_http_client,
    extract_provider_from_model,
    transform_openai_to_provider,
    transform_provider_to_openai,
    validate_model_name,
)
from uutel.core.logging_config import get_logger

logger = get_logger(__name__)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "tests" / "data" / "providers"

RECORDED_FIXTURES: list[dict[str, Any]] = [
    {
        "label": "Codex (GPT-4o)",
        "key": "codex",
        "engine": "codex",
        "prompt": "Write a sorter",
        "path": FIXTURE_ROOT / "codex" / "simple_completion.json",
        "live_hint": 'uutel complete --prompt "Write a sorter" --engine codex',
    },
    {
        "label": "Claude Code (Sonnet 4)",
        "key": "claude",
        "engine": "claude",
        "prompt": "Say hello",
        "path": FIXTURE_ROOT / "claude" / "simple_completion.json",
        "live_hint": 'uutel complete --prompt "Say hello" --engine claude',
    },
    {
        "label": "Gemini CLI (2.5 Pro)",
        "key": "gemini",
        "engine": "gemini",
        "prompt": "Summarise Gemini API",
        "path": FIXTURE_ROOT / "gemini" / "simple_completion.json",
        "live_hint": 'uutel complete --prompt "Summarise Gemini API" --engine gemini',
    },
    {
        "label": "Cloud Code (Gemini 2.5 Pro)",
        "key": "cloud_code",
        "engine": "cloud",
        "prompt": "Deployment checklist",
        "path": FIXTURE_ROOT / "cloud_code" / "simple_completion.json",
        "live_hint": 'uutel complete --prompt "Deployment checklist" --engine cloud',
    },
]


def truncate(text: str, limit: int = 120) -> str:
    """Return a compact preview of text for terminal output."""

    text = " ".join(text.strip().split())
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return f"{text[: limit - 3]}..."


def _normalise_structured_content(content: Any) -> str:
    """Collapse structured message content (lists/dicts) into a plain string."""

    parts: list[str] = []
    _function_keys = {"functionCall", "function_call", "toolCall", "tool_calls"}

    def _collect(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, str):
            parts.append(node)
            return
        if isinstance(node, dict):
            node_type = node.get("type")
            if node_type in {"tool_call", "tool_calls"}:
                return
            if any(key in node for key in _function_keys):
                return
            if "text" in node:
                _collect(node.get("text"))
                return
            if "content" in node:
                _collect(node.get("content"))
                return
            for key, value in node.items():
                if key in _function_keys:
                    continue
                _collect(value)
            return
        if isinstance(node, list):
            for item in node:
                _collect(item)
            return
        text_value = str(node)
        if text_value:
            parts.append(text_value)

    _collect(content)
    combined = "".join(parts)
    return combined.strip()


def _sum_positive_tokens(*values: Any) -> int | None:
    """Return the sum of provided token values when at least one is positive."""

    total = 0
    seen = False

    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            int_value = int(value)
            if int_value > 0:
                total += int_value
                seen = True

    return total if seen and total > 0 else None


def extract_recorded_text(key: str, payload: dict[str, Any]) -> tuple[str, int]:
    """Pull the primary text and token count from a recorded payload."""

    if key == "codex":
        message = payload.get("choices", [{}])[0].get("message", {})
        text = _normalise_structured_content(message.get("content"))
        usage = payload.get("usage") or {}
        total_tokens = usage.get("total_tokens")
        if not isinstance(total_tokens, int) or total_tokens <= 0:
            fallback = _sum_positive_tokens(
                usage.get("prompt_tokens"), usage.get("completion_tokens")
            )
            total_tokens = fallback if fallback is not None else 0
        return text, total_tokens

    if key == "claude":
        text = payload.get("result", "")
        usage = payload.get("usage", {})
        tokens = usage.get("total_tokens")
        if tokens is None:
            tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        return text, tokens

    if key == "gemini":
        candidate = payload.get("candidates", [{}])[0]
        parts = candidate.get("content", {}).get("parts", [])
        text = "".join(
            segment
            for segment in (_normalise_structured_content(part) for part in parts)
            if segment
        )
        usage_metadata = payload.get("usageMetadata") or {}
        total_tokens = usage_metadata.get("totalTokenCount")
        if not isinstance(total_tokens, int) or total_tokens <= 0:
            fallback = _sum_positive_tokens(
                usage_metadata.get("promptTokenCount"),
                usage_metadata.get("candidatesTokenCount"),
            )
            total_tokens = fallback if fallback is not None else 0
        return text, total_tokens

    if key == "cloud_code":
        response = payload.get("response", {})
        candidate = response.get("candidates", [{}])[0]
        parts = candidate.get("content", {}).get("parts", [])
        text = "".join(
            segment
            for segment in (_normalise_structured_content(part) for part in parts)
            if segment
        )
        usage_metadata = response.get("usageMetadata") or {}
        total_tokens = usage_metadata.get("totalTokenCount")
        if not isinstance(total_tokens, int) or total_tokens <= 0:
            fallback = _sum_positive_tokens(
                usage_metadata.get("promptTokenCount"),
                usage_metadata.get("candidatesTokenCount"),
            )
            total_tokens = fallback if fallback is not None else 0
        return text, total_tokens

    raise ValueError(f"Unknown provider key '{key}'")


def _env_flag(name: str) -> bool:
    """Interpret an environment variable as a boolean flag."""

    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on", "y", "t"}


def _resolve_stub_dir() -> Path | None:
    """Return the live fixtures directory when provided."""

    raw = os.getenv("UUTEL_LIVE_FIXTURES_DIR", "").strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_dir() else None


def _load_stub_payload(
    fixture: dict[str, Any], stub_dir: Path | None
) -> tuple[dict[str, Any] | None, Path | None]:
    """Load stub payload for a fixture when stub directory is configured."""

    if not stub_dir:
        return None, None

    candidate = stub_dir / fixture["path"].parent.name / fixture["path"].name
    if not candidate.is_file():
        return None, None

    try:
        raw_text = candidate.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        message = f"Unable to decode stub {candidate}: {exc}"
        logger.warning(message)
        return {"error": message}, candidate
    except OSError as exc:
        reason = getattr(exc, "strerror", None) or str(exc)
        message = f"Unable to read stub {candidate}: {reason}"
        logger.warning(message)
        return {"error": message}, candidate

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        message = f"Invalid stub JSON: {exc}"
        logger.warning(message)
        return {"error": message}, candidate
    return payload, candidate


def _invoke_live_completion(fixture: dict[str, Any]) -> tuple[str, int]:
    """Perform a live completion using LiteLLM providers."""

    import litellm

    from uutel.__main__ import validate_engine  # Local import to avoid CLI startup cost

    canonical_engine = validate_engine(fixture["engine"])
    response = litellm.completion(
        model=canonical_engine,
        messages=[{"role": "user", "content": fixture["prompt"]}],
        max_tokens=fixture.get("max_tokens", 256),
        temperature=fixture.get("temperature", 0.7),
    )

    payload = response.model_dump() if hasattr(response, "model_dump") else response
    text, tokens = extract_recorded_text(fixture["key"], payload)
    return text, tokens


def _gather_live_runs(
    fixtures: list[dict[str, Any]], stub_dir: Path | None
) -> list[dict[str, Any]]:
    """Collect live or stubbed responses for the configured fixtures."""

    entries: list[dict[str, Any]] = []
    cli = None

    for fixture in fixtures:
        label = fixture["label"]
        payload, payload_path = _load_stub_payload(fixture, stub_dir)

        if payload is not None and "error" not in payload:
            text, tokens = extract_recorded_text(fixture["key"], payload)
            entries.append(
                {
                    "label": label,
                    "status": "stub",
                    "text": text,
                    "tokens": tokens,
                    "source": f"Stub fixture {payload_path}",
                }
            )
            continue

        if cli is None:
            from uutel.__main__ import UUTELCLI

            cli = UUTELCLI()

        ready, guidance = cli._check_provider_readiness(fixture["engine"])
        if payload is not None and "error" in payload:
            entry = {
                "label": label,
                "status": "error",
                "message": payload["error"],
            }
            if guidance:
                entry["guidance"] = guidance
            entries.append(entry)
            continue

        if not ready:
            entries.append(
                {
                    "label": label,
                    "status": "unready",
                    "guidance": guidance,
                }
            )
            continue

        try:
            text, tokens = _invoke_live_completion(fixture)
            entries.append(
                {
                    "label": label,
                    "status": "live",
                    "text": text,
                    "tokens": tokens,
                    "source": f"Live via {fixture['engine']}",
                }
            )
        except Exception as exc:  # pragma: no cover - relies on external services
            entry = {
                "label": label,
                "status": "error",
                "message": f"Live call failed: {exc}",
            }
            if guidance:
                entry["guidance"] = guidance
            entries.append(entry)

    return entries


def demonstrate_core_functionality() -> None:
    """Demonstrate key utilities and replay recorded provider outputs."""

    print("🚀 UUTEL Basic Usage Example")
    print("=" * 50)

    # 1. Model validation
    print("\n1️⃣ Model Name Validation")
    models = [
        "uutel-codex/gpt-4o",
        "uutel-claude/claude-sonnet-4",
        "uutel-gemini/gemini-2.5-pro",
        "invalid/model",
    ]
    for model in models:
        status = "✅ Valid" if validate_model_name(model) else "❌ Invalid"
        print(f"   {model}: {status}")

    # 2. Provider extraction
    print("\n2️⃣ Provider/Model Extraction")
    for model in models[:3]:
        provider, short_model = extract_provider_from_model(model)
        print(f"   {model} → Provider: {provider}, Model: {short_model}")

    # 3. Message transformation
    print("\n3️⃣ Message Transformation")
    sample_messages = [
        {"role": "system", "content": "You are a precise assistant."},
        {"role": "user", "content": "List two reliability tips."},
    ]
    provider_format = transform_openai_to_provider(sample_messages, "codex")
    round_trip = transform_provider_to_openai(provider_format, "codex")
    print(f"   Original messages: {len(sample_messages)}")
    print(f"   Return trip preserves content: {sample_messages == round_trip}")

    # 4. HTTP client creation
    print("\n4️⃣ HTTP Client Creation")
    sync_client = create_http_client(async_client=False, timeout=10.0)
    async_client = create_http_client(async_client=True, timeout=10.0)
    print(f"   Sync client type: {type(sync_client).__name__}")
    print(f"   Async client type: {type(async_client).__name__}")

    # 5. Recorded provider responses
    print("\n5️⃣ Recorded Provider Responses")
    for fixture in RECORDED_FIXTURES:
        payload = json.loads(fixture["path"].read_text(encoding="utf-8"))
        text, tokens = extract_recorded_text(fixture["key"], payload)
        print(f"   {fixture['label']}: {truncate(text)}")
        print(f"      Tokens: {tokens}")
        print(f"      Live run: {fixture['live_hint']}")

    # 6. Async functionality demo
    print("\n6️⃣ Async Client Demo")

    async def _async_demo() -> None:
        client = create_http_client(async_client=True, timeout=5.0)
        print(f"   Created async client: {type(client).__name__}")
        if hasattr(client, "aclose"):
            await client.aclose()
        print("   ✅ Async client closed cleanly")

    asyncio.run(_async_demo())

    # 7. Optional live provider runs
    print("\n7️⃣ Live Provider Runs")
    live_mode = _env_flag("UUTEL_LIVE_EXAMPLE")
    stub_dir = _resolve_stub_dir()

    if not live_mode:
        print(
            "   💤 Live mode disabled. Set UUTEL_LIVE_EXAMPLE=1 to perform live calls."
        )
        return

    if stub_dir:
        print(f"   📂 Using stub directory: {stub_dir}")

    entries = _gather_live_runs(RECORDED_FIXTURES, stub_dir)
    if not entries:
        print("   ⚠️ No live responses produced.")
        return

    for entry in entries:
        label = entry["label"]
        status = entry["status"]

        if status in {"live", "stub"}:
            text = entry.get("text", "")
            tokens = entry.get("tokens", 0)
            source = entry.get("source", "")
            print(f"   {label}: {truncate(text)}")
            print(f"      Tokens: {tokens}")
            if source:
                print(f"      Source: {source}")
            continue

        if status == "unready":
            print(f"   {label}: ⚠️ Provider prerequisites missing")
            for hint in entry.get("guidance", []):
                print(f"      {hint}")
            continue

        print(f"   {label}: ⚠️ {entry.get('message', 'Live call failed')}")
        for hint in entry.get("guidance", []):
            print(f"      {hint}")


if __name__ == "__main__":
    demonstrate_core_functionality()
