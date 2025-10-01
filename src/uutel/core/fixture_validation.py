# this_file: src/uutel/core/fixture_validation.py
"""Schema validation helpers for recorded provider fixtures."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from jsonschema import Draft202012Validator, exceptions

_OPENAI_COMPLETION_SCHEMA = {
    "type": "object",
    "required": ["id", "object", "model", "choices", "usage"],
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "object": {"type": "string", "minLength": 1},
        "model": {"type": "string", "minLength": 1},
        "choices": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["message"],
                "properties": {
                    "message": {
                        "type": "object",
                        "required": ["role", "content"],
                        "properties": {
                            "role": {"type": "string", "minLength": 1},
                            "content": {
                                "anyOf": [
                                    {"type": "string", "minLength": 1},
                                    {"type": "array", "minItems": 1},
                                    {"type": "object", "minProperties": 1},
                                ]
                            },
                        },
                        "additionalProperties": True,
                    },
                    "finish_reason": {"type": ["string", "null"]},
                },
                "additionalProperties": True,
            },
        },
        "usage": {
            "type": "object",
            "required": ["prompt_tokens", "completion_tokens", "total_tokens"],
            "properties": {
                "prompt_tokens": {"type": "integer", "minimum": 1},
                "completion_tokens": {"type": "integer", "minimum": 1},
                "total_tokens": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    "additionalProperties": True,
}

_CLAUDE_COMPLETION_SCHEMA = {
    "type": "object",
    "required": ["result", "usage"],
    "properties": {
        "result": {"type": "string", "minLength": 1},
        "usage": {
            "type": "object",
            "required": ["input_tokens", "output_tokens"],
            "properties": {
                "input_tokens": {"type": "integer", "minimum": 1},
                "output_tokens": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    "additionalProperties": True,
}

_GEMINI_COMPLETION_SCHEMA = {
    "type": "object",
    "required": ["candidates", "usageMetadata"],
    "properties": {
        "candidates": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["content"],
                "properties": {
                    "content": {
                        "type": "object",
                        "required": ["parts"],
                        "properties": {
                            "parts": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "object",
                                    "required": ["text"],
                                    "properties": {
                                        "text": {"type": "string", "minLength": 1}
                                    },
                                    "additionalProperties": True,
                                },
                            },
                        },
                        "additionalProperties": True,
                    }
                },
                "additionalProperties": True,
            },
        },
        "usageMetadata": {
            "type": "object",
            "required": [
                "promptTokenCount",
                "candidatesTokenCount",
                "totalTokenCount",
            ],
            "properties": {
                "promptTokenCount": {"type": "integer", "minimum": 1},
                "candidatesTokenCount": {"type": "integer", "minimum": 1},
                "totalTokenCount": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    "additionalProperties": True,
}

_CLOUD_CODE_COMPLETION_SCHEMA = {
    "type": "object",
    "required": ["response"],
    "properties": {
        "response": {
            "type": "object",
            "required": ["candidates", "usageMetadata"],
            "properties": {
                "candidates": _GEMINI_COMPLETION_SCHEMA["properties"]["candidates"],
                "usageMetadata": _GEMINI_COMPLETION_SCHEMA["properties"][
                    "usageMetadata"
                ],
            },
            "additionalProperties": True,
        }
    },
    "additionalProperties": True,
}

_COMPLETION_FIXTURE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "anyOf": [
        _OPENAI_COMPLETION_SCHEMA,
        _CLAUDE_COMPLETION_SCHEMA,
        _GEMINI_COMPLETION_SCHEMA,
        _CLOUD_CODE_COMPLETION_SCHEMA,
    ],
}

_VALIDATOR = Draft202012Validator(_COMPLETION_FIXTURE_SCHEMA)


def _iter_leaf_errors(
    error: exceptions.ValidationError,
) -> Iterator[exceptions.ValidationError]:
    """Yield the deepest validation errors for an `anyOf` failure tree."""

    if error.context:
        for child in error.context:
            yield from _iter_leaf_errors(child)
    else:
        yield error


def _extract_missing_property(error: exceptions.ValidationError) -> str | None:
    """Return the property name that failed a required check, when available."""

    if error.validator != "required":
        return None

    message = getattr(error, "message", "")
    if not isinstance(message, str):
        return None

    # JSON Schema renders messages like "'total_tokens' is a required property"
    if "'" in message:
        parts = message.split("'")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return None


def _format_error_path(error: exceptions.ValidationError) -> str:
    """Return a human readable dotted path for a schema violation."""

    path_parts: Iterable[str] = [str(part) for part in error.absolute_path]
    base_path = ".".join(path_parts)

    missing_property = _extract_missing_property(error)
    if missing_property:
        base_path = f"{base_path}.{missing_property}" if base_path else missing_property

    return base_path or "<root>"


def _check_total_consistency(
    *,
    payload: dict[str, Any],
    total_value: Any,
    first_value: Any,
    second_value: Any,
    total_path: str,
    first_path: str,
    second_path: str,
) -> None:
    """Raise a validation error when aggregated usage totals diverge from components."""

    if not isinstance(total_value, int):
        return
    if not isinstance(first_value, int) or not isinstance(second_value, int):
        return

    expected_total = first_value + second_value
    if total_value == expected_total:
        return

    message = (
        f"{total_path} mismatch: expected {first_path} ({first_value}) + {second_path} ({second_value}) = {expected_total}, "
        f"got {total_value}"
    )
    raise exceptions.ValidationError(
        message,
        instance=payload,
        schema=_COMPLETION_FIXTURE_SCHEMA,
    )


def _enforce_usage_totals(payload: dict[str, Any]) -> None:
    """Ensure provider usage totals align with their component counts."""

    usage = payload.get("usage")
    if isinstance(usage, dict):
        _check_total_consistency(
            payload=payload,
            total_value=usage.get("total_tokens"),
            first_value=usage.get("prompt_tokens"),
            second_value=usage.get("completion_tokens"),
            total_path="usage.total_tokens",
            first_path="usage.prompt_tokens",
            second_path="usage.completion_tokens",
        )

    usage_metadata = payload.get("usageMetadata")
    if isinstance(usage_metadata, dict):
        _check_total_consistency(
            payload=payload,
            total_value=usage_metadata.get("totalTokenCount"),
            first_value=usage_metadata.get("promptTokenCount"),
            second_value=usage_metadata.get("candidatesTokenCount"),
            total_path="usageMetadata.totalTokenCount",
            first_path="usageMetadata.promptTokenCount",
            second_path="usageMetadata.candidatesTokenCount",
        )

    response = payload.get("response")
    if isinstance(response, dict):
        nested_usage = response.get("usageMetadata")
        if isinstance(nested_usage, dict):
            _check_total_consistency(
                payload=payload,
                total_value=nested_usage.get("totalTokenCount"),
                first_value=nested_usage.get("promptTokenCount"),
                second_value=nested_usage.get("candidatesTokenCount"),
                total_path="response.usageMetadata.totalTokenCount",
                first_path="response.usageMetadata.promptTokenCount",
                second_path="response.usageMetadata.candidatesTokenCount",
            )


def _raise_blank_text(payload: dict[str, Any], path: str) -> None:
    """Raise a validation error indicating whitespace-only text at a path."""

    raise exceptions.ValidationError(
        f"{path} must contain non-empty text",
        instance=payload,
        schema=_COMPLETION_FIXTURE_SCHEMA,
    )


def _ensure_meaningful_text(payload: dict[str, Any]) -> None:
    """Ensure key textual fields contain more than whitespace."""

    choices = payload.get("choices")
    if isinstance(choices, list):
        for index, choice in enumerate(choices):
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and not content.strip():
                _raise_blank_text(payload, f"choices.{index}.message.content")

    result = payload.get("result")
    if isinstance(result, str) and not result.strip():
        _raise_blank_text(payload, "result")

    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for c_index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            for p_index, part in enumerate(parts):
                if isinstance(part, dict):
                    text_value = part.get("text")
                    if isinstance(text_value, str) and not text_value.strip():
                        _raise_blank_text(
                            payload,
                            f"candidates.{c_index}.content.parts.{p_index}.text",
                        )

    response = payload.get("response")
    if isinstance(response, dict):
        nested = response.get("candidates")
        if isinstance(nested, list):
            for c_index, candidate in enumerate(nested):
                if not isinstance(candidate, dict):
                    continue
                content = candidate.get("content")
                if not isinstance(content, dict):
                    continue
                parts = content.get("parts")
                if not isinstance(parts, list):
                    continue
                for p_index, part in enumerate(parts):
                    if isinstance(part, dict):
                        text_value = part.get("text")
                        if isinstance(text_value, str) and not text_value.strip():
                            _raise_blank_text(
                                payload,
                                (
                                    "response.candidates"
                                    f".{c_index}.content.parts.{p_index}.text"
                                ),
                            )


def validate_completion_fixture(payload: Any) -> None:
    """Validate a recorded completion fixture against the shared schema.

    Args:
        payload: Parsed JSON payload from a provider fixture.

    Raises:
        jsonschema.exceptions.ValidationError: When the payload violates the schema.
        TypeError: If a non-mapping payload is provided.
    """

    if not isinstance(payload, dict):
        actual_type = type(payload).__name__
        raise TypeError(
            "Fixture payload must be a mapping produced from JSON object "
            f"(got {actual_type})"
        )

    raw_errors = list(_VALIDATOR.iter_errors(payload))
    errors: list[exceptions.ValidationError] = []

    for error in raw_errors:
        errors.extend(list(_iter_leaf_errors(error)))

    if errors:
        formatted_errors = ", ".join(
            f"{_format_error_path(error)}: {error.message}" for error in errors
        )
        raise exceptions.ValidationError(
            f"Fixture payload failed validation: {formatted_errors}",
            instance=payload,
            schema=_COMPLETION_FIXTURE_SCHEMA,
        )

    _enforce_usage_totals(payload)
    _ensure_meaningful_text(payload)


__all__ = ["validate_completion_fixture"]
