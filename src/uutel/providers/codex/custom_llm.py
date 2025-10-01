# this_file: src/uutel/providers/codex/custom_llm.py
"""LiteLLM CustomLLM implementation for CodexUU provider.

This module provides a thin adapter that bridges the gap between LiteLLM's
CustomLLM interface and UUTEL's provider implementations.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from difflib import get_close_matches
from typing import Any

import litellm
from litellm import CustomLLM
from litellm.types.utils import GenericStreamingChunk, ModelResponse

from uutel.core.exceptions import UUTELError
from uutel.core.logging_config import get_logger
from uutel.providers.codex import CodexUU

logger = get_logger(__name__)

_ENGINE_MODEL_MAP: dict[str, str] = {
    "my-custom-llm/codex-large": "gpt-4o",
    "my-custom-llm/codex-mini": "gpt-4o-mini",
    "my-custom-llm/codex-turbo": "gpt-4-turbo",
    "my-custom-llm/codex-fast": "gpt-3.5-turbo",
    "my-custom-llm/codex-preview": "o1-preview",
    "codex-large": "gpt-4o",
    "codex-mini": "gpt-4o-mini",
    "codex-turbo": "gpt-4-turbo",
    "codex-fast": "gpt-3.5-turbo",
    "codex-preview": "o1-preview",
    "uutel-codex/gpt-4o": "gpt-4o",
    "uutel-codex/gpt-4o-mini": "gpt-4o-mini",
    "uutel-codex/gpt-4-turbo": "gpt-4-turbo",
    "uutel-codex/gpt-3.5-turbo": "gpt-3.5-turbo",
    "uutel-codex/o1-preview": "o1-preview",
    "uutel-codex/o1-mini": "o1-mini",
}

_ENGINE_MODEL_MAP_CASEFOLD: dict[str, str] = {
    key.casefold(): value for key, value in _ENGINE_MODEL_MAP.items()
}

_LITELLM_EXCEPTION_TYPE = getattr(litellm, "LiteLLMException", None)


class CodexCustomLLM(CustomLLM):
    """LiteLLM adapter delegating to the real Codex provider implementation."""

    def __init__(self, provider: CodexUU | None = None) -> None:
        super().__init__()
        self._provider = provider or CodexUU()
        self.provider_name = "codex"
        self.supported_models = list(self._provider.supported_models)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_model_response(
        self, model_response: ModelResponse | None
    ) -> ModelResponse:
        response = model_response or litellm.ModelResponse()
        if not getattr(response, "choices", None):
            choice = litellm.utils.Choices()
            choice.message = litellm.utils.Message()
            response.choices = [choice]
        elif not getattr(response.choices[0], "message", None):
            response.choices[0].message = litellm.utils.Message()
        return response

    def _map_model_name(self, model: Any) -> str:
        if not isinstance(model, str):
            raise litellm.BadRequestError(
                "Model must be a string for Codex provider",
                model="",
                llm_provider=self.provider_name,
            )

        trimmed_model = model.strip()
        if not trimmed_model:
            raise litellm.BadRequestError(
                "Model name is required for Codex provider",
                model="",
                llm_provider=self.provider_name,
            )

        normalised_model = trimmed_model.casefold()

        if normalised_model in _ENGINE_MODEL_MAP_CASEFOLD:
            return _ENGINE_MODEL_MAP_CASEFOLD[normalised_model]

        supported_lookup = {
            supported.casefold(): supported
            for supported in self._provider.supported_models
        }

        if "/" in trimmed_model:
            candidate = trimmed_model.rsplit("/", 1)[-1]
            candidate_key = candidate.casefold()
            if candidate_key in supported_lookup:
                return supported_lookup[candidate_key]

        if normalised_model in supported_lookup:
            return supported_lookup[normalised_model]

        available_candidates = list(_ENGINE_MODEL_MAP.keys()) + list(
            self._provider.supported_models
        )
        display_map = {
            candidate.casefold(): candidate for candidate in available_candidates
        }
        candidate_keys = list(display_map.keys())

        suggestions: list[str] = []

        best_match_keys = get_close_matches(
            normalised_model,
            candidate_keys,
            n=1,
            cutoff=0.6,
        )
        for key in best_match_keys:
            candidate = display_map.get(key)
            if candidate and candidate not in suggestions:
                suggestions.append(candidate)

        prefix = None
        if "-" in trimmed_model:
            prefix = trimmed_model.split("-", 1)[0].casefold()
        elif "/" in trimmed_model:
            prefix = trimmed_model.split("/", 1)[0].casefold()

        if prefix:
            for candidate in available_candidates:
                key = candidate.casefold()
                if key.startswith(prefix) and candidate not in suggestions:
                    suggestions.append(candidate)
                if len(suggestions) >= 3:
                    break

        if len(suggestions) < 3:
            for candidate in available_candidates:
                if candidate not in suggestions:
                    suggestions.append(candidate)
                if len(suggestions) >= 3:
                    break

        suggestion_hint = ""
        if suggestions:
            joined = ", ".join(suggestions)
            suggestion_hint = f" Did you mean: {joined}?"

        raise litellm.BadRequestError(
            f"Unsupported Codex model '{trimmed_model}'.{suggestion_hint}",
            model=trimmed_model,
            llm_provider=self.provider_name,
        )

    def _prepare_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(kwargs)
        resolved_model = self._map_model_name(prepared.get("model", ""))
        prepared["model"] = resolved_model
        prepared["model_response"] = self._ensure_model_response(
            prepared.get("model_response")
        )
        return prepared

    # ------------------------------------------------------------------
    # LiteLLM interface implementation
    # ------------------------------------------------------------------

    def completion(self, *args: Any, **kwargs: Any) -> ModelResponse:
        prepared_kwargs = self._prepare_kwargs(kwargs)
        try:
            return self._provider.completion(*args, **prepared_kwargs)
        except UUTELError as exc:
            logger.error("Codex provider error: %s", exc)
            raise litellm.APIConnectionError(
                str(exc),
                llm_provider=self.provider_name,
                model=prepared_kwargs.get("model", ""),
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive
            if _LITELLM_EXCEPTION_TYPE and isinstance(exc, _LITELLM_EXCEPTION_TYPE):
                raise
            logger.exception("Codex provider unexpected failure")
            raise litellm.APIConnectionError(
                f"Codex provider failure: {exc}",
                llm_provider=self.provider_name,
                model=prepared_kwargs.get("model", ""),
            ) from exc

    async def acompletion(self, *args: Any, **kwargs: Any) -> ModelResponse:
        prepared_kwargs = self._prepare_kwargs(kwargs)
        try:
            return await self._provider.acompletion(*args, **prepared_kwargs)
        except UUTELError as exc:
            logger.error("Codex provider error: %s", exc)
            raise litellm.APIConnectionError(
                str(exc),
                llm_provider=self.provider_name,
                model=prepared_kwargs.get("model", ""),
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive
            if _LITELLM_EXCEPTION_TYPE and isinstance(exc, _LITELLM_EXCEPTION_TYPE):
                raise
            logger.exception("Codex provider unexpected failure")
            raise litellm.APIConnectionError(
                f"Codex provider failure: {exc}",
                llm_provider=self.provider_name,
                model=prepared_kwargs.get("model", ""),
            ) from exc

    def streaming(self, *args: Any, **kwargs: Any) -> Iterator[GenericStreamingChunk]:
        prepared_kwargs = self._prepare_kwargs(kwargs)
        try:
            iterator = self._provider.streaming(*args, **prepared_kwargs)
            yield from iterator
        except UUTELError as exc:
            logger.error("Codex provider error: %s", exc)
            raise litellm.APIConnectionError(
                str(exc),
                llm_provider=self.provider_name,
                model=prepared_kwargs.get("model", ""),
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive
            if _LITELLM_EXCEPTION_TYPE and isinstance(exc, _LITELLM_EXCEPTION_TYPE):
                raise
            logger.exception("Codex provider unexpected failure")
            raise litellm.APIConnectionError(
                f"Codex provider failure: {exc}",
                llm_provider=self.provider_name,
                model=prepared_kwargs.get("model", ""),
            ) from exc

    async def astreaming(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[GenericStreamingChunk]:
        prepared_kwargs = self._prepare_kwargs(kwargs)
        try:
            async for chunk in self._provider.astreaming(*args, **prepared_kwargs):
                yield chunk
        except UUTELError as exc:
            logger.error("Codex provider error: %s", exc)
            raise litellm.APIConnectionError(
                str(exc),
                llm_provider=self.provider_name,
                model=prepared_kwargs.get("model", ""),
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive
            if _LITELLM_EXCEPTION_TYPE and isinstance(exc, _LITELLM_EXCEPTION_TYPE):
                raise
            logger.exception("Codex provider unexpected failure")
            raise litellm.APIConnectionError(
                f"Codex provider failure: {exc}",
                llm_provider=self.provider_name,
                model=prepared_kwargs.get("model", ""),
            ) from exc
