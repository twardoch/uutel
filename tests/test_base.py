# this_file: tests/test_base.py
"""Tests for UUTEL base classes."""

from __future__ import annotations

import pytest
from litellm import CustomLLM  # type: ignore[attr-defined]
from litellm.types.utils import ModelResponse

from uutel.core.base import BaseUU


class TestBaseUU:
    """Test the BaseUU base class."""

    def test_base_uu_inherits_from_custom_llm(self) -> None:
        """Test that BaseUU properly inherits from LiteLLM's CustomLLM."""
        base_uu = BaseUU()
        assert isinstance(base_uu, CustomLLM)
        assert hasattr(base_uu, "completion")
        assert hasattr(base_uu, "acompletion")
        assert hasattr(base_uu, "streaming")
        assert hasattr(base_uu, "astreaming")

    def test_base_uu_has_provider_name(self) -> None:
        """Test that BaseUU has a provider_name attribute."""
        base_uu = BaseUU()
        assert hasattr(base_uu, "provider_name")
        assert base_uu.provider_name == "base"

    def test_base_uu_has_supported_models(self) -> None:
        """Test that BaseUU has a supported_models attribute."""
        base_uu = BaseUU()
        assert hasattr(base_uu, "supported_models")
        assert isinstance(base_uu.supported_models, list)

    def test_base_uu_completion_not_implemented(self) -> None:
        """Test that BaseUU completion method raises NotImplementedError."""
        base_uu = BaseUU()

        with pytest.raises(NotImplementedError):
            base_uu.completion(
                model="test-model",
                messages=[{"role": "user", "content": "Hello"}],
                api_base="",
                custom_prompt_dict={},
                model_response=ModelResponse(),
                print_verbose=lambda x: None,
                encoding=None,
                api_key=None,
                logging_obj=None,
                optional_params={},
            )

    def test_base_uu_acompletion_not_implemented(self) -> None:
        """Test that BaseUU acompletion method raises NotImplementedError."""
        base_uu = BaseUU()

        # Test that the method exists and has correct signature
        assert hasattr(base_uu, "acompletion")
        assert callable(base_uu.acompletion)
        # For now we just test that the method exists, async testing will be added later
