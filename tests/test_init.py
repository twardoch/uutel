# this_file: tests/test_init.py
"""Test suite for uutel.__init__ module."""

import sys
from unittest.mock import patch


def test_version_import_success() -> None:
    """Test that __version__ can be imported successfully."""
    import uutel

    assert hasattr(uutel, "__version__")
    assert isinstance(uutel.__version__, str)
    assert uutel.__version__ != ""


def test_version_import_fallback() -> None:
    """Test version fallback when _version module is not available."""
    # This test verifies the fallback behavior in the __init__.py
    # Let's test it by temporarily making _version unavailable

    original_version_module = None
    if "uutel._version" in sys.modules:
        original_version_module = sys.modules["uutel._version"]
        del sys.modules["uutel._version"]

    # Remove main module to force reimport
    if "uutel" in sys.modules:
        del sys.modules["uutel"]

    try:
        # Mock the _version import to raise ImportError
        with patch.dict("sys.modules", {"uutel._version": None}):
            # Now import uutel which should trigger fallback
            import uutel

            # Should fall back to the default version
            assert uutel.__version__ == "0.0.0+unknown"
    finally:
        # Restore the original state
        if original_version_module is not None:
            sys.modules["uutel._version"] = original_version_module


def test_all_exports_available() -> None:
    """Test that all exported items are available and importable."""
    import uutel

    # Test that __all__ exists and contains expected exports
    assert hasattr(uutel, "__all__")
    assert isinstance(uutel.__all__, list)
    assert len(uutel.__all__) > 0

    # Test that all items in __all__ are actually available
    for item in uutel.__all__:
        assert hasattr(uutel, item), f"Export {item} not found in uutel module"

    # Test specific core exports
    expected_exports = [
        "__version__",
        "BaseUU",
        "BaseAuth",
        "UUTELError",
        "AuthenticationError",
        "create_http_client",
        "validate_tool_schema",
    ]

    for export in expected_exports:
        assert export in uutel.__all__
        assert hasattr(uutel, export)


def test_core_imports_work() -> None:
    """Test that core imports from submodules work correctly."""
    from uutel import (
        BaseAuth,
        BaseUU,
        UUTELError,
        create_http_client,
        validate_tool_schema,
    )

    # Test that imports are not None
    assert BaseAuth is not None
    assert BaseUU is not None
    assert UUTELError is not None
    assert create_http_client is not None
    assert validate_tool_schema is not None

    # Test that they're the expected types
    assert callable(create_http_client)
    assert callable(validate_tool_schema)
    assert issubclass(UUTELError, Exception)


def test_exception_imports() -> None:
    """Test that all exception classes can be imported."""
    from uutel import (
        AuthenticationError,
        ModelError,
        NetworkError,
        ProviderError,
        RateLimitError,
        UUTELError,
        ValidationError,
    )

    # Test exception hierarchy
    exceptions = [
        AuthenticationError,
        ModelError,
        NetworkError,
        ProviderError,
        RateLimitError,
        ValidationError,
    ]

    for exc_class in exceptions:
        assert issubclass(exc_class, UUTELError)
        assert issubclass(exc_class, Exception)


def test_utility_function_imports() -> None:
    """Test that utility functions can be imported and are callable."""
    from uutel import (
        create_tool_call_response,
        extract_provider_from_model,
        extract_tool_calls_from_response,
        format_error_message,
        get_error_debug_info,
        transform_openai_to_provider,
        transform_openai_tools_to_provider,
        transform_provider_to_openai,
        transform_provider_tools_to_openai,
        validate_model_name,
        validate_tool_schema,
    )

    utilities = [
        create_tool_call_response,
        extract_provider_from_model,
        extract_tool_calls_from_response,
        format_error_message,
        get_error_debug_info,
        transform_openai_to_provider,
        transform_openai_tools_to_provider,
        transform_provider_to_openai,
        transform_provider_tools_to_openai,
        validate_model_name,
        validate_tool_schema,
    ]

    for func in utilities:
        assert callable(func)
