# this_file: tests/test_providers_init.py
"""Test suite for uutel.providers.__init__ module."""


def test_providers_module_import() -> None:
    """Test that providers module can be imported successfully."""
    import uutel.providers

    # Module should import without errors
    assert uutel.providers is not None


def test_providers_module_has_all() -> None:
    """Test that providers module has __all__ attribute."""
    import uutel.providers

    assert hasattr(uutel.providers, "__all__")
    assert isinstance(uutel.providers.__all__, list)


def test_providers_module_empty_exports() -> None:
    """Test that providers module currently has no exports (as expected)."""
    import uutel.providers

    # Currently the module should have empty __all__
    # since no providers are implemented yet
    assert uutel.providers.__all__ == []


def test_providers_module_structure() -> None:
    """Test that providers module has the expected structure."""
    import uutel.providers

    # Should be a valid module
    assert hasattr(uutel.providers, "__name__")
    assert hasattr(uutel.providers, "__file__")
    assert "providers" in uutel.providers.__name__


def test_providers_module_docstring() -> None:
    """Test that providers module has proper documentation."""
    import uutel.providers

    assert hasattr(uutel.providers, "__doc__")
    assert uutel.providers.__doc__ is not None
    assert "providers" in uutel.providers.__doc__.lower()
