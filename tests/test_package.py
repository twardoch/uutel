"""Test suite for uutel."""


def test_version() -> None:
    """Verify package exposes version."""
    import uutel

    assert hasattr(uutel, "__version__")
    assert uutel.__version__
