# this_file: tests/test_auth.py
"""Tests for UUTEL authentication framework."""

from __future__ import annotations

import pytest

from uutel.core.auth import AuthResult, BaseAuth


class TestBaseAuth:
    """Test the BaseAuth base class."""

    def test_base_auth_initialization(self) -> None:
        """Test that BaseAuth can be initialized properly."""
        auth = BaseAuth()
        assert auth.provider_name == "base"
        assert auth.auth_type == "unknown"

    def test_base_auth_has_required_methods(self) -> None:
        """Test that BaseAuth has all required methods."""
        auth = BaseAuth()
        assert hasattr(auth, "authenticate")
        assert hasattr(auth, "get_headers")
        assert hasattr(auth, "refresh_token")
        assert hasattr(auth, "is_valid")
        assert callable(auth.authenticate)
        assert callable(auth.get_headers)
        assert callable(auth.refresh_token)
        assert callable(auth.is_valid)

    def test_base_auth_authenticate_not_implemented(self) -> None:
        """Test that BaseAuth authenticate method raises NotImplementedError."""
        auth = BaseAuth()
        with pytest.raises(NotImplementedError):
            auth.authenticate()

    def test_base_auth_get_headers_not_implemented(self) -> None:
        """Test that BaseAuth get_headers method raises NotImplementedError."""
        auth = BaseAuth()
        with pytest.raises(NotImplementedError):
            auth.get_headers()

    def test_base_auth_refresh_token_not_implemented(self) -> None:
        """Test that BaseAuth refresh_token method raises NotImplementedError."""
        auth = BaseAuth()
        with pytest.raises(NotImplementedError):
            auth.refresh_token()

    def test_base_auth_is_valid_not_implemented(self) -> None:
        """Test that BaseAuth is_valid method raises NotImplementedError."""
        auth = BaseAuth()
        with pytest.raises(NotImplementedError):
            auth.is_valid()


class TestAuthResult:
    """Test the AuthResult data class."""

    def test_auth_result_creation(self) -> None:
        """Test that AuthResult can be created with required fields."""
        result = AuthResult(
            success=True, token="test-token", expires_at=None, error=None
        )
        assert result.success is True
        assert result.token == "test-token"
        assert result.expires_at is None
        assert result.error is None

    def test_auth_result_failure(self) -> None:
        """Test that AuthResult can represent authentication failure."""
        result = AuthResult(
            success=False, token=None, expires_at=None, error="Authentication failed"
        )
        assert result.success is False
        assert result.token is None
        assert result.error == "Authentication failed"
