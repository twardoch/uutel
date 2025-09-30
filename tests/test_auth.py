# this_file: tests/test_auth.py
"""Tests for UUTEL authentication framework."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from uutel.core.auth import (
    AuthResult,
    BaseAuth,
    get_api_key_from_env,
    load_cli_credentials,
)
from uutel.core.runners import SubprocessResult


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


class TestAuthHelpers:
    """Tests for helper utilities in auth module."""

    def test_get_api_key_from_env_returns_first_non_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_api_key_from_env should pick the first populated variable."""

        monkeypatch.setenv("PRIMARY_KEY", "")
        monkeypatch.setenv("SECONDARY_KEY", "example")
        key = get_api_key_from_env(["PRIMARY_KEY", "SECONDARY_KEY"])
        assert key == "example"

    def test_load_cli_credentials_returns_payload(self, tmp_path: Path) -> None:
        """Credentials should load when file exists and contains keys."""

        auth_file = tmp_path / "auth.json"
        auth_file.write_text(
            json.dumps({"token": "abc", "account_id": "acc"}),
            encoding="utf-8",
        )
        located_path, payload = load_cli_credentials(
            provider="codex",
            candidate_paths=[auth_file],
            required_keys=["token", "account_id"],
        )
        assert located_path == auth_file
        assert payload["token"] == "abc"

    def test_load_cli_credentials_runs_refresh_when_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Refresh command should be invoked when credential file missing."""

        auth_file = tmp_path / "auth.json"

        def _runner(command: list[str], **_: object) -> SubprocessResult:
            auth_file.write_text(
                json.dumps({"token": "xyz", "account_id": "acc"}),
                encoding="utf-8",
            )
            return SubprocessResult(tuple(command), 0, "", "", 0.0)

        located_path, payload = load_cli_credentials(
            provider="codex",
            candidate_paths=[auth_file],
            required_keys=["token", "account_id"],
            refresh_command=["codex", "login"],
            runner=_runner,
        )

        assert located_path == auth_file
        assert payload["token"] == "xyz"

    def test_load_cli_credentials_refreshes_expired_token(self, tmp_path: Path) -> None:
        """Expired credentials should trigger refresh command and reload data."""

        auth_file = tmp_path / "oauth.json"
        expired_payload = {"token": "old", "expires_at": "2000-01-01T00:00:00"}
        fresh_payload = {"token": "new", "expires_at": "2999-01-01T00:00:00"}
        auth_file.write_text(json.dumps(expired_payload), encoding="utf-8")

        refresh_calls: list[list[str]] = []

        def _runner(command: list[str], **_: object) -> SubprocessResult:
            refresh_calls.append(command)
            auth_file.write_text(json.dumps(fresh_payload), encoding="utf-8")
            return SubprocessResult(tuple(command), 0, "", "", 0.0)

        located_path, payload = load_cli_credentials(
            provider="gemini",
            candidate_paths=[auth_file],
            required_keys=["token", "expires_at"],
            refresh_command=["gemini", "login"],
            runner=_runner,
        )

        assert refresh_calls == [["gemini", "login"]]
        assert located_path == auth_file
        assert payload["token"] == "new"

    def test_load_cli_credentials_raises_when_missing_keys(
        self, tmp_path: Path
    ) -> None:
        """Missing required keys should raise a descriptive error."""

        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({"token": "abc"}), encoding="utf-8")

        with pytest.raises(Exception) as excinfo:
            load_cli_credentials(
                provider="codex",
                candidate_paths=[auth_file],
                required_keys=["token", "account_id"],
            )

        assert "account_id" in str(excinfo.value)
