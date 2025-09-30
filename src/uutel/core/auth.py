# this_file: src/uutel/core/auth.py
"""UUTEL authentication framework.

This module provides base classes and utilities for handling authentication
across different AI providers. Each provider should implement its own
authentication class extending BaseAuth.

Example usage:
    Creating a custom authentication class:
        from uutel.core.auth import BaseAuth, AuthResult
        from datetime import datetime, timedelta

        class MyProviderAuth(BaseAuth):
            def authenticate(self, **kwargs) -> AuthResult:
                api_key = kwargs.get("api_key")
                if not api_key:
                    return AuthResult(
                        success=False,
                        error="API key required"
                    )

                # Perform authentication logic
                return AuthResult(
                    success=True,
                    token=f"Bearer {api_key}",
                    expires_at=datetime.now() + timedelta(hours=1)
                )

            def get_headers(self) -> dict[str, str]:
                token = self.get_token()
                return {"Authorization": token} if token else {}

    Using authentication:
        auth = MyProviderAuth()
        result = auth.authenticate(api_key="your-api-key")

        if result.success:
            headers = auth.get_headers()
            # Use headers in HTTP requests
        else:
            print(f"Authentication failed: {result.error}")

    Checking token validity:
        if auth.is_authenticated():
            # Token is valid and not expired
            headers = auth.get_headers()
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from uutel.core.exceptions import UUTELError
from uutel.core.runners import SubprocessResult, run_subprocess


@dataclass
class AuthResult:
    """Result of an authentication attempt.

    Attributes:
        success: Whether authentication was successful
        token: The authentication token (if successful)
        expires_at: When the token expires (if applicable)
        error: Error message (if authentication failed)
    """

    success: bool
    token: str | None
    expires_at: datetime | None
    error: str | None


class BaseAuth:
    """Base class for all UUTEL authentication implementations.

    This class provides the foundation for implementing authentication
    for different AI providers. Each provider should extend this class
    and implement the required methods.

    Attributes:
        provider_name: Name of the provider (e.g., "claude-code", "gemini-cli")
        auth_type: Type of authentication (e.g., "oauth", "api-key", "service-account")
    """

    def __init__(self) -> None:
        """Initialize BaseAuth instance."""
        self.provider_name: str = "base"
        self.auth_type: str = "unknown"

    def authenticate(self, **kwargs: Any) -> AuthResult:
        """Perform authentication and return result.

        This method should be implemented by each provider to handle
        their specific authentication flow.

        Args:
            **kwargs: Authentication parameters specific to the provider

        Returns:
            AuthResult containing the authentication outcome

        Raises:
            NotImplementedError: This base method must be overridden
        """
        raise NotImplementedError(
            f"authenticate method must be implemented by {self.__class__.__name__}"
        )

    def get_headers(self) -> dict[str, str]:
        """Get HTTP headers for authenticated requests.

        This method should return the appropriate headers to include
        in API requests for this provider.

        Returns:
            Dictionary of HTTP headers

        Raises:
            NotImplementedError: This base method must be overridden
        """
        raise NotImplementedError(
            f"get_headers method must be implemented by {self.__class__.__name__}"
        )

    def refresh_token(self) -> AuthResult:
        """Refresh the authentication token if supported.

        This method should refresh the current authentication token
        if the provider supports token refresh.

        Returns:
            AuthResult containing the refresh outcome

        Raises:
            NotImplementedError: This base method must be overridden
        """
        raise NotImplementedError(
            f"refresh_token method must be implemented by {self.__class__.__name__}"
        )

    def is_valid(self) -> bool:
        """Check if the current authentication is valid.

        This method should check whether the current authentication
        is still valid and usable.

        Returns:
            True if authentication is valid, False otherwise

        Raises:
            NotImplementedError: This base method must be overridden
        """
        raise NotImplementedError(
            f"is_valid method must be implemented by {self.__class__.__name__}"
        )


def get_api_key_from_env(
    env_vars: Sequence[str], environ: Mapping[str, str] | None = None
) -> str | None:
    """Return the first non-empty API key from the specified environment variables."""

    lookup = environ or os.environ
    for env_var in env_vars:
        value = lookup.get(env_var)
        if value:
            return value
    return None


def _normalize_paths(paths: Sequence[Path | str]) -> list[Path]:
    return [Path(path).expanduser() for path in paths]


def _load_json_file(path: Path, provider: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:  # pragma: no cover - guarded by existence checks
        raise UUTELError(
            f"Credential file not found at {path}", provider=provider
        ) from exc
    except json.JSONDecodeError as exc:
        raise UUTELError(
            f"Credential file {path} contains invalid JSON: {exc}",
            provider=provider,
        ) from exc


def _extract_expiry_timestamp(payload: Mapping[str, Any]) -> datetime | None:
    """Attempt to extract an expiry timestamp from a credential payload."""

    expiry_candidates = (
        "expires_at",
        "expiry",
        "expiration",
        "token_expiry",
        "tokenExpiry",
    )

    for key in expiry_candidates:
        value = payload.get(key)
        if value is not None:
            parsed = _parse_expiry_value(value)
            if parsed is not None:
                return parsed

    tokens = payload.get("tokens")
    if isinstance(tokens, Mapping):
        for key in expiry_candidates:
            value = tokens.get(key)
            if value is not None:
                parsed = _parse_expiry_value(value)
                if parsed is not None:
                    return parsed
    return None


def _parse_expiry_value(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        normalised = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalised)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _credentials_are_expired(payload: Mapping[str, Any]) -> bool:
    expiry = _extract_expiry_timestamp(payload)
    if expiry is None:
        return False
    now = datetime.now(tz=timezone.utc)
    # Treat tokens expiring within 60 seconds as expired to trigger refresh early.
    return expiry <= now + timedelta(seconds=60)


def load_cli_credentials(
    *,
    provider: str,
    candidate_paths: Sequence[Path | str],
    required_keys: Sequence[str],
    refresh_command: Sequence[str] | None = None,
    runner: Callable[[Sequence[str]], SubprocessResult] = run_subprocess,
) -> tuple[Path, dict[str, Any]]:
    """Load CLI-managed credentials, running refresh command when necessary."""

    paths = _normalize_paths(candidate_paths)

    attempts = 2 if refresh_command else 1
    for attempt in range(attempts):
        credential_path = next((path for path in paths if path.exists()), None)

        if credential_path is None:
            if refresh_command and attempt == 0:
                runner(refresh_command, check=True)
                continue
            raise UUTELError(
                "Credential file not found. Run the provider's login command first.",
                provider=provider,
            )

        payload = _load_json_file(credential_path, provider)

        missing_keys = [key for key in required_keys if key not in payload]
        if missing_keys:
            raise UUTELError(
                f"Credential file {credential_path} missing required keys: {', '.join(missing_keys)}",
                provider=provider,
            )

        if _credentials_are_expired(payload):
            if refresh_command and attempt == 0:
                runner(refresh_command, check=True)
                continue
            raise UUTELError(
                "Credential token expired. Re-run the provider login command.",
                provider=provider,
            )

        return credential_path, payload

    raise UUTELError(
        "Unable to load refreshed credentials after running login command.",
        provider=provider,
    )
