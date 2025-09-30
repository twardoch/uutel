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

from dataclasses import dataclass
from datetime import datetime
from typing import Any


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
