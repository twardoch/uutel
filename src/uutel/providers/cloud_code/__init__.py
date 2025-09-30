# this_file: src/uutel/providers/cloud_code/__init__.py
"""Google Cloud Code provider for UUTEL.

This module implements the CloudCodeUU provider for integrating with Google Cloud Code
via OAuth or API key authentication.
"""

from __future__ import annotations

from uutel.providers.cloud_code.provider import CloudCodeUU

__all__: list[str] = ["CloudCodeUU"]
