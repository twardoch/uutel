# this_file: src/uutel/core/logging_config.py
"""Simple logging configuration for UUTEL.

This module provides basic logging functionality focused on core needs only.
"""

from __future__ import annotations

# Standard library imports
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Get a basic logger for a specific module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Basic Python logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
