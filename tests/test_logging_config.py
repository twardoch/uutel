# this_file: tests/test_logging_config.py
"""Test suite for UUTEL logging configuration."""

from __future__ import annotations

import logging

from uutel.core.logging_config import get_logger


class TestBasicLogging:
    """Test basic logging functionality."""

    def test_get_logger(self) -> None:
        """Test get_logger returns a logger instance."""
        logger = get_logger("test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_get_logger_configured(self) -> None:
        """Test that logger is properly configured."""
        logger = get_logger("test_configured")

        # Logger should have at least one handler
        assert len(logger.handlers) >= 1

        # Should have appropriate log level
        assert logger.level <= logging.INFO

    def test_multiple_loggers(self) -> None:
        """Test getting multiple loggers with different names."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        assert logger1.name == "module1"
        assert logger2.name == "module2"
        assert logger1 is not logger2
