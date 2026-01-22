# pylint: disable=unused-variable
"""
Tests for logger configuration utilities.

This module tests the logging configuration functions.
"""

import logging
from src.utils.logger_config import configure_logging, setup_logger


class TestConfigureLogging:
    """Test configure_logging function."""

    def test_configure_logging_default(self):
        """Test configuring logging with default level."""
        configure_logging()

        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

    def test_configure_logging_info(self):
        """Test configuring logging with INFO level."""
        configure_logging("INFO")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_configure_logging_debug(self):
        """Test configuring logging with DEBUG level."""
        configure_logging("DEBUG")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_configure_logging_error(self):
        """Test configuring logging with ERROR level."""
        configure_logging("ERROR")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.ERROR

    def test_configure_logging_invalid_level(self):
        """Test configuring logging with invalid level (should default to WARNING)."""
        configure_logging("INVALID_LEVEL")

        root_logger = logging.getLogger()
        # Should default to WARNING
        assert root_logger.level == logging.WARNING

    def test_configure_logging_multiple_calls(self):
        """Test that multiple calls update the level."""
        configure_logging("INFO")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

        configure_logging("DEBUG")
        assert root_logger.level == logging.DEBUG


class TestSetupLogger:
    """Test setup_logger function."""

    def test_setup_logger_creates_logger(self):
        """Test that setup_logger creates a logger."""
        logger = setup_logger("test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_setup_logger_propagates(self):
        """Test that logger propagates to root logger."""
        logger = setup_logger("test_module")

        assert logger.propagate is True

    def test_setup_logger_auto_configures(self):
        """Test that setup_logger auto-configures if root logger not configured."""
        # Clear any existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        logger = setup_logger("test_module")

        # Should have configured root logger
        assert len(root_logger.handlers) > 0

    def test_setup_logger_different_names(self):
        """Test creating loggers with different names."""
        logger1 = setup_logger("module1")
        logger2 = setup_logger("module2")

        assert logger1.name == "module1"
        assert logger2.name == "module2"
        assert logger1 is not logger2

    def test_setup_logger_same_name_returns_same_logger(self):
        """Test that same name returns the same logger instance."""
        logger1 = setup_logger("test_module")
        logger2 = setup_logger("test_module")

        assert logger1 is logger2

