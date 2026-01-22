"""Logger configuration utility for consistent console and file logging across the project."""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


_CONFIGURED = False
_FILE_HANDLER_ADDED = False


def configure_logging(level: str = "WARNING", log_file: str = None) -> None:
    """
    Configure the root logger with console and optional file handlers.

    Args:
        level: Log level name (e.g., "DEBUG", "INFO", "WARNING", "ERROR").
        log_file: Optional path to log file. If provided, logs will be written to this file
                  in addition to console. Uses rotating file handler (max 10MB, 5 backups).
    """
    global _CONFIGURED, _FILE_HANDLER_ADDED
    root_logger = logging.getLogger()

    # Parse level string safely; default to WARNING if invalid
    level_value = getattr(logging, str(level).upper(), logging.WARNING)
    root_logger.setLevel(level_value)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add console handler if not already present
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level_value)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    else:
        # Update existing console handlers to reflect new level
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(level_value)

    # Add file handler if log_file is provided and not already added
    if log_file and not _FILE_HANDLER_ADDED:
        try:
            log_path = Path(log_file)
            # Create parent directory if it doesn't exist
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use rotating file handler to prevent huge log files
            # maxBytes=10MB, backupCount=5 means we keep current + 5 backups (60MB total)
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(level_value)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            _FILE_HANDLER_ADDED = True
            
            # Log that file logging is enabled
            root_logger.info("Logging to file: %s", log_path)
        except (OSError, PermissionError, IOError) as e:
            # If file logging fails, log error but don't crash
            root_logger.warning("Failed to set up file logging to %s: %s", log_file, e)
    elif log_file and _FILE_HANDLER_ADDED:
        # Update existing file handler level if already added
        for handler in root_logger.handlers:
            if isinstance(handler, RotatingFileHandler):
                handler.setLevel(level_value)

    _CONFIGURED = True


def setup_logger(name: str) -> logging.Logger:
    """
    Return a named logger that propagates to the root logger.
    If the root logger has not been configured yet, configure with default WARNING.
    """
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        configure_logging("WARNING")
    logger = logging.getLogger(name)
    logger.propagate = True
    return logger
