"""Centralized logging configuration."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

import rich.traceback
from rich.logging import RichHandler

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"
_LOG_DIR = Path("logs")

rich.traceback.install(show_locals=False)


def setup_logging(
    level: LogLevel = "INFO",
    log_file: str | None = "app.log",
) -> None:
    """Configure root logger with rich console output and rotating file handler.

    Idempotent: safe to call multiple times; re-applies the handler config.
    """
    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    root.addHandler(RichHandler(level=level, show_time=True, show_path=True, rich_tracebacks=True))

    if log_file is not None:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            _LOG_DIR / log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(fmt=_FILE_FORMAT, datefmt=_FILE_DATEFMT))
        file_handler.setLevel(level)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)
