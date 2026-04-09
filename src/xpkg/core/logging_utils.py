"""Small logging helpers for the extracted xpkg package."""

from __future__ import annotations

import logging
import os
import threading

_LOCK = threading.RLock()
_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def ensure_basic_config() -> None:
    global _CONFIGURED
    with _LOCK:
        if _CONFIGURED:
            return
        level_name = str(os.environ.get("POSETTA_LOG_LEVEL", "WARNING")).strip().upper()
        level = _LEVELS.get(level_name, logging.WARNING)
        logging.basicConfig(level=level, format=_FORMAT)
        _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    ensure_basic_config()
    logger = logging.getLogger(name if name else "xpkg")
    if not any(isinstance(handler, logging.NullHandler) for handler in logger.handlers):
        logger.addHandler(logging.NullHandler())
    return logger


__all__ = ["ensure_basic_config", "get_logger"]
