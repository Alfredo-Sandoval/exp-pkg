"""Progress-callback adapters shared by converter entry points."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

from xpkg._core.logging_utils import get_logger

ProgressCallback = Callable[[str], None]
PercentProgressCallback = Callable[[int, str], None]

_LOGGER = get_logger(__name__)


def emit_progress(callback: ProgressCallback | None, message: str) -> None:
    """Emit a progress message via callback, logger, or stdout."""
    if callback is not None:
        callback(message)
        return
    if _LOGGER.hasHandlers():
        _LOGGER.info(message)
        return
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def bridge_progress_callback(
    callback: PercentProgressCallback | None,
    markers: Sequence[tuple[str, int]],
) -> ProgressCallback | None:
    """Translate marker-bearing converter messages into percentage callbacks."""
    if callback is None:
        return None

    ordered_markers = tuple(markers)
    last_progress = 0

    def _emit(message: str) -> None:
        nonlocal last_progress
        for marker, progress in ordered_markers:
            if marker in message:
                last_progress = progress
                break
        callback(last_progress, message)

    return _emit


__all__ = [
    "PercentProgressCallback",
    "ProgressCallback",
    "bridge_progress_callback",
    "emit_progress",
]
