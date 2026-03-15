"""Progress callback helpers for converter adapters."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from posetta.io.converters.converter_helpers import ProgressCallback

PercentProgressCallback = Callable[[int, str], None]


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


__all__ = ["PercentProgressCallback", "bridge_progress_callback"]
