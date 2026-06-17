"""Filesystem discovery helpers shared by reader detectors."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def find_first_file(
    path: str | Path,
    detector: Callable[[Path], bool],
) -> Path | None:
    """Return the first file under ``path`` accepted by ``detector``."""

    root = Path(path)
    if root.is_file():
        return root if detector(root) else None
    if not root.is_dir():
        return None
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file() and detector(candidate):
            return candidate
    return None


__all__ = ["find_first_file"]
