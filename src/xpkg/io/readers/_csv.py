"""Canonical bounded CSV loading for external reader boundaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def csv_size_bytes(
    path: str | Path,
    *,
    max_mb: float | None,
    error_label: str = "CSV file",
) -> int:
    """Return file size after enforcing an optional positive load limit."""
    source = Path(path)
    size_bytes = source.stat().st_size
    if max_mb is None:
        return size_bytes
    max_bytes = int(float(max_mb) * 1024 * 1024)
    if max_bytes <= 0:
        raise ValueError(f"max_mb must be positive when provided, got {max_mb!r}.")
    if size_bytes > max_bytes:
        raise ValueError(f"{error_label} '{source}' exceeds max load size ({max_mb} MB).")
    return size_bytes


def read_csv_table(
    path: str | Path,
    *,
    max_mb: float | None = None,
    error_label: str = "CSV file",
) -> tuple[pd.DataFrame, int]:
    """Read one bounded CSV file and return its frame and source byte size."""
    source = Path(path)
    size_bytes = csv_size_bytes(source, max_mb=max_mb, error_label=error_label)
    return pd.read_csv(source), size_bytes


__all__ = ["csv_size_bytes", "read_csv_table"]
