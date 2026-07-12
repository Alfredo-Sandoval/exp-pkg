"""Canonical normalization helpers shared by external-format readers."""

from __future__ import annotations


def time_scale(unit: str) -> float:
    """Return the seconds multiplier for a declared seconds or milliseconds unit."""
    normalized = unit.lower()
    if normalized in {"s", "sec", "second", "seconds"}:
        return 1.0
    if normalized in {"ms", "millisecond", "milliseconds"}:
        return 0.001
    raise ValueError(f"Unsupported time_unit {unit!r}; expected seconds or milliseconds.")


def photometry_excitation(name: str) -> str:
    """Return the known excitation wavelength token embedded in ``name``."""
    lowered = name.lower()
    for token in ("405", "410", "415", "465", "470", "560"):
        if token in lowered:
            return token
    return ""


def normalize_file_type(file_type: str) -> str:
    """Return a non-empty lowercase file-type token."""
    normalized = str(file_type).strip().lower()
    if not normalized:
        raise ValueError("file_type must be a non-empty string.")
    return normalized


__all__ = ["normalize_file_type", "photometry_excitation", "time_scale"]
