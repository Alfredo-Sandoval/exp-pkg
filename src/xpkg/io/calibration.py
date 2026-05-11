"""JSON serialization helpers for xpkg calibration payloads."""

from __future__ import annotations

from pathlib import Path

from xpkg.model.calibration import Calibration

from .._core.json_utils import load_json_dict, write_json


def read_calibration_json(path: str | Path) -> Calibration:
    """Read a canonical xpkg ``Calibration.json`` file."""

    return Calibration.from_dict(load_json_dict(path))


def write_calibration_json(
    calibration: Calibration,
    path: str | Path,
    *,
    indent: int = 2,
) -> None:
    """Write a canonical xpkg ``Calibration.json`` file."""

    if not isinstance(calibration, Calibration):
        raise TypeError(f"calibration must be a Calibration, got {calibration!r}.")
    write_json(path, calibration.to_dict(), indent=indent, sort_keys=False)


__all__ = ["read_calibration_json", "write_calibration_json"]
