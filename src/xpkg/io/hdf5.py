"""Typed boundary helpers for HDF5 containers."""

from __future__ import annotations

from typing import SupportsFloat, cast

import h5py
import numpy as np


def require_dataset(container: h5py.Group, name: str) -> h5py.Dataset:
    """Return a named dataset or raise when the HDF5 object has another type."""
    value = container.get(name)
    if not isinstance(value, h5py.Dataset):
        raise TypeError(f"HDF5 object {name!r} must be a dataset")
    return value


def text_attribute(value: object, *, name: str) -> str:
    """Parse a scalar HDF5 string attribute into text."""
    scalar = _scalar_attribute(value, name=name)
    if isinstance(scalar, str):
        return scalar
    if isinstance(scalar, bytes | bytearray):
        return bytes(scalar).decode("utf-8")
    raise TypeError(f"HDF5 attribute {name!r} must contain text")


def float_attribute(value: object, *, name: str) -> float:
    """Parse a scalar HDF5 numeric attribute into a float."""
    scalar = _scalar_attribute(value, name=name)
    if isinstance(scalar, bool | str | bytes | bytearray):
        raise TypeError(f"HDF5 attribute {name!r} must contain a number")
    try:
        return float(cast(SupportsFloat, scalar))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"HDF5 attribute {name!r} must contain a number") from exc


def _scalar_attribute(value: object, *, name: str) -> object:
    array = np.asarray(value)
    if array.ndim != 0:
        raise TypeError(f"HDF5 attribute {name!r} must be scalar")
    return array.item()


__all__ = ["float_attribute", "require_dataset", "text_attribute"]
