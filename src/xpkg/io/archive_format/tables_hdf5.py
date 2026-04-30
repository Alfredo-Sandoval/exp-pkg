# pyright: reportMissingImports=false
# Justification: h5py lacks packaged type stubs in this environment.

"""Precision-preserving DataFrame tables in arbitrary HDF5 groups."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

_STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def decode_hdf5_string(value: bytes | str) -> str:
    """Normalize HDF5 string values, which may round-trip as bytes or str."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _column_array(series: pd.Series, *, name: str) -> tuple[np.ndarray, np.dtype | None]:
    if series.isna().any():
        raise ValueError(
            f"Cannot write table column {name!r}: missing values are not representable "
            "in the HDF5 table format. "
            "Fill or drop missing values explicitly before persistence."
        )
    if pd.api.types.is_bool_dtype(series):
        return np.asarray(series, dtype=bool), None
    if pd.api.types.is_numeric_dtype(series):
        if pd.api.types.is_integer_dtype(series):
            return np.asarray(series, dtype=int), None
        arr = series.to_numpy()
        if arr.dtype == np.float32:
            return arr, None
        return np.asarray(series, dtype=float), None
    values = series.astype(str).to_numpy()
    return values, _STRING_DTYPE


def write_hdf5_table_group(handle: h5py.File, group_name: str, df: pd.DataFrame) -> None:
    """Write a DataFrame into an open HDF5 file at the requested group path."""
    if group_name in handle:
        del handle[group_name]
    group = handle.create_group(group_name)
    group.attrs["columns"] = json.dumps([str(col) for col in df.columns])

    for col in df.columns:
        values, dtype_override = _column_array(df[col], name=str(col))
        kwargs: dict[str, object] = {}
        if dtype_override is not None:
            kwargs["dtype"] = dtype_override
        if values.dtype.kind not in {"U", "S", "O"} and values.size:
            kwargs["compression"] = "lzf"
        group.create_dataset(str(col), data=values, **kwargs)


def read_hdf5_table_group(handle: h5py.File, group_name: str) -> pd.DataFrame:
    """Read a DataFrame from an open HDF5 file group."""
    group = handle[group_name]
    columns_attr = group.attrs.get("columns")
    if columns_attr is None:
        raise KeyError(
            f"Table group {group_name!r} is missing required 'columns' metadata; "
            "cannot determine column order."
        )
    columns = json.loads(decode_hdf5_string(columns_attr))

    data: dict[str, Any] = {}
    for col in columns:
        values = group[col][()]
        if isinstance(values, bytes):
            data[col] = values.decode("utf-8")
            continue
        if hasattr(values, "dtype"):
            if values.dtype.kind == "S":
                data[col] = values.astype(str)
                continue
            if values.dtype.kind == "O" and len(values) > 0 and isinstance(values.flat[0], bytes):
                data[col] = np.array(
                    [v.decode("utf-8") if isinstance(v, bytes) else v for v in values]
                )
                continue
        data[col] = values
    return pd.DataFrame(data)


def write_hdf5_table(*, path: str | Path, group_name: str, df: pd.DataFrame) -> Path:
    """Write a DataFrame into an HDF5 file at the requested table group."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(resolved, "a") as handle:
        write_hdf5_table_group(handle, group_name, df)
    return resolved


def read_hdf5_table(*, path: str | Path, group_name: str) -> pd.DataFrame:
    """Read a DataFrame from an HDF5 file table group."""
    with h5py.File(Path(path), "r") as handle:
        return read_hdf5_table_group(handle, group_name)


__all__ = [
    "decode_hdf5_string",
    "read_hdf5_table",
    "read_hdf5_table_group",
    "write_hdf5_table",
    "write_hdf5_table_group",
]
