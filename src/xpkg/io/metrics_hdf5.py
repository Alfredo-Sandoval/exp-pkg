"""HDF5 metrics table helpers owned by xpkg."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd

METRICS_GROUP = "metrics"


class MetricsError(RuntimeError):
    """Base class for xpkg metrics HDF5 errors."""


class MetricsReadError(MetricsError):
    """Raised when a metrics table cannot be read."""


class MissingMetricsGroupError(MetricsReadError):
    """Raised when a bundle has no metrics group."""

    def __init__(self, bundle_path: str | Path) -> None:
        self.bundle_path = Path(bundle_path)
        self.group = METRICS_GROUP
        super().__init__(f"Missing /{METRICS_GROUP} group in {self.bundle_path}")


class MissingMetricsTableError(MetricsReadError):
    """Raised when a named metrics table is missing."""

    def __init__(self, bundle_path: str | Path, table: str) -> None:
        self.bundle_path = Path(bundle_path)
        self.table = str(table)
        super().__init__(f"Missing metrics table {self.table!r} in {self.bundle_path}")


@dataclass(frozen=True)
class PartitionedColumns:
    numeric_names: list[str]
    numeric_values: np.ndarray
    string_columns: dict[str, np.ndarray]


def _compute_chunk_shape(row_count: int, column_count: int) -> tuple[int, int] | None:
    if row_count <= 0 or column_count <= 0:
        return None
    rows = min(max(64, min(int(row_count), 4096)), int(row_count))
    cols = min(int(column_count), 64)
    return (rows, cols)


def _compute_string_chunk(row_count: int) -> tuple[int] | None:
    if row_count <= 0:
        return None
    return (min(max(64, min(int(row_count), 4096)), int(row_count)),)


def _partition_columns(df: pd.DataFrame) -> PartitionedColumns:
    numeric_names: list[str] = []
    string_columns: dict[str, np.ndarray] = {}

    for raw_name in df.columns:
        name = str(raw_name)
        series = df[raw_name]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numeric_names.append(name)
            continue
        values = series.astype("string").fillna("").to_numpy(dtype=object)
        string_columns[name] = values.astype(str)

    if numeric_names:
        numeric_values = df[numeric_names].to_numpy(dtype=np.float64, copy=True)
    else:
        numeric_values = np.empty((len(df), 0), dtype=np.float64)

    return PartitionedColumns(
        numeric_names=numeric_names,
        numeric_values=numeric_values,
        string_columns=string_columns,
    )


def has_metrics_group(path: str | Path) -> bool:
    bundle_path = Path(path)
    if not bundle_path.is_file():
        return False
    with h5py.File(bundle_path, "r") as handle:
        return METRICS_GROUP in handle and isinstance(handle[METRICS_GROUP], h5py.Group)


def _require_metrics_group(handle: h5py.File, bundle_path: str | Path) -> h5py.Group:
    metrics_group = handle.get(METRICS_GROUP)
    if not isinstance(metrics_group, h5py.Group):
        raise MissingMetricsGroupError(bundle_path)
    return metrics_group


def iter_tables(path: str | Path) -> Iterator[str]:
    bundle_path = Path(path)
    with h5py.File(bundle_path, "r") as handle:
        metrics_group = _require_metrics_group(handle, bundle_path)
        for name in sorted(metrics_group.keys()):
            if isinstance(metrics_group[name], h5py.Group):
                yield str(name)


def _write_table_group(metrics_group: h5py.Group, table: str, df: pd.DataFrame) -> None:
    if table in metrics_group:
        del metrics_group[table]
    table_group = metrics_group.create_group(table)
    partitioned = _partition_columns(df)
    column_order = [str(column) for column in df.columns]

    table_group.attrs["column_order"] = json.dumps(column_order)
    table_group.attrs["numeric_names"] = json.dumps(partitioned.numeric_names)
    table_group.attrs["string_names"] = json.dumps(list(partitioned.string_columns))
    table_group.attrs["row_count"] = int(len(df))

    table_group.create_dataset(
        "numeric_values",
        data=partitioned.numeric_values,
        chunks=_compute_chunk_shape(*partitioned.numeric_values.shape),
    )

    strings_group = table_group.create_group("strings")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    for name, values in partitioned.string_columns.items():
        strings_group.create_dataset(
            name,
            data=values.astype(object),
            dtype=string_dtype,
            chunks=_compute_string_chunk(len(values)),
        )


def _read_table_group(table_group: h5py.Group) -> pd.DataFrame:
    column_order = json.loads(str(table_group.attrs.get("column_order", "[]")))
    numeric_names = json.loads(str(table_group.attrs.get("numeric_names", "[]")))
    string_names = json.loads(str(table_group.attrs.get("string_names", "[]")))

    data: dict[str, Any] = {}
    numeric_values = np.asarray(table_group["numeric_values"])
    for idx, name in enumerate(numeric_names):
        data[str(name)] = numeric_values[:, idx]

    if string_names:
        strings_group = table_group.get("strings")
        if not isinstance(strings_group, h5py.Group):
            raise MetricsReadError("Metrics table is missing its strings group")
        for name in string_names:
            raw_values = strings_group[str(name)].asstr()[()]
            data[str(name)] = np.asarray(raw_values, dtype=object)

    ordered = {str(name): data[str(name)] for name in column_order if str(name) in data}
    return pd.DataFrame(ordered)


def read_table(path: str | Path, table: str) -> pd.DataFrame:
    bundle_path = Path(path)
    with h5py.File(bundle_path, "r") as handle:
        metrics_group = _require_metrics_group(handle, bundle_path)
        table_group = metrics_group.get(table)
        if not isinstance(table_group, h5py.Group):
            raise MissingMetricsTableError(bundle_path, table)
        return _read_table_group(table_group)


def write_table_to_handle(
    handle: h5py.File,
    table: str,
    df: pd.DataFrame,
    *,
    mode: Literal["replace", "append"] = "replace",
) -> None:
    if mode not in {"replace", "append"}:
        raise ValueError("mode must be 'replace' or 'append'")

    metrics_group = handle.require_group(METRICS_GROUP)
    table_df = df.copy()
    if mode == "append" and table in metrics_group:
        existing_group = metrics_group[table]
        if not isinstance(existing_group, h5py.Group):
            raise MissingMetricsTableError(Path(handle.filename), table)
        table_df = pd.concat([_read_table_group(existing_group), table_df], ignore_index=True)

    _write_table_group(metrics_group, table, table_df)


def write_table(
    path: str | Path,
    table: str,
    df: pd.DataFrame,
    *,
    mode: Literal["replace", "append"] = "replace",
) -> None:
    if mode not in {"replace", "append"}:
        raise ValueError("mode must be 'replace' or 'append'")
    with h5py.File(Path(path), "a") as handle:
        write_table_to_handle(handle, table, df, mode=mode)


class MetricStore:
    """Small convenience wrapper around xpkg HDF5 metrics tables."""

    def __init__(self, bundle_path: str | Path) -> None:
        self.bundle_path = Path(bundle_path)

    def has_metrics_group(self) -> bool:
        return has_metrics_group(self.bundle_path)

    def iter_tables(self) -> Iterator[str]:
        return iter_tables(self.bundle_path)

    def read_table(self, table: str) -> pd.DataFrame:
        return read_table(self.bundle_path, table)

    def write_table(
        self,
        table: str,
        df: pd.DataFrame,
        *,
        mode: Literal["replace", "append"] = "replace",
    ) -> None:
        write_table(self.bundle_path, table, df, mode=mode)

    @staticmethod
    def _normalize_prediction_errors(df: pd.DataFrame) -> pd.DataFrame | None:
        if df.empty:
            return None
        normalized = df.copy()
        rename_map: dict[str, str] = {}
        if "bodypart" in normalized.columns and "keypoint" not in normalized.columns:
            rename_map["bodypart"] = "keypoint"
        if "error" in normalized.columns and "error_px" not in normalized.columns:
            rename_map["error"] = "error_px"
        if rename_map:
            normalized = normalized.rename(columns=rename_map)
        return normalized

    @staticmethod
    def _coerce_frame_index(value: Any) -> int:
        if isinstance(value, str):
            match = re.search(r"(?:^|#frame=)(\d+)$", value)
            if match is None:
                raise ValueError(f"Cannot coerce frame index from {value!r}")
            return int(match.group(1))
        return MetricStore._coerce_int(value)

    @staticmethod
    def _coerce_float(value: Any) -> float:
        if value is None:
            raise TypeError("Expected a finite float, got None")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"Expected a finite float, got {value!r}")
        return result

    @staticmethod
    def _coerce_int(value: Any) -> int:
        if value is None:
            raise TypeError("Expected an integer-compatible value, got None")
        return int(round(float(value)))

    @staticmethod
    def _table_has_epoch_v2(df: pd.DataFrame) -> bool:
        return "epoch" in {str(column) for column in df.columns}

    @staticmethod
    def _match_column(columns: Sequence[str], keywords: Sequence[str]) -> str | None:
        lowered_keywords = tuple(keyword.lower() for keyword in keywords)
        for column in columns:
            lowered_column = str(column).lower()
            if all(keyword in lowered_column for keyword in lowered_keywords):
                return str(column)
        return None


__all__ = [
    "MetricStore",
    "MetricsError",
    "MetricsReadError",
    "MissingMetricsGroupError",
    "MissingMetricsTableError",
    "_compute_chunk_shape",
    "_compute_string_chunk",
    "_partition_columns",
    "has_metrics_group",
    "iter_tables",
    "read_table",
    "write_table",
    "write_table_to_handle",
]
