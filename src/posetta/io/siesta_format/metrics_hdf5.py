# pyright: reportMissingImports=false
# Justification: h5py lacks bundled type stubs in this environment.

"""
Utilities for managing metrics tables stored inside `.sta` archives.

This module centralizes HDF5 interactions for the `/metrics` group and provides a
consistent, compressed storage layout for tabular metrics. Consumers interact with
simple Pandas DataFrames while this layer handles chunk sizing, compression, and the
separation of numeric payloads from string metadata columns.

Chunk heuristic
---------------
Datasets default to `compression="gzip"` with `compression_opts=4`. Chunk rows are
derived from the table size using a square-root heuristic: we pick the largest power
of two not exceeding `sqrt(n_rows)` but always clamp between 64 and 4096. This keeps
chunk sizes large enough for sequential scans while avoiding excessive memory use on
small tables.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd

from posetta.config.loaders import load_json_config
from posetta.core.logging_utils import get_logger

logger = get_logger(__name__)

_METRICS_GROUP = "metrics"
_VALUES_DATASET = "values"
_NUMERIC_COLUMNS_DATASET = "columns"
_INDEX_DATASET = "index"
_STRINGS_GROUP = "strings"
_STRING_COLUMNS_DATASET = "string_columns"
_COLUMN_ORDER_DATASET = "column_order"
_DEFAULT_COMPRESSION = "gzip"

_METRICS_DEFAULTS = load_json_config("defaults/app_defaults.json")["metrics"]
_DEFAULT_COMPRESSION_OPTS = int(_METRICS_DEFAULTS["hdf5_compression_opts"])


class MetricsError(Exception):
    """Base class for metrics storage errors."""


class MissingMetricsGroupError(MetricsError):
    """Raised when the `/metrics` group is absent."""

    def __init__(self, bundle_path: Path, group: str = _METRICS_GROUP) -> None:
        super().__init__(f"{bundle_path} does not contain /{group}")
        self.bundle_path = bundle_path
        self.group = group


class MissingMetricsTableError(MetricsError):
    """Raised when a requested metrics table is missing."""

    def __init__(self, bundle_path: Path, table: str) -> None:
        super().__init__(f"{bundle_path} does not contain metrics table '{table}'")
        self.bundle_path = bundle_path
        self.table = table


class MetricsWriteError(MetricsError):
    """Raised when writing a metrics table fails."""


class MetricsReadError(MetricsError):
    """Raised when reading a metrics table fails."""


@dataclass(frozen=True)
class _ColumnPartition:
    numeric_names: list[str]
    numeric_values: np.ndarray
    string_columns: dict[str, np.ndarray]


def _require_group(parent: h5py.Group | h5py.File, name: str) -> h5py.Group:
    node = parent.get(name)
    if node is None:
        raise MetricsReadError(f"Metrics group missing '{name}' group")
    if not isinstance(node, h5py.Group):
        raise MetricsReadError(f"Metrics node '{name}' is not a group")
    return node


def _require_dataset(parent: h5py.Group | h5py.File, name: str) -> h5py.Dataset:
    node = parent.get(name)
    if node is None:
        raise MetricsReadError(f"Missing required dataset '{name}'")
    if not isinstance(node, h5py.Dataset):
        raise MetricsReadError(f"HDF5 node '{name}' is not a dataset")
    return node


def _to_numeric_series(
    series: pd.Series, *, error_cls: type[MetricsError], label: str
) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    if not isinstance(converted, pd.Series):
        raise error_cls(f"{label} numeric conversion did not return a Series")
    return converted


def _series_to_string_array(series: pd.Series) -> np.ndarray:
    string_dtype = pd.StringDtype("python")
    return series.astype(string_dtype).fillna("").to_numpy()


def _coerce_text(value: Any) -> str:
    """Convert arbitrary HDF5 scalar values into UTF-8 text.

    Args:
        value: The value to convert.

    Returns:
        The converted string.
    """

    if isinstance(value, bytes | bytearray):
        return value.decode("utf-8")
    return str(value)


def ensure_group(bundle_path: str | Path) -> None:
    """Guarantee that a `.sta` archive exposes a `/metrics` group.

    Args:
        bundle_path: Path to the .sta archive.
    """
    path = Path(bundle_path)
    with h5py.File(str(path), "a") as handle:
        _ensure_metrics_group_handle(handle)


def _ensure_metrics_group_handle(handle: h5py.File) -> h5py.Group:
    """Guarantee and return the `/metrics` group on an open HDF5 handle."""
    group = handle.require_group(_METRICS_GROUP)
    current_version = int(group.attrs.get("schema_version", 0) or 0)
    if current_version < 2:
        group.attrs["schema_version"] = 2
    return group


def has_metrics_group(bundle_path: str | Path) -> bool:
    """Return True when the archive exposes a `/metrics` group.

    Args:
        bundle_path: Path to the .sta archive.

    Returns:
        True if the group exists, False otherwise.
    """
    path = Path(bundle_path)
    if not path.exists():
        return False
    with h5py.File(str(path), "r") as handle:
        metrics_group = handle.get(_METRICS_GROUP)
        return metrics_group is not None


def has_metrics_table(bundle_path: str | Path, name: str) -> bool:
    """Return True when the archive exposes a given metrics table.

    Args:
        bundle_path: Path to the .sta archive.
        name: Name of the table to check.

    Returns:
        True if the table exists, False otherwise.
    """
    path = Path(bundle_path)
    if not path.exists():
        return False
    with h5py.File(str(path), "r") as handle:
        metrics_group = handle.get(_METRICS_GROUP)
        return (
            metrics_group is not None
            and isinstance(metrics_group, h5py.Group)
            and name in metrics_group
        )


def read_table(bundle_path: str | Path, name: str) -> pd.DataFrame:
    """Load a metrics table from a `.sta` archive.

    Args:
        bundle_path: Path to the .sta archive.
        name: Name of the table to read.

    Returns:
        A pandas DataFrame containing the metrics.

    Raises:
        MissingMetricsGroupError: If the /metrics group is missing.
        MissingMetricsTableError: If the requested table is missing.
        MetricsReadError: If the table or group is malformed.
    """
    path = Path(bundle_path)
    with h5py.File(str(path), "r") as handle:
        metrics_group = handle.get(_METRICS_GROUP)
        if metrics_group is None:
            logger.debug("Archive '%s' does not expose /%s", path, _METRICS_GROUP)
            raise MissingMetricsGroupError(path)
        if not isinstance(metrics_group, h5py.Group):
            raise MetricsReadError(f"Node '/{_METRICS_GROUP}' is not a group")
        if name not in metrics_group:
            logger.debug("Archive '%s' missing metrics table '%s'", path, name)
            raise MissingMetricsTableError(path, name)
        table_group = metrics_group[name]
        if not isinstance(table_group, h5py.Group):
            raise MetricsReadError(f"Table '{name}' is not a group")
        return _group_to_dataframe(table_group)


def write_table_to_handle(
    handle: h5py.File,
    name: str,
    dataframe: pd.DataFrame,
    *,
    mode: Literal["append", "replace"] = "append",
) -> None:
    """Persist a metrics DataFrame into an already-open HDF5 file handle.

    Args:
        handle: Open `.sta` HDF5 handle.
        name: Name of the table to write under `/metrics`.
        dataframe: The pandas DataFrame to persist.
        mode: Either 'append' or 'replace'.

    Raises:
        ValueError: If mode is invalid.
        MetricsWriteError: If an existing table node is malformed.
    """
    if mode not in {"append", "replace"}:
        raise ValueError("mode must be either 'append' or 'replace'")

    df = dataframe.copy()
    metrics_group = _ensure_metrics_group_handle(handle)

    if mode == "append" and name in metrics_group:
        existing_node = metrics_group[name]
        if not isinstance(existing_node, h5py.Group):
            raise MetricsWriteError(f"Table '{name}' is not a group")
        existing = _group_to_dataframe(existing_node)
        if not existing.empty:
            df = pd.concat([existing, df], ignore_index=True)

    partition = _partition_columns(df)

    if name in metrics_group:
        del metrics_group[name]
    table_group = metrics_group.create_group(name)
    _dataframe_to_group(table_group, df.index, partition)


def write_table(
    bundle_path: str | Path,
    name: str,
    dataframe: pd.DataFrame,
    *,
    mode: Literal["append", "replace"] = "append",
) -> None:
    """Persist a metrics DataFrame as `/metrics/<name>`.

    Args:
        bundle_path: Path to the .sta archive.
        name: Name of the table to write.
        dataframe: The pandas DataFrame to persist.
        mode: Either 'append' or 'replace'.

    Raises:
        ValueError: If mode is invalid.
    """
    path = Path(bundle_path)
    with h5py.File(str(path), "a") as handle:
        write_table_to_handle(handle, name, dataframe, mode=mode)


def _partition_columns(df: pd.DataFrame) -> _ColumnPartition:
    """Split a metrics frame into numeric payloads and string side tables.

    Args:
        df: The DataFrame to partition.

    Returns:
        A _ColumnPartition object containing the split data.
    """

    numeric_names: list[str] = []
    numeric_arrays: list[np.ndarray] = []
    string_columns: dict[str, np.ndarray] = {}

    if not df.columns.tolist():
        numeric_values = np.empty((len(df), 0), dtype=np.float32)
        return _ColumnPartition(numeric_names, numeric_values, string_columns)

    for name in df.columns:
        series = df[name]
        if pd.api.types.is_numeric_dtype(series):
            numeric_names.append(str(name))
            converted = _to_numeric_series(
                series,
                error_cls=MetricsWriteError,
                label=f"metrics column '{name}'",
            )
            numeric_arrays.append(converted.to_numpy(dtype=np.float32))
        else:
            strings = _series_to_string_array(series)
            string_columns[str(name)] = strings

    if numeric_arrays:
        numeric_values = np.column_stack(numeric_arrays)
    else:
        numeric_values = np.empty((len(df), 0), dtype=np.float32)

    return _ColumnPartition(numeric_names, numeric_values, string_columns)


def _compute_chunk_shape(n_rows: int, n_cols: int) -> tuple[int, int] | None:
    """Return an HDF5 chunk shape based on the square-root heuristic.

    Args:
        n_rows: Number of rows in the dataset.
        n_cols: Number of columns in the dataset.

    Returns:
        A tuple representing the chunk shape, or None if the table is empty.
    """

    if n_rows <= 0 or n_cols <= 0:
        return None
    approx = max(1, int(np.sqrt(n_rows)))
    chunk_rows = 1 << (approx - 1).bit_length()
    chunk_rows = min(n_rows, max(64, min(chunk_rows, 4096)))
    chunk_cols = max(1, min(n_cols, 256))
    return (chunk_rows, chunk_cols)


def _dataframe_to_group(
    group: h5py.Group,
    index: pd.Index,
    partition: _ColumnPartition,
) -> None:
    """Materialize a partitioned DataFrame into an HDF5 group layout."""

    group.attrs["table_format"] = "dataframe:v2"
    group.attrs["row_count"] = len(index)
    group.attrs["column_count"] = len(partition.numeric_names) + len(partition.string_columns)

    str_dtype = h5py.string_dtype(encoding="utf-8")

    column_order = list(partition.numeric_names) + list(partition.string_columns.keys())
    group.create_dataset(
        _COLUMN_ORDER_DATASET,
        data=np.asarray(column_order, dtype=object),
        dtype=str_dtype,
    )

    numeric_chunk = _compute_chunk_shape(
        partition.numeric_values.shape[0], partition.numeric_values.shape[1]
    )
    group.create_dataset(
        _VALUES_DATASET,
        data=partition.numeric_values,
        dtype=np.float32,
        compression=_DEFAULT_COMPRESSION,
        compression_opts=_DEFAULT_COMPRESSION_OPTS,
        shuffle=True,
        chunks=numeric_chunk,
    )
    group.create_dataset(
        _NUMERIC_COLUMNS_DATASET,
        data=np.asarray(partition.numeric_names, dtype=object),
        dtype=str_dtype,
    )

    index_values = index.astype("string[python]").fillna("").to_numpy()
    group.create_dataset(_INDEX_DATASET, data=index_values, dtype=str_dtype)

    if partition.string_columns:
        string_group = group.create_group(_STRINGS_GROUP)
        string_names = []
        for name, values in partition.string_columns.items():
            string_names.append(name)
            chunk_len = _compute_string_chunk(len(values))
            string_group.create_dataset(
                name,
                data=np.asarray(values, dtype=object),
                dtype=str_dtype,
                compression=_DEFAULT_COMPRESSION,
                compression_opts=_DEFAULT_COMPRESSION_OPTS,
                shuffle=True,
                chunks=chunk_len,
            )
        group.create_dataset(
            _STRING_COLUMNS_DATASET,
            data=np.asarray(string_names, dtype=object),
            dtype=str_dtype,
        )


def _compute_string_chunk(length: int) -> tuple[int] | None:
    """Return a 1D chunk length for string datasets given ``length`` rows.

    Args:
        length: Number of rows in the string dataset.

    Returns:
        A tuple containing the chunk length, or None if length is 0.
    """

    if length <= 0:
        return None
    approx = max(1, int(np.sqrt(length)))
    chunk = 1 << (approx - 1).bit_length()
    chunk = min(length, max(64, min(chunk, 4096)))
    return (chunk,)


def _series_is_integer(series: pd.Series) -> bool:
    """Return True when a numeric series is effectively integral-valued.

    Args:
        series: The pandas Series to check.

    Returns:
        True if all finite values are integers, False otherwise.
    """
    if series.empty:
        return False
    values = series.to_numpy(dtype=np.float32, copy=False)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return False
    fractional = np.modf(values[finite_mask])[0]
    return bool(np.all(np.isclose(fractional, 0.0, atol=1e-6)))


def _group_to_dataframe(group: h5py.Group) -> pd.DataFrame:
    """Reconstruct a metrics DataFrame from an HDF5 group layout.

    Args:
        group: The HDF5 group containing the table data.

    Returns:
        A pandas DataFrame reconstructed from the group.

    Raises:
        MetricsReadError: If core datasets are missing.
    """
    if _VALUES_DATASET not in group or _NUMERIC_COLUMNS_DATASET not in group:
        raise MetricsReadError("Metrics group is missing core datasets")

    numeric_values = _require_dataset(group, _VALUES_DATASET)[...]
    numeric_names = [
        _coerce_text(v) for v in _require_dataset(group, _NUMERIC_COLUMNS_DATASET)[...]
    ]
    if numeric_values.ndim == 1:
        numeric_values = numeric_values.reshape(-1, 1)
    numeric_df = pd.DataFrame(numeric_values, columns=numeric_names)

    index_values = _require_dataset(group, _INDEX_DATASET)[...] if _INDEX_DATASET in group else None
    if index_values is not None:
        index_series = pd.Index([_coerce_text(v) for v in index_values])
    else:
        index_series = pd.RangeIndex(len(numeric_df))
    numeric_df.index = index_series

    for name in numeric_df.columns:
        series = _to_numeric_series(
            numeric_df[name],
            error_cls=MetricsReadError,
            label=f"metrics column '{name}'",
        )
        if _series_is_integer(series):
            numeric_df[name] = series.astype("Int64")
        else:
            numeric_df[name] = series.astype(np.float32)

    string_columns = {}
    if _STRING_COLUMNS_DATASET in group:
        string_names = [
            _coerce_text(v) for v in _require_dataset(group, _STRING_COLUMNS_DATASET)[...]
        ]
        strings_group = _require_group(group, _STRINGS_GROUP)
        for name in string_names:
            if name not in strings_group:
                logger.warning(
                    "String column '%s' missing in metrics group '%s'",
                    name,
                    group.name,
                )
                continue
            values = _require_dataset(strings_group, name)[...]
            string_columns[name] = [_coerce_text(v) for v in values]
    elif "split" in group:
        values = _require_dataset(group, "split")[...]
        string_columns["set"] = [_coerce_text(v) for v in values]

    column_order = None
    if _COLUMN_ORDER_DATASET in group:
        column_order = [
            _coerce_text(v) for v in _require_dataset(group, _COLUMN_ORDER_DATASET)[...]
        ]

    df = numeric_df.copy()
    if column_order is None:
        column_order = list(numeric_df.columns) + list(string_columns.keys())

    for name in column_order:
        if name in numeric_df.columns:
            df[name] = numeric_df[name]
        elif name in string_columns:
            series = pd.Series(string_columns[name], index=index_series)
            df[name] = series.astype(pd.StringDtype("python"))
        else:
            logger.debug("Skipping missing column '%s' while materializing metrics", name)

    return df.reset_index(drop=True)


def iter_tables(bundle_path: str | Path) -> Iterable[str]:
    """Yield table names under `/metrics`.

    Args:
        bundle_path: Path to the .sta archive.

    Yields:
        The names of the metrics tables found in the archive.

    Raises:
        MissingMetricsGroupError: If the /metrics group is missing.
    """
    path = Path(bundle_path)
    with h5py.File(str(path), "r") as handle:
        metrics_group = handle.get(_METRICS_GROUP)
        if metrics_group is None:
            raise MissingMetricsGroupError(path)
        if not isinstance(metrics_group, h5py.Group):
            raise MetricsReadError(f"Node '/{_METRICS_GROUP}' is not a group")
        yield from metrics_group.keys()


class MetricStore:
    """Read and write typed metrics tables from `.sta` archives."""

    @staticmethod
    def read_prediction_errors(bundle_path: Path) -> pd.DataFrame | None:
        """Read and normalize `/metrics/prediction_errors` from the archive."""
        if not bundle_path.exists():
            raise FileNotFoundError(f"Archive not found: {bundle_path}")
        df = read_table(bundle_path, "prediction_errors")
        return MetricStore._normalize_prediction_errors(df)

    @staticmethod
    def read_pixel_error(bundle_path: Path) -> pd.DataFrame | None:
        """Read and normalize `/metrics/pixel_error` from the archive."""
        if not bundle_path.exists():
            raise FileNotFoundError(f"Archive not found: {bundle_path}")
        if not has_metrics_table(bundle_path, "pixel_error"):
            return None
        df = read_table(bundle_path, "pixel_error")
        return MetricStore._normalize_pixel_error(df)

    @staticmethod
    def write_prediction_errors(
        bundle_path: Path,
        df: pd.DataFrame,
        *,
        mode: Literal["append", "replace"] = "append",
    ) -> None:
        """Persist `/metrics/prediction_errors`."""
        write_table(bundle_path, "prediction_errors", df, mode=mode)

    @staticmethod
    def read_training_metrics(bundle_path: Path) -> pd.DataFrame | None:
        """Read epoch-style training metrics from archive tables."""
        if not has_metrics_group(bundle_path):
            return None

        table_names = list(iter_tables(bundle_path))
        preferred = ("training", "training_metrics", "trainer", "metrics")
        df: pd.DataFrame | None = None

        for name in preferred:
            if name in table_names:
                candidate = read_table(bundle_path, name)
                if MetricStore._table_has_epoch_v2(candidate):
                    df = candidate
                    break

        if df is None:
            for name in table_names:
                if name in preferred:
                    continue
                candidate = read_table(bundle_path, name)
                if MetricStore._table_has_epoch_v2(candidate):
                    df = candidate
                    break

        if df is None or df.empty:
            return None

        epoch_col = MetricStore._match_column(df.columns, ("epoch",))
        if not epoch_col:
            return None
        train_col = MetricStore._match_column(
            df.columns, ("train", "loss")
        ) or MetricStore._match_column(df.columns, ("loss",))
        val_col = MetricStore._match_column(
            df.columns, ("val", "loss")
        ) or MetricStore._match_column(df.columns, ("validation", "loss"))

        rows: list[dict[str, float | int]] = []
        for _, record in df.iterrows():
            epoch_val = MetricStore._coerce_int(record[epoch_col])
            row: dict[str, float | int] = {"epoch": epoch_val}
            if train_col:
                row["train_loss"] = MetricStore._coerce_float(record[train_col])
            if val_col:
                row["val_loss"] = MetricStore._coerce_float(record[val_col])
            rows.append(row)

        if not rows:
            return None

        return pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)

    @staticmethod
    def _normalize_prediction_errors(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None or df.empty:
            return None
        working = df.copy()
        if "bodypart" in working.columns and "keypoint" not in working.columns:
            working = working.rename(columns={"bodypart": "keypoint"})
        if "error" in working.columns and "error_px" not in working.columns:
            working = working.rename(columns={"error": "error_px"})
        expected = {"keypoint", "error_px"}
        if not expected.issubset(set(working.columns)):
            return None
        ordered = ["keypoint", "frame_idx", "error_px", "set"]
        final_cols = [col for col in ordered if col in working.columns]
        for name in working.columns:
            if name not in final_cols:
                final_cols.append(name)
        return working[final_cols]

    @staticmethod
    def _normalize_pixel_error(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None or df.empty:
            return None
        working = df.copy()
        working.columns = [str(col) for col in working.columns]
        if "set" in working.columns:
            split_labels = working["set"].astype(str)
            working = working.drop(columns=["set"])
        else:
            split_labels = None
        if "error_px" in working.columns or "error" in working.columns:
            if split_labels is not None:
                working["set"] = split_labels
            return MetricStore._normalize_prediction_errors(working)
        stacked = working.stack().reset_index()
        stacked.columns = ["frame_label", "keypoint", "error_px"]
        stacked["error_px"] = pd.to_numeric(stacked["error_px"], errors="coerce")
        stacked["frame_idx"] = stacked["frame_label"].apply(MetricStore._coerce_frame_index)
        stacked = stacked.drop(columns=["frame_label"])
        stacked = stacked.dropna(subset=["error_px"]).reset_index(drop=True)
        if split_labels is not None:
            repeats = np.repeat(split_labels.to_numpy(), working.shape[1])
            stacked["set"] = repeats
        return MetricStore._normalize_prediction_errors(stacked)

    @staticmethod
    def _table_has_epoch_v2(df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            return False
        columns = [str(col).strip().lower() for col in df.columns]
        return any("epoch" in col for col in columns)

    @staticmethod
    def _match_column(columns: Iterable[str], keywords: tuple[str, ...]) -> str | None:
        for col in columns:
            lowered = str(col).strip().lower()
            if all(keyword in lowered for keyword in keywords):
                return str(col)
        return None

    @staticmethod
    def _coerce_float(value: Any) -> float:
        if value is None or isinstance(value, bool | np.bool_):
            raise TypeError(f"Expected numeric value, got {value!r}")
        val = float(value)
        if not math.isfinite(val):
            raise ValueError(f"Expected finite float, got {value!r}")
        return val

    @staticmethod
    def _coerce_int(value: Any) -> int:
        val = MetricStore._coerce_float(value)
        return round(val)

    @staticmethod
    def _coerce_frame_index(value: Any) -> int:
        if value is None or isinstance(value, bool | np.bool_):
            raise TypeError(f"Expected frame index value, got {value!r}")
        if isinstance(value, int | np.integer):
            return int(value)
        text = str(value).strip()
        if not text:
            raise ValueError("Frame index value is empty")
        if text.isdigit():
            return int(text)
        if "#frame=" in text:
            suffix = text.split("#frame=", 1)[1]
            digits = "".join(ch for ch in suffix if ch.isdigit())
            if digits:
                return int(digits)
            raise ValueError(f"No digits found after #frame=: {value!r}")
        raise ValueError(f"Unable to parse frame index from {value!r}")


class LabelStore:
    """Read and write labeled-frame metadata tables from `.sta` archives."""

    @staticmethod
    def read_labels_metadata(bundle_path: Path) -> pd.DataFrame | None:
        if not bundle_path.exists():
            raise FileNotFoundError(f"Archive not found: {bundle_path}")
        return read_table(bundle_path, "labels_lp_images")

    @staticmethod
    def write_labels_metadata(bundle_path: Path, df: pd.DataFrame) -> None:
        write_table(bundle_path, "labels_lp_images", df, mode="replace")


__all__ = [
    "LabelStore",
    "MetricStore",
    "MetricsError",
    "MetricsReadError",
    "MetricsWriteError",
    "MissingMetricsGroupError",
    "MissingMetricsTableError",
    "ensure_group",
    "has_metrics_group",
    "has_metrics_table",
    "iter_tables",
    "read_table",
    "write_table",
    "write_table_to_handle",
]
