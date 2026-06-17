"""Direct readers for source-neutral photometry and event CSV files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from xpkg.io.readers._columns import (
    column_by_name,
    first_matching_column,
    resolve_column,
)
from xpkg.model import (
    Event,
    EventTable,
    PhotometryChannel,
    PhotometryRecording,
    Timeline,
    TimeSeries,
)

TimeUnit = Literal["s", "sec", "second", "seconds", "ms", "millisecond", "milliseconds"]

_TIME_COLUMN_CANDIDATES = (
    "time",
    "timestamp",
    "timestamps",
    "event_time",
    "event",
    "events",
)
_EVENT_LABEL_CANDIDATES = (
    "label",
    "event_label",
    "name",
    "event_name",
    "description",
    "condition",
    "behavior",
    "behaviour",
    "eventname",
    "state",
)
_EVENT_KIND_CANDIDATES = ("kind", "type", "event_type", "category")
_EVENT_DURATION_CANDIDATES = ("duration", "duration_s", "endurance")


def _read_csv(path: str | Path, *, max_mb: float | None) -> tuple[pd.DataFrame, int]:
    source_path = Path(path)
    size_bytes = source_path.stat().st_size
    if max_mb is not None:
        max_bytes = int(float(max_mb) * 1024 * 1024)
        if max_bytes <= 0:
            raise ValueError(f"max_mb must be positive when provided, got {max_mb!r}.")
        if size_bytes > max_bytes:
            raise ValueError(f"CSV file '{source_path}' exceeds max load size ({max_mb} MB).")
    return pd.read_csv(source_path), size_bytes


def _numeric_column(frame: pd.DataFrame, column: str, *, role: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{role} column {column!r} must be one-dimensional.")
    if not np.isfinite(values).all():
        raise ValueError(f"{role} column {column!r} contains non-finite values.")
    return values


def _time_scale(unit: TimeUnit) -> float:
    normalized = unit.lower()
    if normalized in {"s", "sec", "second", "seconds"}:
        return 1.0
    if normalized in {"ms", "millisecond", "milliseconds"}:
        return 0.001
    raise ValueError(f"Unsupported time_unit {unit!r}; expected seconds or milliseconds.")


def _resolve_time_column(
    frame: pd.DataFrame,
    *,
    time_column: str | None,
    sample_rate_hz: float | None,
) -> str | None:
    if time_column is not None:
        return column_by_name(frame, time_column)
    matched = first_matching_column(frame, _TIME_COLUMN_CANDIDATES)
    if matched is not None:
        return matched
    if sample_rate_hz is None and frame.shape[1] > 1:
        return str(frame.columns[0])
    return None


def _resolve_signal_columns(
    frame: pd.DataFrame,
    *,
    signal_columns: Sequence[str] | None,
    time_column: str | None,
) -> tuple[str, ...]:
    if signal_columns is not None:
        return tuple(column_by_name(frame, column) for column in signal_columns)
    excluded = {time_column} if time_column is not None else set()
    columns = tuple(str(column) for column in frame.columns if str(column) not in excluded)
    if not columns:
        raise ValueError("No signal columns were found in the photometry CSV.")
    return columns


def _units_for_columns(
    columns: Sequence[str],
    units: Mapping[str, str] | Sequence[str] | None,
) -> tuple[str, ...]:
    if units is None:
        return ("",) * len(columns)
    if isinstance(units, Mapping):
        unit_mapping = {str(key): str(value).strip() for key, value in units.items()}
        return tuple(unit_mapping.get(column, "") for column in columns)
    units_tuple = tuple(str(unit).strip() for unit in units)
    if len(units_tuple) != len(columns):
        raise ValueError(f"units length {len(units_tuple)} does not match {len(columns)} columns.")
    return units_tuple


def read_photometry_csv(
    path: str | Path,
    *,
    time_column: str | None = None,
    signal_columns: Sequence[str] | None = None,
    signal_channel: str | None = None,
    reference_channel: str | None = None,
    sample_rate_hz: float | None = None,
    start_s: float = 0.0,
    time_unit: TimeUnit = "s",
    units: Mapping[str, str] | Sequence[str] | None = None,
    name: str = "photometry",
    max_mb: float | None = None,
) -> PhotometryRecording:
    """Read a photometry CSV into a source-neutral recording object."""

    frame, size_bytes = _read_csv(path, max_mb=max_mb)
    if frame.empty:
        raise ValueError(f"Photometry CSV '{path}' is empty.")
    resolved_time = _resolve_time_column(
        frame,
        time_column=time_column,
        sample_rate_hz=sample_rate_hz,
    )
    resolved_signals = _resolve_signal_columns(
        frame,
        signal_columns=signal_columns,
        time_column=resolved_time,
    )
    values = np.column_stack(
        [_numeric_column(frame, column, role="signal") for column in resolved_signals]
    )

    if resolved_time is not None:
        timestamps = _numeric_column(frame, resolved_time, role="time") * _time_scale(time_unit)
        timeline = Timeline(timestamps_s=timestamps)
    elif sample_rate_hz is not None:
        timeline = Timeline.from_sample_rate(
            n_samples=values.shape[0],
            sample_rate_hz=sample_rate_hz,
            start_s=start_s,
        )
    else:
        raise ValueError("Provide a time column or sample_rate_hz for photometry CSV data.")

    column_units = _units_for_columns(resolved_signals, units)
    channels = tuple(
        PhotometryChannel(name=column, unit=unit)
        for column, unit in zip(resolved_signals, column_units, strict=True)
    )
    series = TimeSeries(
        values=values,
        channels=channels,
        timeline=timeline,
        name=name,
        provenance={
            "source": {"type": "photometry_csv", "path": str(path), "size_bytes": size_bytes},
            "time_column": resolved_time,
            "signal_columns": list(resolved_signals),
        },
    )
    return PhotometryRecording(
        series=series,
        signal_channel=signal_channel or resolved_signals[0],
        reference_channel=reference_channel,
        metadata={"source_type": "photometry_csv"},
    )


def _optional_text(frame: pd.DataFrame, column: str | None, index: int) -> str | None:
    if column is None:
        return None
    value = frame[column].iloc[index]
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _optional_event_label(
    frame: pd.DataFrame,
    column: str | None,
    index: int,
) -> str | None:
    text = _optional_text(frame, column, index)
    if text is None or column is None:
        return text
    if pd.api.types.is_numeric_dtype(frame[column]):
        return f"{column}={text}"
    return text


def read_events_csv(
    path: str | Path,
    *,
    time_column: str | None = None,
    kind_column: str | None = None,
    label_column: str | None = None,
    duration_column: str | None = None,
    default_kind: str = "event",
    time_unit: TimeUnit = "s",
    max_mb: float | None = None,
) -> EventTable:
    """Read an event CSV into an ``EventTable``."""

    frame, _size_bytes = _read_csv(path, max_mb=max_mb)
    if frame.empty:
        return EventTable()
    resolved_time = resolve_column(frame, time_column, _TIME_COLUMN_CANDIDATES)
    if resolved_time is None:
        raise ValueError(
            "Event CSV must include a timestamp column named one of: "
            f"{', '.join(_TIME_COLUMN_CANDIDATES)}."
        )
    resolved_kind = resolve_column(frame, kind_column, _EVENT_KIND_CANDIDATES)
    resolved_label = resolve_column(frame, label_column, _EVENT_LABEL_CANDIDATES)
    resolved_duration = resolve_column(
        frame,
        duration_column,
        _EVENT_DURATION_CANDIDATES,
    )

    scale = _time_scale(time_unit)
    starts = _numeric_column(frame, resolved_time, role="event time") * scale
    durations = (
        _numeric_column(frame, resolved_duration, role="event duration") * scale
        if resolved_duration is not None
        else np.zeros_like(starts)
    )
    events = [
        Event(
            kind=_optional_text(frame, resolved_kind, index) or default_kind,
            start_s=float(start),
            duration_s=float(duration),
            label=_optional_event_label(frame, resolved_label, index),
            metadata={"source": {"type": "events_csv", "path": str(path)}},
        )
        for index, (start, duration) in enumerate(zip(starts, durations, strict=True))
    ]
    return EventTable.from_events(events)


__all__ = ["read_events_csv", "read_photometry_csv"]
