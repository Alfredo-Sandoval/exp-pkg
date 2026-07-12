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
from xpkg.io.readers._normalization import time_scale as _time_scale
from xpkg.model import (
    Event,
    EventTable,
    PhotometryChannel,
    PhotometryRecording,
    Timeline,
    TimeSeries,
)

TimeUnit = Literal["s", "sec", "second", "seconds", "ms", "millisecond", "milliseconds"]

_PHOTOMETRY_TIME_COLUMN_CANDIDATES = (
    "time",
    "timestamp",
    "timestamps",
)
_EVENT_TIME_COLUMN_CANDIDATES = (
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


def _positive_sample_rate(value: float) -> float:
    try:
        sample_rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"sample_rate_hz must be a positive finite number, got {value!r}."
        ) from exc
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError(f"sample_rate_hz must be a positive finite number, got {value!r}.")
    return sample_rate


def _clean_optional_column_name(value: object, *, role: str) -> str | None:
    if value is None:
        return None
    return _clean_required_column_name(value, role=role)


def _clean_required_column_name(value: object, *, role: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{role} must be a string.")
    if not value:
        raise ValueError(f"{role} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{role} must not contain surrounding whitespace.")
    return value


def _clean_column_names(value: object, *, role: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raise TypeError(f"{role} must be a sequence of strings, not a string.")
    if not isinstance(value, Sequence):
        raise TypeError(f"{role} must be a sequence of strings or None.")
    cleaned = tuple(
        _clean_required_column_name(item, role=f"{role}[{index}]")
        for index, item in enumerate(value)
    )
    if not cleaned:
        raise ValueError(f"{role} must be a non-empty sequence of strings.")
    return cleaned


def _uniform_timeline_sample_rate(
    timeline: Timeline,
    *,
    source: str,
    expected_sample_rate_hz: float | None = None,
) -> tuple[float, str]:
    expected = (
        _positive_sample_rate(expected_sample_rate_hz)
        if expected_sample_rate_hz is not None
        else None
    )
    sample_rate = timeline.estimated_sample_rate_hz
    if sample_rate is None:
        if timeline.n_samples < 2:
            if expected is not None:
                return expected, "sample_rate_hz.argument"
            raise ValueError(
                f"{source} timestamps require at least two samples to derive "
                "sampling_rate_hz; provide sample_rate_hz for single-sample "
                "photometry CSV data."
            )
        raise ValueError(f"{source} timestamps must be uniformly sampled.")
    if expected is not None:
        if not np.isclose(sample_rate, expected, rtol=1e-4, atol=1e-9):
            raise ValueError(
                f"{source} timestamps-derived sampling rate {sample_rate:g} Hz "
                f"does not match sample_rate_hz {expected:g} Hz."
            )
    return float(sample_rate), f"{source}.timestamps_uniform"


def _resolve_time_column(
    frame: pd.DataFrame,
    *,
    time_column: str | None,
    sample_rate_hz: float | None,
    time_column_candidates: Sequence[str],
    allow_implicit_time_column: bool,
) -> str | None:
    if time_column is not None:
        return column_by_name(frame, time_column)
    matched = first_matching_column(frame, time_column_candidates)
    if matched is not None:
        return matched
    if allow_implicit_time_column and sample_rate_hz is None and frame.shape[1] > 1:
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
    time_column_candidates: Sequence[str] = _PHOTOMETRY_TIME_COLUMN_CANDIDATES,
    allow_implicit_time_column: bool = True,
    units: Mapping[str, str] | Sequence[str] | None = None,
    name: str = "photometry",
    max_mb: float | None = None,
) -> PhotometryRecording:
    """Read a photometry CSV into a source-neutral recording object."""

    clean_time_column = _clean_optional_column_name(time_column, role="time_column")
    clean_signal_columns = _clean_column_names(signal_columns, role="signal_columns")
    frame, size_bytes = _read_csv(path, max_mb=max_mb)
    if frame.empty:
        raise ValueError(f"Photometry CSV '{path}' is empty.")
    resolved_time = _resolve_time_column(
        frame,
        time_column=clean_time_column,
        sample_rate_hz=sample_rate_hz,
        time_column_candidates=time_column_candidates,
        allow_implicit_time_column=allow_implicit_time_column,
    )
    resolved_signals = _resolve_signal_columns(
        frame,
        signal_columns=clean_signal_columns,
        time_column=resolved_time,
    )
    values = np.column_stack(
        [_numeric_column(frame, column, role="signal") for column in resolved_signals]
    )

    if resolved_time is not None:
        timestamps = _numeric_column(frame, resolved_time, role="time") * _time_scale(time_unit)
        timeline = Timeline(timestamps_s=timestamps)
        sampling_rate_hz, sampling_rate_source = _uniform_timeline_sample_rate(
            timeline,
            source=resolved_time,
            expected_sample_rate_hz=sample_rate_hz,
        )
    elif sample_rate_hz is not None:
        sampling_rate_hz = _positive_sample_rate(sample_rate_hz)
        timeline = Timeline.from_sample_rate(
            n_samples=values.shape[0],
            sample_rate_hz=sampling_rate_hz,
            start_s=start_s,
        )
        sampling_rate_source = "sample_rate_hz.argument"
    else:
        if allow_implicit_time_column:
            raise ValueError("Provide a time column or sample_rate_hz for photometry CSV data.")
        raise ValueError(
            "Photometry CSV must include a time column named one of: "
            f"{', '.join(time_column_candidates)}."
        )

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
        metadata={
            "source_type": "photometry_csv",
            "source": {"type": "photometry_csv", "path": str(path)},
            "time_column": resolved_time,
            "signal_columns": list(resolved_signals),
            "columns": [str(column) for column in frame.columns],
            "rows": int(len(frame)),
            "size_bytes": size_bytes,
            "time_unit": time_unit,
            "sampling_rate_hz": sampling_rate_hz,
            "sampling_rate_source": sampling_rate_source,
        },
    )


def _event_text_error(*, role: str, column: str, row: int) -> str:
    return (
        f"Event CSV {role} column {column!r} at row {row} must be a non-empty "
        "string without surrounding whitespace."
    )


def _required_event_text(
    frame: pd.DataFrame,
    column: str,
    index: int,
    *,
    role: str,
) -> str:
    value = frame[column].iloc[index]
    message = _event_text_error(role=role, column=column, row=index)
    if pd.isna(value) or not isinstance(value, str):
        raise ValueError(message)
    if not value or value != value.strip():
        raise ValueError(message)
    return value


def _clean_default_kind(default_kind: str) -> str:
    if (
        not isinstance(default_kind, str)
        or not default_kind
        or default_kind != default_kind.strip()
    ):
        raise ValueError(
            "Event CSV default_kind must be a non-empty string without surrounding whitespace."
        )
    return default_kind


def _optional_event_label(
    frame: pd.DataFrame,
    column: str | None,
    index: int,
) -> str | None:
    if column is None:
        return None
    if pd.api.types.is_numeric_dtype(frame[column]):
        value = frame[column].iloc[index]
        if pd.isna(value) or not np.isfinite(float(value)):
            raise ValueError(_event_text_error(role="label", column=column, row=index))
        return f"{column}={value}"
    return _required_event_text(frame, column, index, role="label")


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

    clean_time_column = _clean_optional_column_name(time_column, role="time_column")
    clean_kind_column = _clean_optional_column_name(kind_column, role="kind_column")
    clean_label_column = _clean_optional_column_name(label_column, role="label_column")
    clean_duration_column = _clean_optional_column_name(
        duration_column,
        role="duration_column",
    )
    frame, _size_bytes = _read_csv(path, max_mb=max_mb)
    resolved_time = resolve_column(frame, clean_time_column, _EVENT_TIME_COLUMN_CANDIDATES)
    if resolved_time is None:
        raise ValueError(
            "Event CSV must include a timestamp column named one of: "
            f"{', '.join(_EVENT_TIME_COLUMN_CANDIDATES)}."
        )
    resolved_kind = resolve_column(frame, clean_kind_column, _EVENT_KIND_CANDIDATES)
    resolved_label = resolve_column(frame, clean_label_column, _EVENT_LABEL_CANDIDATES)
    resolved_duration = resolve_column(
        frame,
        clean_duration_column,
        _EVENT_DURATION_CANDIDATES,
    )
    default_kind_text = _clean_default_kind(default_kind)
    table_metadata = {
        "source_type": "events_csv",
        "source": {"type": "events_csv", "path": str(path)},
        "time_column": resolved_time,
        "kind_column": resolved_kind,
        "label_column": resolved_label,
        "duration_column": resolved_duration,
        "columns": [str(column) for column in frame.columns],
        "rows": int(len(frame)),
        "event_records": [],
        "time_unit": time_unit,
        "default_kind": default_kind_text,
    }
    if frame.empty:
        return EventTable(metadata=table_metadata)

    scale = _time_scale(time_unit)
    starts = _numeric_column(frame, resolved_time, role="event time") * scale
    durations = (
        _numeric_column(frame, resolved_duration, role="event duration") * scale
        if resolved_duration is not None
        else np.zeros_like(starts)
    )
    event_rows = [
        (
            index,
            Event(
                kind=(
                    _required_event_text(frame, resolved_kind, index, role="kind")
                    if resolved_kind is not None
                    else default_kind_text
                ),
                start_s=float(start),
                duration_s=float(duration),
                label=_optional_event_label(frame, resolved_label, index),
                metadata={
                    "source": {"type": "events_csv", "path": str(path)},
                    "source_row": int(index),
                },
            ),
        )
        for index, (start, duration) in enumerate(zip(starts, durations, strict=True))
    ]
    event_rows = sorted(
        event_rows,
        key=lambda row: (row[1].start_s, row[1].end_s, row[1].kind),
    )
    table_metadata["event_records"] = [
        {
            "source_row": int(source_row),
            "time_s": float(event.start_s),
            "duration_s": float(event.duration_s),
            "kind": event.kind,
            "label": event.label,
        }
        for source_row, event in event_rows
    ]
    events = tuple(event for _source_row, event in event_rows)
    return EventTable(events=tuple(events), metadata=table_metadata)


__all__ = ["read_events_csv", "read_photometry_csv"]
