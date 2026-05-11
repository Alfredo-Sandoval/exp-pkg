"""Generic CSV reader for patch-clamp recordings.

This is the source-neutral escape hatch: when an ephys vendor exports a CSV of
samples and (optionally) a sweep index, this reader maps it into the canonical
:class:`EphysRecording` shape without committing to any vendor format.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd

from xpkg.model import (
    EphysRecording,
    RecordingMode,
    SignalChannel,
    Sweep,
    SweepSet,
    Timeline,
    TimeSeries,
)
from xpkg.model.ephys import detect_recording_mode, normalize_signal_units

TimeUnit = Literal["s", "sec", "second", "seconds", "ms", "millisecond", "milliseconds"]

_TIME_COLUMN_CANDIDATES = ("time", "timestamp", "timestamps", "time_s", "time_ms")
_SWEEP_COLUMN_CANDIDATES = ("sweep", "sweep_index", "sweep_id", "trace")


def _column(frame: pd.DataFrame, name: str) -> str:
    lookup = {str(column).lower(): str(column) for column in frame.columns}
    key = name.lower()
    if key not in lookup:
        raise ValueError(
            f"Column {name!r} was not found. Available columns: {list(frame.columns)}."
        )
    return lookup[key]


def _first_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    lookup = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        match = lookup.get(candidate.lower())
        if match is not None:
            return match
    return None


def _numeric(frame: pd.DataFrame, column: str, *, role: str) -> np.ndarray:
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


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path, *, max_mb: float | None) -> tuple[pd.DataFrame, int]:
    size_bytes = path.stat().st_size
    if max_mb is not None:
        max_bytes = int(float(max_mb) * 1024 * 1024)
        if max_bytes <= 0:
            raise ValueError(f"max_mb must be positive when provided, got {max_mb!r}.")
        if size_bytes > max_bytes:
            raise ValueError(f"CSV file '{path}' exceeds max load size ({max_mb} MB).")
    return pd.read_csv(path), size_bytes


def _resolve_signal_columns(
    frame: pd.DataFrame,
    *,
    signal_columns: Sequence[str] | None,
    excluded: set[str],
) -> tuple[str, ...]:
    if signal_columns is not None:
        resolved = tuple(_column(frame, column) for column in signal_columns)
        if not resolved:
            raise ValueError("signal_columns must include at least one channel.")
        return resolved
    columns = tuple(
        str(column)
        for column in frame.columns
        if str(column) not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    )
    if not columns:
        raise ValueError("No numeric signal columns were found in the ephys CSV.")
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
        raise ValueError(
            f"units length {len(units_tuple)} does not match {len(columns)} columns."
        )
    return units_tuple


def _roles_for_columns(
    columns: Sequence[str],
    roles: Mapping[str, str] | None,
) -> dict[str, str]:
    if roles is None:
        return {}
    available = set(columns)
    resolved: dict[str, str] = {}
    for channel, role in roles.items():
        channel_text = str(channel).strip()
        if channel_text not in available:
            raise ValueError(
                f"channel_roles references unknown channel {channel!r}; "
                f"available signal columns are {list(columns)}."
            )
        resolved[channel_text] = str(role).strip()
    return resolved


def _build_timeline(
    *,
    times_s: np.ndarray | None,
    n_samples: int,
    sample_rate_hz: float | None,
    start_s: float,
) -> Timeline:
    if times_s is not None:
        return Timeline(timestamps_s=np.asarray(times_s, dtype=np.float64))
    if sample_rate_hz is None:
        raise ValueError(
            "Provide a time column or sample_rate_hz so an ephys timeline can be built."
        )
    return Timeline.from_sample_rate(
        n_samples=n_samples,
        sample_rate_hz=sample_rate_hz,
        start_s=start_s,
    )


def read_ephys_csv(
    path: str | Path,
    *,
    time_column: str | None = None,
    sweep_column: str | None = None,
    signal_columns: Sequence[str] | None = None,
    units: Mapping[str, str] | Sequence[str] | None = None,
    channel_roles: Mapping[str, str] | None = None,
    sample_rate_hz: float | None = None,
    start_s: float = 0.0,
    time_unit: TimeUnit = "s",
    mode: RecordingMode | None = None,
    name: str = "ephys",
    max_mb: float | None = None,
) -> EphysRecording:
    """Read a patch-clamp CSV into a normalized :class:`EphysRecording`.

    The CSV may either describe one continuous sweep or carry a ``sweep_column``
    that splits the rows into multiple sweeps with shared channel layout.
    Voltage columns are normalized to mV and current columns to pA; every
    applied scale is recorded in the recording's ``conversion_log``.
    """

    source_path = Path(path)
    frame, size_bytes = _read_csv(source_path, max_mb=max_mb)
    if frame.empty:
        raise ValueError(f"Ephys CSV '{source_path}' is empty.")

    resolved_time = (
        _column(frame, time_column)
        if time_column is not None
        else _first_column(frame, _TIME_COLUMN_CANDIDATES)
    )
    resolved_sweep = (
        _column(frame, sweep_column)
        if sweep_column is not None
        else _first_column(frame, _SWEEP_COLUMN_CANDIDATES)
    )
    excluded: set[str] = set()
    if resolved_time is not None:
        excluded.add(resolved_time)
    if resolved_sweep is not None:
        excluded.add(resolved_sweep)
    resolved_signals = _resolve_signal_columns(
        frame,
        signal_columns=signal_columns,
        excluded=excluded,
    )
    column_units = _units_for_columns(resolved_signals, units)
    roles = _roles_for_columns(resolved_signals, channel_roles)

    if mode is None:
        unit_lookup = dict(zip(resolved_signals, column_units, strict=True))
        resolved_mode: str = detect_recording_mode(
            channel_roles=roles, channel_units=unit_lookup
        )
    else:
        resolved_mode = str(mode).strip()

    time_scale = _time_scale(time_unit)
    sweep_groups: list[tuple[int, pd.DataFrame]]
    if resolved_sweep is not None:
        sweep_indices = pd.to_numeric(frame[resolved_sweep], errors="raise").to_numpy(
            dtype=np.int64
        )
        unique_sweeps = sorted({int(index) for index in sweep_indices.tolist()})
        sweep_groups = [
            (
                sweep_index,
                frame.iloc[np.where(sweep_indices == sweep_index)[0]].reset_index(drop=True),
            )
            for sweep_index in unique_sweeps
        ]
    else:
        sweep_groups = [(0, frame.reset_index(drop=True))]

    sweeps: list[Sweep] = []
    cumulative_log: list[dict[str, object]] = []
    sweep_offset_s = 0.0
    for sweep_index, sweep_frame in sweep_groups:
        if sweep_frame.empty:
            continue
        values = np.column_stack(
            [
                _numeric(sweep_frame, column, role="signal")
                for column in resolved_signals
            ]
        )
        if resolved_time is not None:
            times = _numeric(sweep_frame, resolved_time, role="time") * time_scale
        else:
            times = None
        timeline = _build_timeline(
            times_s=times,
            n_samples=values.shape[0],
            sample_rate_hz=sample_rate_hz,
            start_s=start_s + sweep_offset_s if times is None else 0.0,
        )
        raw_channels = tuple(
            SignalChannel(name=column, unit=unit, metadata={"source_column": column})
            for column, unit in zip(resolved_signals, column_units, strict=True)
        )
        scaled_values, normalized_channels, log_entries = normalize_signal_units(
            values, raw_channels, roles=roles
        )
        cumulative_log.extend(
            {**entry, "sweep_index": int(sweep_index)} for entry in log_entries
        )
        series = TimeSeries(
            values=scaled_values,
            channels=normalized_channels,
            timeline=timeline,
            name=f"{name}_sweep_{int(sweep_index)}",
            provenance={
                "source": {
                    "type": "ephys_csv",
                    "path": str(source_path),
                    "size_bytes": size_bytes,
                },
                "sweep_index": int(sweep_index),
                "time_column": resolved_time,
                "sweep_column": resolved_sweep,
            },
        )
        sweeps.append(
            Sweep(
                index=int(sweep_index),
                series=series,
                sweep_start_s=float(sweep_offset_s) if times is None else float(timeline.start_s),
            )
        )
        if times is None:
            sweep_offset_s += float(timeline.duration_s)

    if not sweeps:
        raise ValueError(f"Ephys CSV '{source_path}' did not yield any sweeps.")

    sweep_set = SweepSet.from_sweeps(sweeps)
    provenance = {
        "source": {
            "type": "ephys_csv",
            "path": str(source_path),
            "size_bytes": size_bytes,
            "sha256": _file_hash(source_path),
        },
        "time_column": resolved_time,
        "sweep_column": resolved_sweep,
        "signal_columns": list(resolved_signals),
    }
    return EphysRecording(
        sweeps=sweep_set,
        mode=cast(RecordingMode, resolved_mode),
        channel_roles=roles,
        conversion_log=tuple(cumulative_log),
        provenance=provenance,
        metadata={
            "source_type": "ephys_csv",
            "n_sweeps": sweep_set.n_sweeps,
            "name": name,
        },
    )


__all__ = ["read_ephys_csv"]
