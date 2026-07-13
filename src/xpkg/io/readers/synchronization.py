"""Reader for paired synchronization observations in two timebases."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from xpkg.io.readers._columns import resolve_column
from xpkg.io.readers._csv import read_csv_table
from xpkg.io.readers._normalization import time_scale
from xpkg.model.session import (
    AlignmentModel,
    SynchronizationMethod,
    TimebaseAlignment,
    TimebaseCorrespondence,
)
from xpkg.model.session_actions import fit_timebase_alignment
from xpkg.model.time import Timebase

TimeUnit = Literal["s", "sec", "second", "seconds", "ms", "millisecond", "milliseconds"]

_SOURCE_TIME_CANDIDATES = (
    "source_time_s",
    "source_time",
    "source_timestamp",
    "time_source",
    "time_a",
)
_TARGET_TIME_CANDIDATES = (
    "target_time_s",
    "target_time",
    "target_timestamp",
    "time_target",
    "time_b",
)
_CORRESPONDENCE_ID_CANDIDATES = (
    "correspondence_id",
    "pulse_id",
    "event_id",
    "sync_id",
    "id",
)


def read_synchronization_csv(
    path: str | Path,
    *,
    source_timebase: Timebase,
    target_timebase: Timebase,
    model: AlignmentModel = AlignmentModel.AFFINE,
    method: SynchronizationMethod = SynchronizationMethod.PULSES,
    name: str | None = None,
    source_time_column: str | None = None,
    target_time_column: str | None = None,
    correspondence_id_column: str | None = None,
    source_time_unit: TimeUnit = "s",
    target_time_unit: TimeUnit = "s",
    max_mb: float | None = None,
) -> TimebaseAlignment:
    """Read paired clock observations and fit a typed timebase alignment."""
    source_path = Path(path)
    frame, size_bytes = read_csv_table(source_path, max_mb=max_mb)
    if frame.empty:
        raise ValueError(f"Synchronization CSV '{source_path}' is empty.")
    source_column = _required_column(
        frame,
        source_time_column,
        _SOURCE_TIME_CANDIDATES,
        role="source time",
    )
    target_column = _required_column(
        frame,
        target_time_column,
        _TARGET_TIME_CANDIDATES,
        role="target time",
    )
    id_column = resolve_column(
        frame,
        correspondence_id_column,
        _CORRESPONDENCE_ID_CANDIDATES,
    )
    source_times = _numeric_times(
        frame, source_column, scale=time_scale(source_time_unit), role="source time"
    )
    target_times = _numeric_times(
        frame, target_column, scale=time_scale(target_time_unit), role="target time"
    )
    evidence = tuple(
        TimebaseCorrespondence(
            source_time_s=float(source_time),
            target_time_s=float(target_time),
            correspondence_id=_correspondence_id(frame, id_column, index),
            metadata={"source_row": index},
        )
        for index, (source_time, target_time) in enumerate(
            zip(source_times, target_times, strict=True)
        )
    )
    return fit_timebase_alignment(
        name=name or f"{source_timebase.name}-to-{target_timebase.name}",
        source=source_timebase,
        target=target_timebase,
        model=model,
        method=method,
        evidence=evidence,
        metadata={
            "source": {
                "type": "synchronization_csv",
                "path": str(source_path),
                "size_bytes": size_bytes,
            },
            "source_time_column": source_column,
            "target_time_column": target_column,
            "correspondence_id_column": id_column,
            "source_time_unit": source_time_unit,
            "target_time_unit": target_time_unit,
        },
    )


def _required_column(
    frame: pd.DataFrame,
    explicit: str | None,
    candidates: tuple[str, ...],
    *,
    role: str,
) -> str:
    column = resolve_column(frame, explicit, candidates)
    if column is None:
        raise ValueError(
            f"Synchronization CSV must include a {role} column named one of: "
            f"{', '.join(candidates)}."
        )
    return column


def _numeric_times(
    frame: pd.DataFrame,
    column: str,
    *,
    scale: float,
    role: str,
) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"Synchronization {role} column {column!r} must contain finite values.")
    return values * scale


def _correspondence_id(frame: pd.DataFrame, column: str | None, index: int) -> str | None:
    if column is None:
        return None
    value = frame[column].iloc[index]
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


__all__ = ["read_synchronization_csv"]
