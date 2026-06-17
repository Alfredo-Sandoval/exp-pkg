"""Direct readers for common fiber-photometry acquisition exports."""

from __future__ import annotations

import importlib
import re
from collections.abc import Iterable, Mapping, Sequence
from json import JSONDecodeError
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, NamedTuple

import h5py
import numpy as np
import pandas as pd

from xpkg._core.json_utils import parse_json_dict
from xpkg.io.readers._discovery import find_first_file
from xpkg.model import (
    Event,
    EventTable,
    PhotometryChannel,
    PhotometryRecording,
    RecordingSession,
    SignalChannel,
    Timeline,
    TimeSeries,
)

TimeUnit = Literal["s", "sec", "second", "seconds", "ms", "millisecond", "milliseconds"]

_TIME_COLUMNS = ("time", "timestamp", "timestamps", "Time", "TimeStamp", "Timestamp")
_NPM_TIME_COLUMNS = (
    "Timestamp",
    "SystemTimestamp",
    "ComputerTimestamp",
    "Time",
    "TimeStamp",
)
_NPM_METADATA_COLUMNS = {
    "computertimestamp",
    "framecounter",
    "flags",
    "ledstate",
    "systemtimestamp",
    "time",
    "timestamp",
}
_NPM_STATE_COLUMNS = ("Flags", "LedState")
_NPM_DEFAULT_LED_CODE_TO_NM = {1: 415, 2: 470, 4: 560}
_NPM_SIGNAL_NM = 470
_NPM_REFERENCE_NM = 415
_NPM_DEMUXED_LABEL_SUFFIX_RE = re.compile(r"_(?:\d+nm|led_state_-?\d+)$")
_NPM_DEMUXED_LABEL_PARTS_RE = re.compile(
    r"_(?:(?P<wavelength_nm>\d+)nm|led_state_(?P<led_state_code>-?\d+))$"
)
_RWD_SIGNAL_SUFFIXES = ("-470", "_470", "-560", "_560")
_RWD_REFERENCE_SUFFIXES = ("-410", "_410", "-405", "_405", "-415", "_415")
_TELEOPTO_EVENT_KEYS = ("ct1", "ct2", "ct3", "ct4", "ar1", "ar2")
_TELEOPTO_REQUIRED_KEYS = frozenset({"d1", "num", "st1", "str"})


class NeurophotometricsChannelSelection(NamedTuple):
    """Parsed source column and LED identity from a Neurophotometrics label."""

    source_column: str | None
    wavelength_nm: int | None = None
    led_state_code: int | None = None


def _read_csv(path: str | Path, *, max_mb: float | None = None) -> pd.DataFrame:
    source_path = Path(path)
    size_bytes = source_path.stat().st_size
    if max_mb is not None:
        max_bytes = int(float(max_mb) * 1024 * 1024)
        if max_bytes <= 0:
            raise ValueError(f"max_mb must be positive when provided, got {max_mb!r}.")
        if size_bytes > max_bytes:
            raise ValueError(f"CSV file '{source_path}' exceeds max load size ({max_mb} MB).")
    return pd.read_csv(source_path)


def _time_scale(unit: TimeUnit) -> float:
    normalized = unit.lower()
    if normalized in {"s", "sec", "second", "seconds"}:
        return 1.0
    if normalized in {"ms", "millisecond", "milliseconds"}:
        return 0.001
    raise ValueError(f"Unsupported time_unit {unit!r}; expected seconds or milliseconds.")


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


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Column {column!r} must be one-dimensional.")
    if not np.isfinite(values).all():
        raise ValueError(f"Column {column!r} contains non-finite values.")
    return values


def _pmat_event_label(value: object, *, column: str, row: int) -> str:
    message = (
        f"pMAT event label column {column!r} at row {row} must be a non-empty "
        "string without surrounding whitespace."
    )
    if pd.isna(value) or not isinstance(value, str):
        raise ValueError(message)
    if not value or value != value.strip():
        raise ValueError(message)
    return value


def _rwd_event_label(value: object, *, row: int) -> str:
    message = (
        f"RWD Events.csv Name column at row {row} must be a non-empty string "
        "without surrounding whitespace."
    )
    if pd.isna(value) or not isinstance(value, str):
        raise ValueError(message)
    if not value or value != value.strip():
        raise ValueError(message)
    return value


def _timeline_from_time(values: np.ndarray) -> Timeline:
    return Timeline(timestamps_s=np.asarray(values, dtype=np.float64))


def _uniform_timeline_sample_rate(timeline: Timeline, source: str) -> tuple[float, str]:
    sample_rate = timeline.estimated_sample_rate_hz
    if sample_rate is None:
        raise ValueError(f"{source} must be uniformly sampled.")
    return float(sample_rate), f"{source}.timestamps_uniform"


def _excitation_from_name(name: str) -> str:
    lowered = name.lower()
    for token in ("405", "410", "415", "465", "470", "560"):
        if token in lowered:
            return token
    return ""


def _photometry_recording(
    *,
    values: np.ndarray,
    channel_names: Sequence[str],
    timeline: Timeline,
    source_type: str,
    source_path: Path,
    signal_channel: str | None = None,
    reference_channel: str | None = None,
    units: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PhotometryRecording:
    units_tuple = tuple(units) if units is not None else ("",) * len(channel_names)
    if len(units_tuple) != len(channel_names):
        raise ValueError("units length must match channel_names length.")
    channels = tuple(
        PhotometryChannel(
            name=name,
            unit=unit,
            excitation=_excitation_from_name(name),
            metadata={"source_column": name},
        )
        for name, unit in zip(channel_names, units_tuple, strict=True)
    )
    series = TimeSeries(
        values=values,
        channels=channels,
        timeline=timeline,
        name="photometry",
        provenance={"source": {"type": source_type, "path": str(source_path)}},
    )
    selected_signal = signal_channel or str(channel_names[0])
    return PhotometryRecording(
        series=series,
        signal_channel=selected_signal,
        reference_channel=reference_channel,
        metadata={"source_type": source_type, **dict(metadata or {})},
    )


def _events_from_map(
    events: Mapping[str, Iterable[float]],
    *,
    source_type: str,
    source_path: Path,
    kind: str = "event",
) -> EventTable:
    rows: list[Event] = []
    for label, values in events.items():
        for value in values:
            rows.append(
                Event(
                    kind=kind,
                    start_s=float(value),
                    label=str(label),
                    metadata={"source": {"type": source_type, "path": str(source_path)}},
                )
            )
    return EventTable.from_events(rows)


def read_pmat_photometry_csv(
    path: str | Path,
    *,
    time_column: str | None = None,
    signal_column: str | None = None,
    reference_column: str | None = None,
    time_unit: TimeUnit = "s",
    max_mb: float | None = None,
) -> PhotometryRecording:
    """Read a pMAT-compatible photometry CSV export."""

    source_path = Path(path)
    frame = _read_csv(source_path, max_mb=max_mb)
    if frame.empty:
        raise ValueError(f"pMAT photometry CSV '{source_path}' is empty.")
    columns = [str(column) for column in frame.columns]
    resolved_time = _column(frame, time_column) if time_column else columns[0]
    resolved_signal = _column(frame, signal_column) if signal_column else columns[1]
    resolved_reference = (
        _column(frame, reference_column)
        if reference_column is not None
        else (columns[2] if len(columns) > 2 else None)
    )
    signal_columns = [resolved_signal]
    if resolved_reference is not None:
        signal_columns.append(resolved_reference)
    timestamps = _numeric(frame, resolved_time) * _time_scale(time_unit)
    values = np.column_stack([_numeric(frame, column) for column in signal_columns])
    timeline = _timeline_from_time(timestamps)
    sample_rate, sample_rate_source = _uniform_timeline_sample_rate(
        timeline,
        resolved_time,
    )
    return _photometry_recording(
        values=values,
        channel_names=signal_columns,
        timeline=timeline,
        source_type="pmat_photometry_csv",
        source_path=source_path,
        signal_channel=resolved_signal,
        reference_channel=resolved_reference,
        metadata={
            "time_column": resolved_time,
            "signal_column": resolved_signal,
            "reference_column": resolved_reference,
            "signal_columns": signal_columns,
            "columns": columns,
            "rows": int(len(frame)),
            "time_unit": time_unit,
            "sampling_rate_hz": sample_rate,
            "sampling_rate_source": sample_rate_source,
        },
    )


def read_pmat_events_csv(
    path: str | Path,
    *,
    label_column: str | None = None,
    onset_column: str | None = None,
    offset_column: str | None = None,
    time_unit: TimeUnit = "s",
    max_mb: float | None = None,
) -> EventTable:
    """Read a pMAT-compatible event CSV export."""

    source_path = Path(path)
    frame = _read_csv(source_path, max_mb=max_mb)
    columns = [str(column) for column in frame.columns]
    resolved_label = _column(frame, label_column) if label_column else columns[0]
    resolved_onset = _column(frame, onset_column) if onset_column else columns[1]
    resolved_offset = (
        _column(frame, offset_column)
        if offset_column is not None
        else (columns[2] if len(columns) > 2 else None)
    )
    table_metadata = {
        "source_type": "pmat_events_csv",
        "label_column": resolved_label,
        "onset_column": resolved_onset,
        "offset_column": resolved_offset,
        "columns": columns,
        "rows": int(len(frame)),
        "event_records": [],
        "time_unit": time_unit,
    }
    if frame.empty:
        return EventTable(metadata=table_metadata)
    scale = _time_scale(time_unit)
    starts = _numeric(frame, resolved_onset) * scale
    if resolved_offset:
        offsets = _numeric(frame, resolved_offset) * scale
    else:
        offsets = starts
    before_onset = np.flatnonzero(offsets < starts)
    if before_onset.size:
        row = int(before_onset[0])
        raise ValueError(
            f"pMAT event offset column {resolved_offset!r} at row {row} must be "
            f"greater than or equal to onset column {resolved_onset!r}."
        )
    rows = []
    for index, (start, offset) in enumerate(zip(starts, offsets, strict=True)):
        duration = float(offset - start)
        label = _pmat_event_label(
            frame[resolved_label].iloc[index],
            column=resolved_label,
            row=index,
        )
        rows.append(
            Event(
                kind="event",
                start_s=float(start),
                duration_s=duration,
                label=label,
                metadata={
                    "source": {"type": "pmat_events_csv", "path": str(source_path)},
                    "source_row": int(index),
                    "offset_s": float(offset),
                },
            )
        )
    rows = sorted(rows, key=lambda event: (event.start_s, event.end_s, event.kind))
    table_metadata["event_records"] = [
        {
            "source_row": int(event.metadata["source_row"]),
            "time_s": float(event.start_s),
            "duration_s": float(event.duration_s),
            "offset_s": float(event.metadata["offset_s"]),
            "kind": event.kind,
            "label": event.label,
        }
        for event in rows
    ]
    return EventTable(events=tuple(rows), metadata=table_metadata)


def _npm_led_code_map(value: Mapping[int, int] | None) -> dict[int, int]:
    source = _NPM_DEFAULT_LED_CODE_TO_NM if value is None else value
    result: dict[int, int] = {}
    for raw_code, raw_nm in source.items():
        try:
            code = int(raw_code)
            nm = int(raw_nm)
        except (TypeError, ValueError) as exc:
            raise ValueError("Neurophotometrics LED code map must contain integers.") from exc
        result[code] = nm
    return result


def is_neurophotometrics_csv(path: str | Path) -> bool:
    """Return whether ``path`` has the Neurophotometrics/Bonsai CSV state contract."""

    source_path = Path(path)
    if not source_path.is_file() or source_path.suffix.lower() != ".csv":
        return False
    try:
        frame = pd.read_csv(source_path, nrows=0)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError):
        return False
    state_columns = {column.lower() for column in _NPM_STATE_COLUMNS}
    columns = {str(column).strip().strip('"').lower() for column in frame.columns}
    return bool(columns & state_columns)


def find_first_neurophotometrics_csv(path: str | Path) -> Path | None:
    """Return the first Neurophotometrics/Bonsai CSV under ``path``."""

    return find_first_file(path, is_neurophotometrics_csv)


def _npm_state_values(frame: pd.DataFrame, state_column: str) -> np.ndarray:
    values = _numeric(frame, state_column)
    rounded = np.rint(values)
    if not np.allclose(values, rounded, rtol=0.0, atol=0.0):
        raise ValueError(
            f"Neurophotometrics state column {state_column!r} must contain integer LED codes."
        )
    return rounded.astype(np.int64)


def _npm_channel_name(roi_column: str, *, code: int, nm: int | None) -> str:
    if nm is None:
        return f"{roi_column}_led_state_{code}"
    return f"{roi_column}_{nm}nm"


def neurophotometrics_source_column_from_label(label: str | None) -> str | None:
    """Return the raw ROI column for a Neurophotometrics channel label."""

    return neurophotometrics_channel_selection_from_label(label).source_column


def _npm_clean_optional_label(label: object) -> str | None:
    if label is None:
        return None
    if not isinstance(label, str):
        raise TypeError("Neurophotometrics label must be a string or None.")
    if label == "":
        return None
    if label != label.strip():
        raise ValueError(
            "Neurophotometrics label must be a non-empty string without "
            "surrounding whitespace, or None."
        )
    return label


def neurophotometrics_channel_selection_from_label(
    label: str | None,
) -> NeurophotometricsChannelSelection:
    """Return the ROI column plus LED identity encoded in a demuxed label."""

    clean_label = _npm_clean_optional_label(label)
    if clean_label is None:
        return NeurophotometricsChannelSelection(None, None, None)
    match = _NPM_DEMUXED_LABEL_PARTS_RE.search(clean_label)
    if match is None:
        return NeurophotometricsChannelSelection(clean_label, None, None)
    source_column = clean_label[: match.start()]
    if not source_column:
        raise ValueError(
            "Neurophotometrics label must include a source column before the demux suffix."
        )
    raw_wavelength = match.group("wavelength_nm")
    raw_code = match.group("led_state_code")
    wavelength_nm = int(raw_wavelength) if raw_wavelength is not None else None
    led_state_code = int(raw_code) if raw_code is not None else None
    return NeurophotometricsChannelSelection(
        source_column,
        wavelength_nm,
        led_state_code,
    )


def _npm_resolve_led_code(
    *,
    codes: Sequence[int],
    code_map: Mapping[int, int],
    requested_code: int | None,
    requested_nm: int,
    role: str,
    source_path: Path,
    required: bool,
) -> int | None:
    if requested_code is not None:
        code = int(requested_code)
        if code not in codes:
            raise ValueError(
                f"Neurophotometrics CSV '{source_path}' has no requested "
                f"{role} LedState code {code} (found LedState codes {list(codes)})."
            )
        return code

    code = next((item for item in codes if code_map.get(item) == requested_nm), None)
    if code is None and required:
        raise ValueError(
            f"Neurophotometrics CSV '{source_path}' has no {requested_nm} nm "
            f"{role} LED (found LedState codes {list(codes)}). Pass "
            "led_code_to_nm if this rig uses a non-default encoding."
        )
    return code


def _npm_demux_photometry(
    *,
    frame: pd.DataFrame,
    timestamps: np.ndarray,
    state: np.ndarray,
    source_path: Path,
    roi_columns: Sequence[str],
    signal_column: str,
    reference_column: str | None,
    state_column: str,
    time_column: str,
    code_map: Mapping[int, int],
    signal_nm: int,
    reference_nm: int,
    signal_led_code: int | None = None,
    reference_led_code: int | None = None,
) -> PhotometryRecording:
    codes = sorted({int(code) for code in np.unique(state) if int(code) != 0})
    signal_code = _npm_resolve_led_code(
        codes=codes,
        code_map=code_map,
        requested_code=signal_led_code,
        requested_nm=signal_nm,
        role="signal",
        source_path=source_path,
        required=True,
    )
    if signal_code is None:
        raise ValueError(
            f"Neurophotometrics CSV '{source_path}' has no {signal_nm} nm signal "
            f"LED (found LedState codes {codes}). Pass led_code_to_nm to map the "
            "signal LED if this rig uses a non-default encoding."
        )
    resolved_signal_nm = code_map.get(signal_code)
    reference_code = _npm_resolve_led_code(
        codes=codes,
        code_map=code_map,
        requested_code=reference_led_code,
        requested_nm=reference_nm,
        role="reference",
        source_path=source_path,
        required=reference_led_code is not None,
    )
    if reference_code == signal_code:
        raise ValueError(
            "Neurophotometrics signal and reference selections resolve to the "
            f"same LedState code {signal_code}. Use distinct signal_nm and "
            "reference_nm values or distinct signal_led_code and "
            "reference_led_code values."
        )
    resolved_reference_nm = code_map.get(reference_code) if reference_code is not None else None
    signal_values = _numeric(frame, signal_column)
    reference_roi = reference_column or signal_column
    reference_values = _numeric(frame, reference_roi)

    streams: dict[int, tuple[int | None, str, np.ndarray, np.ndarray]] = {}
    for code in codes:
        nm = code_map.get(code)
        roi_column = reference_roi if code == reference_code else signal_column
        values = reference_values if code == reference_code else signal_values
        mask = state == code
        streams[code] = (nm, roi_column, timestamps[mask], values[mask])

    channel_order = [signal_code]
    if reference_code is not None and reference_code not in channel_order:
        channel_order.append(reference_code)
    channel_order.extend(code for code in codes if code not in channel_order)

    aligned_count = min(streams[code][3].size for code in channel_order)
    if aligned_count <= 0:
        raise ValueError("Neurophotometrics LED demux produced an empty signal stream.")
    signal_times = streams[signal_code][2][:aligned_count]

    channel_names: list[str] = []
    arrays: list[np.ndarray] = []
    original_counts: dict[str, int] = {}
    for code in channel_order:
        nm, roi_column, _stream_time, stream_values = streams[code]
        channel_name = _npm_channel_name(roi_column, code=code, nm=nm)
        channel_names.append(channel_name)
        arrays.append(stream_values[:aligned_count])
        original_counts[channel_name] = int(stream_values.size)

    signal_name = _npm_channel_name(
        signal_column,
        code=signal_code,
        nm=resolved_signal_nm,
    )
    reference_name = (
        _npm_channel_name(reference_roi, code=reference_code, nm=resolved_reference_nm)
        if reference_code is not None
        else None
    )
    values = np.column_stack(arrays)
    timeline = _timeline_from_time(signal_times)
    sample_rate, sample_rate_source = _uniform_timeline_sample_rate(
        timeline,
        f"{time_column}.demux_signal",
    )
    return _photometry_recording(
        values=values,
        channel_names=channel_names,
        timeline=timeline,
        source_type="neurophotometrics_csv",
        source_path=source_path,
        signal_channel=signal_name,
        reference_channel=reference_name,
        metadata={
            "raw_roi_columns": list(roi_columns),
            "sampling_rate_hz": sample_rate,
            "sampling_rate_source": sample_rate_source,
            "led_demux": {
                "applied": True,
                "state_column": state_column,
                "code_to_nm": {str(code): nm for code, nm in sorted(code_map.items())},
                "codes_present": codes,
                "signal_code": signal_code,
                "reference_code": reference_code,
                "signal_nm": resolved_signal_nm,
                "reference_nm": resolved_reference_nm,
                "signal_roi_column": signal_column,
                "reference_roi_column": reference_roi if reference_code is not None else None,
                "signal_selection": (
                    "led_state_code" if signal_led_code is not None else "wavelength_nm"
                ),
                "reference_selection": (
                    "led_state_code"
                    if reference_led_code is not None
                    else ("wavelength_nm" if reference_code is not None else None)
                ),
                "aligned_sample_count": int(aligned_count),
                "raw_sample_counts": original_counts,
            },
        },
    )


def read_neurophotometrics_csv(
    path: str | Path,
    *,
    time_column: str | None = None,
    signal_column: str | None = None,
    reference_column: str | None = None,
    time_unit: TimeUnit = "s",
    led_code_to_nm: Mapping[int, int] | None = None,
    signal_nm: int = _NPM_SIGNAL_NM,
    reference_nm: int = _NPM_REFERENCE_NM,
    signal_led_code: int | None = None,
    reference_led_code: int | None = None,
) -> RecordingSession:
    """Read a Neurophotometrics/Bonsai photometry writer CSV export."""

    source_path = Path(path)
    frame = pd.read_csv(source_path)
    if frame.empty:
        raise ValueError(f"Neurophotometrics CSV '{source_path}' is empty.")
    resolved_time = (
        _column(frame, time_column) if time_column else _first_column(frame, _NPM_TIME_COLUMNS)
    )
    if resolved_time is None:
        raise ValueError(
            "Neurophotometrics CSV missing a timestamp column. "
            f"Available columns: {list(frame.columns)}."
        )
    state_column = _first_column(frame, _NPM_STATE_COLUMNS)
    roi_columns = [
        str(column)
        for column in frame.columns
        if str(column).lower() not in _NPM_METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not roi_columns:
        raise ValueError("Neurophotometrics CSV does not contain numeric ROI columns.")
    resolved_signal = _column(frame, signal_column) if signal_column else roi_columns[0]
    resolved_reference = _column(frame, reference_column) if reference_column else None
    timestamps = _numeric(frame, resolved_time) * _time_scale(time_unit)
    led_map = _npm_led_code_map(led_code_to_nm)
    if state_column is None:
        values = np.column_stack([_numeric(frame, column) for column in roi_columns])
        timeline = _timeline_from_time(timestamps)
        sample_rate, sample_rate_source = _uniform_timeline_sample_rate(
            timeline,
            resolved_time,
        )
        photometry = _photometry_recording(
            values=values,
            channel_names=roi_columns,
            timeline=timeline,
            source_type="neurophotometrics_csv",
            source_path=source_path,
            signal_channel=resolved_signal,
            reference_channel=resolved_reference,
            metadata={
                "sampling_rate_hz": sample_rate,
                "sampling_rate_source": sample_rate_source,
                "led_demux": {"applied": False},
            },
        )
        state_values = None
    else:
        state_values = _npm_state_values(frame, state_column)
        photometry = _npm_demux_photometry(
            frame=frame,
            timestamps=timestamps,
            state=state_values,
            source_path=source_path,
            roi_columns=roi_columns,
            signal_column=resolved_signal,
            reference_column=resolved_reference,
            state_column=state_column,
            time_column=resolved_time,
            code_map=led_map,
            signal_nm=int(signal_nm),
            reference_nm=int(reference_nm),
            signal_led_code=signal_led_code,
            reference_led_code=reference_led_code,
        )
    signals: dict[str, TimeSeries | PhotometryRecording] = {"photometry": photometry}
    if state_column is not None and state_values is not None:
        signals["flags"] = TimeSeries(
            values=state_values,
            channels=(SignalChannel(name=state_column, unit="state"),),
            timeline=_timeline_from_time(timestamps),
            name="neurophotometrics_state",
            provenance={"source": {"type": "neurophotometrics_csv", "path": str(source_path)}},
        )
    metadata: dict[str, Any] = {
        "source": {"type": "neurophotometrics_csv", "path": str(source_path)},
        "columns": [str(column) for column in frame.columns],
        "time_column": resolved_time,
        "state_column": state_column,
        "sampling_rate_hz": photometry.metadata["sampling_rate_hz"],
        "sampling_rate_source": photometry.metadata["sampling_rate_source"],
    }
    frame_counter = _first_column(frame, ("FrameCounter",))
    if frame_counter is not None:
        metadata["frame_counter"] = _numeric(frame, frame_counter).astype(int).tolist()
    return RecordingSession(
        session_id=source_path.stem,
        signals=signals,
        metadata=metadata,
    )


# Magnitude threshold (seconds) used only as a last resort when no declared
# ``Fps`` metadata is available to anchor the RWD timestamp unit deterministically.
_RWD_AMBIGUOUS_DELTA_SECONDS = 1.0


def _rwd_unit_scale_from_fps(median_delta: float, declared_fps: float) -> float:
    """Pick seconds vs milliseconds by closeness to the declared FPS spacing."""
    expected_delta_s = 1.0 / declared_fps
    seconds_residual = abs(float(np.log(median_delta * 1.0 / expected_delta_s)))
    millis_residual = abs(float(np.log(median_delta * 0.001 / expected_delta_s)))
    return 1.0 if seconds_residual <= millis_residual else 0.001


def _rwd_timebase(
    values: np.ndarray,
    declared_fps: float | None,
) -> tuple[np.ndarray, float, float, str, float | None]:
    if values.size == 0:
        raise ValueError("RWD OFRS time column is empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("RWD OFRS time column contains non-finite values.")
    if values.size <= 1:
        if declared_fps is None:
            raise ValueError(
                "RWD OFRS time column has a single sample and no declared 'Fps' "
                "metadata; cannot determine a sampling rate."
            )
        if values[0] != 0.0:
            raise ValueError(
                "RWD OFRS time column has a single non-zero timestamp; cannot "
                "determine whether the value is seconds or milliseconds."
            )
        return (
            values.astype(np.float64),
            float(declared_fps),
            1.0,
            "declared_fps_zero_start_single_sample",
            None,
        )
    diffs = np.diff(values)
    if np.any(diffs <= 0):
        raise ValueError("RWD OFRS time column must be strictly increasing.")
    median_delta = float(np.median(diffs))
    if declared_fps is not None:
        # Declared metadata is authoritative: it fixes both the unit and fs, so a
        # slow (>= 1 s/sample) acquisition in seconds is never divided by 1000 by
        # accident. The sampling rate is the declared Fps, not a re-estimate.
        scale = _rwd_unit_scale_from_fps(median_delta, declared_fps)
        inference = "declared_fps_seconds" if scale == 1.0 else "declared_fps_milliseconds"
        return values * scale, float(declared_fps), scale, inference, median_delta
    # No declared Fps: only the magnitude heuristic remains. An ambiguous (>= 1 s)
    # spacing could be slow-seconds or fast-milliseconds; refuse to silently guess.
    if median_delta >= _RWD_AMBIGUOUS_DELTA_SECONDS:
        raise ValueError(
            "RWD OFRS Fluorescence.csv has no declared 'Fps' metadata and the "
            f"median timestamp spacing ({median_delta:g}) is ambiguous between "
            "seconds and milliseconds. Provide 'Fps' metadata to anchor the timebase."
        )
    time_s = values * 0.001
    return (
        time_s,
        float(1.0 / np.median(np.diff(time_s))),
        0.001,
        "subsecond_spacing_milliseconds",
        median_delta,
    )


def _rwd_metadata_line(path: Path) -> tuple[str | None, dict[str, Any] | None]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return None, None
    first_line = lines[0].strip()
    if not first_line.startswith("{"):
        return None, None
    try:
        return first_line, parse_json_dict(first_line.replace(";", ","))
    except JSONDecodeError:
        return first_line, None


def _rwd_event_table(events_path: Path, time_scale: float) -> tuple[EventTable, dict[str, Any]]:
    event_metadata: dict[str, Any] = {
        "present": events_path.is_file(),
        "path": str(events_path),
        "row_count": 0,
        "time_column": None,
        "unique_labels": [],
        "time_monotonic": None,
    }
    if not events_path.is_file():
        return EventTable(), event_metadata
    frame = pd.read_csv(events_path)
    if frame.empty:
        return EventTable(), event_metadata
    event_metadata["row_count"] = int(frame.shape[0])
    time_column = _first_column(frame, _TIME_COLUMNS)
    if time_column is None:
        raise ValueError(f"RWD Events.csv missing time column. Available: {list(frame.columns)}.")
    event_metadata["time_column"] = time_column
    raw_starts = _numeric(frame, time_column)
    if raw_starts.size >= 2:
        event_metadata["time_monotonic"] = bool(np.all(np.diff(raw_starts) >= 0))
    starts = raw_starts * time_scale
    if "Name" in frame.columns:
        raw_labels = frame["Name"]
        labels = pd.Series(
            [_rwd_event_label(value, row=index) for index, value in enumerate(raw_labels)],
            dtype=str,
        )
    else:
        labels = pd.Series(["event"] * len(frame), dtype=str)
    event_metadata["unique_labels"] = sorted(set(labels.tolist()))
    states = None
    if "State" in frame.columns:
        states = pd.to_numeric(frame["State"], errors="raise")
        state_values = states.to_numpy(dtype=np.float64)
        invalid_states = sorted(
            {
                float(state)
                for state in state_values
                if not np.isfinite(state) or not state.is_integer() or int(state) not in {0, 1}
            }
        )
        if invalid_states:
            raise ValueError(
                "RWD Events.csv State values must be 0 for onset or 1 for offset; "
                f"got {invalid_states}."
            )
    rows: list[Event] = []
    for index, start in enumerate(starts):
        state = None if states is None else int(states.iloc[index])
        suffix = "_offset" if state == 1 else ""
        rows.append(
            Event(
                kind="behavior",
                start_s=float(start),
                label=f"{labels.iloc[index]}{suffix}",
                metadata={"source": {"type": "rwd_ofrs_events", "path": str(events_path)}},
            )
        )
    return EventTable.from_events(rows), event_metadata


def is_rwd_ofrs_session(path: str | Path) -> bool:
    """Return whether ``path`` has the RWD/OFRS CSV session contract."""

    session_path = Path(path)
    fluorescence_path = session_path / "Fluorescence.csv"
    if not fluorescence_path.is_file():
        return False
    try:
        with fluorescence_path.open("r", encoding="utf-8", errors="ignore") as handle:
            first_line = handle.readline()
            second_line = handle.readline()
    except OSError:
        return False

    header_line = second_line if first_line.lstrip().startswith("{") else first_line
    if not header_line:
        return False

    columns = [token.strip() for token in header_line.strip().split(",") if token.strip()]
    time_columns = {candidate.lower() for candidate in _TIME_COLUMNS}
    has_time = any(column.lower() in time_columns for column in columns)
    has_signal = any(column.endswith(_RWD_SIGNAL_SUFFIXES) for column in columns)
    return has_time and has_signal


def read_rwd_ofrs_session(path: str | Path) -> RecordingSession:
    """Read an RWD OFRS CSV session bundle."""

    session_path = Path(path)
    fluorescence_path = session_path / "Fluorescence.csv"
    if not fluorescence_path.is_file():
        raise FileNotFoundError(f"RWD OFRS Fluorescence.csv not found in {session_path}.")
    metadata_line, parsed_metadata = _rwd_metadata_line(fluorescence_path)
    frame = pd.read_csv(fluorescence_path, skiprows=1 if metadata_line is not None else 0)
    frame = frame.loc[
        :,
        [column for column in frame.columns if not str(column).startswith("Unnamed:")],
    ]
    frame.columns = [str(column).strip() for column in frame.columns]
    time_column = _first_column(frame, _TIME_COLUMNS)
    if time_column is None:
        raise ValueError(f"RWD OFRS Fluorescence.csv missing time column: {list(frame.columns)}.")
    declared_fps = None
    if isinstance(parsed_metadata, Mapping):
        raw_fps = parsed_metadata.get("Fps")
        if isinstance(raw_fps, int | float) and np.isfinite(float(raw_fps)) and float(raw_fps) > 0:
            declared_fps = float(raw_fps)
    (
        timestamps,
        sample_rate,
        time_scale,
        time_scale_inference,
        median_raw_time_delta,
    ) = _rwd_timebase(_numeric(frame, time_column), declared_fps)
    excluded = {time_column.lower(), "events", "event", "name", "state"}
    signal_columns = [
        str(column) for column in frame.columns if str(column).lower() not in excluded
    ]
    if not signal_columns:
        raise ValueError("RWD OFRS Fluorescence.csv does not contain signal columns.")
    signal_column = next(
        (column for column in signal_columns if column.endswith(_RWD_SIGNAL_SUFFIXES)),
        signal_columns[0],
    )
    reference_column = next(
        (column for column in signal_columns if column.endswith(_RWD_REFERENCE_SUFFIXES)),
        None,
    )
    values = np.column_stack([_numeric(frame, column) for column in signal_columns])
    photometry = _photometry_recording(
        values=values,
        channel_names=signal_columns,
        timeline=_timeline_from_time(timestamps),
        source_type="rwd_ofrs",
        source_path=session_path,
        signal_channel=signal_column,
        reference_channel=reference_column,
        metadata={
            "sampling_rate_hz": sample_rate,
            "time_scale": time_scale,
            "time_scale_inference": time_scale_inference,
            "declared_fps_hz": declared_fps,
            "median_raw_time_delta": median_raw_time_delta,
            "metadata_line": metadata_line,
            "metadata": parsed_metadata,
        },
    )
    videos = {}
    video_path = session_path / "Video.mp4"
    if video_path.is_file():
        videos["behavior"] = {"path": str(video_path)}
    event_table, event_metadata = _rwd_event_table(session_path / "Events.csv", time_scale)
    return RecordingSession(
        session_id=session_path.name,
        signals={"photometry": photometry},
        videos=videos,
        events=event_table,
        metadata={
            "source": {"type": "rwd_ofrs", "path": str(session_path)},
            "time_column": time_column,
            "signal_columns": signal_columns,
            "events_csv": event_metadata,
        },
    )


def _walk_hdf5_datasets(handle: h5py.File) -> dict[str, h5py.Dataset]:
    datasets: dict[str, h5py.Dataset] = {}

    def visitor(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets[name] = obj

    handle.visititems(visitor)
    return datasets


def _numeric_1d_dataset(dataset: h5py.Dataset) -> bool:
    return dataset.ndim == 1 and np.issubdtype(dataset.dtype, np.number)


def _doric_dataset_selector(value: object, *, role: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Doric {role} must be a string or None.")
    if not value:
        raise ValueError(f"Doric {role} must be a non-empty string or None.")
    if value != value.strip():
        raise ValueError(f"Doric {role} must not contain surrounding whitespace.")
    return value


def is_doric_photometry_file(path: str | Path) -> bool:
    """Return whether ``path`` is a Doric photometry HDF5 container."""

    source_path = Path(path)
    if not source_path.is_file() or source_path.suffix.lower() != ".doric":
        return False
    try:
        with h5py.File(source_path, "r") as handle:
            return any(
                _numeric_1d_dataset(dataset) for dataset in _walk_hdf5_datasets(handle).values()
            )
    except OSError:
        return False


def find_first_doric_photometry_file(path: str | Path) -> Path | None:
    """Return the first Doric photometry file under ``path``."""

    return find_first_file(path, is_doric_photometry_file)


def _dataset_sample_rate_with_source(dataset: h5py.Dataset) -> tuple[float | None, str | None]:
    for key in ("SamplingRate", "sampling_rate", "Rate", "Fs", "fs"):
        if key in dataset.attrs:
            value = float(dataset.attrs[key])
            if np.isfinite(value) and value > 0:
                return value, f"{dataset.name.lstrip('/')}.attrs.{key}"
    return None, None


def _doric_time_dataset_sample_rate(timeline: Timeline, time_dataset: str) -> tuple[float, str]:
    sample_rate = timeline.estimated_sample_rate_hz
    if sample_rate is None:
        raise ValueError(f"Doric time dataset {time_dataset!r} must be uniformly sampled.")
    return float(sample_rate), f"{time_dataset}.timestamps_uniform"


# Wavelength tokens are the most specific/authoritative for Doric exports, so
# they outrank indicator/colour names and the generic "signal"/"control" words.
_DORIC_SIGNAL_TOKENS = ("470", "465", "560", "gcamp", "green", "signal")
_DORIC_REFERENCE_TOKENS = ("405", "415", "410", "isosbestic", "iso", "control", "reference")


def _preferred_doric_signal(paths: Sequence[str]) -> tuple[str, bool]:
    """Return the signal path and whether it matched a wavelength/name token."""
    for token in _DORIC_SIGNAL_TOKENS:
        for path in paths:
            if token in path.lower():
                return path, True
    return str(paths[0]), False


def _preferred_doric_reference(paths: Sequence[str], signal_path: str) -> tuple[str | None, bool]:
    """Return the reference path and whether it matched a token (vs storage order)."""
    candidates = [path for path in paths if path != signal_path]
    for token in _DORIC_REFERENCE_TOKENS:
        for path in candidates:
            if token in path.lower():
                return path, True
    return (candidates[0], False) if candidates else (None, False)


def read_doric_photometry(
    path: str | Path,
    *,
    signal_path: str | None = None,
    reference_path: str | None = None,
    time_path: str | None = None,
) -> RecordingSession:
    """Read a Doric ``.doric`` HDF5 photometry container."""

    source_path = Path(path)
    clean_signal_path = _doric_dataset_selector(signal_path, role="signal_path")
    clean_reference_path = _doric_dataset_selector(reference_path, role="reference_path")
    clean_time_path = _doric_dataset_selector(time_path, role="time_path")
    with h5py.File(source_path, "r") as handle:
        datasets = _walk_hdf5_datasets(handle)
        numeric_paths = [name for name, dataset in datasets.items() if _numeric_1d_dataset(dataset)]
        if not numeric_paths:
            raise ValueError(f"Doric file '{source_path}' contains no 1D numeric datasets.")
        resolved_time = clean_time_path
        if resolved_time is None:
            resolved_time = next((name for name in numeric_paths if "time" in name.lower()), None)
        signal_candidates = [name for name in numeric_paths if name != resolved_time]
        signal_matched = True
        signal_explicit = clean_signal_path is not None
        if clean_signal_path is not None:
            resolved_signal = clean_signal_path
        else:
            resolved_signal, signal_matched = _preferred_doric_signal(signal_candidates)
        if resolved_signal not in datasets:
            raise ValueError(f"Doric signal dataset {resolved_signal!r} was not found.")
        resolved_reference = clean_reference_path
        reference_matched = True
        reference_explicit = clean_reference_path is not None
        if resolved_reference is None and len(signal_candidates) > 1:
            resolved_reference, reference_matched = _preferred_doric_reference(
                signal_candidates, resolved_signal
            )
        if resolved_time is not None and resolved_signal == resolved_time:
            raise ValueError("Doric signal dataset must not be the time dataset.")
        if resolved_reference is not None:
            if resolved_reference == resolved_signal:
                raise ValueError("Doric reference dataset must differ from signal dataset.")
            if resolved_time is not None and resolved_reference == resolved_time:
                raise ValueError("Doric reference dataset must not be the time dataset.")
        # Signal/control identity is a swap risk worth surfacing when it falls back
        # to storage order rather than a wavelength/name token.
        channel_inference = (
            "explicit_dataset"
            if signal_explicit
            else ("wavelength_tokens" if signal_matched else "storage_order")
        )
        reference_channel_inference = None
        if resolved_reference is not None:
            reference_channel_inference = (
                "explicit_dataset"
                if reference_explicit
                else ("wavelength_tokens" if reference_matched else "storage_order")
            )
        root_attributes = {str(key): handle.attrs[key] for key in handle.attrs}

        signal = np.asarray(datasets[resolved_signal], dtype=np.float64)
        values = [signal]
        names = [resolved_signal]
        if resolved_reference is not None:
            if resolved_reference not in datasets:
                raise ValueError(f"Doric reference dataset {resolved_reference!r} was not found.")
            reference = np.asarray(datasets[resolved_reference], dtype=np.float64)
            if reference.shape != signal.shape:
                raise ValueError("Doric reference dataset must match signal dataset length.")
            values.append(reference)
            names.append(resolved_reference)
        if resolved_time is not None:
            timestamps = np.asarray(datasets[resolved_time], dtype=np.float64)
            if timestamps.shape != signal.shape:
                raise ValueError("Doric time dataset must match signal dataset length.")
            timeline = _timeline_from_time(timestamps)
            sample_rate, sample_rate_source = _doric_time_dataset_sample_rate(
                timeline,
                resolved_time,
            )
        else:
            sample_rate, sample_rate_source = _dataset_sample_rate_with_source(
                datasets[resolved_signal]
            )
            if sample_rate is None:
                raise ValueError(
                    "Doric file missing time dataset and signal sampling-rate attribute."
                )
            timeline = Timeline.from_sample_rate(n_samples=signal.size, sample_rate_hz=sample_rate)

    photometry = _photometry_recording(
        values=np.column_stack(values),
        channel_names=names,
        timeline=timeline,
        source_type="doric_photometry",
        source_path=source_path,
        signal_channel=resolved_signal,
        reference_channel=resolved_reference,
        metadata={
            "datasets": numeric_paths,
            "time_dataset": resolved_time,
            "sampling_rate_hz": sample_rate,
            "sampling_rate_source": sample_rate_source,
            "channel_inference": channel_inference,
            "reference_channel_inference": reference_channel_inference,
            "root_attributes": root_attributes,
        },
    )
    return RecordingSession(
        session_id=source_path.stem,
        signals={"photometry": photometry},
        metadata={
            "source": {"type": "doric_photometry", "path": str(source_path)},
            "sampling_rate_hz": sample_rate,
            "sampling_rate_source": sample_rate_source,
        },
    )


def _decode_h5_strings(values: np.ndarray) -> list[str]:
    result: list[str] = []
    for value in values.ravel():
        if isinstance(value, bytes | bytearray):
            result.append(value.decode("utf-8", errors="ignore"))
        else:
            result.append(str(value))
    return result


def _finite_vector(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _teleopto_sample_rate(num: np.ndarray, st1: np.ndarray, n_samples: int) -> float:
    sample_rate, _source = _teleopto_sample_rate_with_source(num, st1, n_samples)
    return sample_rate


def _teleopto_sample_rate_with_source(
    num: np.ndarray,
    st1: np.ndarray,
    n_samples: int,
) -> tuple[float, str]:
    if num.size > 1 and np.isfinite(num[1]) and num[1] > 0:
        return float(num[1]), "num[1]"
    if st1.size and np.isfinite(st1[0]) and st1[0] > 0:
        return float(n_samples / st1[0]), "st1_duration"
    raise ValueError("Teleopto H5 missing sampling rate and duration metadata.")


def _teleopto_ttl_edges(
    values: np.ndarray,
    sample_rate_hz: float,
    *,
    debounce_s: float = 0.005,
    min_width_s: float = 0.005,
    min_amplitude: float = 0.5,
) -> np.ndarray:
    signal = np.asarray(values, dtype=np.float64).ravel()
    if not np.isfinite(signal).all():
        raise ValueError("Teleopto H5 d2 TTL channel must contain only finite values.")
    if signal.size < 2:
        return np.asarray([], dtype=np.float64)
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("Teleopto H5 sampling rate must be positive and finite.")
    low = float(np.percentile(signal, 5))
    high = float(np.percentile(signal, 99.5))
    span = high - low
    if span < min_amplitude:
        return np.asarray([], dtype=np.float64)
    threshold = low + 0.5 * span
    digital = signal >= threshold
    min_width_samples = max(1, round(float(min_width_s) * sample_rate_hz))
    if min_width_samples > 1:
        edges = np.flatnonzero(np.diff(np.concatenate(([0], digital.view(np.int8), [0]))))
        for start, stop in edges.reshape((-1, 2)):
            if stop - start < min_width_samples:
                digital[start:stop] = False
    indices = np.where((~digital[:-1]) & digital[1:])[0] + 1
    if indices.size == 0:
        return np.asarray([], dtype=np.float64)
    debounce_samples = max(1, round(float(debounce_s) * sample_rate_hz))
    kept: list[int] = []
    previous = -debounce_samples
    for index in indices:
        if int(index) - previous >= debounce_samples:
            kept.append(int(index))
            previous = int(index)
    return np.asarray(kept, dtype=np.float64) / sample_rate_hz


def _teleopto_press_events(
    values: np.ndarray,
    sample_rate_hz: float,
    reinforcement_times: np.ndarray,
) -> dict[str, np.ndarray]:
    signal = np.asarray(values, dtype=np.float64).ravel()
    if signal.size < 2:
        return {}
    high = signal > 0.5
    rising = np.flatnonzero((~high[:-1]) & high[1:]) + 1
    falling = np.flatnonzero(high[:-1] & (~high[1:])) + 1
    on_times = rising.astype(np.float64) / sample_rate_hz
    off_times = falling.astype(np.float64) / sample_rate_hz
    aligned_on: list[float] = []
    aligned_off: list[float] = []
    off_index = 0
    for on_time in on_times:
        while off_index < off_times.size and off_times[off_index] <= on_time:
            off_index += 1
        if off_index >= off_times.size:
            break
        off_time = float(off_times[off_index])
        aligned_on.append(float(on_time))
        aligned_off.append(off_time)
        off_index += 1
    event_map: dict[str, np.ndarray] = {}
    if aligned_on:
        press_on = np.asarray(aligned_on, dtype=np.float64)
        event_map["press_on_times"] = press_on
        event_map["press_off_times"] = np.asarray(aligned_off, dtype=np.float64)
        if reinforcement_times.size:
            rewards = np.sort(np.asarray(reinforcement_times, dtype=np.float64))
            reinforced: list[float] = []
            non_reinforced: list[float] = []
            reward_index = 0
            for press in press_on:
                while reward_index < rewards.size and rewards[reward_index] < press:
                    reward_index += 1
                if (
                    reward_index < rewards.size
                    and 0.0 <= float(rewards[reward_index] - press) <= 1.0
                ):
                    reinforced.append(float(press))
                else:
                    non_reinforced.append(float(press))
            event_map["press_reinforced"] = np.asarray(reinforced, dtype=np.float64)
            event_map["press_non_reinforced"] = np.asarray(non_reinforced, dtype=np.float64)
    return event_map


def read_teleopto_h5(
    path: str | Path,
    *,
    extract_ttl_from_secondary: bool = True,
) -> RecordingSession:
    """Read a Teleopto/PMAT-style HDF5 photometry export."""

    source_path = Path(path)
    with h5py.File(source_path, "r") as handle:
        return parse_teleopto_h5_arrays(
            handle,
            session_id=source_path.stem,
            source_path=source_path,
            extract_ttl_from_secondary=extract_ttl_from_secondary,
        )


def is_teleopto_h5(path: str | Path) -> bool:
    """Return whether ``path`` has the Teleopto/PMAT HDF5 dataset contract."""

    source_path = Path(path)
    if not source_path.is_file() or not h5py.is_hdf5(source_path):
        return False
    try:
        with h5py.File(source_path, "r") as handle:
            return _TELEOPTO_REQUIRED_KEYS.issubset(handle.keys())
    except OSError:
        return False


def find_first_teleopto_h5(path: str | Path) -> Path | None:
    """Return the first Teleopto/PMAT HDF5 file under ``path``."""

    return find_first_file(path, is_teleopto_h5)


def parse_teleopto_h5_arrays(
    datasets: Mapping[str, object],
    *,
    session_id: str = "teleopto_h5",
    source_path: str | Path | None = None,
    extract_ttl_from_secondary: bool = True,
) -> RecordingSession:
    """Parse Teleopto/PMAT-style HDF5 datasets already loaded in memory."""

    missing = sorted(_TELEOPTO_REQUIRED_KEYS.difference(datasets.keys()))
    if missing:
        raise ValueError(f"Teleopto H5 missing required datasets: {missing}.")
    d1 = _finite_vector(datasets["d1"], "Teleopto H5 d1")
    d2 = _finite_vector(datasets["d2"], "Teleopto H5 d2") if "d2" in datasets else None
    num = _finite_vector(datasets["num"], "Teleopto H5 num")
    st1 = _finite_vector(datasets["st1"], "Teleopto H5 st1")
    labels = _decode_h5_strings(np.asarray(datasets["str"]))
    event_map: dict[str, np.ndarray] = {}
    for key in _TELEOPTO_EVENT_KEYS:
        if key not in datasets:
            continue
        values = _finite_vector(datasets[key], f"Teleopto H5 event channel {key}")
        if values.size:
            event_map[key] = np.sort(values)
    sample_rate, sample_rate_source = _teleopto_sample_rate_with_source(
        num,
        st1,
        d1.size,
    )
    timeline = Timeline.from_sample_rate(n_samples=d1.size, sample_rate_hz=sample_rate)
    names = [labels[0] if labels else "d1"]
    values = [d1]
    secondary = None
    if d2 is not None and d2.size:
        if d2.shape != d1.shape:
            raise ValueError("Teleopto d2 length must match d1 length.")
        secondary = labels[2] if len(labels) > 2 else "d2"
        names.append(secondary)
        values.append(d2)
        if extract_ttl_from_secondary:
            ttl_times = _teleopto_ttl_edges(d2, sample_rate)
            if ttl_times.size:
                event_map[f"{secondary}_ttl"] = ttl_times
            reinforcement_arrays = [event_map[key] for key in ("ar1", "ar2") if key in event_map]
            reinforcement = (
                np.concatenate(reinforcement_arrays)
                if reinforcement_arrays
                else np.asarray([], dtype=np.float64)
            )
            event_map.update(_teleopto_press_events(d2, sample_rate, reinforcement))
    resolved_source = Path(source_path) if source_path is not None else Path(session_id)
    photometry = _photometry_recording(
        values=np.column_stack(values),
        channel_names=names,
        timeline=timeline,
        source_type="teleopto_h5",
        source_path=resolved_source,
        signal_channel=names[0],
        reference_channel=None,
        metadata={
            "channel_labels": labels,
            "secondary_channel": secondary,
            "sampling_rate_hz": sample_rate,
            "sampling_rate_source": sample_rate_source,
            "event_label_scheme": "teleopto_native",
            "num": num.tolist(),
            "st1": st1.tolist(),
        },
    )
    return RecordingSession(
        session_id=session_id,
        signals={"photometry": photometry},
        events=_events_from_map(event_map, source_type="teleopto_h5", source_path=resolved_source),
        metadata={
            "source": {"type": "teleopto_h5", "path": str(resolved_source)},
            "sampling_rate_hz": sample_rate,
            "sampling_rate_source": sample_rate_source,
            "event_label_scheme": "teleopto_native",
        },
    )


def _tdt_module(module: Any | None) -> Any:
    if module is not None:
        return module
    try:
        return importlib.import_module("tdt")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TDT support requires the optional 'tdt' package. Install exp-pkg[tdt] "
            "or install tdt in the active environment."
        ) from exc


def _tdt_block_ids(root: Path, suffix: str) -> set[tuple[Path, str]]:
    return {
        (path.parent, path.stem.lower())
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == suffix
    }


def _tdt_file_dirs(root: Path, suffix: str) -> set[Path]:
    return {
        path.parent for path in root.rglob("*") if path.is_file() and path.suffix.lower() == suffix
    }


def _is_tdt_block_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    tsq_stems: set[str] = set()
    tev_stems: set[str] = set()
    has_sev = False
    for child in path.iterdir():
        if not child.is_file():
            continue
        suffix = child.suffix.lower()
        if suffix == ".tsq":
            tsq_stems.add(child.stem.lower())
        elif suffix == ".tev":
            tev_stems.add(child.stem.lower())
        elif suffix == ".sev":
            has_sev = True
    return bool(tsq_stems & tev_stems) or has_sev


def is_tdt_block(path: str | Path) -> bool:
    """Return whether ``path`` contains TDT block files."""

    root = Path(path)
    if not root.is_dir():
        return False
    tsq_blocks = _tdt_block_ids(root, ".tsq")
    tev_blocks = _tdt_block_ids(root, ".tev")
    if tsq_blocks & tev_blocks:
        return True
    return bool(_tdt_file_dirs(root, ".sev"))


def find_tdt_block_directories(
    path: str | Path,
    *,
    include_hidden_dirs: bool = False,
) -> list[Path]:
    """Return exact TDT block directories under ``path``."""

    root = Path(path)
    if not root.is_dir():
        return []

    entries: list[Path] = []

    def visit(directory: Path) -> None:
        if _is_tdt_block_directory(directory):
            entries.append(directory.resolve())
            return
        for child in sorted(directory.iterdir()):
            if not child.is_dir():
                continue
            if not include_hidden_dirs and child.name.startswith("."):
                continue
            visit(child)

    visit(root)
    return sorted(set(entries), key=lambda entry: str(entry))


def resolve_tdt_block_path(path: str | Path) -> Path:
    """Resolve ``path`` to one exact TDT block directory."""

    entries = find_tdt_block_directories(path)
    if not entries:
        raise ValueError(f"No TDT block found in {str(path)!r}.")
    if len(entries) == 1:
        return entries[0]

    entry_list = ", ".join(str(entry) for entry in entries)
    raise ValueError(
        f"Multiple TDT blocks found in {str(path)!r}; pass an exact block "
        f"directory. Found: {entry_list}."
    )


def _iter_tdt_streams(streams_obj: Any) -> Iterable[tuple[str, Any]]:
    for name, obj in getattr(streams_obj, "__dict__", {}).items():
        if hasattr(obj, "data") and hasattr(obj, "fs"):
            yield name, obj


_TDT_SIGNAL_TOKENS = (
    "465",
    "470",
    "gcamp",
    "dlight",
    "grab",
    "signal",
    "response",
    "fluorescence",
)
_TDT_REFERENCE_TOKENS = (
    "405",
    "410",
    "415",
    "iso",
    "isos",
    "isosbestic",
    "control",
    "reference",
)
_TDT_BROAD_PHOTOMETRY_TOKENS = ("pho", "fiber", "fi")


def _tdt_stream_rank(name: str, order: int) -> tuple[int, int, str]:
    lowered = name.lower().lstrip("_")
    if any(token in lowered for token in _TDT_SIGNAL_TOKENS):
        return (0, order, name)
    if any(token in lowered for token in _TDT_REFERENCE_TOKENS):
        return (1, order, name)
    if any(token in lowered for token in _TDT_BROAD_PHOTOMETRY_TOKENS):
        return (2, order, name)
    return (3, order, name)


def _rank_tdt_streams(names: Sequence[str]) -> list[str]:
    scored = [_tdt_stream_rank(name, order) for order, name in enumerate(names)]
    scored.sort()
    return [name for _category, _order, name in scored]


def _tdt_channel_inference(name: str, *, explicit: bool) -> str:
    if explicit:
        return "explicit_store"
    lowered = name.lower().lstrip("_")
    if any(token in lowered for token in _TDT_SIGNAL_TOKENS + _TDT_REFERENCE_TOKENS):
        return "wavelength_tokens"
    return "storage_order"


def _one_dimensional(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array
    if array.shape[0] < array.shape[1]:
        array = array.T
    return array[:, 0]


def _tdt_stream_start(stream: Any, *, allow_missing_zero: bool = False) -> tuple[float, str]:
    # A missing or non-finite start_time would silently shift every event
    # timestamp by the wrong offset, so fail fast instead of defaulting to 0.0.
    # A legitimately present, finite 0.0 (stream starts at recording onset) is
    # valid and returned as-is.
    if not hasattr(stream, "start_time"):
        if allow_missing_zero:
            return 0.0, "tdt.read_sev.t1_default"
        raise ValueError("TDT stream is missing start_time.")
    start = stream.start_time
    if isinstance(start, (list, tuple, np.ndarray)):
        flat = np.asarray(start, dtype=np.float64).ravel()
        if not flat.size:
            raise ValueError("TDT stream start_time is empty.")
        start = float(flat[0])
    try:
        value = float(start)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"TDT stream start_time is not numeric: {start!r}.") from exc
    if not np.isfinite(value):
        raise ValueError(f"TDT stream start_time must be finite, got {value!r}.")
    return value, "streams.<signal>.start_time"


def _tdt_store_name(value: object, *, role: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"TDT {role} must be a string.")
    if not value:
        raise ValueError(f"TDT {role} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"TDT {role} must not contain surrounding whitespace.")
    return value


def _tdt_event_store_names(event_stores: Sequence[str] | None) -> tuple[str, ...] | None:
    if event_stores is None:
        return None
    if isinstance(event_stores, str):
        raise TypeError("TDT event_stores must be a sequence of strings, not a string.")
    return tuple(
        _tdt_store_name(store, role=f"event_stores[{index}]")
        for index, store in enumerate(event_stores)
    )


def _tdt_read_block_kwargs(
    *,
    evtype: Sequence[str],
    stores: Sequence[str] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"evtype": list(evtype)}
    if stores is not None:
        kwargs["store"] = list(dict.fromkeys(stores))
    return kwargs


def _tdt_read_streams_and_epocs(
    module: Any,
    block_path: Path,
    *,
    signal_store: str | None,
    reference_store: str | None,
    event_stores: Sequence[str] | None,
) -> Any:
    stream_stores = (
        [signal_store, reference_store]
        if signal_store is not None and reference_store is not None
        else None
    )
    epoc_stores = list(event_stores) if event_stores is not None else None
    path = str(block_path)

    if stream_stores is None and epoc_stores is None:
        return module.read_block(
            path,
            **_tdt_read_block_kwargs(evtype=["streams", "epocs"]),
        )

    if stream_stores is not None and epoc_stores is not None:
        return module.read_block(
            path,
            **_tdt_read_block_kwargs(
                evtype=["streams", "epocs"],
                stores=[*stream_stores, *epoc_stores],
            ),
        )

    stream_data = module.read_block(
        path,
        **_tdt_read_block_kwargs(evtype=["streams"], stores=stream_stores),
    )
    if epoc_stores is not None and not epoc_stores:
        epocs = SimpleNamespace()
    else:
        epoc_data = module.read_block(
            path,
            **_tdt_read_block_kwargs(evtype=["epocs"], stores=epoc_stores),
        )
        epocs = getattr(epoc_data, "epocs", SimpleNamespace())
    return SimpleNamespace(
        streams=getattr(stream_data, "streams", stream_data),
        epocs=epocs,
        _xpkg_allow_missing_stream_start=not hasattr(stream_data, "streams"),
    )


def _tdt_epoc_times(
    epocs_obj: Any,
    event_stores: Sequence[str] | None,
    *,
    stream_start_s: float = 0.0,
) -> dict[str, np.ndarray]:
    events: dict[str, np.ndarray] = {}
    explicit_event_stores = event_stores is not None
    names = (
        tuple(event_stores)
        if explicit_event_stores
        else tuple(
            name
            for name, obj in getattr(epocs_obj, "__dict__", {}).items()
            if hasattr(obj, "onset") or hasattr(obj, "data")
        )
    )
    for name in names:
        obj = getattr(epocs_obj, name, None)
        if obj is None:
            if explicit_event_stores:
                raise ValueError(f"TDT event store {name!r} was not found.")
            continue
        if hasattr(obj, "onset"):
            values = np.asarray(obj.onset, dtype=np.float64).ravel()
        elif hasattr(obj, "data"):
            values = np.asarray(obj.data, dtype=np.float64).ravel()
        else:
            if explicit_event_stores:
                raise ValueError(f"TDT event store {name!r} has no onset or data timestamps.")
            continue
        values = values - float(stream_start_s)
        values = values[np.isfinite(values) & (values >= 0.0)]
        if values.size:
            events[str(name)] = np.sort(values)
    return events


def read_tdt_photometry_block(
    path: str | Path,
    *,
    signal_store: str | None = None,
    reference_store: str | None = None,
    event_stores: Sequence[str] | None = None,
    tdt_module: Any | None = None,
) -> RecordingSession:
    """Read photometry streams from a TDT tank/block folder."""

    block_path = Path(path)
    signal_store_name = (
        _tdt_store_name(signal_store, role="signal_store") if signal_store is not None else None
    )
    reference_store_name = (
        _tdt_store_name(reference_store, role="reference_store")
        if reference_store is not None
        else None
    )
    event_store_names = _tdt_event_store_names(event_stores)
    module = _tdt_module(tdt_module)
    data = _tdt_read_streams_and_epocs(
        module,
        block_path,
        signal_store=signal_store_name,
        reference_store=reference_store_name,
        event_stores=event_store_names,
    )
    has_streams_container = hasattr(data, "streams")
    streams_obj = data.streams if has_streams_container else data
    stream_map = dict(_iter_tdt_streams(streams_obj))
    if not stream_map:
        raise ValueError("TDT block did not contain readable stream data.")
    ranked = _rank_tdt_streams(tuple(stream_map))
    signal_explicit = signal_store_name is not None
    resolved_signal = signal_store_name or ranked[0]
    if resolved_signal not in stream_map:
        raise ValueError(f"TDT signal store {resolved_signal!r} was not found.")
    signal = stream_map[resolved_signal]
    values = [_one_dimensional(signal.data)]
    names = [resolved_signal]
    allow_missing_stream_start = bool(
        getattr(data, "_xpkg_allow_missing_stream_start", not has_streams_container)
    )
    stream_start_s, stream_start_source = _tdt_stream_start(
        signal,
        allow_missing_zero=allow_missing_stream_start,
    )
    resolved_stream_start_source = (
        stream_start_source.replace("<signal>", resolved_signal)
        if "<signal>" in stream_start_source
        else stream_start_source
    )
    sample_rate = float(getattr(signal, "fs", 0.0) or 0.0)
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError(f"TDT stream {resolved_signal!r} is missing a positive sampling rate.")
    sample_rate_source = f"streams.{resolved_signal}.fs"
    reference_explicit = reference_store_name is not None
    resolved_reference = reference_store_name
    if resolved_reference is None:
        resolved_reference = next((name for name in ranked if name != resolved_signal), None)
    if resolved_reference is not None:
        if resolved_reference not in stream_map:
            raise ValueError(f"TDT reference store {resolved_reference!r} was not found.")
        reference = _one_dimensional(stream_map[resolved_reference].data)
        if reference.shape == values[0].shape:
            values.append(reference)
            names.append(resolved_reference)
        elif reference_store_name is not None:
            raise ValueError("TDT reference store length must match signal store length.")
        else:
            resolved_reference = None
    timeline = Timeline.from_sample_rate(n_samples=values[0].size, sample_rate_hz=sample_rate)
    photometry = _photometry_recording(
        values=np.column_stack(values),
        channel_names=names,
        timeline=timeline,
        source_type="tdt_block",
        source_path=block_path,
        signal_channel=resolved_signal,
        reference_channel=resolved_reference,
        metadata={
            "stores": ranked,
            "stream_start_s": stream_start_s,
            "stream_start_source": resolved_stream_start_source,
            "sampling_rate_hz": sample_rate,
            "sampling_rate_source": sample_rate_source,
            "channel_inference": _tdt_channel_inference(
                resolved_signal,
                explicit=signal_explicit,
            ),
            "reference_channel_inference": (
                _tdt_channel_inference(
                    resolved_reference,
                    explicit=reference_explicit,
                )
                if resolved_reference is not None
                else None
            ),
        },
    )
    event_map = _tdt_epoc_times(
        getattr(data, "epocs", object()),
        event_store_names,
        stream_start_s=stream_start_s,
    )
    return RecordingSession(
        session_id=block_path.name,
        signals={"photometry": photometry},
        events=_events_from_map(event_map, source_type="tdt_block", source_path=block_path),
        metadata={
            "source": {"type": "tdt_block", "path": str(block_path)},
            "sampling_rate_hz": sample_rate,
            "sampling_rate_source": sample_rate_source,
            "stream_start_s": stream_start_s,
            "stream_start_source": resolved_stream_start_source,
        },
    )


__all__ = [
    "find_first_doric_photometry_file",
    "find_first_neurophotometrics_csv",
    "find_first_teleopto_h5",
    "find_tdt_block_directories",
    "is_teleopto_h5",
    "is_doric_photometry_file",
    "is_neurophotometrics_csv",
    "is_rwd_ofrs_session",
    "is_tdt_block",
    "neurophotometrics_channel_selection_from_label",
    "neurophotometrics_source_column_from_label",
    "read_doric_photometry",
    "read_neurophotometrics_csv",
    "parse_teleopto_h5_arrays",
    "read_pmat_events_csv",
    "read_pmat_photometry_csv",
    "read_rwd_ofrs_session",
    "read_tdt_photometry_block",
    "read_teleopto_h5",
    "resolve_tdt_block_path",
]
