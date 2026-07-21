"""NWB fiber-photometry reader."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeGuard

import h5py
import numpy as np

from xpkg.io.hdf5 import float_attribute
from xpkg.io.readers._discovery import find_first_file
from xpkg.io.readers._normalization import photometry_excitation as _excitation_from_name
from xpkg.model import (
    Event,
    EventTable,
    PhotometryChannel,
    PhotometryRecording,
    RecordingSession,
    SessionEventStream,
    SessionSignal,
    Timeline,
    TimeSeries,
)

_CONTROL_HINTS = ("isosbestic", "_405", "405nm", "405 nm", "reference", "control")
_SIGNAL_HINTS = ("dfoverf", "dff", "response", "green", "470", "465", "560", "signal", "gcamp")
_SIGNAL_EXCLUDE = (*_CONTROL_HINTS, "fluorescence")
_EVENT_CHANNEL_NAMES = frozenset(
    {
        "airpuff",
        "air_puff",
        "cue",
        "footshock",
        "lever",
        "lever_press",
        "leverpress",
        "nose_poke",
        "nosepoke",
        "reward",
        "shock",
        "stimulus",
        "tone",
    }
)
_NAMLAB_EVENTLOG_NAMESPACE = "ndx-eventlog-namlab"
_NAMLAB_EVENTLOG_TYPE = "Eventlog"
_NAMLAB_EVENT_TABLE_TYPE = "AnnotatedEventsTable"
_NAMLAB_DFF_NAMESPACE = "ndx-photometry-namlab"
_NAMLAB_DFF_TYPE = "DffSeries"
NonfinitePolicy = Literal["raise", "interpolate_sparse", "drop_rows"]
NwbPhotometryClassification = Literal["photometry", "event_only", "unsupported"]
_NAMLAB_DFF_MAX_RELATIVE_INTERVAL_DEVIATION = 0.10


@dataclass(frozen=True, slots=True)
class _Series:
    name: str
    path: str
    group: h5py.Group


@dataclass(frozen=True, slots=True)
class NwbPhotometryInspection:
    """Read-only structural and numerical inspection of one NWB asset."""

    path: str
    size_bytes: int
    classification: NwbPhotometryClassification
    candidate_series_count: int
    signal_series: str | None
    control_series: str | None
    signal: Mapping[str, Any] | None
    control: Mapping[str, Any] | None
    namlab_eventlogs: tuple[Mapping[str, Any], ...]
    issue_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable inspection payload."""
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "classification": self.classification,
            "candidate_series_count": self.candidate_series_count,
            "signal_series": self.signal_series,
            "control_series": self.control_series,
            "signal": None if self.signal is None else dict(self.signal),
            "control": None if self.control is None else dict(self.control),
            "namlab_eventlogs": [dict(item) for item in self.namlab_eventlogs],
            "issue_codes": list(self.issue_codes),
        }


def _decode(value: object) -> object:
    if isinstance(value, bytes | bytearray):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, np.generic):
        return value.item()
    array = np.asarray(value)
    if array.shape == ():
        return _decode(array.item())
    return value


def _scalar_dataset(group: h5py.Group, name: str) -> object:
    dataset = group[name]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"NWB dataset '{dataset.name}' is not a dataset.")
    return _decode(np.asarray(dataset).reshape(-1)[0])


def _series_candidates(handle: h5py.File) -> dict[str, _Series]:
    candidates: dict[str, _Series] = {}

    def visit(root_name: str) -> None:
        root = handle.get(root_name)
        if not isinstance(root, h5py.Group):
            return

        def visitor(name: str, obj: object) -> None:
            if isinstance(obj, h5py.Group) and isinstance(obj.get("data"), h5py.Dataset):
                candidate_name = Path(name).name
                key = candidate_name
                if key in candidates:
                    key = name
                candidates[key] = _Series(
                    name=candidate_name,
                    path=f"{root_name}/{name}",
                    group=obj,
                )

        root.visititems(visitor)

    visit("acquisition")
    visit("processing")
    return candidates


def _select_series(
    candidates: dict[str, _Series],
    hints: tuple[str, ...],
    *,
    exclude: tuple[str, ...] = (),
) -> str | None:
    best: tuple[int, str] | None = None
    for key, series in candidates.items():
        haystack = f"{series.name} {series.path}".lower()
        if any(token in haystack for token in exclude):
            continue
        rank = next((index for index, token in enumerate(hints) if token in haystack), None)
        if rank is not None and (best is None or rank < best[0]):
            best = (rank, key)
    return best[1] if best is not None else None


def _shared_prefix_length(left: str, right: str) -> int:
    shared = 0
    for left_char, right_char in zip(left, right, strict=False):
        if left_char != right_char:
            break
        shared += 1
    return shared


def _select_control(candidates: dict[str, _Series], signal_key: str) -> str | None:
    signal = candidates[signal_key]
    controls = [
        key
        for key, series in candidates.items()
        if key != signal_key
        and any(token in f"{series.name} {series.path}".lower() for token in _CONTROL_HINTS)
    ]
    if not controls:
        return None
    return max(
        controls,
        key=lambda key: _shared_prefix_length(signal.path.lower(), candidates[key].path.lower()),
    )


def is_nwb_photometry_file(path: str | Path) -> bool:
    """Return whether ``path`` is an NWB HDF5 file with photometry series."""

    source_path = Path(path)
    if not source_path.is_file() or source_path.suffix.lower() != ".nwb":
        return False
    try:
        with h5py.File(str(source_path), "r") as handle:
            candidates = _series_candidates(handle)
            return _select_series(candidates, _SIGNAL_HINTS, exclude=_SIGNAL_EXCLUDE) is not None
    except OSError:
        return False


def find_first_nwb_photometry_file(path: str | Path) -> Path | None:
    """Return the first NWB photometry file under ``path``."""

    return find_first_file(path, is_nwb_photometry_file)


_MAX_NONFINITE_FRACTION = 0.01


def _series_values(
    series: _Series,
    *,
    nonfinite_policy: NonfinitePolicy,
    max_nonfinite_fraction: float,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    data = np.asarray(series.group["data"], dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape((-1, 1))
    if data.ndim != 2:
        raise ValueError(f"NWB TimeSeries '{series.path}' data must be 1D or 2D, got {data.shape}.")
    if data.shape[0] == 0:
        raise ValueError(f"NWB TimeSeries '{series.path}' contains no samples.")
    if np.isfinite(data).all():
        return data, None
    if nonfinite_policy == "raise":
        raise ValueError(
            f"NWB TimeSeries '{series.path}' contains non-finite samples. "
            "Pass nonfinite_policy='interpolate_sparse' to repair sparse internal "
            "gaps or nonfinite_policy='drop_rows' to preserve only stored finite rows."
        )
    if nonfinite_policy == "drop_rows":
        return data, None
    if nonfinite_policy != "interpolate_sparse":
        raise ValueError(
            "nonfinite_policy must be 'raise', 'interpolate_sparse', or "
            f"'drop_rows', got {nonfinite_policy!r}."
        )
    return _repair_sparse_nonfinite(data, series.path, max_nonfinite_fraction)


def _repair_sparse_nonfinite(
    data: np.ndarray,
    path: str,
    max_nonfinite_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not np.isfinite(max_nonfinite_fraction) or not 0.0 <= max_nonfinite_fraction <= 1.0:
        raise ValueError(
            "max_nonfinite_fraction must be finite and between 0 and 1, "
            f"got {max_nonfinite_fraction!r}."
        )
    nonfinite = ~np.isfinite(data)
    fraction = float(nonfinite.mean())
    if fraction > max_nonfinite_fraction:
        raise ValueError(
            f"NWB TimeSeries '{path}' is {fraction:.1%} non-finite "
            f"(>{max_nonfinite_fraction:.0%}); refusing to interpolate corrupt data."
        )
    repaired = np.array(data, dtype=np.float64)
    index = np.arange(repaired.shape[0])
    for column in range(repaired.shape[1]):
        finite = ~nonfinite[:, column]
        if not finite.any():
            raise ValueError(f"NWB TimeSeries '{path}' column {column} has no finite samples.")
        if not finite[0] or not finite[-1]:
            raise ValueError(
                f"NWB TimeSeries '{path}' column {column} has non-finite edge samples; "
                "cannot interpolate without extrapolation."
            )
        repaired[~finite, column] = np.interp(
            index[~finite], index[finite], repaired[finite, column]
        )
    return repaired, {
        "policy": "interpolate_sparse",
        "nonfinite_samples": int(nonfinite.sum()),
        "nonfinite_fraction": fraction,
        "max_nonfinite_fraction": float(max_nonfinite_fraction),
    }


def _series_timeline(series: _Series, n_samples: int) -> Timeline:
    group = series.group
    if "timestamps" in group:
        stamps = np.asarray(group["timestamps"], dtype=np.float64).reshape(-1)
        if stamps.size != n_samples:
            raise ValueError(
                f"NWB TimeSeries '{series.path}' timestamps length must match data samples."
            )
        return Timeline(timestamps_s=stamps)
    if "starting_time" not in group:
        raise ValueError(f"NWB TimeSeries '{series.path}' is missing timestamps or starting_time.")
    starting_time = group["starting_time"]
    if not isinstance(starting_time, h5py.Dataset):
        raise ValueError(f"NWB TimeSeries '{series.path}' starting_time is not a dataset.")
    rate = starting_time.attrs.get("rate")
    if rate is None:
        raise ValueError(f"NWB TimeSeries '{series.path}' starting_time is missing rate.")
    start = float(np.asarray(starting_time).reshape(-1)[0])
    sample_rate = float_attribute(rate, name=f"{series.path}.starting_time.rate")
    if not np.isfinite(start):
        raise ValueError(f"NWB TimeSeries '{series.path}' starting_time is not finite.")
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError(f"NWB TimeSeries '{series.path}' rate must be positive and finite.")
    return Timeline.from_sample_rate(
        n_samples=n_samples,
        sample_rate_hz=sample_rate,
        start_s=start,
    )


def _sampling_rate_with_source(
    timeline: Timeline,
    series_path: str,
) -> tuple[float | None, str | None]:
    if timeline.sample_rate_hz is not None:
        return float(timeline.sample_rate_hz), f"{series_path}.starting_time.rate"
    sample_rate = timeline.estimated_sample_rate_hz
    if sample_rate is None:
        return None, None
    return float(sample_rate), f"{series_path}.timestamps_uniform"


def _read_series(
    series: _Series,
    *,
    nonfinite_policy: NonfinitePolicy,
    max_nonfinite_fraction: float,
) -> tuple[np.ndarray, Timeline, dict[str, Any] | None]:
    values, repair = _series_values(
        series,
        nonfinite_policy=nonfinite_policy,
        max_nonfinite_fraction=max_nonfinite_fraction,
    )
    timestamp_nonfinite = False
    if nonfinite_policy == "drop_rows" and "timestamps" in series.group:
        stamps = np.asarray(series.group["timestamps"], dtype=np.float64).reshape(-1)
        timestamp_nonfinite = not np.isfinite(stamps).all()
    if nonfinite_policy == "drop_rows" and (
        not np.isfinite(values).all() or timestamp_nonfinite
    ):
        return _drop_nonfinite_rows(
            series,
            values,
            max_nonfinite_fraction=max_nonfinite_fraction,
        )
    return values, _series_timeline(series, values.shape[0]), repair


def _is_namlab_dff_series(series: _Series) -> bool:
    return (
        str(_decode(series.group.attrs.get("namespace", "")))
        == _NAMLAB_DFF_NAMESPACE
        and str(_decode(series.group.attrs.get("neurodata_type", "")))
        == _NAMLAB_DFF_TYPE
    )


def _dropped_row_timeline(
    series: _Series,
    *,
    retained: np.ndarray,
) -> tuple[Timeline, str]:
    if "timestamps" not in series.group:
        original = _series_timeline(series, retained.size)
        if original.sample_rate_hz is None:
            raise RuntimeError(
                f"NWB TimeSeries '{series.path}' rate timeline lost its sampling rate."
            )
        return (
            Timeline(
                timestamps_s=original.timestamps_s[retained],
                sample_rate_hz=original.sample_rate_hz,
            ),
            f"{series.path}.starting_time.rate",
        )

    stamps = np.asarray(series.group["timestamps"], dtype=np.float64).reshape(-1)
    if stamps.size != retained.size:
        raise ValueError(
            f"NWB TimeSeries '{series.path}' timestamps length must match data samples."
        )
    finite_pairs = np.isfinite(stamps[:-1]) & np.isfinite(stamps[1:])
    intervals = np.diff(stamps)[finite_pairs]
    if intervals.size == 0 or not np.all(intervals > 0.0):
        raise ValueError(
            f"NWB TimeSeries '{series.path}' cannot establish a positive nominal "
            "sample interval from stored timestamps."
        )
    median_interval = float(np.median(intervals))
    if np.allclose(intervals, median_interval, rtol=1e-6, atol=1e-9):
        sample_interval = median_interval
        rate_source = f"{series.path}.timestamps_uniform_before_row_drop"
    elif _is_namlab_dff_series(series):
        sample_interval = float(np.mean(intervals))
        relative_deviation = np.abs(intervals - sample_interval) / sample_interval
        maximum_deviation = float(np.max(relative_deviation))
        if maximum_deviation > _NAMLAB_DFF_MAX_RELATIVE_INTERVAL_DEVIATION:
            raise ValueError(
                f"NWB NAML DffSeries '{series.path}' finite timestamp intervals "
                f"deviate by {maximum_deviation:.1%}, exceeding the "
                f"{_NAMLAB_DFF_MAX_RELATIVE_INTERVAL_DEVIATION:.0%} nominal-clock "
                "limit."
            )
        rate_source = (
            f"{series.path}.timestamps_mean_finite_adjacent_namlab_dff"
        )
    else:
        raise ValueError(
            f"NWB TimeSeries '{series.path}' has irregular timestamps and no "
            "format-specific nominal-rate contract."
        )
    return (
        Timeline(
            timestamps_s=stamps[retained],
            sample_rate_hz=1.0 / sample_interval,
        ),
        rate_source,
    )


def _drop_nonfinite_rows(
    series: _Series,
    data: np.ndarray,
    *,
    max_nonfinite_fraction: float,
) -> tuple[np.ndarray, Timeline, dict[str, Any]]:
    if not np.isfinite(max_nonfinite_fraction) or not 0.0 <= max_nonfinite_fraction <= 1.0:
        raise ValueError(
            "max_nonfinite_fraction must be finite and between 0 and 1, "
            f"got {max_nonfinite_fraction!r}."
        )
    data_finite = np.isfinite(data).all(axis=1)
    if "timestamps" in series.group:
        stamps = np.asarray(series.group["timestamps"], dtype=np.float64).reshape(-1)
        if stamps.size != data.shape[0]:
            raise ValueError(
                f"NWB TimeSeries '{series.path}' timestamps length must match data samples."
            )
        timestamp_finite = np.isfinite(stamps)
    else:
        timestamp_finite = np.ones(data.shape[0], dtype=bool)
    retained = data_finite & timestamp_finite
    dropped_count = int((~retained).sum())
    dropped_fraction = dropped_count / int(retained.size)
    if dropped_fraction > max_nonfinite_fraction:
        raise ValueError(
            f"NWB TimeSeries '{series.path}' requires dropping {dropped_fraction:.1%} "
            f"of rows (>{max_nonfinite_fraction:.0%}); refusing the requested repair."
        )
    if int(retained.sum()) < 2:
        raise ValueError(
            f"NWB TimeSeries '{series.path}' has fewer than two finite rows."
        )
    timeline, rate_source = _dropped_row_timeline(series, retained=retained)
    sample_rate = timeline.sample_rate_hz
    if sample_rate is None:
        raise RuntimeError(
            f"NWB TimeSeries '{series.path}' row-drop timeline lost its sample rate."
        )
    expected_step = 1.0 / sample_rate
    output_intervals = np.diff(timeline.timestamps_s)
    timeline_kind = (
        "uniform"
        if np.allclose(output_intervals, expected_step, rtol=1e-6, atol=1e-9)
        else "irregular_preserved"
    )
    return data[retained], timeline, {
        "policy": "drop_rows",
        "original_sample_count": int(data.shape[0]),
        "retained_sample_count": int(retained.sum()),
        "dropped_row_count": dropped_count,
        "dropped_row_fraction": dropped_fraction,
        "max_nonfinite_fraction": float(max_nonfinite_fraction),
        "signal_nonfinite_row_count": int((~data_finite).sum()),
        "timestamp_nonfinite_count": int((~timestamp_finite).sum()),
        "sampling_rate_hz": sample_rate,
        "sampling_rate_source": rate_source,
        "timeline_kind": timeline_kind,
    }


def _timelines_match(left: Timeline, right: Timeline) -> bool:
    if left.n_samples != right.n_samples:
        return False
    return bool(np.allclose(left.timestamps_s, right.timestamps_s, rtol=1e-9, atol=1e-12))


def _channel_names(base: str, n_columns: int) -> list[str]:
    return [base, *(f"{base}_fiber{index}" for index in range(1, n_columns))]


def _photometry_recording(
    *,
    signal_series: _Series,
    signal_values: np.ndarray,
    timeline: Timeline,
    source_path: Path,
    control_series: _Series | None = None,
    control_values: np.ndarray | None = None,
    signal_nonfinite_repair: dict[str, Any] | None = None,
    control_nonfinite_repair: dict[str, Any] | None = None,
) -> PhotometryRecording:
    signal_names = _channel_names(signal_series.name, signal_values.shape[1])
    values = signal_values
    reference_channel = None
    source_paths = [signal_series.path] * len(signal_names)
    if control_series is not None and control_values is not None:
        control_names = _channel_names(control_series.name, control_values.shape[1])
        values = np.column_stack([signal_values, control_values])
        reference_channel = control_names[0]
        source_paths.extend([control_series.path] * len(control_names))
    else:
        control_names = []
    channel_names = [*signal_names, *control_names]
    channels = tuple(
        PhotometryChannel(
            name=name,
            unit="a.u.",
            excitation=_excitation_from_name(name),
            metadata={"source_path": source_path},
        )
        for name, source_path in zip(channel_names, source_paths, strict=True)
    )
    series = TimeSeries(
        values=values,
        channels=channels,
        timeline=timeline,
        name="photometry",
        provenance={"source": {"type": "nwb_photometry", "path": str(source_path)}},
    )
    nonfinite_repairs: dict[str, Any] = {}
    if signal_nonfinite_repair is not None:
        nonfinite_repairs[signal_series.path] = signal_nonfinite_repair
    if control_series is not None and control_nonfinite_repair is not None:
        nonfinite_repairs[control_series.path] = control_nonfinite_repair
    sampling_rate_hz, sampling_rate_source = _sampling_rate_with_source(
        timeline,
        signal_series.path,
    )
    timeline_kind = "irregular"
    if sampling_rate_hz is not None:
        expected_step = 1.0 / sampling_rate_hz
        timeline_kind = (
            "uniform"
            if timeline.n_samples < 2
            or np.allclose(
                np.diff(timeline.timestamps_s),
                expected_step,
                rtol=1e-6,
                atol=1e-9,
            )
            else "irregular_preserved"
        )
    if signal_nonfinite_repair is not None:
        repair_rate_source = signal_nonfinite_repair.get("sampling_rate_source")
        repair_timeline_kind = signal_nonfinite_repair.get("timeline_kind")
        if isinstance(repair_rate_source, str):
            sampling_rate_source = repair_rate_source
        if isinstance(repair_timeline_kind, str):
            timeline_kind = repair_timeline_kind
    metadata: dict[str, Any] = {
        "source_type": "nwb_photometry",
        "signal_series": signal_series.path,
        "control_series": None if control_series is None else control_series.path,
        "signal_is_dff": any(
            token in f"{signal_series.name} {signal_series.path}".lower()
            for token in ("dfoverf", "dff")
        ),
        "n_fibers": int(signal_values.shape[1]),
        "timeline_kind": timeline_kind,
    }
    if sampling_rate_hz is not None:
        metadata["sampling_rate_hz"] = sampling_rate_hz
        metadata["sampling_rate_source"] = sampling_rate_source
    if nonfinite_repairs:
        metadata["nonfinite_repairs"] = nonfinite_repairs
    return PhotometryRecording(
        series=series,
        signal_channel=signal_names[0],
        reference_channel=reference_channel,
        metadata=metadata,
    )


def _pulse_onsets(
    values: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    frac: float = 0.5,
    refractory_s: float = 0.5,
    min_peak: float = 0.2,
) -> np.ndarray:
    signal = np.asarray(values, dtype=np.float64).reshape(-1)
    if signal.size != timestamps_s.size:
        raise ValueError("NWB event channel length must match its timeline.")
    if signal.size < 2:
        return np.asarray([], dtype=np.float64)
    if not np.isfinite(signal).all():
        raise ValueError("NWB event channel contains non-finite samples.")
    baseline = float(np.median(signal))
    peak = float(np.max(signal))
    if peak - baseline < min_peak:
        return np.asarray([], dtype=np.float64)
    high = signal > baseline + frac * (peak - baseline)
    rises = np.flatnonzero((~high[:-1]) & high[1:]) + 1
    if rises.size == 0:
        return np.asarray([], dtype=np.float64)
    times = np.asarray(timestamps_s[rises], dtype=np.float64)
    kept = [float(times[0])]
    for time_s in times[1:]:
        if float(time_s) - kept[-1] >= refractory_s:
            kept.append(float(time_s))
    return np.asarray(kept, dtype=np.float64)


def _event_channel_events(
    candidates: dict[str, _Series],
    *,
    source_path: Path,
    nonfinite_policy: NonfinitePolicy,
    max_nonfinite_fraction: float,
) -> list[Event]:
    rows: list[Event] = []
    for series in candidates.values():
        if series.path.split("/", maxsplit=1)[0] != "acquisition":
            continue
        if series.name.lower() not in _EVENT_CHANNEL_NAMES:
            continue
        values, timeline, _repair = _read_series(
            series,
            nonfinite_policy=nonfinite_policy,
            max_nonfinite_fraction=max_nonfinite_fraction,
        )
        if values.shape[1] != 1:
            raise ValueError(
                f"NWB event channel '{series.path}' must be one-dimensional, "
                f"got {values.shape[1]} columns."
            )
        onsets = _pulse_onsets(values[:, 0], timeline.timestamps_s)
        for onset in onsets:
            rows.append(
                Event(
                    event_id=f"nwb-pulse-{len(rows):06d}",
                    kind="stimulus",
                    start_s=float(onset),
                    label=series.name,
                    metadata={"source": {"type": "nwb_photometry", "path": str(source_path)}},
                )
            )
    return rows


def _annotation_series_events(
    candidates: dict[str, _Series],
    *,
    source_path: Path,
) -> list[Event]:
    rows: list[Event] = []
    for series in candidates.values():
        if series.path.split("/", maxsplit=1)[0] != "acquisition":
            continue
        neurodata_type = str(_decode(series.group.attrs.get("neurodata_type", "")))
        if (
            series.name.lower() not in {"events", "annotations"}
            and neurodata_type != "AnnotationSeries"
        ):
            continue
        if "timestamps" not in series.group:
            raise ValueError(f"NWB AnnotationSeries '{series.path}' is missing timestamps.")
        labels = np.asarray(series.group["data"]).reshape(-1)
        stamps = np.asarray(series.group["timestamps"], dtype=np.float64).reshape(-1)
        if labels.size != stamps.size:
            raise ValueError(f"NWB AnnotationSeries '{series.path}' label/time lengths differ.")
        if not np.isfinite(stamps).all():
            raise ValueError(f"NWB AnnotationSeries '{series.path}' contains non-finite times.")
        for index, (label, stamp) in enumerate(zip(labels, stamps, strict=True)):
            decoded_label = _nwb_annotation_label(label, path=series.path, row=index)
            rows.append(
                Event(
                    event_id=f"nwb-annotation-{len(rows):06d}",
                    kind="event",
                    start_s=float(stamp),
                    label=decoded_label,
                    metadata={"source": {"type": "nwb_photometry", "path": str(source_path)}},
                )
            )
    return rows


def _nwb_annotation_label(value: object, *, path: str, row: int) -> str:
    decoded = _decode(value)
    message = (
        f"NWB AnnotationSeries '{path}' label at row {row} must be a non-empty "
        "string without surrounding whitespace."
    )
    if not isinstance(decoded, str):
        raise ValueError(message)
    if not decoded or decoded != decoded.strip():
        raise ValueError(message)
    return decoded


def _analysis_timestamp_events(handle: h5py.File, *, source_path: Path) -> list[Event]:
    root = handle.get("analysis")
    if not isinstance(root, h5py.Group):
        return []
    rows: list[Event] = []

    def visitor(name: str, obj: object) -> None:
        if not isinstance(obj, h5py.Group) or "timestamp" not in obj or "data" in obj:
            return
        timestamp_dataset = obj["timestamp"]
        if not isinstance(timestamp_dataset, h5py.Dataset):
            raise ValueError(f"NWB analysis event table '{obj.name}' timestamp is not a dataset.")
        stamps = np.asarray(timestamp_dataset, dtype=np.float64).reshape(-1)
        if not np.isfinite(stamps).all():
            raise ValueError(f"NWB analysis event table '{obj.name}' contains non-finite times.")
        label = Path(name).name
        for stamp in np.sort(stamps):
            rows.append(
                Event(
                    event_id=f"nwb-analysis-{len(rows):06d}",
                    kind="event",
                    start_s=float(stamp),
                    label=label,
                    metadata={"source": {"type": "nwb_photometry", "path": str(source_path)}},
                )
            )

    root.visititems(visitor)
    return rows


def _namlab_contract_group(
    obj: object,
    *,
    namespace: str,
    neurodata_type: str,
) -> TypeGuard[h5py.Group]:
    return bool(
        isinstance(obj, h5py.Group)
        and _decode(obj.attrs.get("namespace", "")) == namespace
        and _decode(obj.attrs.get("neurodata_type", "")) == neurodata_type
    )


def _namlab_vector(group: h5py.Group, name: str) -> np.ndarray:
    obj = group.get(name)
    if not isinstance(obj, h5py.Dataset):
        raise ValueError(f"NWB namlab object '{group.name}' is missing dataset '{name}'.")
    values = np.asarray(obj)
    if values.ndim != 1:
        raise ValueError(
            f"NWB namlab dataset '{obj.name}' must be one-dimensional, got {values.shape}."
        )
    return values


def _namlab_event_table(
    eventlog: h5py.Group,
    tables: list[h5py.Group],
) -> h5py.Group:
    linked = eventlog.get("eventtable")
    if linked is not None:
        if not _namlab_contract_group(
            linked,
            namespace=_NAMLAB_EVENTLOG_NAMESPACE,
            neurodata_type=_NAMLAB_EVENT_TABLE_TYPE,
        ):
            raise ValueError(
                f"NWB namlab Eventlog '{eventlog.name}' has an invalid eventtable link."
            )
        return linked
    if len(tables) != 1:
        raise ValueError(
            f"NWB namlab Eventlog '{eventlog.name}' has no eventtable link and the "
            f"acquisition contains {len(tables)} AnnotatedEventsTable objects; "
            "the event-index mapping is ambiguous."
        )
    return tables[0]


def _namlab_label_map(table: h5py.Group) -> dict[int, str]:
    indices = _namlab_vector(table, "event_index")
    descriptions = _namlab_vector(table, "event_description")
    if indices.size != descriptions.size:
        raise ValueError(f"NWB namlab event table '{table.name}' index/description lengths differ.")
    if indices.dtype.kind not in "iu":
        raise ValueError(f"NWB namlab dataset '{table.name}/event_index' must contain integers.")
    labels: dict[int, str] = {}
    for row, (raw_index, raw_description) in enumerate(zip(indices, descriptions, strict=True)):
        event_index = int(raw_index)
        description = _decode(raw_description)
        if (
            not isinstance(description, str)
            or not description
            or description != description.strip()
        ):
            raise ValueError(
                f"NWB namlab event description at row {row} in '{table.name}' must be "
                "a non-empty string without surrounding whitespace."
            )
        if event_index in labels:
            raise ValueError(
                f"NWB namlab event table '{table.name}' repeats event index {event_index}."
            )
        labels[event_index] = description
    return labels


def _validate_namlab_clock_signal(signal: _Series) -> None:
    if not _namlab_contract_group(
        signal.group,
        namespace=_NAMLAB_DFF_NAMESPACE,
        neurodata_type=_NAMLAB_DFF_TYPE,
    ) or not isinstance(signal.group.get("timestamps"), h5py.Dataset):
        raise ValueError(
            "NWB ndx-eventlog-namlab events require the selected signal to be an "
            "ndx-photometry-namlab DffSeries with synchronized timestamps."
        )


def _namlab_eventlog_events(
    handle: h5py.File,
    *,
    source_path: Path,
    signal: _Series,
) -> tuple[list[Event], list[dict[str, Any]]]:
    acquisition = handle.get("acquisition")
    if not isinstance(acquisition, h5py.Group):
        return [], []
    eventlogs = [
        obj
        for obj in acquisition.values()
        if _namlab_contract_group(
            obj,
            namespace=_NAMLAB_EVENTLOG_NAMESPACE,
            neurodata_type=_NAMLAB_EVENTLOG_TYPE,
        )
    ]
    if not eventlogs:
        return [], []
    _validate_namlab_clock_signal(signal)
    tables = [
        obj
        for obj in acquisition.values()
        if _namlab_contract_group(
            obj,
            namespace=_NAMLAB_EVENTLOG_NAMESPACE,
            neurodata_type=_NAMLAB_EVENT_TABLE_TYPE,
        )
    ]
    rows: list[Event] = []
    provenance: list[dict[str, Any]] = []
    for log_number, eventlog in enumerate(eventlogs):
        times = _namlab_vector(eventlog, "eventtime")
        indices = _namlab_vector(eventlog, "eventindex")
        flags = _namlab_vector(eventlog, "nonsolenoidflag")
        if not (times.size == indices.size == flags.size):
            raise ValueError(f"NWB namlab Eventlog '{eventlog.name}' column lengths differ.")
        if times.dtype.kind not in "fiu" or not np.isfinite(times).all():
            raise ValueError(
                f"NWB namlab dataset '{eventlog.name}/eventtime' must contain finite numbers."
            )
        times_s = np.asarray(times, dtype=np.float64)
        if times_s.size > 1 and np.any(np.diff(times_s) < 0.0):
            raise ValueError(
                f"NWB namlab dataset '{eventlog.name}/eventtime' must be non-decreasing."
            )
        if indices.dtype.kind not in "iu" or flags.dtype.kind not in "iub":
            raise ValueError(
                f"NWB namlab Eventlog '{eventlog.name}' indices and flags must be integers."
            )
        table = _namlab_event_table(eventlog, tables)
        labels = _namlab_label_map(table)
        time_dataset = eventlog["eventtime"]
        assert isinstance(time_dataset, h5py.Dataset)
        declared_unit = _decode(time_dataset.attrs.get("unit", ""))
        if declared_unit not in {"milliseconds", "seconds"}:
            raise ValueError(
                f"NWB namlab dataset '{time_dataset.name}' has unsupported unit "
                f"{declared_unit!r}; expected 'milliseconds' or 'seconds'."
            )
        for source_row, (time_s, raw_index, raw_flag) in enumerate(
            zip(times_s, indices, flags, strict=True)
        ):
            event_index = int(raw_index)
            if event_index not in labels:
                raise ValueError(
                    f"NWB namlab Eventlog '{eventlog.name}' row {source_row} references "
                    f"unknown event index {event_index}."
                )
            rows.append(
                Event(
                    event_id=f"nwb-namlab-{log_number:03d}-{source_row:06d}",
                    kind="event",
                    start_s=float(time_s),
                    label=labels[event_index],
                    metadata={
                        "source": {"type": "nwb_photometry", "path": str(source_path)},
                        "source_path": eventlog.name,
                        "source_row": source_row,
                        "event_index": event_index,
                        "nonsolenoid_flag": int(raw_flag),
                        "declared_time_unit": declared_unit,
                        "interpreted_time_unit": "seconds",
                    },
                )
            )
        provenance.append(
            {
                "source_path": eventlog.name,
                "event_table_path": table.name,
                "event_count": int(times_s.size),
                "declared_time_unit": declared_unit,
                "interpreted_time_unit": "seconds",
                "clock_signal_path": signal.path,
                "clock_contract": "anccr_processed_dff_timestamps",
                "source_analysis": "https://github.com/namboodirilab/ANCCR",
            }
        )
    return rows, provenance


def _invalid_runs(mask: np.ndarray) -> list[dict[str, int | str]]:
    invalid = np.asarray(mask, dtype=bool).reshape(-1)
    if not invalid.any():
        return []
    starts = np.flatnonzero(invalid & ~np.r_[False, invalid[:-1]])
    stops = np.flatnonzero(invalid & ~np.r_[invalid[1:], False]) + 1
    runs: list[dict[str, int | str]] = []
    for start_raw, stop_raw in zip(starts, stops, strict=True):
        start = int(start_raw)
        stop = int(stop_raw)
        if start == 0:
            position = "leading"
        elif stop == invalid.size:
            position = "trailing"
        else:
            position = "internal"
        runs.append(
            {
                "start_index": start,
                "stop_index_exclusive": stop,
                "length": stop - start,
                "position": position,
            }
        )
    return runs


def _explicit_timeline_inspection(
    series: _Series,
    *,
    n_samples: int,
) -> tuple[dict[str, Any], np.ndarray | None]:
    stamps = np.asarray(series.group["timestamps"], dtype=np.float64).reshape(-1)
    length_matches = stamps.size == n_samples
    nonfinite = ~np.isfinite(stamps)
    finite_stamps = stamps[~nonfinite]
    finite_values_monotonic = bool(finite_stamps.size < 2 or np.all(np.diff(finite_stamps) > 0.0))
    strict_monotonic = bool(
        length_matches
        and not nonfinite.any()
        and (stamps.size < 2 or np.all(np.diff(stamps) > 0.0))
    )

    adjacent_finite = np.isfinite(stamps[:-1]) & np.isfinite(stamps[1:])
    finite_deltas = np.diff(stamps)[adjacent_finite]
    median_interval = float(np.median(finite_deltas)) if finite_deltas.size else None
    uniform = bool(
        strict_monotonic
        and finite_deltas.size > 0
        and median_interval is not None
        and np.allclose(finite_deltas, median_interval, rtol=1e-6, atol=1e-9)
    )
    if nonfinite.any():
        clock_class = "explicit_nonfinite"
    elif not strict_monotonic:
        clock_class = "explicit_non_monotonic"
    elif uniform:
        clock_class = "explicit_uniform"
    else:
        clock_class = "explicit_irregular"

    interval_summary: dict[str, float | int | None] = {
        "count": int(finite_deltas.size),
        "minimum_s": None,
        "median_s": median_interval,
        "p95_s": None,
        "maximum_s": None,
        "nonpositive_count": int((finite_deltas <= 0.0).sum()),
        "over_1_5x_median_count": 0,
    }
    if finite_deltas.size:
        interval_summary.update(
            {
                "minimum_s": float(np.min(finite_deltas)),
                "p95_s": float(np.percentile(finite_deltas, 95.0)),
                "maximum_s": float(np.max(finite_deltas)),
                "over_1_5x_median_count": (
                    int((finite_deltas > 1.5 * median_interval).sum())
                    if median_interval is not None and median_interval > 0.0
                    else 0
                ),
            }
        )
    payload = {
        "source": "timestamps",
        "clock_class": clock_class,
        "sample_count": int(stamps.size),
        "length_matches_data": length_matches,
        "nonfinite_count": int(nonfinite.sum()),
        "nonfinite_fraction": float(nonfinite.mean()) if stamps.size else 0.0,
        "nonfinite_runs": _invalid_runs(nonfinite),
        "finite_values_monotonic": finite_values_monotonic,
        "strictly_increasing": strict_monotonic,
        "uniform": uniform,
        "sampling_rate_hz": (
            float(1.0 / median_interval)
            if uniform and median_interval is not None and median_interval > 0.0
            else None
        ),
        "intervals": interval_summary,
    }
    return payload, nonfinite if length_matches else None


def _series_inspection(series: _Series) -> dict[str, Any]:
    data = np.asarray(series.group["data"], dtype=np.float64)
    if data.ndim == 1:
        values = data.reshape((-1, 1))
    elif data.ndim == 2:
        values = data
    else:
        raise ValueError(f"NWB TimeSeries '{series.path}' data must be 1D or 2D, got {data.shape}.")
    if values.shape[0] == 0:
        raise ValueError(f"NWB TimeSeries '{series.path}' contains no samples.")

    nonfinite_values = ~np.isfinite(values)
    nonfinite_rows = nonfinite_values.any(axis=1)
    if "timestamps" in series.group:
        timeline, nonfinite_time = _explicit_timeline_inspection(
            series,
            n_samples=values.shape[0],
        )
    else:
        strict_timeline = _series_timeline(series, values.shape[0])
        sampling_rate = strict_timeline.sample_rate_hz
        if sampling_rate is None:
            raise RuntimeError(
                f"NWB TimeSeries '{series.path}' rate timeline lost its sampling rate."
            )
        interval = 1.0 / sampling_rate if strict_timeline.n_samples >= 2 else None
        timeline = {
            "source": "starting_time_rate",
            "clock_class": "generated_uniform",
            "sample_count": strict_timeline.n_samples,
            "length_matches_data": True,
            "nonfinite_count": 0,
            "nonfinite_fraction": 0.0,
            "nonfinite_runs": [],
            "finite_values_monotonic": True,
            "strictly_increasing": True,
            "uniform": True,
            "sampling_rate_hz": sampling_rate,
            "intervals": {
                "count": max(0, strict_timeline.n_samples - 1),
                "minimum_s": interval,
                "median_s": interval,
                "p95_s": interval,
                "maximum_s": interval,
                "nonpositive_count": 0,
                "over_1_5x_median_count": 0,
            },
        }
        nonfinite_time = np.zeros(values.shape[0], dtype=bool)

    joint_runs = (
        _invalid_runs(nonfinite_rows | nonfinite_time) if nonfinite_time is not None else []
    )
    return {
        "name": series.name,
        "path": series.path,
        "data_shape": list(data.shape),
        "sample_count": int(values.shape[0]),
        "channel_count": int(values.shape[1]),
        "nonfinite_value_count": int(nonfinite_values.sum()),
        "nonfinite_value_fraction": float(nonfinite_values.mean()),
        "nonfinite_row_count": int(nonfinite_rows.sum()),
        "nonfinite_row_fraction": float(nonfinite_rows.mean()),
        "nonfinite_row_runs": _invalid_runs(nonfinite_rows),
        "joint_invalid_runs": joint_runs,
        "timeline": timeline,
    }


def _namlab_eventlog_inventory(handle: h5py.File) -> tuple[dict[str, Any], ...]:
    acquisition = handle.get("acquisition")
    if not isinstance(acquisition, h5py.Group):
        return ()
    eventlogs = [
        obj
        for obj in acquisition.values()
        if _namlab_contract_group(
            obj,
            namespace=_NAMLAB_EVENTLOG_NAMESPACE,
            neurodata_type=_NAMLAB_EVENTLOG_TYPE,
        )
    ]
    tables = [
        obj
        for obj in acquisition.values()
        if _namlab_contract_group(
            obj,
            namespace=_NAMLAB_EVENTLOG_NAMESPACE,
            neurodata_type=_NAMLAB_EVENT_TABLE_TYPE,
        )
    ]
    inventory: list[dict[str, Any]] = []
    for eventlog in eventlogs:
        times = _namlab_vector(eventlog, "eventtime")
        indices = _namlab_vector(eventlog, "eventindex")
        flags = _namlab_vector(eventlog, "nonsolenoidflag")
        if not (times.size == indices.size == flags.size):
            raise ValueError(f"NWB namlab Eventlog '{eventlog.name}' column lengths differ.")
        if times.dtype.kind not in "fiu" or not np.isfinite(times).all():
            raise ValueError(
                f"NWB namlab dataset '{eventlog.name}/eventtime' must contain finite numbers."
            )
        if indices.dtype.kind not in "iu" or flags.dtype.kind not in "iub":
            raise ValueError(
                f"NWB namlab Eventlog '{eventlog.name}' indices and flags must be integers."
            )
        table = _namlab_event_table(eventlog, tables)
        labels = _namlab_label_map(table)
        unknown = sorted(set(int(value) for value in indices) - labels.keys())
        if unknown:
            raise ValueError(
                f"NWB namlab Eventlog '{eventlog.name}' references unknown event indices {unknown}."
            )
        time_dataset = eventlog["eventtime"]
        assert isinstance(time_dataset, h5py.Dataset)
        declared_unit = _decode(time_dataset.attrs.get("unit", ""))
        inventory.append(
            {
                "source_path": eventlog.name,
                "event_table_path": table.name,
                "event_count": int(times.size),
                "declared_time_unit": declared_unit,
                "label_counts": {
                    labels[index]: count
                    for index, count in sorted(Counter(int(value) for value in indices).items())
                },
                "nonsolenoid_flag_counts": {
                    str(flag): count
                    for flag, count in sorted(Counter(int(value) for value in flags).items())
                },
            }
        )
    return tuple(inventory)


def _inspection_issue_codes(
    signal: Mapping[str, Any] | None,
    *,
    classification: NwbPhotometryClassification,
) -> tuple[str, ...]:
    if signal is None:
        return (
            "event_only_no_photometry"
            if classification == "event_only"
            else "no_photometry_signal",
        )
    codes: list[str] = []
    if int(signal["nonfinite_row_count"]) > 0:
        codes.append("signal_nonfinite")
    timeline = signal["timeline"]
    if not isinstance(timeline, Mapping):
        raise TypeError("NWB inspection timeline payload must be a mapping.")
    if not bool(timeline["length_matches_data"]):
        codes.append("timestamp_length_mismatch")
    if int(timeline["nonfinite_count"]) > 0:
        codes.append("timestamp_nonfinite")
    if not bool(timeline["strictly_increasing"]):
        codes.append("timestamp_not_strictly_increasing")
    elif not bool(timeline["uniform"]):
        codes.append("timestamp_irregular")
    joint_runs = signal["joint_invalid_runs"]
    if not isinstance(joint_runs, list):
        raise TypeError("NWB inspection joint_invalid_runs must be a list.")
    positions = {str(run["position"]) for run in joint_runs}
    if positions & {"leading", "trailing"}:
        codes.append("edge_nonfinite")
    if "internal" in positions:
        codes.append("internal_nonfinite")
    return tuple(codes)


def inspect_nwb_photometry(path: str | Path) -> NwbPhotometryInspection:
    """Inspect NWB photometry, clock, gaps, and namlab events without repair."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"NWB file does not exist: {source_path}")
    with h5py.File(str(source_path), "r") as handle:
        candidates = _series_candidates(handle)
        signal_key = _select_series(candidates, _SIGNAL_HINTS, exclude=_SIGNAL_EXCLUDE)
        eventlogs = _namlab_eventlog_inventory(handle)
        if signal_key is None:
            classification: NwbPhotometryClassification = (
                "event_only" if eventlogs else "unsupported"
            )
            signal_series = None
            control_series = None
            signal_summary = None
            control_summary = None
        else:
            classification = "photometry"
            signal = candidates[signal_key]
            control_key = _select_control(candidates, signal_key)
            control = candidates[control_key] if control_key is not None else None
            signal_series = signal.path
            control_series = None if control is None else control.path
            signal_summary = _series_inspection(signal)
            control_summary = None if control is None else _series_inspection(control)
        issue_codes = _inspection_issue_codes(
            signal_summary,
            classification=classification,
        )
    return NwbPhotometryInspection(
        path=str(source_path),
        size_bytes=int(source_path.stat().st_size),
        classification=classification,
        candidate_series_count=len(candidates),
        signal_series=signal_series,
        control_series=control_series,
        signal=signal_summary,
        control=control_summary,
        namlab_eventlogs=eventlogs,
        issue_codes=issue_codes,
    )


def _metadata(handle: h5py.File, path: Path, photometry: PhotometryRecording) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source": {"type": "nwb_photometry", "path": str(path)},
        "signal_series": photometry.metadata["signal_series"],
        "control_series": photometry.metadata["control_series"],
    }
    for key in ("sampling_rate_hz", "sampling_rate_source"):
        if key in photometry.metadata:
            metadata[key] = photometry.metadata[key]
    for key in ("identifier", "session_description"):
        if key in handle and isinstance(handle[key], h5py.Dataset):
            metadata[key] = _scalar_dataset(handle, key)
    subject = handle.get("general/subject")
    if isinstance(subject, h5py.Group):
        subject_values: dict[str, Any] = {}
        for key, value in subject.items():
            if isinstance(value, h5py.Dataset):
                subject_values[key] = _decode(np.asarray(value).reshape(-1)[0])
        if subject_values:
            metadata["subject"] = subject_values
    return metadata


def read_nwb_photometry(
    path: str | Path,
    *,
    nonfinite_policy: NonfinitePolicy = "raise",
    max_nonfinite_fraction: float = _MAX_NONFINITE_FRACTION,
) -> RecordingSession:
    """Read fiber-photometry data from a standard NWB HDF5 file."""
    source_path = Path(path)
    with h5py.File(str(source_path), "r") as handle:
        candidates = _series_candidates(handle)
        signal_key = _select_series(candidates, _SIGNAL_HINTS, exclude=_SIGNAL_EXCLUDE)
        if signal_key is None:
            raise ValueError(f"NWB file '{source_path}' has no recognisable photometry signal.")
        signal = candidates[signal_key]
        signal_values, timeline, signal_repair = _read_series(
            signal,
            nonfinite_policy=nonfinite_policy,
            max_nonfinite_fraction=max_nonfinite_fraction,
        )
        control_key = _select_control(candidates, signal_key)
        control = candidates[control_key] if control_key is not None else None
        control_values = None
        control_repair = None
        if control is not None:
            control_values, control_timeline, control_repair = _read_series(
                control,
                nonfinite_policy=nonfinite_policy,
                max_nonfinite_fraction=max_nonfinite_fraction,
            )
            if not _timelines_match(timeline, control_timeline):
                raise ValueError("NWB control series timeline must match signal series timeline.")
        photometry = _photometry_recording(
            signal_series=signal,
            signal_values=signal_values,
            timeline=timeline,
            source_path=source_path,
            control_series=control,
            control_values=control_values,
            signal_nonfinite_repair=signal_repair,
            control_nonfinite_repair=control_repair,
        )
        namlab_rows, namlab_provenance = _namlab_eventlog_events(
            handle,
            source_path=source_path,
            signal=signal,
        )
        rows = [
            *_event_channel_events(
                candidates,
                source_path=source_path,
                nonfinite_policy=nonfinite_policy,
                max_nonfinite_fraction=max_nonfinite_fraction,
            ),
            *_annotation_series_events(candidates, source_path=source_path),
            *_analysis_timestamp_events(handle, source_path=source_path),
            *namlab_rows,
        ]
        session_metadata = _metadata(handle, source_path, photometry)
        if namlab_provenance:
            session_metadata["namlab_eventlogs"] = namlab_provenance
        session_id = str(session_metadata.get("identifier") or source_path.stem)
    return RecordingSession(
        session_id=session_id,
        signals=(SessionSignal("photometry", photometry),),
        event_streams=(SessionEventStream("events", EventTable.from_events(rows)),),
        metadata=session_metadata,
    )


__all__ = [
    "NwbPhotometryInspection",
    "find_first_nwb_photometry_file",
    "inspect_nwb_photometry",
    "is_nwb_photometry_file",
    "read_nwb_photometry",
]
