"""NWB fiber-photometry reader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
NonfinitePolicy = Literal["raise", "interpolate_sparse"]


@dataclass(frozen=True, slots=True)
class _Series:
    name: str
    path: str
    group: h5py.Group


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
            "Pass nonfinite_policy='interpolate_sparse' to repair sparse internal gaps."
        )
    if nonfinite_policy != "interpolate_sparse":
        raise ValueError(
            f"nonfinite_policy must be 'raise' or 'interpolate_sparse', got {nonfinite_policy!r}."
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
    return values, _series_timeline(series, values.shape[0]), repair


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
    metadata: dict[str, Any] = {
        "source_type": "nwb_photometry",
        "signal_series": signal_series.path,
        "control_series": None if control_series is None else control_series.path,
        "signal_is_dff": any(
            token in f"{signal_series.name} {signal_series.path}".lower()
            for token in ("dfoverf", "dff")
        ),
        "n_fibers": int(signal_values.shape[1]),
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
        rows = [
            *_event_channel_events(
                candidates,
                source_path=source_path,
                nonfinite_policy=nonfinite_policy,
                max_nonfinite_fraction=max_nonfinite_fraction,
            ),
            *_annotation_series_events(candidates, source_path=source_path),
            *_analysis_timestamp_events(handle, source_path=source_path),
        ]
        session_metadata = _metadata(handle, source_path, photometry)
        session_id = str(session_metadata.get("identifier") or source_path.stem)
    return RecordingSession(
        session_id=session_id,
        signals=(SessionSignal("photometry", photometry),),
        event_streams=(SessionEventStream("events", EventTable.from_events(rows)),),
        metadata=session_metadata,
    )


__all__ = [
    "find_first_nwb_photometry_file",
    "is_nwb_photometry_file",
    "read_nwb_photometry",
]
