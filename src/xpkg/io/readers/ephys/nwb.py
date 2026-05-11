"""NWB ICEphys reader for whole-cell patch-clamp recordings.

This module reads NWB files via :mod:`h5py` directly rather than going through
:mod:`pynwb`. The reference implementation is correct but pays multi-second
schema-validation cost on every ``read()``; for interactive trace browsing we
walk the ICEphys group layout directly:

  - ``/acquisition/<name>`` — :class:`CurrentClampSeries`, :class:`IZeroClampSeries`,
    or :class:`VoltageClampSeries` per sweep.
  - ``/stimulus/presentation/<name>`` — paired command waveform (optional).
  - Each acquisition carries ``data`` (with attributes ``conversion``, ``offset``,
    and ``unit``), ``rate``, ``starting_time``, and a ``stimulus_description``.

Unit normalisation happens at the model boundary (``volts`` → mV,
``amperes`` → pA) using xpkg's own helpers, with every applied scale recorded
in the recording's ``conversion_log``.

``h5py`` is an optional dependency; install ``exp-pkg[ephys]`` to pick it up.
"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from xpkg.model import (
    EphysRecording,
    RecordingMode,
    SignalChannel,
    Sweep,
    SweepSet,
    Timeline,
    TimeSeries,
)
from xpkg.model.ephys import normalize_signal_units

_ELECTRODE_CHANNEL = "primary"
_STIMULUS_CHANNEL = "stimulus_monitor"

# Map NWB neurodata_type strings to xpkg's recording-mode taxonomy.
_NWB_CLASS_TO_MODE: dict[str, RecordingMode] = {
    "CurrentClampSeries": "current_clamp",
    "IZeroClampSeries": "current_clamp",
    "VoltageClampSeries": "voltage_clamp",
}

# NWB stores SI strings; xpkg recognises symbol forms. We translate at the
# boundary so the rest of xpkg's unit machinery applies unchanged.
_NWB_UNIT_TO_SYMBOL: dict[str, str] = {
    "volts": "V",
    "volt": "V",
    "amperes": "A",
    "ampere": "A",
}


def _h5py_module(module: Any | None) -> Any:
    if module is not None:
        return module
    try:
        return importlib.import_module("h5py")
    except ModuleNotFoundError as exc:  # pragma: no cover — depends on env
        raise ModuleNotFoundError(
            "NWB support requires the optional 'h5py' package. Install "
            "exp-pkg[ephys] or install h5py in the active environment."
        ) from exc


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _nwb_class(group: Any) -> str:
    """Return the NWB ``neurodata_type`` for an HDF5 group, or empty."""
    attr = group.attrs.get("neurodata_type")
    return _decode(attr).strip() if attr is not None else ""


def _session_datetime(root: Any) -> datetime | None:
    raw = root.attrs.get("session_start_time")
    if raw is None and "session_start_time" in root:
        raw = root["session_start_time"][()]
    if raw is None:
        return None
    text = _decode(raw).strip()
    if not text:
        return None
    try:
        # NWB writes ISO-8601 with timezone; sometimes "Z" suffix.
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class _AcquisitionEntry:
    """Lightweight record of one NWB acquisition before array loading."""

    name: str
    group: Any  # h5py.Group
    mode: RecordingMode
    rate: float
    n_samples: int
    starting_time: float
    stimulus_label: str


def _list_acquisitions(root: Any, *, path: Path) -> list[_AcquisitionEntry]:
    if "acquisition" not in root:
        raise ValueError(
            f"NWB file {path} has no /acquisition group; not an ICEphys file?"
        )
    acquisition = root["acquisition"]
    entries: list[_AcquisitionEntry] = []
    for name in acquisition:
        group = acquisition[name]
        cls = _nwb_class(group)
        mode = _NWB_CLASS_TO_MODE.get(cls)
        if mode is None:
            # Non-ICEphys acquisition (e.g. behavior); skip silently.
            continue
        data = group.get("data")
        if data is None:
            continue
        st_dataset = group.get("starting_time")
        rate = (
            float(st_dataset.attrs["rate"])
            if st_dataset is not None and "rate" in st_dataset.attrs
            else 0.0
        )
        if rate <= 0:
            # Fall back to top-level rate attr if present.
            rate = float(group.attrs.get("rate", 0.0) or 0.0)
        if rate <= 0:
            raise ValueError(
                f"NWB acquisition {name!r} in {path} has non-positive rate."
            )
        if st_dataset is not None:
            starting_time = float(np.asarray(st_dataset[()]).reshape(-1)[0])
        else:
            starting_time = 0.0
        stimulus_label_attr = group.attrs.get("stimulus_description")
        stimulus_label = (
            _decode(stimulus_label_attr).strip()
            if stimulus_label_attr is not None
            else ""
        )
        entries.append(
            _AcquisitionEntry(
                name=name,
                group=group,
                mode=mode,
                rate=rate,
                n_samples=int(data.shape[0]),
                starting_time=starting_time,
                stimulus_label=stimulus_label,
            )
        )
    entries.sort(key=lambda e: (e.starting_time, e.name))
    return entries


def _paired_stimulus(root: Any, acquisition_name: str) -> Any | None:
    """Find the stimulus series paired with an acquisition.

    Allen/DANDI convention: ``data_00000_AD0`` ↔ ``data_00000_DA0``. Returns
    the HDF5 group or None if no obvious match exists.
    """
    stimulus_root = root.get("stimulus")
    if stimulus_root is None:
        return None
    presentation = stimulus_root.get("presentation")
    if presentation is None:
        return None
    if "_AD" in acquisition_name:
        candidate = acquisition_name.replace("_AD", "_DA", 1)
        if candidate in presentation:
            return presentation[candidate]
    return None


def _scale_to_si(values: np.ndarray, group: Any) -> np.ndarray:
    """Apply NWB ``value_SI = data * conversion + offset``."""
    conversion = float(group.attrs.get("conversion", 1.0) or 1.0)
    offset = float(group.attrs.get("offset", 0.0) or 0.0)
    out = values.astype(np.float64, copy=True)
    if conversion != 1.0:
        out *= conversion
    if offset != 0.0:
        out += offset
    return out


def _xpkg_unit(group: Any) -> str:
    raw = group.attrs.get("unit") or group.attrs.get("unit_in_data")
    if raw is None and "data" in group:
        raw = group["data"].attrs.get("unit")
    text = _decode(raw).strip() if raw is not None else ""
    return _NWB_UNIT_TO_SYMBOL.get(text.lower(), text)


def peek_nwb_modes(
    path: str | Path, *, h5py_module: Any | None = None
) -> tuple[RecordingMode, ...]:
    """Enumerate the recording modes present in an NWB file.

    Cheap: walks ``/acquisition/`` reading only the ``neurodata_type``
    attribute, never the sample data.
    """
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"NWB file not found: {source_path}.")
    h5 = _h5py_module(h5py_module)
    with h5.File(str(source_path), "r") as root:
        acquisition = root.get("acquisition")
        if acquisition is None:
            return ()
        modes: set[RecordingMode] = set()
        for name in acquisition:
            cls = _nwb_class(acquisition[name])
            mode = _NWB_CLASS_TO_MODE.get(cls)
            if mode is not None:
                modes.add(mode)
    return tuple(sorted(modes))


def read_nwb(
    path: str | Path,
    *,
    mode: RecordingMode | None = None,
    name: str = "ephys",
    h5py_module: Any | None = None,
) -> EphysRecording:
    """Read an NWB ICEphys file into a normalised :class:`EphysRecording`.

    Allen/DANDI files commonly carry both current-clamp and voltage-clamp
    acquisitions in the same NWB; pass ``mode`` to read one mode-coherent
    slice. Without ``mode``, a mixed-mode file raises ``ValueError`` so the
    caller has to commit.

    Voltage data is normalised to ``mV`` and current data to ``pA``; every
    applied scale lands in ``EphysRecording.conversion_log``.
    """
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"NWB file not found: {source_path}.")
    h5 = _h5py_module(h5py_module)
    handle = h5.File(str(source_path), "r")
    try:
        entries = _list_acquisitions(handle, path=source_path)
        if not entries:
            raise ValueError(
                f"NWB file {source_path} has no recognised ICEphys acquisitions."
            )

        if mode is not None:
            entries = [e for e in entries if e.mode == mode]
            if not entries:
                raise ValueError(
                    f"NWB file {source_path} contains no acquisitions of "
                    f"mode {mode!r}."
                )
            chosen_mode: RecordingMode = mode
        else:
            modes = {e.mode for e in entries}
            if len(modes) != 1:
                raise ValueError(
                    f"NWB file {source_path} contains mixed recording modes "
                    f"{sorted(modes)}; pass mode=... to filter."
                )
            chosen_mode = next(iter(modes))

        rate_reference = entries[0].rate
        sweeps: list[Sweep] = []
        conversion_log: list[dict[str, Any]] = []
        protocol_labels: set[str] = set()

        for index, entry in enumerate(entries):
            primary_raw = np.asarray(entry.group["data"][:], dtype=np.float64)
            primary_si = _scale_to_si(primary_raw, entry.group["data"])
            primary_unit = _xpkg_unit(entry.group)

            channel_arrays: list[np.ndarray] = [primary_si]
            channel_objs: list[SignalChannel] = [
                SignalChannel(
                    name=_ELECTRODE_CHANNEL,
                    unit=primary_unit,
                    description=entry.name,
                )
            ]
            channel_role_pairs: list[tuple[str, str]] = [
                (_ELECTRODE_CHANNEL, "electrode")
            ]

            stim_group = _paired_stimulus(handle, entry.name)
            if stim_group is not None and "data" in stim_group:
                stim_raw = np.asarray(stim_group["data"][:], dtype=np.float64)
                stim_si = _scale_to_si(stim_raw, stim_group["data"])
                stim_unit = _xpkg_unit(stim_group)
                channel_arrays.append(stim_si)
                channel_objs.append(
                    SignalChannel(
                        name=_STIMULUS_CHANNEL,
                        unit=stim_unit,
                        description=stim_group.name,
                    )
                )
                channel_role_pairs.append((_STIMULUS_CHANNEL, "stimulus_monitor"))

            values = np.column_stack(channel_arrays)
            roles = dict(channel_role_pairs)
            scaled, normalised_channels, log_entries = normalize_signal_units(
                values, tuple(channel_objs), roles=roles
            )
            for log_entry in log_entries:
                log_entry.setdefault("sweep_index", index)
                conversion_log.append(log_entry)

            # Preserve NWB's absolute sweep offset while storing the exact
            # acquisition rate as a Timeline hint. That avoids precision loss
            # when estimating the rate from large-offset float timestamps.
            timeline = Timeline.from_sample_rate(
                n_samples=entry.n_samples,
                sample_rate_hz=entry.rate,
                start_s=entry.starting_time,
            )
            series = TimeSeries(
                values=scaled,
                channels=normalised_channels,
                timeline=timeline,
                name=name,
            )
            sweeps.append(
                Sweep(
                    index=index,
                    series=series,
                    sweep_start_s=entry.starting_time,
                    metadata={
                        "source_name": entry.name,
                        "stimulus_description": entry.stimulus_label,
                    },
                )
            )
            if entry.stimulus_label:
                protocol_labels.add(entry.stimulus_label)

        provenance = {
            "source_format": "nwb",
            "source_path": str(source_path.resolve()),
            "source_sha256": _file_hash(source_path),
            "session_start_time": _session_datetime(handle),
            "sample_rate_hz": float(rate_reference),
        }
        metadata = {
            "stimulus_descriptions": sorted(protocol_labels),
            "protocol": (
                ",".join(sorted(protocol_labels)) if protocol_labels else ""
            ),
        }
        return EphysRecording(
            sweeps=SweepSet(sweeps=tuple(sweeps)),
            mode=chosen_mode,
            channel_roles={
                _ELECTRODE_CHANNEL: "electrode",
                _STIMULUS_CHANNEL: "stimulus_monitor",
            },
            conversion_log=tuple(conversion_log),
            provenance=provenance,
            metadata=metadata,
        )
    finally:
        handle.close()


__all__ = ["peek_nwb_modes", "read_nwb"]
