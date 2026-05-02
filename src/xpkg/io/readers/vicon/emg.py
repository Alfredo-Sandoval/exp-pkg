"""Vicon analog-channel extraction for source-neutral EMG signals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from xpkg.model import EMGSignalData, ViconRecording

_EMG_HINTS = (
    "emg",
    "ers",
    "rf",
    "vl",
    "st",
    "bf",
    "ta",
    "gas",
    "gastroc",
    "soleus",
    "quad",
    "ham",
    "glut",
)
_EXCLUDED_HINTS = ("force", "moment", "footswitch", "sync", "trigger")


@dataclass(frozen=True, slots=True)
class _MappedChannel:
    channel_name: str
    muscle_name: str
    side: str


def _clean_required_string(value: object, *, field: str) -> str:
    if value is None:
        raise ValueError(f"Vicon EMG channel mapping field {field!r} must be non-empty.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Vicon EMG channel mapping field {field!r} must be non-empty.")
    return text


def _channel_from_mapping_key(key: str, value: Mapping[str, object]) -> str:
    return _clean_required_string(
        value.get("analog_channel")
        or value.get("analog_channel_name")
        or value.get("channel")
        or value.get("channel_name")
        or key,
        field="analog_channel",
    )


def _mapped_channel_from_item(
    key: str,
    value: object,
) -> _MappedChannel:
    if isinstance(value, Mapping):
        mapping_value = cast(Mapping[str, object], value)
        channel_name = _channel_from_mapping_key(key, mapping_value)
        muscle_name = _clean_required_string(
            mapping_value.get("muscle") or mapping_value.get("muscle_name"),
            field="muscle_name",
        )
        side = _clean_required_string(mapping_value.get("side"), field="side")
        return _MappedChannel(channel_name=channel_name, muscle_name=muscle_name, side=side)

    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(
            "Vicon EMG channel mapping values must provide muscle name and side."
        )
    values = tuple(value)
    if len(values) == 2:
        channel_name = _clean_required_string(key, field="analog_channel")
        muscle_name, side = values
    elif len(values) == 3:
        channel_name, muscle_name, side = values
    else:
        raise ValueError(
            "Vicon EMG tuple mappings must be (muscle_name, side) or "
            "(analog_channel, muscle_name, side)."
        )
    return _MappedChannel(
        channel_name=_clean_required_string(channel_name, field="analog_channel"),
        muscle_name=_clean_required_string(muscle_name, field="muscle_name"),
        side=_clean_required_string(side, field="side"),
    )


def _signal_unit(channel_units: Sequence[str]) -> str:
    units = tuple(str(unit).strip() for unit in channel_units if str(unit).strip())
    if len(set(units)) == 1:
        return units[0]
    return "V"


def candidate_vicon_emg_channels(channel_names: Sequence[str]) -> tuple[str, ...]:
    """Return candidate Vicon analog names that may warrant explicit EMG mapping."""

    candidates: list[str] = []
    for channel_name in channel_names:
        label = str(channel_name).strip()
        normalized = label.lower()
        if not label or any(excluded in normalized for excluded in _EXCLUDED_HINTS):
            continue
        if any(hint in normalized for hint in _EMG_HINTS):
            candidates.append(label)
    return tuple(candidates)


def extract_vicon_emg(
    recording: ViconRecording,
    channel_map: Mapping[str, object],
) -> EMGSignalData:
    """Extract explicitly mapped raw EMG channels from Vicon analog data."""

    if recording.analog is None:
        raise ValueError("Vicon recording has no analog data to extract EMG from.")
    if not channel_map:
        raise ValueError("Vicon EMG extraction requires an explicit channel mapping.")

    analog = recording.analog
    mapped_channels = tuple(
        _mapped_channel_from_item(key, value) for key, value in channel_map.items()
    )
    channel_indices = tuple(
        analog.channel_index(mapped_channel.channel_name) for mapped_channel in mapped_channels
    )
    channel_names = tuple(mapped_channel.channel_name for mapped_channel in mapped_channels)
    signals = analog.values[:, channel_indices].copy()
    sample_times_s = np.arange(analog.n_samples, dtype=np.float64) / float(analog.fps)
    selected_units = tuple(analog.channel_units[index] for index in channel_indices)

    return EMGSignalData(
        sample_times_s=sample_times_s,
        signals=signals,
        channel_names=channel_names,
        muscle_names=tuple(mapped_channel.muscle_name for mapped_channel in mapped_channels),
        sides=tuple(mapped_channel.side for mapped_channel in mapped_channels),
        sample_rate_hz=float(analog.fps),
        units=(("signal", _signal_unit(selected_units)),),
        processing_state="raw",
        provenance=(
            ("source_path", str(recording.path)),
            ("reader", "extract_vicon_emg"),
            ("source_channels", ",".join(channel_names)),
        ),
    )


__all__ = ["candidate_vicon_emg_channels", "extract_vicon_emg"]
