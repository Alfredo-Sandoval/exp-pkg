"""Patch-clamp electrophysiology models for experiment sessions.

Patch-clamp recordings are sampled signals with an explicit stimulus protocol.
This module reuses :class:`xpkg.model.signals.TimeSeries` for the per-sweep
sample data and adds protocol structure (sweeps, sweep sets, stimulus epochs)
plus the small amount of metadata the wider session contract needs: recording
mode, channel roles, and a per-recording unit conversion log.

Time is kept in seconds for cross-modal alignment with pose, video, photometry,
and events. Signal units are normalized to ``mV`` for voltage and ``pA`` for
current at the reader boundary; the original recorded unit and the scale used
to reach the canonical unit are captured in the recording's
:attr:`EphysRecording.conversion_log`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from xpkg.model.signals import SignalChannel, TimeSeries
from xpkg.model.time import Timeline, TimeRange

RecordingMode = Literal["current_clamp", "voltage_clamp", "unknown"]
"""Acquisition mode for a patch-clamp recording."""

_RECORDING_MODES: frozenset[str] = frozenset({"current_clamp", "voltage_clamp", "unknown"})

ChannelRole = Literal[
    "electrode",
    "stimulus_monitor",
    "command",
    "ttl",
    "auxiliary",
    "unknown",
]
"""Documented channel roles. Readers may also use other strings; unknown roles
are preserved as-recorded."""

_KNOWN_CHANNEL_ROLES: frozenset[str] = frozenset(
    {"electrode", "stimulus_monitor", "command", "ttl", "auxiliary", "unknown"}
)

_VOLTAGE_UNITS: frozenset[str] = frozenset({"mV", "V", "uV", "µV", "nV"})
_CURRENT_UNITS: frozenset[str] = frozenset({"pA", "nA", "uA", "µA", "mA", "A"})


def _name(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


def _metadata(value: Mapping[str, Any] | None, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or None.")
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key).strip()
        if not key_text:
            raise ValueError(f"{name} keys must be non-empty strings.")
        normalized[key_text] = item
    return normalized


def _finite_float(value: Any, *, name: str) -> float:
    coerced = float(value)
    if not np.isfinite(coerced):
        raise ValueError(f"{name} must be finite, got {coerced!r}.")
    return coerced


def _validate_role(value: str, *, field_name: str) -> str:
    role = _name(value, field_name=field_name)
    return role


def is_voltage_unit(unit: str) -> bool:
    """Return whether ``unit`` is a recognized voltage unit string."""
    return str(unit).strip() in _VOLTAGE_UNITS


def is_current_unit(unit: str) -> bool:
    """Return whether ``unit`` is a recognized current unit string."""
    return str(unit).strip() in _CURRENT_UNITS


def voltage_scale_to_mV(unit: str) -> float:  # noqa: N802 — SI unit symbol case is meaningful
    """Return the multiplicative scale converting ``unit`` to millivolts."""
    text = str(unit).strip()
    if text == "mV":
        return 1.0
    if text == "V":
        return 1_000.0
    if text in {"uV", "µV"}:
        return 1.0e-3
    if text == "nV":
        return 1.0e-6
    raise ValueError(f"Unknown voltage unit {unit!r}; expected one of {sorted(_VOLTAGE_UNITS)}.")


def current_scale_to_pA(unit: str) -> float:  # noqa: N802 — SI unit symbol case is meaningful
    """Return the multiplicative scale converting ``unit`` to picoamperes."""
    text = str(unit).strip()
    if text == "pA":
        return 1.0
    if text == "nA":
        return 1_000.0
    if text in {"uA", "µA"}:
        return 1.0e6
    if text == "mA":
        return 1.0e9
    if text == "A":
        return 1.0e12
    raise ValueError(f"Unknown current unit {unit!r}; expected one of {sorted(_CURRENT_UNITS)}.")


@dataclass(frozen=True, slots=True)
class StimulusEpoch:
    """One protocol epoch as recorded by the acquisition software.

    Epochs are preserved as-recorded; xpkg does not classify protocols
    ("FI", "ramp", "rheobase") — that semantic step belongs to downstream
    analysis tools.
    """

    index: int
    kind: str
    start_s: float
    duration_s: float
    level: float = 0.0
    level_unit: str = ""
    channel: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        index = int(self.index)
        if index < 0:
            raise ValueError(f"stimulus epoch index must be non-negative, got {index}.")
        kind = _name(self.kind, field_name="stimulus epoch kind")
        start_s = _finite_float(self.start_s, name="stimulus epoch start_s")
        duration_s = _finite_float(self.duration_s, name="stimulus epoch duration_s")
        if duration_s < 0.0:
            raise ValueError(
                f"stimulus epoch duration_s must be non-negative, got {duration_s}."
            )
        level = _finite_float(self.level, name="stimulus epoch level")
        level_unit = str(self.level_unit).strip()
        channel = None if self.channel is None else _name(self.channel, field_name="channel")
        metadata = _metadata(self.metadata, name="stimulus epoch metadata")

        object.__setattr__(self, "index", index)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "start_s", start_s)
        object.__setattr__(self, "duration_s", duration_s)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "level_unit", level_unit)
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "metadata", metadata)

    @property
    def end_s(self) -> float:
        """Return the epoch end time in seconds, relative to the sweep start."""
        return float(self.start_s + self.duration_s)

    @property
    def time_range(self) -> TimeRange:
        """Return the epoch span as a :class:`TimeRange` (sweep-relative)."""
        return TimeRange(self.start_s, self.end_s)


@dataclass(frozen=True, slots=True)
class Sweep:
    """One sweep of a patch-clamp recording.

    A sweep is a multi-channel sample window with an optional stimulus protocol.
    The sample data is held as a :class:`TimeSeries` so it composes cleanly with
    the rest of the signal model; stimulus epochs are sweep-relative.
    """

    index: int
    series: TimeSeries
    epochs: tuple[StimulusEpoch, ...] = ()
    sweep_start_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        index = int(self.index)
        if index < 0:
            raise ValueError(f"sweep index must be non-negative, got {index}.")
        if not isinstance(self.series, TimeSeries):
            raise TypeError(f"sweep series must be a TimeSeries, got {self.series!r}.")
        epochs = tuple(self.epochs)
        for epoch in epochs:
            if not isinstance(epoch, StimulusEpoch):
                raise TypeError(
                    f"sweep epochs entries must be StimulusEpoch, got {epoch!r}."
                )
        epochs = tuple(sorted(epochs, key=lambda epoch: (epoch.start_s, epoch.index)))
        sweep_start_s = _finite_float(self.sweep_start_s, name="sweep_start_s")
        metadata = _metadata(self.metadata, name="sweep metadata")

        object.__setattr__(self, "index", index)
        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(self, "sweep_start_s", sweep_start_s)
        object.__setattr__(self, "metadata", metadata)

    @property
    def timeline(self) -> Timeline:
        """Return the sweep sample timeline."""
        return self.series.timeline

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Return channel names in storage order."""
        return self.series.channel_names

    @property
    def n_samples(self) -> int:
        """Return the number of samples in the sweep."""
        return self.series.n_samples

    @property
    def sample_rate_hz(self) -> float | None:
        """Return the sweep sample rate when timestamps are evenly spaced."""
        return self.series.sample_rate_hz

    @property
    def duration_s(self) -> float:
        """Return the elapsed time across the sweep timeline."""
        return self.series.timeline.duration_s


@dataclass(frozen=True, slots=True)
class SweepSet:
    """Ordered collection of sweeps that share channel layout."""

    sweeps: tuple[Sweep, ...] = ()

    def __post_init__(self) -> None:
        sweeps = tuple(self.sweeps)
        for sweep in sweeps:
            if not isinstance(sweep, Sweep):
                raise TypeError(f"sweep set entries must be Sweep, got {sweep!r}.")
        if sweeps:
            reference = sweeps[0].channel_names
            for sweep in sweeps[1:]:
                if sweep.channel_names != reference:
                    raise ValueError(
                        "sweep set sweeps must share channel layout: "
                        f"{sweep.channel_names} vs {reference}."
                    )
            indices = [sweep.index for sweep in sweeps]
            if len(set(indices)) != len(indices):
                raise ValueError("sweep set sweep indices must be unique.")
        object.__setattr__(self, "sweeps", sweeps)

    @classmethod
    def from_sweeps(cls, sweeps: Iterable[Sweep]) -> SweepSet:
        """Build a sweep set from any iterable of sweeps."""
        return cls(sweeps=tuple(sorted(sweeps, key=lambda sweep: sweep.index)))

    def __len__(self) -> int:
        return len(self.sweeps)

    def __iter__(self):
        return iter(self.sweeps)

    def __getitem__(self, index: int) -> Sweep:
        return self.sweeps[index]

    @property
    def n_sweeps(self) -> int:
        """Return the number of sweeps."""
        return len(self.sweeps)

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Return the shared channel layout, or an empty tuple if no sweeps."""
        if not self.sweeps:
            return ()
        return self.sweeps[0].channel_names

    @property
    def total_samples(self) -> int:
        """Return the total number of samples across all sweeps."""
        return int(sum(sweep.n_samples for sweep in self.sweeps))


@dataclass(frozen=True, slots=True)
class EphysRecording:
    """Patch-clamp recording with sweeps, channel roles, and a conversion log."""

    sweeps: SweepSet
    mode: RecordingMode = "unknown"
    channel_roles: dict[str, str] = field(default_factory=dict)
    conversion_log: tuple[dict[str, Any], ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sweeps, SweepSet):
            raise TypeError(f"ephys recording sweeps must be a SweepSet, got {self.sweeps!r}.")
        mode = str(self.mode).strip()
        if mode not in _RECORDING_MODES:
            raise ValueError(
                f"ephys recording mode must be one of {sorted(_RECORDING_MODES)}, got {mode!r}."
            )
        channel_roles_raw = self.channel_roles
        if not isinstance(channel_roles_raw, Mapping):
            raise TypeError(
                f"channel_roles must be a mapping, got {channel_roles_raw!r}."
            )
        channel_roles: dict[str, str] = {}
        for channel, role in channel_roles_raw.items():
            channel_name = _name(channel, field_name="channel_roles channel")
            channel_roles[channel_name] = _validate_role(role, field_name="channel role")
        if self.sweeps.channel_names:
            unknown = sorted(set(channel_roles) - set(self.sweeps.channel_names))
            if unknown:
                raise ValueError(
                    f"channel_roles references unknown channels: {unknown}; "
                    f"available channels are {list(self.sweeps.channel_names)}."
                )

        conversion_log: list[dict[str, Any]] = []
        for entry in self.conversion_log:
            if not isinstance(entry, Mapping):
                raise TypeError(
                    f"conversion_log entries must be mappings, got {entry!r}."
                )
            conversion_log.append(dict(entry))

        provenance = _metadata(self.provenance, name="ephys provenance")
        metadata = _metadata(self.metadata, name="ephys metadata")

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "channel_roles", channel_roles)
        object.__setattr__(self, "conversion_log", tuple(conversion_log))
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "metadata", metadata)

    @property
    def n_sweeps(self) -> int:
        """Return the number of sweeps."""
        return self.sweeps.n_sweeps

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Return the shared channel layout for this recording."""
        return self.sweeps.channel_names

    @property
    def sample_rate_hz(self) -> float | None:
        """Return the per-sweep sample rate when consistent across sweeps.

        Returns ``None`` if the recording has no sweeps or if sweeps have
        irregular or differing sample rates.
        """
        if not self.sweeps.sweeps:
            return None
        rates = {sweep.sample_rate_hz for sweep in self.sweeps}
        if len(rates) != 1:
            return None
        rate = next(iter(rates))
        return None if rate is None else float(rate)

    @property
    def duration_s(self) -> float:
        """Return the total sweep duration in seconds."""
        return float(sum(sweep.duration_s for sweep in self.sweeps))

    @property
    def electrode_channel(self) -> str | None:
        """Return the first channel mapped to the ``electrode`` role, if any."""
        for channel, role in self.channel_roles.items():
            if role == "electrode":
                return channel
        return None

    @property
    def stimulus_monitor_channel(self) -> str | None:
        """Return the first channel mapped to the ``stimulus_monitor`` role, if any."""
        for channel, role in self.channel_roles.items():
            if role == "stimulus_monitor":
                return channel
        return None

    def channel_unit(self, channel: str) -> str:
        """Return the canonical unit string for a named channel."""
        target = _name(channel, field_name="channel")
        if not self.sweeps.sweeps:
            raise KeyError(f"channel {target!r} not found; recording has no sweeps.")
        first = self.sweeps[0]
        index = first.series.channel_index(target)
        return first.series.channels[index].unit


def detect_recording_mode(
    *,
    channel_roles: Mapping[str, str],
    channel_units: Mapping[str, str],
) -> RecordingMode:
    """Infer ``current_clamp``/``voltage_clamp`` from channel roles and units.

    Rule: locate the channel with role ``electrode``. If its unit is a voltage
    unit, the recording is current-clamp (the amplifier is reading membrane
    voltage). If its unit is a current unit, it is voltage-clamp. Otherwise
    return ``"unknown"`` so the caller can override explicitly.
    """
    electrode = next(
        (channel for channel, role in channel_roles.items() if role == "electrode"),
        None,
    )
    if electrode is None:
        return "unknown"
    unit = str(channel_units.get(electrode, "")).strip()
    if is_voltage_unit(unit):
        return "current_clamp"
    if is_current_unit(unit):
        return "voltage_clamp"
    return "unknown"


def normalize_signal_units(
    values: np.ndarray,
    channels: tuple[SignalChannel, ...],
    *,
    roles: Mapping[str, str],
) -> tuple[np.ndarray, tuple[SignalChannel, ...], list[dict[str, Any]]]:
    """Normalize voltage channels to mV and current channels to pA.

    Returns the rescaled values, channels with canonical units, and a list of
    conversion log entries describing each applied scale. Channels with units
    that already match the canonical form, or with unrecognized units, are
    left untouched (no log entry).
    """
    if values.shape[1] != len(channels):
        raise ValueError(
            "values column count must match channels length: "
            f"{values.shape[1]} vs {len(channels)}."
        )
    scaled = values.astype(np.float64, copy=True)
    log_entries: list[dict[str, Any]] = []
    new_channels: list[SignalChannel] = []
    for index, channel in enumerate(channels):
        unit = channel.unit
        role = roles.get(channel.name, "")
        target_unit = unit
        scale = 1.0
        # Voltage channels (electrode in current-clamp, stimulus_monitor in
        # voltage-clamp, or any explicit voltage unit) -> mV.
        if is_voltage_unit(unit) and unit != "mV":
            scale = voltage_scale_to_mV(unit)
            target_unit = "mV"
        elif is_current_unit(unit) and unit != "pA":
            scale = current_scale_to_pA(unit)
            target_unit = "pA"
        if scale != 1.0:
            scaled[:, index] *= scale
            log_entries.append(
                {
                    "channel": channel.name,
                    "role": role,
                    "from_unit": unit,
                    "to_unit": target_unit,
                    "scale": scale,
                }
            )
            new_channels.append(
                SignalChannel(
                    name=channel.name,
                    unit=target_unit,
                    description=channel.description,
                    metadata={**channel.metadata, "original_unit": unit},
                )
            )
        else:
            new_channels.append(channel)
    return scaled, tuple(new_channels), log_entries


__all__ = [
    "ChannelRole",
    "EphysRecording",
    "RecordingMode",
    "StimulusEpoch",
    "Sweep",
    "SweepSet",
    "current_scale_to_pA",
    "detect_recording_mode",
    "is_current_unit",
    "is_voltage_unit",
    "normalize_signal_units",
    "voltage_scale_to_mV",
]
