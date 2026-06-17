"""Source-neutral sampled signal models for experiment sessions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from xpkg.model._metadata_validation import (
    metadata_dict,
)
from xpkg.model.time import Timeline


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace.")
    return value


def _optional_text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if value != value.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace.")
    return value


@dataclass(frozen=True, slots=True)
class SignalChannel:
    """One named channel in a sampled signal recording."""

    name: str
    unit: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _required_text(self.name, name="signal channel name")
        unit = _optional_text(self.unit, name="signal channel unit")
        description = _optional_text(self.description, name="signal channel description")
        metadata = metadata_dict(self.metadata, name="signal channel metadata")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True, slots=True)
class PhotometryChannel(SignalChannel):
    """Signal channel known to come from a fiber photometry recording."""

    excitation: str = ""

    def __post_init__(self) -> None:
        SignalChannel.__post_init__(self)
        excitation = _optional_text(self.excitation, name="photometry channel excitation")
        object.__setattr__(self, "excitation", excitation)


@dataclass(frozen=True, slots=True)
class TimeSeries:
    """Sampled values aligned to a timeline."""

    values: np.ndarray
    channels: tuple[SignalChannel, ...]
    timeline: Timeline
    name: str = "signals"
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape((-1, 1))
        if values.ndim != 2:
            raise ValueError(
                "time series values must have shape (samples, channels), "
                f"got {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError("time series values must contain only finite values.")
        channels = tuple(self.channels)
        for channel in channels:
            if not isinstance(channel, SignalChannel):
                raise TypeError(
                    "time series channels must be SignalChannel objects, "
                    f"got {channel!r}."
                )
        if len(channels) != values.shape[1]:
            raise ValueError(
                "time series channel count must match values columns: "
                f"{len(channels)} vs {values.shape[1]}."
            )
        channel_names = tuple(channel.name for channel in channels)
        if len(set(channel_names)) != len(channel_names):
            raise ValueError("time series channel names must be unique.")
        if not isinstance(self.timeline, Timeline):
            raise TypeError(f"time series timeline must be a Timeline, got {self.timeline!r}.")
        if self.timeline.n_samples != values.shape[0]:
            raise ValueError(
                "time series sample count must match timeline length: "
                f"{values.shape[0]} vs {self.timeline.n_samples}."
            )
        name = _required_text(self.name, name="time series name")
        provenance = metadata_dict(self.provenance, name="time series provenance")

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "provenance", provenance)

    @classmethod
    def from_samples(
        cls,
        values: object,
        *,
        sample_rate_hz: float,
        channel_names: Sequence[str],
        start_s: float = 0.0,
        name: str = "signals",
        units: Sequence[str] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> TimeSeries:
        """Build a regular time series from sampled values and channel names."""
        values_array = np.asarray(values, dtype=np.float64)
        if values_array.ndim == 1:
            values_array = values_array.reshape((-1, 1))
        if values_array.ndim != 2:
            raise ValueError(
                "values must have shape (samples, channels), "
                f"got {values_array.shape}."
            )
        channel_names_tuple = tuple(channel_names)
        if len(channel_names_tuple) != values_array.shape[1]:
            raise ValueError(
                "channel_names length must match values channel count: "
                f"{len(channel_names_tuple)} vs {values_array.shape[1]}."
            )
        units_tuple = tuple(units) if units is not None else ("",) * values_array.shape[1]
        if len(units_tuple) != values_array.shape[1]:
            raise ValueError(
                "units length must match values channel count: "
                f"{len(units_tuple)} vs {values_array.shape[1]}."
            )
        channels = tuple(
            SignalChannel(name=channel_name, unit=unit)
            for channel_name, unit in zip(channel_names_tuple, units_tuple, strict=True)
        )
        timeline = Timeline.from_sample_rate(
            n_samples=values_array.shape[0],
            sample_rate_hz=sample_rate_hz,
            start_s=start_s,
        )
        return cls(
            values=values_array,
            channels=channels,
            timeline=timeline,
            name=name,
            provenance=dict(provenance or {}),
        )

    @property
    def n_samples(self) -> int:
        """Return the number of samples."""
        return int(self.values.shape[0])

    @property
    def n_channels(self) -> int:
        """Return the number of signal channels."""
        return int(self.values.shape[1])

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Return channel names in storage order."""
        return tuple(channel.name for channel in self.channels)

    @property
    def sample_times_s(self) -> np.ndarray:
        """Return sample timestamps in seconds."""
        return self.timeline.timestamps_s

    @property
    def sample_rate_hz(self) -> float | None:
        """Return regular sample rate if the timeline is evenly spaced."""
        return self.timeline.estimated_sample_rate_hz

    def channel_index(self, name: str) -> int:
        """Return the index for a uniquely named channel."""
        target = _required_text(name, name="channel name")
        matches = [index for index, channel in enumerate(self.channels) if channel.name == target]
        if len(matches) == 1:
            return int(matches[0])
        if matches:
            raise KeyError(f"channel {target!r} is ambiguous.")
        raise KeyError(f"channel {target!r} not found in {self.channel_names}.")


@dataclass(frozen=True, slots=True)
class PhotometryRecording:
    """Fiber photometry recording backed by a generic time series."""

    series: TimeSeries
    signal_channel: str | None = None
    reference_channel: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.series, TimeSeries):
            raise TypeError(f"photometry series must be a TimeSeries, got {self.series!r}.")
        signal_channel = None
        if self.signal_channel is not None:
            signal_channel = _required_text(self.signal_channel, name="signal_channel")
            self.series.channel_index(signal_channel)
        reference_channel = None
        if self.reference_channel is not None:
            reference_channel = _required_text(self.reference_channel, name="reference_channel")
            self.series.channel_index(reference_channel)
        metadata = metadata_dict(self.metadata, name="photometry metadata")
        object.__setattr__(self, "signal_channel", signal_channel)
        object.__setattr__(self, "reference_channel", reference_channel)
        object.__setattr__(self, "metadata", metadata)

    @property
    def timeline(self) -> Timeline:
        """Return the recording timeline."""
        return self.series.timeline

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Return photometry channel names."""
        return self.series.channel_names


__all__ = [
    "PhotometryChannel",
    "PhotometryRecording",
    "SignalChannel",
    "TimeSeries",
]
