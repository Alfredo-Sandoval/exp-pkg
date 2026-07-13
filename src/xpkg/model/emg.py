"""Source-neutral EMG signal objects for multimodal sessions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from xpkg.model.time import Timebase, Timeline


class EMGSide(StrEnum):
    """Anatomical side associated with one EMG channel."""

    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"
    BILATERAL = "bilateral"


class EMGProcessingState(StrEnum):
    """Declared processing state of an EMG signal."""

    RAW = "raw"
    FILTERED = "filtered"
    RECTIFIED = "rectified"
    ENVELOPE = "envelope"


def _tuple_str(items: Sequence[str], *, name: str) -> tuple[str, ...]:
    if any(not isinstance(item, str) for item in items):
        raise TypeError(f"{name} entries must be strings.")
    values = tuple(item.strip() for item in items)
    if any(not value for value in values):
        raise ValueError(f"{name} entries must be non-empty strings.")
    return values


def _metadata_pairs(
    items: Iterable[tuple[str, str]],
    *,
    name: str,
) -> tuple[tuple[str, str], ...]:
    raw_pairs = tuple(items)
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in raw_pairs):
        raise TypeError(f"{name} key/value pairs must be strings.")
    pairs = tuple((key.strip(), value.strip()) for key, value in raw_pairs)
    if not pairs:
        raise ValueError(f"{name} must contain at least one key/value pair.")
    if any(not key or not value for key, value in pairs):
        raise ValueError(f"{name} key/value pairs must be non-empty strings.")
    return pairs


def _normalize_sides(items: Sequence[EMGSide]) -> tuple[EMGSide, ...]:
    sides = tuple(items)
    if any(not isinstance(side, EMGSide) for side in sides):
        raise TypeError("EMG sides entries must be EMGSide values.")
    return sides


@dataclass(frozen=True, slots=True)
class EMGSignalData:
    """EMG channels with explicit provenance and processing state."""

    sample_times_s: np.ndarray
    signals: np.ndarray
    channel_names: tuple[str, ...]
    muscle_names: tuple[str, ...]
    sides: tuple[EMGSide, ...]
    sample_rate_hz: float
    units: tuple[tuple[str, str], ...]
    processing_state: EMGProcessingState
    timebase: Timebase = Timebase()

    def __post_init__(self) -> None:
        sample_times_s = np.asarray(self.sample_times_s, dtype=np.float64)
        signals = np.asarray(self.signals, dtype=np.float64)
        channel_names = _tuple_str(self.channel_names, name="EMG channel_names")
        muscle_names = _tuple_str(self.muscle_names, name="EMG muscle_names")
        sides = _normalize_sides(self.sides)
        sample_rate_hz = float(self.sample_rate_hz)
        units = _metadata_pairs(self.units, name="EMG units")
        processing_state = self.processing_state

        if sample_times_s.ndim != 1:
            raise ValueError(f"EMG sample_times_s must be 1D, got shape {sample_times_s.shape}.")
        if sample_times_s.size == 0:
            raise ValueError("EMG sample_times_s must contain at least one sample.")
        if not np.isfinite(sample_times_s).all():
            raise ValueError("EMG sample_times_s must contain only finite values.")
        if sample_times_s.size > 1 and not np.all(np.diff(sample_times_s) > 0):
            raise ValueError("EMG sample_times_s must be strictly increasing.")
        if signals.ndim != 2:
            raise ValueError(
                f"EMG signals must have shape (samples, channels), got {signals.shape}."
            )
        if signals.shape[0] != sample_times_s.shape[0]:
            raise ValueError(
                "EMG signals sample count must match sample_times_s length: "
                f"{signals.shape[0]} vs {sample_times_s.shape[0]}."
            )
        channel_count = signals.shape[1]
        if len(channel_names) != channel_count:
            raise ValueError(
                "EMG channel_names length must match signal channel count: "
                f"{len(channel_names)} vs {channel_count}."
            )
        if len(set(channel_names)) != len(channel_names):
            raise ValueError("EMG channel_names must be unique.")
        if len(muscle_names) != channel_count:
            raise ValueError(
                "EMG muscle_names length must match signal channel count: "
                f"{len(muscle_names)} vs {channel_count}."
            )
        if len(sides) != channel_count:
            raise ValueError(
                "EMG sides length must match signal channel count: "
                f"{len(sides)} vs {channel_count}."
            )
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError(
                f"EMG sample_rate_hz must be finite and positive, got {sample_rate_hz}."
            )
        if not isinstance(processing_state, EMGProcessingState):
            raise TypeError("EMG processing_state must be an EMGProcessingState.")
        if not isinstance(self.timebase, Timebase):
            raise TypeError("EMG timebase must be a Timebase.")

        object.__setattr__(self, "sample_times_s", sample_times_s)
        object.__setattr__(self, "signals", signals)
        object.__setattr__(self, "channel_names", channel_names)
        object.__setattr__(self, "muscle_names", muscle_names)
        object.__setattr__(self, "sides", sides)
        object.__setattr__(self, "sample_rate_hz", sample_rate_hz)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "processing_state", processing_state)

    @property
    def timeline(self) -> Timeline:
        """Return the canonical sampled timeline for this EMG recording."""
        return Timeline(
            timestamps_s=self.sample_times_s,
            timebase=self.timebase,
            sample_rate_hz=self.sample_rate_hz,
        )

    @property
    def n_samples(self) -> int:
        return int(self.signals.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.signals.shape[1])


__all__ = ["EMGProcessingState", "EMGSide", "EMGSignalData"]
