"""Shared timing primitives for multimodal experiment sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _finite_float(value: Any, *, name: str) -> float:
    coerced = float(value)
    if not np.isfinite(coerced):
        raise ValueError(f"{name} must be finite, got {coerced!r}.")
    return coerced


def _nonempty_name(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string.")
    return text


@dataclass(frozen=True, slots=True)
class TimeRange:
    """Half-open time interval in seconds."""

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        start_s = _finite_float(self.start_s, name="time range start_s")
        end_s = _finite_float(self.end_s, name="time range end_s")
        if end_s < start_s:
            raise ValueError(
                f"time range end_s must be >= start_s, got {end_s} < {start_s}."
            )
        object.__setattr__(self, "start_s", start_s)
        object.__setattr__(self, "end_s", end_s)

    @property
    def duration_s(self) -> float:
        """Return the interval duration in seconds."""
        return float(self.end_s - self.start_s)

    def contains(self, time_s: float, *, include_end: bool = False) -> bool:
        """Return whether ``time_s`` falls inside this interval."""
        time_value = _finite_float(time_s, name="time_s")
        if include_end:
            return self.start_s <= time_value <= self.end_s
        return self.start_s <= time_value < self.end_s

    def overlaps(self, other: TimeRange) -> bool:
        """Return whether this interval overlaps ``other``."""
        if not isinstance(other, TimeRange):
            raise TypeError(f"other must be a TimeRange, got {other!r}.")
        return self.start_s < other.end_s and other.start_s < self.end_s


@dataclass(frozen=True, slots=True)
class Timebase:
    """Named time coordinate system for a session or recording."""

    name: str = "session"
    unit: str = "s"
    offset_s: float = 0.0

    def __post_init__(self) -> None:
        name = _nonempty_name(self.name, name="timebase name")
        unit = _nonempty_name(self.unit, name="timebase unit")
        offset_s = _finite_float(self.offset_s, name="timebase offset_s")
        if unit not in {"s", "sec", "second", "seconds"}:
            raise ValueError(
                "timebase unit must be seconds-compatible "
                "('s', 'sec', 'second', or 'seconds')."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "offset_s", offset_s)

    def to_session_time(self, value_s: float) -> float:
        """Convert a local time value into session-relative seconds."""
        return _finite_float(value_s, name="value_s") + self.offset_s

    def from_session_time(self, value_s: float) -> float:
        """Convert session-relative seconds into this timebase."""
        return _finite_float(value_s, name="value_s") - self.offset_s


@dataclass(frozen=True, slots=True)
class Timeline:
    """Strictly increasing sample or frame timestamps.

    ``sample_rate_hz`` is an optional explicit hint stored by constructors
    that know the rate exactly (e.g. ``from_sample_rate``). When set, it
    bypasses the diff-based derivation in ``estimated_sample_rate_hz`` —
    deriving a regular rate from float64 timestamps loses precision at the
    ULP level when the timeline sits at a large session offset.
    """

    timestamps_s: np.ndarray
    timebase: Timebase = field(default_factory=Timebase)
    sample_rate_hz: float | None = None

    def __post_init__(self) -> None:
        timestamps_s = np.asarray(self.timestamps_s, dtype=np.float64)
        if timestamps_s.ndim != 1:
            raise ValueError(
                "timeline timestamps_s must be 1D, "
                f"got shape {timestamps_s.shape}."
            )
        if timestamps_s.size == 0:
            raise ValueError("timeline timestamps_s must contain at least one sample.")
        if not np.isfinite(timestamps_s).all():
            raise ValueError("timeline timestamps_s must contain only finite values.")
        if timestamps_s.size > 1 and not np.all(np.diff(timestamps_s) > 0.0):
            raise ValueError("timeline timestamps_s must be strictly increasing.")
        if not isinstance(self.timebase, Timebase):
            raise TypeError(f"timeline timebase must be a Timebase, got {self.timebase!r}.")
        if self.sample_rate_hz is not None:
            rate = float(self.sample_rate_hz)
            if not np.isfinite(rate) or rate <= 0.0:
                raise ValueError(
                    f"timeline sample_rate_hz must be a positive finite number, "
                    f"got {self.sample_rate_hz!r}."
                )
            object.__setattr__(self, "sample_rate_hz", rate)

        object.__setattr__(self, "timestamps_s", timestamps_s)

    @classmethod
    def from_sample_rate(
        cls,
        *,
        n_samples: int,
        sample_rate_hz: float,
        start_s: float = 0.0,
        timebase: Timebase | None = None,
    ) -> Timeline:
        """Build a regular timeline from sample count and sample rate."""
        sample_count = int(n_samples)
        rate = _finite_float(sample_rate_hz, name="sample_rate_hz")
        start = _finite_float(start_s, name="start_s")
        if sample_count <= 0:
            raise ValueError(f"n_samples must be positive, got {sample_count}.")
        if rate <= 0.0:
            raise ValueError(f"sample_rate_hz must be positive, got {rate}.")
        timestamps = start + (np.arange(sample_count, dtype=np.float64) / rate)
        return cls(
            timestamps_s=timestamps,
            timebase=timebase or Timebase(),
            sample_rate_hz=rate,
        )

    @property
    def n_samples(self) -> int:
        """Return the number of timestamps."""
        return int(self.timestamps_s.shape[0])

    @property
    def start_s(self) -> float:
        """Return the first timestamp in seconds."""
        return float(self.timestamps_s[0])

    @property
    def end_s(self) -> float:
        """Return the final timestamp in seconds."""
        return float(self.timestamps_s[-1])

    @property
    def time_range(self) -> TimeRange:
        """Return the timestamp span as a closed-duration interval."""
        return TimeRange(self.start_s, self.end_s)

    @property
    def duration_s(self) -> float:
        """Return the elapsed time from first to last timestamp."""
        return self.time_range.duration_s

    @property
    def estimated_sample_rate_hz(self) -> float | None:
        """Return the regular sample rate if timestamps are evenly spaced.

        Constructors that know the rate (``from_sample_rate``) stash it on
        ``sample_rate_hz`` so it survives float64 timestamp arithmetic
        intact. When no hint is stored we fall back to the regression form
        ``(n - 1) / (t_last - t_first)`` after a uniform-spacing check —
        more stable than ``1 / median(diff)`` for offset timelines, but
        still subject to ULP-level drift.
        """
        if self.sample_rate_hz is not None:
            return float(self.sample_rate_hz)
        if self.timestamps_s.size < 2:
            return None
        diffs = np.diff(self.timestamps_s)
        step = float(np.median(diffs))
        if step <= 0.0:
            return None
        if not np.allclose(diffs, step, rtol=1e-6, atol=1e-9):
            return None
        n = int(self.timestamps_s.size)
        span = float(self.timestamps_s[-1] - self.timestamps_s[0])
        if span <= 0.0:
            return None
        return float((n - 1) / span)

    def nearest_index(self, time_s: float) -> int:
        """Return the nearest timestamp index for ``time_s``."""
        value = _finite_float(time_s, name="time_s")
        insertion = int(np.searchsorted(self.timestamps_s, value))
        if insertion <= 0:
            return 0
        if insertion >= self.timestamps_s.size:
            return int(self.timestamps_s.size - 1)
        before = insertion - 1
        after = insertion
        if abs(float(self.timestamps_s[after]) - value) < abs(
            value - float(self.timestamps_s[before])
        ):
            return after
        return before


__all__ = ["TimeRange", "Timebase", "Timeline"]
