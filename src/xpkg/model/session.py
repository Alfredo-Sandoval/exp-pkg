"""Session container tying experiment modalities to shared timing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from xpkg.model._metadata_validation import (
    metadata_dict,
)
from xpkg.model.events import EventTable
from xpkg.model.signals import PhotometryRecording, TimeSeries
from xpkg.model.time import Timebase, TimeRange


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace.")
    return value


def _optional_text(value: object | None, *, name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, name=name)


def _strict_mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return {_required_text(key, name=f"{name} key"): item for key, item in value.items()}


@dataclass(frozen=True, slots=True)
class RecordingSession:
    """Multimodal experiment session with shared time semantics."""

    session_id: str
    title: str | None = None
    timebase: Timebase = field(default_factory=Timebase)
    pose: dict[str, Any] = field(default_factory=dict)
    videos: dict[str, Any] = field(default_factory=dict)
    signals: dict[str, TimeSeries | PhotometryRecording] = field(default_factory=dict)
    events: EventTable = field(default_factory=EventTable)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        session_id = _required_text(self.session_id, name="session_id")
        title = _optional_text(self.title, name="session title")
        if not isinstance(self.timebase, Timebase):
            raise TypeError(f"session timebase must be a Timebase, got {self.timebase!r}.")
        pose = _strict_mapping(self.pose, name="session pose")
        videos = _strict_mapping(self.videos, name="session videos")
        raw_signals = _strict_mapping(self.signals, name="session signals")
        signals: dict[str, TimeSeries | PhotometryRecording] = {}
        for key, value in raw_signals.items():
            if not isinstance(value, TimeSeries | PhotometryRecording):
                raise TypeError(
                    "session signals values must be TimeSeries or PhotometryRecording "
                    f"objects; got {key}={value!r}."
                )
            signals[key] = value
        if not isinstance(self.events, EventTable):
            raise TypeError(f"session events must be an EventTable, got {self.events!r}.")
        metadata = metadata_dict(self.metadata, name="session metadata")

        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "pose", pose)
        object.__setattr__(self, "videos", videos)
        object.__setattr__(self, "signals", signals)
        object.__setattr__(self, "metadata", metadata)

    @property
    def modality_names(self) -> tuple[str, ...]:
        """Return modality groups currently represented in this session."""
        names: list[str] = []
        if self.pose:
            names.append("pose")
        if self.videos:
            names.append("videos")
        if self.signals:
            names.append("signals")
        if len(self.events) > 0:
            names.append("events")
        return tuple(names)

    @property
    def time_range(self) -> TimeRange | None:
        """Return the broadest available time range across timed modalities."""
        ranges: list[TimeRange] = [
            signal.timeline.time_range
            if isinstance(signal, PhotometryRecording)
            else signal.timeline.time_range
            for signal in self.signals.values()
        ]
        ranges.extend(event.time_range for event in self.events)
        if not ranges:
            return None
        return TimeRange(
            min(time_range.start_s for time_range in ranges),
            max(time_range.end_s for time_range in ranges),
        )

    def with_signal(
        self,
        name: str,
        signal: TimeSeries | PhotometryRecording,
    ) -> RecordingSession:
        """Return a copy with one signal recording added or replaced."""
        key = _required_text(name, name="signal name")
        if not isinstance(signal, TimeSeries | PhotometryRecording):
            raise TypeError(f"signal must be a TimeSeries or PhotometryRecording, got {signal!r}.")
        signals = dict(self.signals)
        signals[key] = signal
        return RecordingSession(
            session_id=self.session_id,
            title=self.title,
            timebase=self.timebase,
            pose=self.pose,
            videos=self.videos,
            signals=signals,
            events=self.events,
            metadata=self.metadata,
        )

    def with_events(self, events: EventTable) -> RecordingSession:
        """Return a copy with a new event table."""
        if not isinstance(events, EventTable):
            raise TypeError(f"events must be an EventTable, got {events!r}.")
        return RecordingSession(
            session_id=self.session_id,
            title=self.title,
            timebase=self.timebase,
            pose=self.pose,
            videos=self.videos,
            signals=self.signals,
            events=events,
            metadata=self.metadata,
        )


__all__ = ["RecordingSession"]
