"""Session container tying experiment modalities to shared timing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from xpkg.model.events import EventTable
from xpkg.model.signals import PhotometryRecording, TimeSeries
from xpkg.model.time import Timebase, TimeRange


def _name(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


def _mapping(value: Mapping[str, Any] | None, *, name: str) -> dict[str, Any]:
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
        session_id = _name(self.session_id, field_name="session_id")
        title = None if self.title is None else str(self.title).strip()
        if title == "":
            title = None
        if not isinstance(self.timebase, Timebase):
            raise TypeError(f"session timebase must be a Timebase, got {self.timebase!r}.")
        pose = _mapping(self.pose, name="session pose")
        videos = _mapping(self.videos, name="session videos")
        signals = _mapping(self.signals, name="session signals")
        for key, value in signals.items():
            if not isinstance(value, TimeSeries | PhotometryRecording):
                raise TypeError(
                    "session signals values must be TimeSeries or PhotometryRecording "
                    f"objects; got {key}={value!r}."
                )
        if not isinstance(self.events, EventTable):
            raise TypeError(f"session events must be an EventTable, got {self.events!r}.")
        metadata = _mapping(self.metadata, name="session metadata")

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
        key = _name(name, field_name="signal name")
        if not isinstance(signal, TimeSeries | PhotometryRecording):
            raise TypeError(
                f"signal must be a TimeSeries or PhotometryRecording, got {signal!r}."
            )
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
