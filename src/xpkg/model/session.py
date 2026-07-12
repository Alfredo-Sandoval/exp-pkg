"""Ontology objects for a multimodal recording session."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from xpkg.model._metadata_validation import (
    metadata_dict,
)
from xpkg.model._metadata_validation import (
    strict_required_text as _required_text,
)
from xpkg.model.events import EventTable
from xpkg.model.signals import PhotometryRecording, TimeSeries
from xpkg.model.time import Timebase, TimeRange

SessionSignalValue = TimeSeries | PhotometryRecording


@dataclass(frozen=True, slots=True)
class SessionSignal:
    """Named link from a recording session to one sampled signal object."""

    name: str
    recording: SessionSignalValue

    def __post_init__(self) -> None:
        name = _required_text(self.name, name="session signal name")
        if not isinstance(self.recording, TimeSeries | PhotometryRecording):
            raise TypeError(
                "session signal recording must be a TimeSeries or PhotometryRecording; "
                f"got {self.recording!r}."
            )
        object.__setattr__(self, "name", name)


@dataclass(frozen=True, slots=True)
class SessionVideo:
    """Named link from a recording session to one source video path."""

    role: str
    path: Path

    def __post_init__(self) -> None:
        role = _required_text(self.role, name="session video role")
        path = Path(self.path)
        if not path.name:
            raise ValueError("session video path must identify a file.")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "path", path)


@dataclass(frozen=True, slots=True)
class RecordingSession:
    """Multimodal experiment session with explicit typed modality links."""

    session_id: str
    title: str | None = None
    timebase: Timebase = field(default_factory=Timebase)
    signals: tuple[SessionSignal, ...] = ()
    videos: tuple[SessionVideo, ...] = ()
    events: EventTable = field(default_factory=EventTable)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        session_id = _required_text(self.session_id, name="session_id")
        title = None if self.title is None else _required_text(self.title, name="session title")
        if not isinstance(self.timebase, Timebase):
            raise TypeError(f"session timebase must be a Timebase, got {self.timebase!r}.")
        signals = tuple(self.signals)
        videos = tuple(self.videos)
        _require_unique_signals(signals)
        _require_unique_videos(videos)
        if not isinstance(self.events, EventTable):
            raise TypeError(f"session events must be an EventTable, got {self.events!r}.")
        metadata = MappingProxyType(metadata_dict(self.metadata, name="session metadata"))
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "signals", signals)
        object.__setattr__(self, "videos", videos)
        object.__setattr__(self, "metadata", metadata)

    @property
    def modality_names(self) -> tuple[str, ...]:
        """Return modality groups currently represented in this session."""
        names: list[str] = []
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
        ranges = [link.recording.timeline.time_range for link in self.signals]
        ranges.extend(event.time_range for event in self.events)
        if not ranges:
            return None
        return TimeRange(
            min(time_range.start_s for time_range in ranges),
            max(time_range.end_s for time_range in ranges),
        )

    @property
    def signal_names(self) -> tuple[str, ...]:
        """Return signal link names in storage order."""
        return tuple(link.name for link in self.signals)

    def signal(self, name: str) -> SessionSignalValue:
        """Return the signal linked under ``name`` or raise with session context."""
        key = _required_text(name, name="session signal name")
        for link in self.signals:
            if link.name == key:
                return link.recording
        raise KeyError(f"Recording session {self.session_id!r} has no signal named {key!r}.")

    def video(self, role: str) -> SessionVideo:
        """Return the video link for ``role`` or raise with session context."""
        key = _required_text(role, name="session video role")
        for link in self.videos:
            if link.role == key:
                return link
        raise KeyError(f"Recording session {self.session_id!r} has no video role {key!r}.")


def _require_unique_signals(links: tuple[object, ...]) -> None:
    values: list[str] = []
    for link in links:
        if not isinstance(link, SessionSignal):
            raise TypeError(f"session signal entries must be SessionSignal objects, got {link!r}.")
        if link.name in values:
            raise ValueError(f"Duplicate session signal name: {link.name!r}.")
        values.append(link.name)


def _require_unique_videos(links: tuple[object, ...]) -> None:
    values: list[str] = []
    for link in links:
        if not isinstance(link, SessionVideo):
            raise TypeError(f"session video entries must be SessionVideo objects, got {link!r}.")
        if link.role in values:
            raise ValueError(f"Duplicate session video role: {link.role!r}.")
        values.append(link.role)


__all__ = ["RecordingSession", "SessionSignal", "SessionSignalValue", "SessionVideo"]
