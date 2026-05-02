"""Generic event models for aligned experiment sessions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from xpkg.model.time import Timebase, TimeRange


def _normalize_metadata(value: Mapping[str, Any] | None, *, name: str) -> dict[str, Any]:
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


@dataclass(frozen=True, slots=True)
class Event:
    """One labeled event on a session timeline."""

    kind: str
    start_s: float
    duration_s: float = 0.0
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = str(self.kind).strip()
        if not kind:
            raise ValueError("event kind must be a non-empty string.")
        start_s = _finite_float(self.start_s, name="event start_s")
        duration_s = _finite_float(self.duration_s, name="event duration_s")
        if duration_s < 0.0:
            raise ValueError(f"event duration_s must be non-negative, got {duration_s}.")
        label = None if self.label is None else str(self.label).strip()
        if label == "":
            label = None
        metadata = _normalize_metadata(self.metadata, name="event metadata")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "start_s", start_s)
        object.__setattr__(self, "duration_s", duration_s)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "metadata", metadata)

    @property
    def end_s(self) -> float:
        """Return the event end time in seconds."""
        return float(self.start_s + self.duration_s)

    @property
    def time_range(self) -> TimeRange:
        """Return this event as a time interval."""
        return TimeRange(self.start_s, self.end_s)

    def contains(self, time_s: float) -> bool:
        """Return whether ``time_s`` is inside the event interval."""
        include_end = self.duration_s == 0.0
        return self.time_range.contains(time_s, include_end=include_end)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly event payload."""
        payload: dict[str, Any] = {
            "kind": self.kind,
            "start_s": self.start_s,
            "duration_s": self.duration_s,
        }
        if self.label is not None:
            payload["label"] = self.label
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Event:
        """Hydrate an event from a JSON-friendly payload."""
        if not isinstance(payload, Mapping):
            raise TypeError("event payload must be a mapping.")
        raw_metadata = payload.get("metadata")
        if raw_metadata is not None and not isinstance(raw_metadata, Mapping):
            raise TypeError("event metadata must be a mapping when present.")
        return cls(
            kind=str(payload.get("kind", "")),
            start_s=payload.get("start_s", 0.0),
            duration_s=payload.get("duration_s", 0.0),
            label=payload.get("label"),
            metadata=_normalize_metadata(raw_metadata, name="event metadata"),
        )


@dataclass(frozen=True, slots=True)
class EventTable:
    """Sorted collection of events on one timebase."""

    events: tuple[Event, ...] = ()
    timebase: Timebase = field(default_factory=Timebase)

    def __post_init__(self) -> None:
        events = tuple(self.events)
        for event in events:
            if not isinstance(event, Event):
                raise TypeError(f"event table entries must be Event objects, got {event!r}.")
        if not isinstance(self.timebase, Timebase):
            raise TypeError(f"event table timebase must be a Timebase, got {self.timebase!r}.")
        events = tuple(sorted(events, key=lambda event: (event.start_s, event.end_s, event.kind)))
        object.__setattr__(self, "events", events)

    @classmethod
    def from_events(
        cls,
        events: Iterable[Event],
        *,
        timebase: Timebase | None = None,
    ) -> EventTable:
        """Build an event table from an iterable."""
        return cls(events=tuple(events), timebase=timebase or Timebase())

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    def append(self, event: Event) -> EventTable:
        """Return a new table with ``event`` added."""
        return EventTable(events=(*self.events, event), timebase=self.timebase)

    def query(
        self,
        *,
        kind: str | None = None,
        label: str | None = None,
        time_s: float | None = None,
        overlaps: TimeRange | None = None,
    ) -> tuple[Event, ...]:
        """Return events matching optional kind, label, time, and overlap filters."""
        kind_filter = None if kind is None else str(kind).strip()
        label_filter = None if label is None else str(label).strip()
        if time_s is not None:
            time_value = _finite_float(time_s, name="time_s")
        else:
            time_value = None
        if overlaps is not None and not isinstance(overlaps, TimeRange):
            raise TypeError(f"overlaps must be a TimeRange, got {overlaps!r}.")

        result: list[Event] = []
        for event in self.events:
            if kind_filter is not None and event.kind != kind_filter:
                continue
            if label_filter is not None and event.label != label_filter:
                continue
            if time_value is not None and not event.contains(time_value):
                continue
            if overlaps is not None and not event.time_range.overlaps(overlaps):
                continue
            result.append(event)
        return tuple(result)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly event table payload."""
        return {
            "timebase": {
                "name": self.timebase.name,
                "unit": self.timebase.unit,
                "offset_s": self.timebase.offset_s,
            },
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EventTable:
        """Hydrate an event table from a JSON-friendly payload."""
        if not isinstance(payload, Mapping):
            raise TypeError("event table payload must be a mapping.")
        raw_timebase = payload.get("timebase")
        if isinstance(raw_timebase, Mapping):
            timebase = Timebase(
                name=str(raw_timebase.get("name", "session")),
                unit=str(raw_timebase.get("unit", "s")),
                offset_s=raw_timebase.get("offset_s", 0.0),
            )
        else:
            timebase = Timebase()
        raw_events = payload.get("events") or []
        if not isinstance(raw_events, Iterable):
            raise TypeError("event table events must be iterable.")
        return cls(
            events=tuple(Event.from_dict(event) for event in raw_events),
            timebase=timebase,
        )


@dataclass(frozen=True, slots=True)
class SyncEvent(Event):
    """Synchronization pulse or marker event."""

    source: str = "unknown"

    def __post_init__(self) -> None:
        Event.__post_init__(self)
        source = str(self.source).strip()
        if not source:
            raise ValueError("sync event source must be a non-empty string.")
        object.__setattr__(self, "source", source)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly sync event payload."""
        payload = Event.to_dict(self)
        payload["source"] = self.source
        return payload


__all__ = ["Event", "EventTable", "SyncEvent"]
