"""Behavior-label and ethogram model primitives."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from xpkg.model._metadata_validation import (
    finite_float,
    metadata_dict,
    optional_text,
    payload_mapping,
    required_text,
)
from xpkg.model.events import Event, EventTable
from xpkg.model.time import Timebase

BEHAVIOR_LABELS_SCHEMA_VERSION = "xpkg.behavior_labels.v1"


def _optional_finite(value: Any | None, *, name: str) -> float | None:
    if value is None:
        return None
    coerced = float(value)
    if not np.isfinite(coerced):
        raise ValueError(f"{name} must be finite, got {coerced!r}.")
    return coerced


def _optional_index(value: Any | None, *, name: str) -> int | None:
    if value is None:
        return None
    coerced = int(value)
    if coerced < 0:
        raise ValueError(f"{name} must be non-negative, got {coerced}.")
    return coerced


def _required_index(value: Any, *, name: str) -> int:
    coerced = _optional_index(value, name=name)
    if coerced is None:
        raise ValueError(f"{name} must be provided.")
    return coerced


def _path_text(value: str | Path | None, *, name: str) -> str | None:
    if value is None:
        return None
    return required_text(Path(value).as_posix(), name=name)


def _required_clean_text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace.")
    return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence.")
    return value


def _float_tuple(value: object, *, name: str) -> tuple[float, ...]:
    return tuple(finite_float(item, name=f"{name} item") for item in _sequence(value, name=name))


@dataclass(frozen=True, slots=True)
class BehaviorInterval:
    """One behavior label over a time span, frame span, or both."""

    label: str
    start_s: float | None = None
    end_s: float | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    score: float | None = None
    confidence: str | None = None
    source_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        label = required_text(self.label, name="behavior interval label")
        start_s = _optional_finite(self.start_s, name="behavior interval start_s")
        end_s = _optional_finite(self.end_s, name="behavior interval end_s")
        start_frame = _optional_index(self.start_frame, name="behavior interval start_frame")
        end_frame = _optional_index(self.end_frame, name="behavior interval end_frame")
        if start_s is None and start_frame is None:
            raise ValueError("behavior interval requires start_s or start_frame.")
        if start_s is not None and end_s is None:
            end_s = start_s
        if start_s is not None and end_s is not None and end_s < start_s:
            raise ValueError("behavior interval end_s must be >= start_s.")
        if start_frame is not None and end_frame is None:
            end_frame = start_frame
        if start_frame is not None and end_frame is not None and end_frame < start_frame:
            raise ValueError("behavior interval end_frame must be >= start_frame.")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "start_s", start_s)
        object.__setattr__(self, "end_s", end_s)
        object.__setattr__(self, "start_frame", start_frame)
        object.__setattr__(self, "end_frame", end_frame)
        object.__setattr__(
            self,
            "score",
            _optional_finite(self.score, name="behavior interval score"),
        )
        object.__setattr__(self, "confidence", optional_text(self.confidence, name="confidence"))
        object.__setattr__(self, "source_id", optional_text(self.source_id, name="source_id"))
        object.__setattr__(self, "metadata", metadata_dict(self.metadata, name="interval metadata"))

    @property
    def duration_s(self) -> float | None:
        """Return the interval duration in seconds when time coordinates exist."""

        if self.start_s is None or self.end_s is None:
            return None
        return float(self.end_s - self.start_s)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly interval payload."""

        return _compact_payload(
            {
                "label": self.label,
                "start_s": self.start_s,
                "end_s": self.end_s,
                "start_frame": self.start_frame,
                "end_frame": self.end_frame,
                "score": self.score,
                "confidence": self.confidence,
                "source_id": self.source_id,
                "metadata": self.metadata or None,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BehaviorInterval:
        """Hydrate a behavior interval from a JSON-friendly payload."""

        fields = payload_mapping(payload, name="behavior interval payload")
        metadata = fields.get("metadata")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("behavior interval metadata must be a mapping.")
        return cls(
            label=fields.get("label", ""),
            start_s=fields.get("start_s"),
            end_s=fields.get("end_s"),
            start_frame=fields.get("start_frame"),
            end_frame=fields.get("end_frame"),
            score=fields.get("score"),
            confidence=fields.get("confidence"),
            source_id=fields.get("source_id"),
            metadata=metadata_dict(metadata, name="interval metadata"),
        )


@dataclass(frozen=True, slots=True)
class BehaviorFrameLabel:
    """One discrete behavior label assigned to a video frame."""

    frame_index: int
    label: str
    score: float | None = None
    confidence: str | None = None
    source_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frame_index",
            _required_index(self.frame_index, name="frame_index"),
        )
        object.__setattr__(self, "label", required_text(self.label, name="frame label"))
        object.__setattr__(self, "score", _optional_finite(self.score, name="frame label score"))
        object.__setattr__(self, "confidence", optional_text(self.confidence, name="confidence"))
        object.__setattr__(self, "source_id", optional_text(self.source_id, name="source_id"))
        object.__setattr__(self, "metadata", metadata_dict(self.metadata, name="frame metadata"))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly frame-label payload."""

        return _compact_payload(
            {
                "frame_index": self.frame_index,
                "label": self.label,
                "score": self.score,
                "confidence": self.confidence,
                "source_id": self.source_id,
                "metadata": self.metadata or None,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BehaviorFrameLabel:
        """Hydrate a frame label from a JSON-friendly payload."""

        fields = payload_mapping(payload, name="behavior frame-label payload")
        return cls(
            frame_index=fields.get("frame_index", -1),
            label=fields.get("label", ""),
            score=fields.get("score"),
            confidence=fields.get("confidence"),
            source_id=fields.get("source_id"),
            metadata=metadata_dict(fields.get("metadata"), name="frame metadata"),
        )


@dataclass(frozen=True, slots=True)
class BehaviorEmbedding:
    """One per-frame continuous behavior embedding or latent code."""

    frame_index: int
    values: tuple[float, ...]
    space: str = "embedding"
    source_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.values)
        if not values:
            raise ValueError("behavior embedding values must be non-empty.")
        if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
            raise ValueError("behavior embedding values must be finite.")
        object.__setattr__(
            self,
            "frame_index",
            _required_index(self.frame_index, name="frame_index"),
        )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "space", required_text(self.space, name="embedding space"))
        object.__setattr__(self, "source_id", optional_text(self.source_id, name="source_id"))
        object.__setattr__(
            self, "metadata", metadata_dict(self.metadata, name="embedding metadata")
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly embedding payload."""

        return _compact_payload(
            {
                "frame_index": self.frame_index,
                "values": list(self.values),
                "space": self.space,
                "source_id": self.source_id,
                "metadata": self.metadata or None,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BehaviorEmbedding:
        """Hydrate a behavior embedding from a JSON-friendly payload."""

        fields = payload_mapping(payload, name="behavior embedding payload")
        return cls(
            frame_index=fields.get("frame_index", -1),
            values=_float_tuple(fields.get("values", ()), name="embedding values"),
            space=str(fields.get("space", "embedding")),
            source_id=fields.get("source_id"),
            metadata=metadata_dict(fields.get("metadata"), name="embedding metadata"),
        )


@dataclass(frozen=True, slots=True)
class BehaviorLabels:
    """Canonical behavior labels imported from human or model-produced outputs."""

    source_type: str
    intervals: tuple[BehaviorInterval, ...] = ()
    frame_labels: tuple[BehaviorFrameLabel, ...] = ()
    embeddings: tuple[BehaviorEmbedding, ...] = ()
    timebase: Timebase = field(default_factory=Timebase)
    media_path: str | None = None
    subject_id: str | None = None
    annotator: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        intervals = _typed_tuple(self.intervals, BehaviorInterval, name="intervals")
        frame_labels = _typed_tuple(self.frame_labels, BehaviorFrameLabel, name="frame_labels")
        embeddings = _typed_tuple(self.embeddings, BehaviorEmbedding, name="embeddings")
        if not intervals and not frame_labels and not embeddings:
            raise ValueError("behavior labels require intervals, frame_labels, or embeddings.")
        if not isinstance(self.timebase, Timebase):
            raise TypeError(f"behavior labels timebase must be a Timebase, got {self.timebase!r}.")
        object.__setattr__(
            self,
            "source_type",
            _required_clean_text(self.source_type, name="source_type"),
        )
        object.__setattr__(self, "intervals", tuple(sorted(intervals, key=_interval_sort_key)))
        object.__setattr__(
            self,
            "frame_labels",
            tuple(sorted(frame_labels, key=lambda item: item.frame_index)),
        )
        object.__setattr__(
            self,
            "embeddings",
            tuple(sorted(embeddings, key=lambda item: item.frame_index)),
        )
        object.__setattr__(self, "media_path", _path_text(self.media_path, name="media_path"))
        object.__setattr__(self, "subject_id", optional_text(self.subject_id, name="subject_id"))
        object.__setattr__(self, "annotator", optional_text(self.annotator, name="annotator"))
        object.__setattr__(self, "metadata", metadata_dict(self.metadata, name="behavior metadata"))

    @property
    def label_names(self) -> tuple[str, ...]:
        """Return unique behavior names across interval and frame labels."""

        labels = {item.label for item in self.intervals}
        labels.update(item.label for item in self.frame_labels)
        return tuple(sorted(labels))

    def to_event_table(self, *, kind: str = "behavior") -> EventTable:
        """Project time-indexed behavior intervals into the generic event model."""

        events: list[Event] = []
        for interval in self.intervals:
            if interval.start_s is None:
                continue
            events.append(
                Event(
                    kind=kind,
                    start_s=interval.start_s,
                    duration_s=interval.duration_s or 0.0,
                    label=interval.label,
                    metadata=_interval_event_metadata(interval, self.source_type),
                )
            )
        return EventTable.from_events(events, timebase=self.timebase)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly behavior-label payload."""

        return _compact_payload(
            {
                "schema": BEHAVIOR_LABELS_SCHEMA_VERSION,
                "source_type": self.source_type,
                "timebase": _timebase_payload(self.timebase),
                "media_path": self.media_path,
                "subject_id": self.subject_id,
                "annotator": self.annotator,
                "metadata": self.metadata or None,
                "intervals": [item.to_dict() for item in self.intervals],
                "frame_labels": [item.to_dict() for item in self.frame_labels],
                "embeddings": [item.to_dict() for item in self.embeddings],
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BehaviorLabels:
        """Hydrate behavior labels from a JSON-friendly payload."""

        fields = payload_mapping(payload, name="behavior labels payload")
        return cls(
            source_type=fields.get("source_type", ""),
            intervals=_intervals_from_payload(fields.get("intervals", ())),
            frame_labels=_frame_labels_from_payload(fields.get("frame_labels", ())),
            embeddings=_embeddings_from_payload(fields.get("embeddings", ())),
            timebase=_timebase_from_payload(fields.get("timebase")),
            media_path=fields.get("media_path"),
            subject_id=fields.get("subject_id"),
            annotator=fields.get("annotator"),
            metadata=metadata_dict(fields.get("metadata"), name="behavior metadata"),
        )

    @classmethod
    def from_event_table(
        cls,
        events: EventTable,
        *,
        source_type: str = "event_table",
        media_path: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> BehaviorLabels:
        """Create behavior labels from a generic ``EventTable``."""

        if not isinstance(events, EventTable):
            raise TypeError(f"events must be an EventTable, got {events!r}.")
        intervals = tuple(
            BehaviorInterval(
                label=event.label or event.kind,
                start_s=event.start_s,
                end_s=event.end_s,
                metadata={"event_kind": event.kind, **dict(event.metadata)},
            )
            for event in events
        )
        return cls(
            source_type=source_type,
            intervals=intervals,
            timebase=events.timebase,
            media_path=_path_text(media_path, name="media_path"),
            metadata=metadata_dict(metadata, name="behavior metadata"),
        )


def _typed_tuple(values: Iterable[Any], cls: type, *, name: str) -> tuple[Any, ...]:
    result = tuple(values)
    for item in result:
        if not isinstance(item, cls):
            raise TypeError(f"{name} entries must be {cls.__name__} objects, got {item!r}.")
    return result


def _interval_sort_key(interval: BehaviorInterval) -> tuple[float, int, str]:
    start_s = float("inf") if interval.start_s is None else interval.start_s
    start_frame = 10**18 if interval.start_frame is None else interval.start_frame
    return (start_s, start_frame, interval.label)


def _compact_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value not in (None, [], {})}


def _timebase_payload(timebase: Timebase) -> dict[str, Any]:
    return {"name": timebase.name, "unit": timebase.unit, "offset_s": timebase.offset_s}


def _timebase_from_payload(payload: object) -> Timebase:
    if not isinstance(payload, Mapping):
        return Timebase()
    fields = payload_mapping(payload, name="timebase payload")
    return Timebase(
        name=fields.get("name", "session"),
        unit=fields.get("unit", "s"),
        offset_s=fields.get("offset_s", 0.0),
    )


def _interval_event_metadata(interval: BehaviorInterval, source_type: str) -> dict[str, Any]:
    metadata = dict(interval.metadata)
    metadata["source_type"] = source_type
    for key, value in (
        ("source_id", interval.source_id),
        ("score", interval.score),
        ("confidence", interval.confidence),
        ("start_frame", interval.start_frame),
        ("end_frame", interval.end_frame),
    ):
        if value is not None:
            metadata[key] = value
    return metadata


def _intervals_from_payload(payload: object) -> tuple[BehaviorInterval, ...]:
    return tuple(
        BehaviorInterval.from_dict(payload_mapping(item, name="interval payload"))
        for item in _sequence(payload, name="intervals")
    )


def _frame_labels_from_payload(payload: object) -> tuple[BehaviorFrameLabel, ...]:
    return tuple(
        BehaviorFrameLabel.from_dict(payload_mapping(item, name="frame-label payload"))
        for item in _sequence(payload, name="frame_labels")
    )


def _embeddings_from_payload(payload: object) -> tuple[BehaviorEmbedding, ...]:
    return tuple(
        BehaviorEmbedding.from_dict(payload_mapping(item, name="embedding payload"))
        for item in _sequence(payload, name="embeddings")
    )


__all__ = [
    "BEHAVIOR_LABELS_SCHEMA_VERSION",
    "BehaviorEmbedding",
    "BehaviorFrameLabel",
    "BehaviorInterval",
    "BehaviorLabels",
]
