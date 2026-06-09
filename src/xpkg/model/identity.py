"""Identity-provenance records for multi-animal labels."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from xpkg.model._metadata_validation import (
    metadata_dict,
    optional_text,
    payload_mapping,
    required_text,
)

IDENTITY_PROVENANCE_SCHEMA_VERSION = "xpkg.identity_provenance.v1"
IDENTITY_SOURCES = frozenset({"mot", "reid", "manual", "mixed", "unknown"})


def _identity_source(value: object | None, *, name: str) -> str:
    source = str(value or "unknown").strip().lower()
    if source not in IDENTITY_SOURCES:
        allowed = ", ".join(sorted(IDENTITY_SOURCES))
        raise ValueError(f"{name} must be one of {allowed}; got {source!r}.")
    return source


def _frame_index(value: Any, *, name: str) -> int:
    coerced = int(value)
    if coerced < 0:
        raise ValueError(f"{name} must be non-negative, got {coerced}.")
    return coerced


def _optional_confidence(value: Any | None, *, name: str) -> float | None:
    if value is None:
        return None
    coerced = float(value)
    if not np.isfinite(coerced) or not 0.0 <= coerced <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1, got {coerced!r}.")
    return coerced


@dataclass(frozen=True, slots=True)
class IdentityConfidenceSpan:
    """Frame span where an identity source and optional confidence apply."""

    start_frame: int
    end_frame: int
    video_id: str | None = None
    identity_source: str = "unknown"
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        start_frame = _frame_index(self.start_frame, name="identity span start_frame")
        end_frame = _frame_index(self.end_frame, name="identity span end_frame")
        if end_frame < start_frame:
            raise ValueError("identity span end_frame must be >= start_frame.")
        object.__setattr__(self, "start_frame", start_frame)
        object.__setattr__(self, "end_frame", end_frame)
        object.__setattr__(self, "video_id", optional_text(self.video_id, name="video_id"))
        object.__setattr__(
            self,
            "identity_source",
            _identity_source(self.identity_source, name="identity_source"),
        )
        object.__setattr__(
            self,
            "confidence",
            _optional_confidence(self.confidence, name="identity confidence"),
        )
        object.__setattr__(self, "metadata", metadata_dict(self.metadata, name="span metadata"))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "identity_source": self.identity_source,
        }
        if self.video_id is not None:
            payload["video_id"] = self.video_id
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IdentityConfidenceSpan:
        fields = payload_mapping(payload, name="identity span payload")
        return cls(
            start_frame=fields.get("start_frame", 0),
            end_frame=fields.get("end_frame", 0),
            video_id=fields.get("video_id"),
            identity_source=fields.get("identity_source", "unknown"),
            confidence=fields.get("confidence"),
            metadata=metadata_dict(fields.get("metadata"), name="span metadata"),
        )


@dataclass(frozen=True, slots=True)
class IdentityEvent:
    """Event marking identity swaps, corrections, or source-reported changes."""

    kind: str
    frame: int
    video_id: str | None = None
    from_track_id: str | None = None
    to_track_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", required_text(self.kind, name="identity event kind"))
        object.__setattr__(self, "frame", _frame_index(self.frame, name="identity event frame"))
        object.__setattr__(self, "video_id", optional_text(self.video_id, name="video_id"))
        object.__setattr__(
            self,
            "from_track_id",
            optional_text(self.from_track_id, name="from_track_id"),
        )
        object.__setattr__(
            self,
            "to_track_id",
            optional_text(self.to_track_id, name="to_track_id"),
        )
        object.__setattr__(self, "metadata", metadata_dict(self.metadata, name="event metadata"))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind, "frame": self.frame}
        for key in ("video_id", "from_track_id", "to_track_id"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IdentityEvent:
        fields = payload_mapping(payload, name="identity event payload")
        return cls(
            kind=fields.get("kind", ""),
            frame=fields.get("frame", 0),
            video_id=fields.get("video_id"),
            from_track_id=fields.get("from_track_id"),
            to_track_id=fields.get("to_track_id"),
            metadata=metadata_dict(fields.get("metadata"), name="event metadata"),
        )


@dataclass(frozen=True, slots=True)
class IdentityProofreadingSpan:
    """Frame span that was manually reviewed for identity correctness."""

    start_frame: int
    end_frame: int
    video_id: str | None = None
    reviewed: bool = True
    corrected: bool | None = None
    reviewer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        start_frame = _frame_index(self.start_frame, name="proofreading start_frame")
        end_frame = _frame_index(self.end_frame, name="proofreading end_frame")
        if end_frame < start_frame:
            raise ValueError("proofreading end_frame must be >= start_frame.")
        object.__setattr__(self, "start_frame", start_frame)
        object.__setattr__(self, "end_frame", end_frame)
        object.__setattr__(self, "video_id", optional_text(self.video_id, name="video_id"))
        object.__setattr__(self, "reviewed", bool(self.reviewed))
        object.__setattr__(
            self,
            "corrected",
            None if self.corrected is None else bool(self.corrected),
        )
        object.__setattr__(self, "reviewer", optional_text(self.reviewer, name="reviewer"))
        object.__setattr__(
            self,
            "metadata",
            metadata_dict(self.metadata, name="proofreading metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "reviewed": self.reviewed,
        }
        if self.video_id is not None:
            payload["video_id"] = self.video_id
        if self.corrected is not None:
            payload["corrected"] = self.corrected
        if self.reviewer is not None:
            payload["reviewer"] = self.reviewer
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IdentityProofreadingSpan:
        fields = payload_mapping(payload, name="identity proofreading payload")
        return cls(
            start_frame=fields.get("start_frame", 0),
            end_frame=fields.get("end_frame", 0),
            video_id=fields.get("video_id"),
            reviewed=bool(fields.get("reviewed", True)),
            corrected=fields.get("corrected"),
            reviewer=fields.get("reviewer"),
            metadata=metadata_dict(fields.get("metadata"), name="proofreading metadata"),
        )


def _spans(
    values: Iterable[IdentityConfidenceSpan | Mapping[str, Any]] | None,
) -> tuple[IdentityConfidenceSpan, ...]:
    if values is None:
        return ()
    return tuple(
        item if isinstance(item, IdentityConfidenceSpan) else IdentityConfidenceSpan.from_dict(item)
        for item in values
    )


def _events(
    values: Iterable[IdentityEvent | Mapping[str, Any]] | None,
) -> tuple[IdentityEvent, ...]:
    if values is None:
        return ()
    return tuple(
        item if isinstance(item, IdentityEvent) else IdentityEvent.from_dict(item)
        for item in values
    )


def _proofreading(
    values: Iterable[IdentityProofreadingSpan | Mapping[str, Any]] | None,
) -> tuple[IdentityProofreadingSpan, ...]:
    if values is None:
        return ()
    return tuple(
        item
        if isinstance(item, IdentityProofreadingSpan)
        else IdentityProofreadingSpan.from_dict(item)
        for item in values
    )


@dataclass(frozen=True, slots=True)
class IdentityProvenanceRecord:
    """Companion provenance for one labels track identity."""

    track_id: str
    track_name: str | None = None
    source_tool: str | None = None
    source_file: str | None = None
    identity_source: str = "unknown"
    spans: tuple[IdentityConfidenceSpan, ...] = ()
    events: tuple[IdentityEvent, ...] = ()
    proofreading: tuple[IdentityProofreadingSpan, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", required_text(self.track_id, name="track_id"))
        object.__setattr__(self, "track_name", optional_text(self.track_name, name="track_name"))
        object.__setattr__(
            self,
            "source_tool",
            optional_text(self.source_tool, name="source_tool"),
        )
        object.__setattr__(
            self,
            "source_file",
            optional_text(self.source_file, name="source_file"),
        )
        object.__setattr__(
            self,
            "identity_source",
            _identity_source(self.identity_source, name="identity_source"),
        )
        object.__setattr__(self, "spans", _spans(self.spans))
        object.__setattr__(self, "events", _events(self.events))
        object.__setattr__(self, "proofreading", _proofreading(self.proofreading))
        object.__setattr__(self, "metadata", metadata_dict(self.metadata, name="identity metadata"))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "track_id": self.track_id,
            "identity_source": self.identity_source,
        }
        for key in ("track_name", "source_tool", "source_file"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.spans:
            payload["spans"] = [span.to_dict() for span in self.spans]
        if self.events:
            payload["events"] = [event.to_dict() for event in self.events]
        if self.proofreading:
            payload["proofreading"] = [span.to_dict() for span in self.proofreading]
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IdentityProvenanceRecord:
        fields = payload_mapping(payload, name="identity provenance payload")
        return cls(
            track_id=fields.get("track_id", ""),
            track_name=fields.get("track_name"),
            source_tool=fields.get("source_tool"),
            source_file=fields.get("source_file"),
            identity_source=fields.get("identity_source", "unknown"),
            spans=_spans(fields.get("spans")),
            events=_events(fields.get("events")),
            proofreading=_proofreading(fields.get("proofreading")),
            metadata=metadata_dict(fields.get("metadata"), name="identity metadata"),
        )


def _coerce_records(raw_records: object) -> list[IdentityProvenanceRecord]:
    if isinstance(raw_records, str | bytes) or not isinstance(raw_records, Iterable):
        raise TypeError("identity provenance records must be an iterable.")
    records: list[IdentityProvenanceRecord] = []
    for item in raw_records:
        if isinstance(item, IdentityProvenanceRecord):
            records.append(item)
            continue
        if not isinstance(item, Mapping):
            raise TypeError("identity provenance records must be mappings.")
        records.append(
            IdentityProvenanceRecord.from_dict(
                payload_mapping(item, name="identity provenance record")
            )
        )
    return records


def identity_provenance_records(payload: object) -> list[IdentityProvenanceRecord]:
    """Hydrate identity-provenance records from a JSON companion payload."""

    if payload is None:
        return []
    if not isinstance(payload, Mapping):
        raise TypeError("identity provenance payload must be a mapping.")
    fields = payload_mapping(payload, name="identity provenance payload")
    schema = str(fields.get("schema", "")).strip()
    if schema != IDENTITY_PROVENANCE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported identity provenance schema "
            f"{schema!r}; expected {IDENTITY_PROVENANCE_SCHEMA_VERSION!r}."
        )
    return _coerce_records(fields.get("records", ()))


def identity_provenance_payload(
    records: Iterable[IdentityProvenanceRecord | Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Return the JSON companion payload for identity provenance records."""

    hydrated = _coerce_records(records)
    if not hydrated:
        return None
    return {
        "schema": IDENTITY_PROVENANCE_SCHEMA_VERSION,
        "records": [record.to_dict() for record in hydrated],
    }


__all__ = [
    "IDENTITY_PROVENANCE_SCHEMA_VERSION",
    "IDENTITY_SOURCES",
    "IdentityConfidenceSpan",
    "IdentityEvent",
    "IdentityProofreadingSpan",
    "IdentityProvenanceRecord",
    "identity_provenance_payload",
    "identity_provenance_records",
]
