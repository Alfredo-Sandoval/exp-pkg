"""Canonical experiment ontology and first-class scientific relationships."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from xpkg.model._metadata_validation import (
    metadata_dict,
    optional_text,
    strict_required_text,
)
from xpkg.model.events import Event
from xpkg.model.identity import IdentityProvenanceRecord
from xpkg.model.labels import Labels
from xpkg.model.metadata import DatasetShareMetadata
from xpkg.model.session import (
    RecordingSession,
    SessionBehavior,
    SessionEventStream,
    SessionPose,
)
from xpkg.pose.annotations import Track
from xpkg.pose.trajectory import PoseTrack, PoseTrajectory


@dataclass(frozen=True, slots=True)
class Subject:
    """One biological subject that can participate in multiple sessions."""

    subject_id: str
    species: str | None = None
    strain: str | None = None
    sex: str | None = None
    genotype: str | None = None
    date_of_birth: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "subject_id",
            strict_required_text(self.subject_id, name="subject_id"),
        )
        for name in ("species", "strain", "sex", "genotype", "date_of_birth"):
            object.__setattr__(
                self,
                name,
                optional_text(getattr(self, name), name=f"subject {name}"),
            )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(metadata_dict(self.metadata, name="subject metadata")),
        )


@dataclass(frozen=True, slots=True)
class Protocol:
    """Versioned experimental or acquisition procedure."""

    protocol_id: str
    name: str
    version: str | None = None
    uri: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "protocol_id", strict_required_text(self.protocol_id, name="protocol_id")
        )
        object.__setattr__(self, "name", strict_required_text(self.name, name="protocol name"))
        object.__setattr__(self, "version", optional_text(self.version, name="protocol version"))
        object.__setattr__(self, "uri", optional_text(self.uri, name="protocol uri"))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(metadata_dict(self.metadata, name="protocol metadata")),
        )


@dataclass(frozen=True, slots=True)
class ExperimentalCondition:
    """Named treatment, cohort, or environmental condition."""

    condition_id: str
    name: str
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "condition_id",
            strict_required_text(self.condition_id, name="condition_id"),
        )
        object.__setattr__(self, "name", strict_required_text(self.name, name="condition name"))
        object.__setattr__(
            self,
            "description",
            optional_text(self.description, name="condition description"),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(metadata_dict(self.metadata, name="condition metadata")),
        )


@dataclass(frozen=True, slots=True)
class SessionSubjectLink:
    """Participation relationship between one session and one subject."""

    subject: Subject
    role: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.subject, Subject):
            raise TypeError("session subject link must contain a Subject.")
        object.__setattr__(self, "role", optional_text(self.role, name="subject session role"))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(metadata_dict(self.metadata, name="session subject link metadata")),
        )

    @property
    def subject_id(self) -> str:
        return self.subject.subject_id


@dataclass(frozen=True, slots=True)
class SessionProtocolLink:
    """Use relationship between one session and one protocol."""

    protocol: Protocol
    role: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, Protocol):
            raise TypeError("session protocol link must contain a Protocol.")
        object.__setattr__(self, "role", strict_required_text(self.role, name="protocol role"))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(metadata_dict(self.metadata, name="session protocol link metadata")),
        )

    @property
    def protocol_id(self) -> str:
        return self.protocol.protocol_id


@dataclass(frozen=True, slots=True)
class SessionConditionLink:
    """Application of one condition to a session or named session subjects."""

    condition: ExperimentalCondition
    subjects: tuple[Subject, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.condition, ExperimentalCondition):
            raise TypeError("session condition link must contain an ExperimentalCondition.")
        subjects = tuple(self.subjects)
        _require_unique_objects(subjects, Subject, "subject_id", "condition subject")
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(metadata_dict(self.metadata, name="session condition link metadata")),
        )

    @property
    def condition_id(self) -> str:
        return self.condition.condition_id


@dataclass(frozen=True, slots=True)
class SubjectTrackAssignment:
    """Time-bounded biological-identity assignment to one pose track."""

    subject: Subject
    pose: SessionPose
    track: Track | PoseTrack
    start_frame: int
    end_frame: int
    evidence: IdentityProvenanceRecord | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.subject, Subject):
            raise TypeError("subject track link must contain a Subject.")
        if not isinstance(self.pose, SessionPose):
            raise TypeError("subject track assignment must contain a SessionPose.")
        if not isinstance(self.track, Track | PoseTrack):
            raise TypeError("subject track assignment must contain a Track or PoseTrack.")
        for field_name in ("start_frame", "end_frame"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"subject track {field_name} must be an integer.")
            if value < 0:
                raise ValueError(f"subject track {field_name} must be non-negative.")
        if self.end_frame < self.start_frame:
            raise ValueError("subject track end_frame must be >= start_frame.")
        if self.evidence is not None:
            if not isinstance(self.evidence, IdentityProvenanceRecord):
                raise TypeError(
                    "subject track link evidence must be IdentityProvenanceRecord or None."
                )
            if self.evidence.track_id != self.track_id:
                raise ValueError("subject track link evidence must describe the linked track_id.")
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(metadata_dict(self.metadata, name="subject track link metadata")),
        )

    @property
    def subject_id(self) -> str:
        return self.subject.subject_id

    @property
    def pose_name(self) -> str:
        return self.pose.name

    @property
    def track_id(self) -> str:
        if isinstance(self.track, PoseTrack):
            return self.track.track_id
        return str(self.track.id)


@dataclass(frozen=True, slots=True)
class BehaviorSubjectLink:
    """Attribution of one behavior stream to one participating subject."""

    behavior: SessionBehavior
    subject: Subject
    role: str = "actor"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.behavior, SessionBehavior):
            raise TypeError("behavior subject link must contain a SessionBehavior.")
        if not isinstance(self.subject, Subject):
            raise TypeError("behavior subject link must contain a Subject.")
        object.__setattr__(
            self, "role", strict_required_text(self.role, name="behavior subject role")
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(metadata_dict(self.metadata, name="behavior subject metadata")),
        )

    @property
    def subject_id(self) -> str:
        return self.subject.subject_id


class EventRelationshipKind(StrEnum):
    """Supported semantic relationships between session events."""

    PRECEDES = "precedes"
    TRIGGERS = "triggers"
    RESPONSE_TO = "response_to"
    OUTCOME_OF = "outcome_of"
    PART_OF = "part_of"


@dataclass(frozen=True, slots=True)
class EventRelationship:
    """Typed relationship between two events in named session streams."""

    source_stream: SessionEventStream
    source_event: Event
    target_stream: SessionEventStream
    target_event: Event
    kind: EventRelationshipKind
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_stream, SessionEventStream) or not isinstance(
            self.target_stream, SessionEventStream
        ):
            raise TypeError("event relationship streams must be SessionEventStream objects.")
        if not isinstance(self.source_event, Event) or not isinstance(self.target_event, Event):
            raise TypeError("event relationship endpoints must be Event objects.")
        if not _contains_identity(self.source_stream.events.events, self.source_event):
            raise ValueError("event relationship source event is outside its stream.")
        if not _contains_identity(self.target_stream.events.events, self.target_event):
            raise ValueError("event relationship target event is outside its stream.")
        if not isinstance(self.kind, EventRelationshipKind):
            raise TypeError("event relationship kind must be EventRelationshipKind.")
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(metadata_dict(self.metadata, name="event relationship metadata")),
        )


@dataclass(frozen=True, slots=True)
class ExperimentSessionLink:
    """Containment and scientific-context relationship for one recording session."""

    session: RecordingSession
    subjects: tuple[SessionSubjectLink, ...] = ()
    protocols: tuple[SessionProtocolLink, ...] = ()
    conditions: tuple[SessionConditionLink, ...] = ()
    behavior_subjects: tuple[BehaviorSubjectLink, ...] = ()
    subject_track_assignments: tuple[SubjectTrackAssignment, ...] = ()
    event_relationships: tuple[EventRelationship, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.session, RecordingSession):
            raise TypeError("experiment session link must contain a RecordingSession.")
        subjects = tuple(self.subjects)
        protocols = tuple(self.protocols)
        conditions = tuple(self.conditions)
        behavior_subjects = tuple(self.behavior_subjects)
        assignments = tuple(self.subject_track_assignments)
        event_relationships = tuple(self.event_relationships)
        _require_unique_objects(subjects, SessionSubjectLink, "subject_id", "session subject")
        _require_unique_objects(protocols, SessionProtocolLink, "role", "session protocol role")
        _require_unique_objects(
            conditions, SessionConditionLink, "condition_id", "session condition"
        )
        _require_unique_behavior_subjects(behavior_subjects)
        _require_subject_track_assignments(assignments)
        _require_unique_event_relationships(event_relationships)
        participant_ids = {link.subject_id for link in subjects}
        for condition in conditions:
            unknown = {subject.subject_id for subject in condition.subjects} - participant_ids
            if unknown:
                raise ValueError(
                    "Session condition references non-participating subjects: "
                    f"{', '.join(sorted(unknown))}."
                )
        _require_behavior_subjects(self.session, behavior_subjects, participant_ids)
        _require_subject_track_targets(self.session, assignments, participant_ids)
        _require_event_relationships(self.session, event_relationships)
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "protocols", protocols)
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "behavior_subjects", behavior_subjects)
        object.__setattr__(self, "subject_track_assignments", assignments)
        object.__setattr__(self, "event_relationships", event_relationships)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(metadata_dict(self.metadata, name="experiment session link metadata")),
        )

    @property
    def session_id(self) -> str:
        return self.session.session_id


@dataclass(frozen=True, slots=True)
class Experiment:
    """Scientific experiment containing typed entities and recording sessions."""

    experiment_id: str
    title: str
    subjects: tuple[Subject, ...] = ()
    protocols: tuple[Protocol, ...] = ()
    conditions: tuple[ExperimentalCondition, ...] = ()
    session_links: tuple[ExperimentSessionLink, ...] = ()
    dataset_share: DatasetShareMetadata | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        subjects = tuple(self.subjects)
        protocols = tuple(self.protocols)
        conditions = tuple(self.conditions)
        links = tuple(self.session_links)
        _require_unique_objects(subjects, Subject, "subject_id", "experiment subject")
        _require_unique_objects(protocols, Protocol, "protocol_id", "experiment protocol")
        _require_unique_objects(
            conditions, ExperimentalCondition, "condition_id", "experiment condition"
        )
        _require_unique_objects(links, ExperimentSessionLink, "session_id", "experiment session")
        _require_registered_links(links, subjects, protocols, conditions)
        if self.dataset_share is not None and not isinstance(
            self.dataset_share, DatasetShareMetadata
        ):
            raise TypeError("experiment dataset_share must be DatasetShareMetadata or None.")
        object.__setattr__(
            self, "experiment_id", strict_required_text(self.experiment_id, name="experiment_id")
        )
        object.__setattr__(self, "title", strict_required_text(self.title, name="experiment title"))
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "protocols", protocols)
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "session_links", links)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(metadata_dict(self.metadata, name="experiment metadata")),
        )

    @property
    def session_ids(self) -> tuple[str, ...]:
        return tuple(link.session_id for link in self.session_links)

    @property
    def sessions(self) -> tuple[RecordingSession, ...]:
        return tuple(link.session for link in self.session_links)

    def session(self, session_id: str) -> RecordingSession:
        key = strict_required_text(session_id, name="session_id")
        for link in self.session_links:
            if link.session_id == key:
                return link.session
        raise KeyError(f"Experiment {self.experiment_id!r} has no session {key!r}.")


def _require_unique_objects(values: Sequence[object], cls: type, identity: str, name: str) -> None:
    identities: set[str] = set()
    for value in values:
        if not isinstance(value, cls):
            raise TypeError(f"{name} entries must be {cls.__name__} objects.")
        key = getattr(value, identity)
        if key in identities:
            raise ValueError(f"Duplicate {name} {identity}: {key!r}.")
        identities.add(key)


def _require_registered_links(
    links: tuple[ExperimentSessionLink, ...],
    subjects: tuple[Subject, ...],
    protocols: tuple[Protocol, ...],
    conditions: tuple[ExperimentalCondition, ...],
) -> None:
    registered = {
        Subject: {item.subject_id: item for item in subjects},
        Protocol: {item.protocol_id: item for item in protocols},
        ExperimentalCondition: {item.condition_id: item for item in conditions},
    }
    for link in links:
        _require_registered(link.subjects, registered[Subject], "subject", "subject")
        _require_registered(link.protocols, registered[Protocol], "protocol", "protocol")
        _require_registered(
            link.conditions,
            registered[ExperimentalCondition],
            "condition",
            "condition",
        )


def _require_registered(
    links: Sequence[object], registered: Mapping[str, object], attribute: str, name: str
) -> None:
    for link in links:
        value = getattr(link, attribute)
        identity = getattr(value, f"{name}_id")
        if registered.get(identity) is not value:
            raise ValueError(f"Session references unregistered experiment {name} {identity!r}.")


def _require_behavior_subjects(
    session: RecordingSession,
    links: tuple[BehaviorSubjectLink, ...],
    subject_ids: set[str],
) -> None:
    for link in links:
        if not _contains_identity(session.behaviors, link.behavior):
            raise ValueError("Behavior subject link references behavior outside its session.")
        if link.subject_id not in subject_ids:
            raise ValueError(
                f"Behavior subject link references non-participating subject {link.subject_id!r}."
            )


def _require_unique_behavior_subjects(links: tuple[BehaviorSubjectLink, ...]) -> None:
    if any(not isinstance(link, BehaviorSubjectLink) for link in links):
        raise TypeError("session behavior_subjects entries must be BehaviorSubjectLink objects.")
    identities = [(link.behavior.name, link.subject_id, link.role) for link in links]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate session behavior-subject relationship.")


def _require_subject_track_assignments(
    links: tuple[SubjectTrackAssignment, ...],
) -> None:
    if any(not isinstance(link, SubjectTrackAssignment) for link in links):
        raise TypeError(
            "session subject_track_assignments entries must be SubjectTrackAssignment objects."
        )
    for index, link in enumerate(links):
        for other in links[index + 1 :]:
            same_track = (link.pose_name, link.track_id) == (
                other.pose_name,
                other.track_id,
            )
            overlaps = link.start_frame <= other.end_frame and other.start_frame <= link.end_frame
            if same_track and overlaps:
                raise ValueError("Subject track assignments cannot overlap on one pose track.")


def _require_subject_track_targets(
    session: RecordingSession,
    links: tuple[SubjectTrackAssignment, ...],
    participant_ids: set[str],
) -> None:
    for link in links:
        if link.subject_id not in participant_ids:
            raise ValueError(
                f"Subject track link references non-participating subject {link.subject_id!r}."
            )
        if not _contains_identity(session.poses, link.pose):
            raise ValueError("Subject track assignment references a pose outside its session.")
        if not _pose_owns_track(link.pose.data, link.track):
            raise ValueError("Subject track assignment references a track outside its pose.")
        frame_count = _pose_frame_count(link.pose.data)
        if link.end_frame >= frame_count:
            raise ValueError(
                f"Subject track assignment end_frame {link.end_frame} exceeds pose "
                f"{link.pose_name!r} frame range 0..{frame_count - 1}."
            )


def _require_unique_event_relationships(
    links: tuple[EventRelationship, ...],
) -> None:
    if any(not isinstance(link, EventRelationship) for link in links):
        raise TypeError("session event_relationships entries must be EventRelationship objects.")
    identities = [
        (
            link.source_stream.name,
            link.source_event.event_id,
            link.target_stream.name,
            link.target_event.event_id,
            link.kind,
        )
        for link in links
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate session event relationship.")


def _require_event_relationships(
    session: RecordingSession, links: tuple[EventRelationship, ...]
) -> None:
    for link in links:
        if not _contains_identity(session.event_streams, link.source_stream):
            raise ValueError("Event relationship source stream is outside its session.")
        if not _contains_identity(session.event_streams, link.target_stream):
            raise ValueError("Event relationship target stream is outside its session.")


def _pose_owns_track(data: Labels | PoseTrajectory, track: Track | PoseTrack) -> bool:
    if isinstance(data, PoseTrajectory):
        return isinstance(track, PoseTrack) and _contains_identity(data.tracks, track)
    return isinstance(track, Track) and _contains_identity(data.tracks, track)


def _contains_identity(values: Sequence[object], target: object) -> bool:
    return any(value is target for value in values)


def _pose_frame_count(data: Labels | PoseTrajectory) -> int:
    if isinstance(data, PoseTrajectory):
        return data.n_frames
    indexes = [frame.frame_idx for frame in data.labeled_frames]
    if not indexes:
        raise ValueError("Subject track assignments require pose frames.")
    return max(indexes) + 1


__all__ = [
    "Experiment",
    "BehaviorSubjectLink",
    "EventRelationship",
    "EventRelationshipKind",
    "ExperimentalCondition",
    "ExperimentSessionLink",
    "Protocol",
    "SessionConditionLink",
    "SessionProtocolLink",
    "SessionSubjectLink",
    "Subject",
    "SubjectTrackAssignment",
]
