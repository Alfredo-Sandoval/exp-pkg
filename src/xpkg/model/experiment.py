"""Canonical experiment ontology and first-class scientific relationships."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from xpkg.io.labels.model import Labels
from xpkg.model._metadata_validation import (
    metadata_dict,
    optional_text,
    strict_required_text,
)
from xpkg.model.identity import IdentityProvenanceRecord
from xpkg.model.metadata import DatasetShareMetadata
from xpkg.model.session import RecordingSession
from xpkg.pose.trajectory import PoseTrajectory


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
class SubjectTrackLink:
    """Biological-identity assignment from one subject to one pose track."""

    subject: Subject
    pose_name: str
    track_id: str
    evidence: IdentityProvenanceRecord | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.subject, Subject):
            raise TypeError("subject track link must contain a Subject.")
        object.__setattr__(
            self, "pose_name", strict_required_text(self.pose_name, name="pose name")
        )
        object.__setattr__(
            self, "track_id", strict_required_text(self.track_id, name="pose track id")
        )
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


@dataclass(frozen=True, slots=True)
class ExperimentSessionLink:
    """Containment and scientific-context relationship for one recording session."""

    session: RecordingSession
    subjects: tuple[SessionSubjectLink, ...] = ()
    protocols: tuple[SessionProtocolLink, ...] = ()
    conditions: tuple[SessionConditionLink, ...] = ()
    subject_tracks: tuple[SubjectTrackLink, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.session, RecordingSession):
            raise TypeError("experiment session link must contain a RecordingSession.")
        subjects = tuple(self.subjects)
        protocols = tuple(self.protocols)
        conditions = tuple(self.conditions)
        subject_tracks = tuple(self.subject_tracks)
        _require_unique_objects(subjects, SessionSubjectLink, "subject_id", "session subject")
        _require_unique_objects(protocols, SessionProtocolLink, "role", "session protocol role")
        _require_unique_objects(
            conditions, SessionConditionLink, "condition_id", "session condition"
        )
        _require_unique_subject_tracks(subject_tracks)
        participant_ids = {link.subject_id for link in subjects}
        for condition in conditions:
            unknown = {subject.subject_id for subject in condition.subjects} - participant_ids
            if unknown:
                raise ValueError(
                    "Session condition references non-participating subjects: "
                    f"{', '.join(sorted(unknown))}."
                )
        _require_behavior_subjects(self.session, participant_ids)
        _require_subject_tracks(self.session, subject_tracks, participant_ids)
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "protocols", protocols)
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "subject_tracks", subject_tracks)
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


def _require_unique_objects(
    values: Sequence[object], cls: type, identity: str, name: str
) -> None:
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
        if registered.get(identity) != value:
            raise ValueError(f"Session references unregistered experiment {name} {identity!r}.")


def _require_behavior_subjects(session: RecordingSession, subject_ids: set[str]) -> None:
    referenced = {
        behavior.labels.subject_id
        for behavior in session.behaviors
        if behavior.labels.subject_id is not None
    }
    unknown = referenced - subject_ids
    if unknown:
        raise ValueError(
            "Session behavior labels reference non-participating subjects: "
            f"{', '.join(sorted(unknown))}."
        )


def _require_unique_subject_tracks(links: tuple[SubjectTrackLink, ...]) -> None:
    if any(not isinstance(link, SubjectTrackLink) for link in links):
        raise TypeError("session subject_tracks entries must be SubjectTrackLink objects.")
    identities = [(link.pose_name, link.track_id) for link in links]
    if len(set(identities)) != len(identities):
        raise ValueError("Each session pose track can be assigned to at most one subject.")


def _require_subject_tracks(
    session: RecordingSession,
    links: tuple[SubjectTrackLink, ...],
    participant_ids: set[str],
) -> None:
    for link in links:
        if link.subject_id not in participant_ids:
            raise ValueError(
                f"Subject track link references non-participating subject {link.subject_id!r}."
            )
        pose = next((item for item in session.poses if item.name == link.pose_name), None)
        if pose is None:
            raise ValueError(f"Subject track link references unknown pose {link.pose_name!r}.")
        track_ids = _pose_track_ids(pose.data)
        if link.track_id not in track_ids:
            raise ValueError(
                f"Subject track link references unknown track {link.track_id!r} "
                f"on pose {link.pose_name!r}."
            )


def _pose_track_ids(data: Labels | PoseTrajectory) -> set[str]:
    if isinstance(data, PoseTrajectory):
        return set(data.track_ids)
    return {str(track.id) for track in data.tracks}


__all__ = [
    "Experiment",
    "ExperimentalCondition",
    "ExperimentSessionLink",
    "Protocol",
    "SessionConditionLink",
    "SessionProtocolLink",
    "SessionSubjectLink",
    "Subject",
    "SubjectTrackLink",
]
