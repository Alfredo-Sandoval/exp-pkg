"""Machine-readable catalog for the canonical xpkg ontology."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from typing import Any

from xpkg.model.behavior import BehaviorLabels
from xpkg.model.events import Event, EventTable
from xpkg.model.experiment import (
    BehaviorSubjectLink,
    EventRelationship,
    Experiment,
    ExperimentalCondition,
    ExperimentSessionLink,
    Protocol,
    SessionConditionLink,
    SessionProtocolLink,
    SessionSubjectLink,
    Subject,
    SubjectTrackAssignment,
)
from xpkg.model.metadata import SourceProvenance
from xpkg.model.session import (
    RecordingSession,
    SessionBehavior,
    SessionCalibration,
    SessionEventStream,
    SessionPose,
    SessionSignal,
    SessionVideo,
    TimebaseAlignment,
)
from xpkg.model.time import Timebase
from xpkg.pose.trajectory import PoseTrack, PoseTrajectory

ONTOLOGY_SCHEMA_VERSION = 4

_OBJECT_TYPES = (
    Subject,
    Protocol,
    ExperimentalCondition,
    Experiment,
    RecordingSession,
    SessionSignal,
    SessionVideo,
    SessionPose,
    SessionBehavior,
    SessionCalibration,
    SessionEventStream,
    EventTable,
    Event,
    BehaviorLabels,
    PoseTrack,
    PoseTrajectory,
    Timebase,
    TimebaseAlignment,
    SourceProvenance,
    ExperimentSessionLink,
    SessionSubjectLink,
    SessionProtocolLink,
    SessionConditionLink,
    BehaviorSubjectLink,
    SubjectTrackAssignment,
    EventRelationship,
)

_LINK_TYPES = (
    ("ExperimentSessionLink", "Experiment", "RecordingSession", ExperimentSessionLink),
    ("SessionSubjectLink", "RecordingSession", "Subject", SessionSubjectLink),
    ("SessionProtocolLink", "RecordingSession", "Protocol", SessionProtocolLink),
    (
        "SessionConditionLink",
        "RecordingSession",
        "ExperimentalCondition",
        SessionConditionLink,
    ),
    ("RecordingSessionSignal", "RecordingSession", "SessionSignal", None),
    ("RecordingSessionVideo", "RecordingSession", "SessionVideo", None),
    ("RecordingSessionPose", "RecordingSession", "SessionPose", None),
    ("RecordingSessionBehavior", "RecordingSession", "SessionBehavior", None),
    ("RecordingSessionCalibration", "RecordingSession", "SessionCalibration", None),
    ("RecordingSessionEventStream", "RecordingSession", "SessionEventStream", None),
    ("SessionPoseVideo", "SessionPose", "SessionVideo", None),
    ("SessionPoseCalibration", "SessionPose", "SessionCalibration", None),
    ("SessionBehaviorVideo", "SessionBehavior", "SessionVideo", None),
    ("SessionBehaviorPose", "SessionBehavior", "SessionPose", None),
    ("BehaviorSubjectLink", "SessionBehavior", "Subject", BehaviorSubjectLink),
    ("SubjectTrackAssignment", "Subject", "PoseTrack", SubjectTrackAssignment),
    ("EventRelationship", "Event", "Event", EventRelationship),
    ("TimebaseAlignment", "Timebase", "Timebase", TimebaseAlignment),
)


def ontology_document() -> dict[str, Any]:
    """Return the generated object, property, and relationship catalog."""
    return {
        "format": "xpkg.ontology",
        "schema_version": ONTOLOGY_SCHEMA_VERSION,
        "object_types": [_object_type_payload(value) for value in _OBJECT_TYPES],
        "link_types": [
            {
                "name": name,
                "source_type": source,
                "target_type": target,
                "properties": [] if value is None else _field_names(value),
            }
            for name, source, target, value in _LINK_TYPES
        ],
    }


def recording_session_json_schema() -> dict[str, Any]:
    """Return the schema-4 recording-session document envelope."""
    return _document_schema(
        title="xpkg recording session",
        format_name="xpkg.recording-session",
        payload_key="session",
        payload_properties=(
            "session_id",
            "title",
            "acquisition",
            "session_timebase_name",
            "timebases",
            "signals",
            "videos",
            "poses",
            "behaviors",
            "calibrations",
            "alignments",
            "event_streams",
            "metadata",
        ),
        required_payload_properties=(
            "session_id",
            "session_timebase_name",
            "timebases",
            "signals",
            "videos",
            "poses",
            "behaviors",
            "calibrations",
            "alignments",
            "event_streams",
            "metadata",
        ),
    )


def experiment_json_schema() -> dict[str, Any]:
    """Return the schema-4 experiment document envelope."""
    return _document_schema(
        title="xpkg experiment",
        format_name="xpkg.experiment",
        payload_key="experiment",
        payload_properties=(
            "experiment_id",
            "title",
            "subjects",
            "protocols",
            "conditions",
            "sessions",
            "dataset_share",
            "metadata",
        ),
        required_payload_properties=(
            "experiment_id",
            "title",
            "subjects",
            "protocols",
            "conditions",
            "sessions",
            "dataset_share",
            "metadata",
        ),
    )


def ontology_schema_documents() -> dict[str, dict[str, Any]]:
    """Return every generated schema document keyed by output filename."""
    return {
        "ontology.json": ontology_document(),
        "recording-session.schema.json": recording_session_json_schema(),
        "experiment.schema.json": experiment_json_schema(),
    }


def _object_type_payload(value: type) -> dict[str, Any]:
    return {
        "name": value.__name__,
        "properties": _field_names(value),
    }


def _field_names(value: type) -> list[str]:
    if not is_dataclass(value):
        raise TypeError(f"Ontology object type must be a dataclass: {value!r}.")
    return [field.name for field in fields(value)]


def _document_schema(
    *,
    title: str,
    format_name: str,
    payload_key: str,
    payload_properties: Sequence[str],
    required_payload_properties: Sequence[str],
) -> dict[str, Any]:
    properties = {name: {} for name in payload_properties}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "format": {"const": format_name},
            "schema_version": {"const": ONTOLOGY_SCHEMA_VERSION},
            "payload": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    payload_key: {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": properties,
                        "required": list(required_payload_properties),
                    },
                    "metadata": {"type": "object"},
                },
                "required": [payload_key, "metadata"],
            },
        },
        "required": ["format", "schema_version", "payload"],
    }


__all__ = [
    "ONTOLOGY_SCHEMA_VERSION",
    "experiment_json_schema",
    "ontology_document",
    "ontology_schema_documents",
    "recording_session_json_schema",
]
