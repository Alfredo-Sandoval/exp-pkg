"""Versioned JSON exchange for the canonical experiment ontology."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from xpkg._core.json_utils import load_json_dict, write_json
from xpkg.io.session_json import (
    recording_session_document,
    recording_session_from_document,
)
from xpkg.model.experiment import (
    Experiment,
    ExperimentalCondition,
    ExperimentSessionLink,
    Protocol,
    SessionConditionLink,
    SessionProtocolLink,
    SessionSubjectLink,
    Subject,
    SubjectTrackLink,
)
from xpkg.model.metadata import DatasetShareMetadata

EXPERIMENT_FORMAT = "xpkg.experiment"
EXPERIMENT_SCHEMA_VERSION = 3


def experiment_document(
    experiment: Experiment,
    *,
    document_metadata: Mapping[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Serialize one valid experiment into its versioned JSON document."""
    if not isinstance(experiment, Experiment):
        raise TypeError(f"experiment must be an Experiment, got {experiment!r}.")
    root = None if project_root is None else Path(project_root)
    return {
        "format": EXPERIMENT_FORMAT,
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "payload": {
            "experiment": {
                "experiment_id": experiment.experiment_id,
                "title": experiment.title,
                "subjects": [_subject_payload(item) for item in experiment.subjects],
                "protocols": [_protocol_payload(item) for item in experiment.protocols],
                "conditions": [_condition_payload(item) for item in experiment.conditions],
                "sessions": [
                    _session_link_payload(link, root)
                    for link in experiment.session_links
                ],
                "dataset_share": (
                    None
                    if experiment.dataset_share is None
                    else experiment.dataset_share.to_dict()
                ),
                "metadata": dict(experiment.metadata),
            },
            "metadata": dict(document_metadata or {}),
        },
    }


def experiment_from_document(
    document: Mapping[str, Any], *, project_root: str | Path | None = None
) -> Experiment:
    """Parse a versioned document into a valid experiment object."""
    payload = _document_payload(document)
    raw_experiment = _required_mapping(payload, "experiment", context="experiment payload")
    root = None if project_root is None else Path(project_root)
    subjects = tuple(
        _subject_from_payload(item)
        for item in _required_mapping_sequence(raw_experiment, "subjects", "experiment")
    )
    protocols = tuple(
        _protocol_from_payload(item)
        for item in _required_mapping_sequence(raw_experiment, "protocols", "experiment")
    )
    conditions = tuple(
        _condition_from_payload(item)
        for item in _required_mapping_sequence(raw_experiment, "conditions", "experiment")
    )
    return Experiment(
        experiment_id=_required_str(raw_experiment, "experiment_id", context="experiment"),
        title=_required_str(raw_experiment, "title", context="experiment"),
        subjects=subjects,
        protocols=protocols,
        conditions=conditions,
        session_links=tuple(
            _session_link_from_payload(item, root, subjects, protocols, conditions)
            for item in _required_mapping_sequence(raw_experiment, "sessions", "experiment")
        ),
        dataset_share=_dataset_share_from_payload(raw_experiment.get("dataset_share")),
        metadata=dict(_required_mapping(raw_experiment, "metadata", context="experiment")),
    )


def write_experiment_json(
    path: str | Path,
    experiment: Experiment,
    *,
    document_metadata: Mapping[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """Write one experiment document."""
    target = Path(path)
    write_json(
        target,
        experiment_document(
            experiment,
            document_metadata=document_metadata,
            project_root=project_root,
        ),
        indent=None,
        sort_keys=False,
        ensure_ascii=True,
        compact=True,
    )
    return target


def read_experiment_json(
    path: str | Path, *, project_root: str | Path | None = None
) -> Experiment:
    """Read one experiment document."""
    return experiment_from_document(load_json_dict(path), project_root=project_root)


def read_experiment_metadata_json(path: str | Path) -> dict[str, Any]:
    """Read experiment metadata without hydrating nested recording sessions."""
    document = load_json_dict(path)
    payload = _document_payload(document)
    experiment = _required_mapping(payload, "experiment", context="experiment payload")
    return dict(_required_mapping(experiment, "metadata", context="experiment"))


def _session_link_payload(
    link: ExperimentSessionLink, project_root: Path | None
) -> dict[str, Any]:
    return {
        "session": recording_session_document(link.session, project_root=project_root),
        "subjects": [
            {
                "subject_id": item.subject_id,
                "role": item.role,
                "metadata": dict(item.metadata),
            }
            for item in link.subjects
        ],
        "protocols": [
            {
                "protocol_id": item.protocol_id,
                "role": item.role,
                "metadata": dict(item.metadata),
            }
            for item in link.protocols
        ],
        "conditions": [
            {
                "condition_id": item.condition_id,
                "subject_ids": [subject.subject_id for subject in item.subjects],
                "metadata": dict(item.metadata),
            }
            for item in link.conditions
        ],
        "subject_tracks": [
            {
                "subject_id": item.subject_id,
                "pose_name": item.pose_name,
                "track_id": item.track_id,
                "evidence": None if item.evidence is None else item.evidence.to_dict(),
                "metadata": dict(item.metadata),
            }
            for item in link.subject_tracks
        ],
        "metadata": dict(link.metadata),
    }


def _session_link_from_payload(
    payload: Mapping[str, Any],
    project_root: Path | None,
    subjects: tuple[Subject, ...],
    protocols: tuple[Protocol, ...],
    conditions: tuple[ExperimentalCondition, ...],
) -> ExperimentSessionLink:
    subject_lookup = {item.subject_id: item for item in subjects}
    protocol_lookup = {item.protocol_id: item for item in protocols}
    condition_lookup = {item.condition_id: item for item in conditions}
    return ExperimentSessionLink(
        session=recording_session_from_document(
            _required_mapping(payload, "session", context="experiment session link"),
            project_root=project_root,
        ),
        subjects=tuple(
            _subject_link_from_payload(item, subject_lookup)
            for item in _required_mapping_sequence(
                payload, "subjects", "experiment session link"
            )
        ),
        protocols=tuple(
            _protocol_link_from_payload(item, protocol_lookup)
            for item in _required_mapping_sequence(
                payload, "protocols", "experiment session link"
            )
        ),
        conditions=tuple(
            _condition_link_from_payload(item, condition_lookup, subject_lookup)
            for item in _required_mapping_sequence(
                payload, "conditions", "experiment session link"
            )
        ),
        subject_tracks=tuple(
            _subject_track_link_from_payload(item, subject_lookup)
            for item in _required_mapping_sequence(
                payload, "subject_tracks", "experiment session link"
            )
        ),
        metadata=dict(
            _required_mapping(payload, "metadata", context="experiment session link")
        ),
    )


def _subject_payload(subject: Subject) -> dict[str, Any]:
    return {
        "subject_id": subject.subject_id,
        "species": subject.species,
        "strain": subject.strain,
        "sex": subject.sex,
        "genotype": subject.genotype,
        "date_of_birth": subject.date_of_birth,
        "metadata": dict(subject.metadata),
    }


def _subject_from_payload(payload: Mapping[str, Any]) -> Subject:
    return Subject(
        subject_id=_required_str(payload, "subject_id", context="subject"),
        species=_optional_str(payload.get("species"), name="subject species"),
        strain=_optional_str(payload.get("strain"), name="subject strain"),
        sex=_optional_str(payload.get("sex"), name="subject sex"),
        genotype=_optional_str(payload.get("genotype"), name="subject genotype"),
        date_of_birth=_optional_str(
            payload.get("date_of_birth"), name="subject date_of_birth"
        ),
        metadata=dict(_required_mapping(payload, "metadata", context="subject")),
    )


def _protocol_payload(protocol: Protocol) -> dict[str, Any]:
    return {
        "protocol_id": protocol.protocol_id,
        "name": protocol.name,
        "version": protocol.version,
        "uri": protocol.uri,
        "metadata": dict(protocol.metadata),
    }


def _protocol_from_payload(payload: Mapping[str, Any]) -> Protocol:
    return Protocol(
        protocol_id=_required_str(payload, "protocol_id", context="protocol"),
        name=_required_str(payload, "name", context="protocol"),
        version=_optional_str(payload.get("version"), name="protocol version"),
        uri=_optional_str(payload.get("uri"), name="protocol uri"),
        metadata=dict(_required_mapping(payload, "metadata", context="protocol")),
    )


def _condition_payload(condition: ExperimentalCondition) -> dict[str, Any]:
    return {
        "condition_id": condition.condition_id,
        "name": condition.name,
        "description": condition.description,
        "metadata": dict(condition.metadata),
    }


def _condition_from_payload(payload: Mapping[str, Any]) -> ExperimentalCondition:
    return ExperimentalCondition(
        condition_id=_required_str(payload, "condition_id", context="condition"),
        name=_required_str(payload, "name", context="condition"),
        description=_optional_str(
            payload.get("description"), name="condition description"
        ),
        metadata=dict(_required_mapping(payload, "metadata", context="condition")),
    )


def _subject_link_from_payload(
    payload: Mapping[str, Any], subjects: Mapping[str, Subject]
) -> SessionSubjectLink:
    subject_id = _required_str(payload, "subject_id", context="session subject link")
    return SessionSubjectLink(
        subject=_lookup(subjects, subject_id, "subject"),
        role=_optional_str(payload.get("role"), name="session subject role"),
        metadata=dict(_required_mapping(payload, "metadata", context="session subject link")),
    )


def _protocol_link_from_payload(
    payload: Mapping[str, Any], protocols: Mapping[str, Protocol]
) -> SessionProtocolLink:
    protocol_id = _required_str(payload, "protocol_id", context="session protocol link")
    return SessionProtocolLink(
        protocol=_lookup(protocols, protocol_id, "protocol"),
        role=_required_str(payload, "role", context="session protocol link"),
        metadata=dict(
            _required_mapping(payload, "metadata", context="session protocol link")
        ),
    )


def _condition_link_from_payload(
    payload: Mapping[str, Any],
    conditions: Mapping[str, ExperimentalCondition],
    subjects: Mapping[str, Subject],
) -> SessionConditionLink:
    condition_id = _required_str(
        payload, "condition_id", context="session condition link"
    )
    subject_ids = _required_str_sequence(
        payload, "subject_ids", context="session condition link"
    )
    return SessionConditionLink(
        condition=_lookup(conditions, condition_id, "condition"),
        subjects=tuple(_lookup(subjects, subject_id, "subject") for subject_id in subject_ids),
        metadata=dict(
            _required_mapping(payload, "metadata", context="session condition link")
        ),
    )


def _subject_track_link_from_payload(
    payload: Mapping[str, Any], subjects: Mapping[str, Subject]
) -> SubjectTrackLink:
    from xpkg.model.identity import IdentityProvenanceRecord

    subject_id = _required_str(payload, "subject_id", context="subject track link")
    raw_evidence = payload.get("evidence")
    if raw_evidence is not None and not isinstance(raw_evidence, Mapping):
        raise TypeError("subject track link.evidence must be an object or null.")
    return SubjectTrackLink(
        subject=_lookup(subjects, subject_id, "subject"),
        pose_name=_required_str(payload, "pose_name", context="subject track link"),
        track_id=_required_str(payload, "track_id", context="subject track link"),
        evidence=(
            None
            if raw_evidence is None
            else IdentityProvenanceRecord.from_dict(raw_evidence)
        ),
        metadata=dict(_required_mapping(payload, "metadata", context="subject track link")),
    )
def _dataset_share_from_payload(value: object) -> DatasetShareMetadata | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("experiment.dataset_share must be an object or null.")
    return DatasetShareMetadata.from_dict(cast("Mapping[str, Any]", value))


def _lookup[T](values: Mapping[str, T], identity: str, name: str) -> T:
    try:
        return values[identity]
    except KeyError as exc:
        raise ValueError(f"Session link references unknown {name} {identity!r}.") from exc


def _document_payload(document: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(document, Mapping):
        raise TypeError("experiment document must be an object.")
    if document.get("format") != EXPERIMENT_FORMAT:
        raise ValueError(f"Unsupported experiment format: {document.get('format')!r}.")
    if document.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported experiment schema_version: "
            f"{document.get('schema_version')!r}."
        )
    return _required_mapping(document, "payload", context="experiment document")


def _required_mapping(
    payload: Mapping[str, Any], key: str, *, context: str
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{context}.{key} must be an object.")
    return cast("Mapping[str, Any]", value)


def _required_str(payload: Mapping[str, Any], key: str, *, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{context}.{key} must be a non-empty string.")
    return value


def _optional_str(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or null.")
    return value


def _required_str_sequence(
    payload: Mapping[str, Any], key: str, *, context: str
) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{context}.{key} must be an array.")
    if any(not isinstance(item, str) for item in value):
        raise TypeError(f"{context}.{key} must contain only strings.")
    return tuple(cast("Sequence[str]", value))


def _required_mapping_sequence(
    payload: Mapping[str, Any], key: str, context: str
) -> tuple[Mapping[str, Any], ...]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{context}.{key} must be an array.")
    items: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"{context}.{key}[{index}] must be an object.")
        items.append(cast("Mapping[str, Any]", item))
    return tuple(items)


__all__ = [
    "EXPERIMENT_FORMAT",
    "EXPERIMENT_SCHEMA_VERSION",
    "experiment_document",
    "experiment_from_document",
    "read_experiment_json",
    "read_experiment_metadata_json",
    "write_experiment_json",
]
