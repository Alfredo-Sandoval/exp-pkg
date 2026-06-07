"""Detection and introspection routines for project state JSON documents."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

from xpkg.io.labels.json_format import XPKG_LABELS_JSON_FORMAT
from xpkg.project.state_io import PROJECT_COMMIT_ID_KEY

from .._core.json_utils import load_json_dict

ProjectStateKind = Literal["labels"]


def read_project_state_document(path: str | Path) -> dict[str, object]:
    """Read a project state JSON document."""

    return load_json_dict(path)


def project_state_payload_from_document(
    document: Mapping[str, object],
) -> Mapping[str, object]:
    """Return the normalized payload object from a project state document."""

    payload = document.get("payload")
    if isinstance(payload, Mapping):
        return cast("dict[str, object]", dict(payload))
    return document


def project_state_kind_from_document(document: Mapping[str, object]) -> ProjectStateKind:
    """Return the state kind encoded in a project state JSON document."""

    raw_format = str(document.get("format", "")).strip()
    if raw_format == XPKG_LABELS_JSON_FORMAT:
        return "labels"
    raise ValueError(f"Unsupported project state format: {raw_format!r}.")


def project_state_kind(path: str | Path) -> ProjectStateKind:
    """Return the project state kind for a JSON state file."""

    return project_state_kind_from_document(read_project_state_document(path))


def project_state_commit_id_from_document(document: Mapping[str, object]) -> str | None:
    """Return the committed project head id encoded in a state document, if present."""

    payload = project_state_payload_from_document(document)
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    normalized_metadata = cast("dict[str, object]", dict(metadata))
    raw_commit_id = normalized_metadata.get(PROJECT_COMMIT_ID_KEY)
    if not isinstance(raw_commit_id, str):
        return None
    commit_id = raw_commit_id.strip()
    return commit_id or None


__all__ = [
    "ProjectStateKind",
    "read_project_state_document",
    "project_state_commit_id_from_document",
    "project_state_kind",
    "project_state_kind_from_document",
    "project_state_payload_from_document",
]
