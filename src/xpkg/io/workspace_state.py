"""Helpers for detecting and introspecting workspace state documents."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

from xpkg.codecs.vicon import XPKG_VICON_JSON_FORMAT
from xpkg.core.json_utils import load_json_dict
from xpkg.io.labels.json_format import XPKG_LABELS_JSON_FORMAT
from xpkg.io.workspace_snapshot_backend import WORKSPACE_COMMIT_ID_KEY

WorkspaceStateKind = Literal["labels", "vicon"]


def read_workspace_state_document(path: str | Path) -> dict[str, object]:
    """Read a workspace state JSON document."""

    return load_json_dict(path)


def workspace_state_payload_from_document(
    document: Mapping[str, object],
) -> Mapping[str, object]:
    """Return the normalized payload object from a workspace state document."""

    payload = document.get("payload")
    if isinstance(payload, Mapping):
        return cast("dict[str, object]", dict(payload))
    return document


def workspace_state_kind_from_document(document: Mapping[str, object]) -> WorkspaceStateKind:
    """Return the state kind encoded in a workspace state JSON document."""

    raw_format = str(document.get("format", "")).strip()
    if raw_format == XPKG_LABELS_JSON_FORMAT:
        return "labels"
    if raw_format == XPKG_VICON_JSON_FORMAT:
        return "vicon"
    raise ValueError(f"Unsupported workspace state format: {raw_format!r}.")


def workspace_state_kind(path: str | Path) -> WorkspaceStateKind:
    """Return the workspace state kind for a JSON state file."""

    return workspace_state_kind_from_document(read_workspace_state_document(path))


def workspace_state_commit_id_from_document(document: Mapping[str, object]) -> str | None:
    """Return the committed workspace head id encoded in a state document, if present."""

    payload = workspace_state_payload_from_document(document)
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    normalized_metadata = cast("dict[str, object]", dict(metadata))
    raw_commit_id = normalized_metadata.get(WORKSPACE_COMMIT_ID_KEY)
    if not isinstance(raw_commit_id, str):
        return None
    commit_id = raw_commit_id.strip()
    return commit_id or None


__all__ = [
    "WorkspaceStateKind",
    "read_workspace_state_document",
    "workspace_state_commit_id_from_document",
    "workspace_state_kind",
    "workspace_state_kind_from_document",
    "workspace_state_payload_from_document",
]
