"""Project state caching, commit orchestration, and durable-store writes."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xpkg._core.hashing import sha256_file
from xpkg.project.layout import (
    CURRENT_STATE_FILENAME,
    project_current_state_path,
    project_store_root,
)
from xpkg.project.state import (
    project_state_commit_id_from_document,
    project_state_kind,
    project_state_payload_from_document,
    read_project_state_document,
)
from xpkg.project.state_io import (
    PROJECT_COMMIT_ID_KEY,
    normalize_predictions_payload,
    predictions_payload_from_labels,
    project_state_cache_digest_matches,
    read_project_state,
    rewrite_project_metadata_paths,
    write_project_state,
    write_project_state_cache_digest,
    write_project_state_payload,
)
from xpkg.project.store._helpers import (
    _project_store,
    _stage_project_parent,
    current_project_commit_id,
)
from xpkg.project.store.media import (
    _manage_labels_media,
)
from xpkg.project.store.payloads import (
    _predictions_payload_from_state_payload,
    _state_metadata_from_state_payload,
)

if TYPE_CHECKING:
    from xpkg.model import Labels


def _normalized_project_metadata(
    metadata: dict[str, Any] | None,
    *,
    project_root: Path,
    commit_id: str | None,
) -> dict[str, Any]:
    normalized = rewrite_project_metadata_paths(
        metadata,
        project_root=project_root,
    )
    if commit_id is None:
        normalized.pop(PROJECT_COMMIT_ID_KEY, None)
    else:
        normalized[PROJECT_COMMIT_ID_KEY] = str(commit_id)
    return normalized


def _project_state_cache_matches_committed_head(
    project_root: Path,
    state_path: Path,
) -> bool:
    from xpkg.project.durable_store import ProjectDurableStore

    store = ProjectDurableStore.open(project_store_root(project_root))
    commit = store.load_current_commit()
    if not commit.has_root("state"):
        return False
    if project_state_cache_digest_matches(state_path, commit_id=commit.commit_id):
        return True
    root_entry = commit.root_entry("state")
    committed_state_path = store.paths.object_path(root_entry.object_id, ext=root_entry.ext)
    if not committed_state_path.exists():
        return False
    if f"obj_{sha256_file(state_path)}" == root_entry.object_id:
        write_project_state_cache_digest(state_path, commit_id=commit.commit_id)
        return True

    cache_document = read_project_state_document(state_path)
    committed_document = read_project_state_document(committed_state_path)
    if _project_state_documents_match_cache(cache_document, committed_document):
        write_project_state_cache_digest(state_path, commit_id=commit.commit_id)
        return True
    return False


def _metadata_matches_without_commit_id(
    cache_metadata: Mapping[str, Any],
    committed_metadata: Mapping[str, Any],
) -> bool:
    cache_keys = set(cache_metadata)
    committed_keys = set(committed_metadata)
    cache_keys.discard(PROJECT_COMMIT_ID_KEY)
    committed_keys.discard(PROJECT_COMMIT_ID_KEY)
    if cache_keys != committed_keys:
        return False
    return all(cache_metadata[key] == committed_metadata[key] for key in cache_keys)


def _project_payload_matches_cache(
    cache_payload: Mapping[str, Any],
    committed_payload: Mapping[str, Any],
) -> bool:
    cache_keys = set(cache_payload)
    committed_keys = set(committed_payload)
    if cache_keys != committed_keys:
        return False

    for key in cache_keys:
        if key == "metadata":
            continue
        if cache_payload[key] != committed_payload[key]:
            return False

    cache_metadata = cache_payload.get("metadata")
    committed_metadata = committed_payload.get("metadata")
    if isinstance(cache_metadata, Mapping) and isinstance(committed_metadata, Mapping):
        return _metadata_matches_without_commit_id(cache_metadata, committed_metadata)
    return cache_metadata == committed_metadata


def _project_state_documents_match_cache(
    cache_document: Mapping[str, Any],
    committed_document: Mapping[str, Any],
) -> bool:
    cache_payload = project_state_payload_from_document(cache_document)
    committed_payload = project_state_payload_from_document(committed_document)

    if cache_payload is not cache_document or committed_payload is not committed_document:
        cache_keys = set(cache_document)
        committed_keys = set(committed_document)
        if cache_keys != committed_keys:
            return False
        for key in cache_keys:
            if key == "payload":
                continue
            if cache_document[key] != committed_document[key]:
                return False
        return _project_payload_matches_cache(cache_payload, committed_payload)

    return _project_payload_matches_cache(cache_document, committed_document)


def ensure_current_project_state_cache(project_root: Path) -> Path | None:
    """Materialize the current project state cache from the committed head when needed."""

    state_path = project_current_state_path(project_root)
    if state_path.exists():
        current_head = current_project_commit_id(project_root)
        if current_head is None:
            return state_path
        state_document = read_project_state_document(state_path)
        state_head = project_state_commit_id_from_document(state_document)
        if (
            state_head == current_head
            and _project_state_cache_matches_committed_head(
                project_root,
                state_path,
            )
        ):
            return state_path

    state = _current_project_state_payload(project_root)
    if state is None:
        return None
    return rebuild_project_state_cache(project_root)


def _current_project_state_payload(
    project_root: Path,
) -> tuple[dict[str, Any], str] | None:
    store = _project_store(project_root)
    if store.has_durable_store():
        mounted = store.open()
        if mounted.has_current_root("state"):
            state_path = mounted.current_root_path("state")
            state_kind = project_state_kind(state_path)
            if state_kind == "labels":
                return read_project_state(state_path), "state_labels"

    return None


def _project_state_components(
    state_payload: dict[str, Any],
    *,
    source_kind: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if source_kind == "state_labels":
        metadata = _state_metadata_from_state_payload(state_payload)
    else:
        raise ValueError(f"Unsupported project state source: {source_kind!r}")
    predictions = _predictions_payload_from_state_payload(state_payload)
    return metadata, predictions


def _write_project_state(
    project_root: Path,
    *,
    labels: Labels,
    metadata: dict[str, Any] | None = None,
    predictions: dict[str, Any] | None = None,
    commit_id: str | None = None,
) -> Path:
    _manage_labels_media(labels, project_root)
    return write_project_state(
        project_current_state_path(project_root),
        labels=labels,
        project_root=project_root,
        metadata=metadata,
        predictions=predictions,
        commit_id=commit_id,
    )


def _commit_labels_to_project(
    project_root: Path,
    *,
    labels: Labels,
    metadata: dict[str, Any] | None = None,
    predictions: dict[str, Any] | None = None,
    reason: str,
) -> Path:
    _manage_labels_media(labels, project_root)
    normalized_metadata = rewrite_project_metadata_paths(
        metadata,
        project_root=project_root,
    )
    normalized_predictions = (
        predictions_payload_from_labels(labels)
        if predictions is None
        else normalize_predictions_payload(predictions)
    )

    stage_parent = _stage_project_parent(project_root)
    with tempfile.TemporaryDirectory(
        prefix=".project_commit_",
        dir=str(stage_parent),
    ) as tmp_dir:
        staged_state = Path(tmp_dir) / CURRENT_STATE_FILENAME
        write_project_state(
            staged_state,
            labels=labels,
            project_root=project_root,
            metadata=normalized_metadata,
            predictions=normalized_predictions,
        )
        store = _project_store(project_root)
        store.commit_state(staged_state, reason=reason)
        commit_id = store.current_commit_id()

    return _write_project_state(
        project_root,
        labels=labels,
        metadata=normalized_metadata,
        predictions=normalized_predictions,
        commit_id=commit_id,
    )


def _commit_state_metadata_to_project(
    project_root: Path,
    *,
    state_payload: dict[str, Any],
    metadata: dict[str, Any] | None,
    reason: str,
) -> Path:
    normalized_metadata = rewrite_project_metadata_paths(
        metadata,
        project_root=project_root,
    )
    existing_metadata = state_payload.get("metadata")
    if "preferences" not in normalized_metadata:
        preferences = (
            existing_metadata.get("preferences")
            if isinstance(existing_metadata, dict)
            else None
        )
        normalized_metadata["preferences"] = dict(preferences or {})
    staged_payload = deepcopy(state_payload)
    staged_payload["metadata"] = _normalized_project_metadata(
        normalized_metadata,
        project_root=project_root,
        commit_id=None,
    )

    stage_parent = _stage_project_parent(project_root)
    with tempfile.TemporaryDirectory(
        prefix=".project_commit_",
        dir=str(stage_parent),
    ) as tmp_dir:
        staged_state = Path(tmp_dir) / CURRENT_STATE_FILENAME
        write_project_state_payload(staged_state, staged_payload)
        store = _project_store(project_root)
        store.commit_state(staged_state, reason=reason)
        commit_id = store.current_commit_id()

    current_payload = deepcopy(state_payload)
    current_payload["metadata"] = normalized_metadata
    return write_project_state_payload(
        project_current_state_path(project_root),
        current_payload,
        commit_id=commit_id,
    )


def rebuild_project_state_cache(project_root: Path) -> Path:
    state = _current_project_state_payload(project_root)
    if state is None:
        raise FileNotFoundError(f"Project has no committed state: {project_root}")

    state_payload, source_kind = state
    commit_id = _project_store(project_root).current_commit_id()
    if source_kind == "state_labels":
        return write_project_state_payload(
            project_current_state_path(project_root),
            state_payload,
            commit_id=commit_id,
        )

    raise ValueError(f"Unsupported project state source: {source_kind!r}")
