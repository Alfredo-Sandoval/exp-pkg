"""Materialized cache for the canonical experiment project head."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from xpkg.io.experiment_json import read_experiment_json, write_experiment_json
from xpkg.project.layout import project_current_state_path
from xpkg.project.state import (
    PROJECT_COMMIT_ID_KEY,
    project_state_kind,
    read_project_state_document,
)
from xpkg.project.store._helpers import _project_store, current_project_commit_id


def _project_state_cache_matches_committed_head(
    project_root: Path,
    cache_path: Path,
) -> bool:
    """Return whether the materialized cache matches the durable experiment head."""
    store = _project_store(project_root)
    if not store.has_current_state() or not cache_path.is_file():
        return False
    cache_document = read_project_state_document(cache_path)
    committed_document = read_project_state_document(store.current_state_path())
    return _without_cache_commit(cache_document) == _without_cache_commit(
        committed_document
    )


def _without_cache_commit(document: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(document)
    payload = normalized.get("payload")
    if isinstance(payload, dict):
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            metadata.pop(PROJECT_COMMIT_ID_KEY, None)
    return normalized


def ensure_current_project_state_cache(project_root: Path) -> Path | None:
    """Materialize the experiment cache from the durable head when needed."""
    state_path = project_current_state_path(project_root)
    store = _project_store(project_root)
    if not store.has_current_state():
        return None
    if state_path.is_file() and _project_state_cache_matches_committed_head(
        project_root, state_path
    ):
        return state_path
    return rebuild_project_state_cache(project_root)


def rebuild_project_state_cache(project_root: Path) -> Path:
    """Rebuild the canonical experiment cache from the durable head."""
    store = _project_store(project_root)
    if not store.has_current_state():
        raise FileNotFoundError(f"Project has no committed state: {project_root}")
    durable_state = store.current_state_path()
    state_kind = project_state_kind(durable_state)
    if state_kind != "experiment":
        raise ValueError(f"Unsupported project state kind: {state_kind!r}.")
    experiment = read_experiment_json(durable_state, project_root=project_root)
    target = project_current_state_path(project_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    return write_experiment_json(
        target,
        experiment,
        document_metadata={PROJECT_COMMIT_ID_KEY: current_project_commit_id(project_root)},
        project_root=project_root,
    )


__all__ = [
    "ensure_current_project_state_cache",
    "rebuild_project_state_cache",
]
