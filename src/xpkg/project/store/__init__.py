"""Canonical project-store boundary for experiment state."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xpkg.io.experiment_json import read_experiment_metadata_json
from xpkg.model.experiment_actions import replace_experiment_metadata
from xpkg.project.layout import project_current_state_path, resolve_project_root
from xpkg.project.recording import (
    load_project_experiment,
    save_project_experiment,
    save_project_labels,
)
from xpkg.project.store._helpers import ProjectStore as ProjectStore
from xpkg.project.store._helpers import (
    current_project_commit_id as current_project_commit_id,
)
from xpkg.project.store._helpers import ensure_project as ensure_project
from xpkg.project.store._helpers import init_project as init_project
from xpkg.project.store.cache import (
    ensure_current_project_state_cache as ensure_current_project_state_cache,
)
from xpkg.project.store.cache import (
    rebuild_project_state_cache as rebuild_project_state_cache,
)


def current_project_state_path(path: str | Path) -> Path:
    """Return the materialized experiment state path."""
    return project_current_state_path(path)


def load_project_metadata(project: str | Path) -> dict[str, Any]:
    """Return metadata from the canonical experiment head."""
    root = resolve_project_root(project)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")
    store = ProjectStore(root)
    if not store.has_current_state():
        return {}
    return read_experiment_metadata_json(store.current_state_path())


def save_project_metadata(
    project: str | Path,
    metadata: Mapping[str, Any] | None,
    *,
    reason: str = "project.save.metadata",
) -> Path:
    """Replace experiment metadata through the governed action layer."""
    root = resolve_project_root(project)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")
    experiment = load_project_experiment(root)
    updated = replace_experiment_metadata(experiment, metadata)
    return save_project_experiment(root, updated, reason=reason)


__all__ = [
    "ProjectStore",
    "current_project_commit_id",
    "current_project_state_path",
    "ensure_current_project_state_cache",
    "ensure_project",
    "init_project",
    "load_project_metadata",
    "rebuild_project_state_cache",
    "save_project_labels",
    "save_project_metadata",
]
