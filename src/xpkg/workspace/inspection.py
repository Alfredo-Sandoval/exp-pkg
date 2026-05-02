"""Workspace inspection helpers owned by the public workspace package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xpkg.io.project_layout import (
    ProjectDescriptor,
    load_project_descriptor,
    resolve_workspace_root,
)
from xpkg.io.project_validation import (
    ProjectSummary,
    summarize_loaded_project,
    validate_loaded_project,
)
from xpkg.io.project_workspace import (
    current_project_commit_id,
    current_project_state_path,
    load_workspace_metadata,
    load_workspace_payload,
)


@dataclass(frozen=True, slots=True)
class WorkspaceInspection:
    """Normalized inspection result for one workspace root."""

    workspace_root: Path
    descriptor: ProjectDescriptor
    current_state_path: Path
    state_kind: str
    commit_id: str | None
    summary: ProjectSummary | None
    metadata: dict[str, Any]
    is_valid: bool
    invalid_reason: str


def inspect_workspace(path: str | Path) -> WorkspaceInspection:
    """Inspect one workspace using canonical workspace/archive APIs."""
    workspace_root = resolve_workspace_root(path)
    if workspace_root is None:
        raise FileNotFoundError(f"Not an xpkg workspace: {path}")

    descriptor = load_project_descriptor(workspace_root)
    state_path = current_project_state_path(workspace_root)
    metadata = dict(load_workspace_metadata(workspace_root) or {})
    payload = load_workspace_payload(workspace_root)

    if "recording" in payload:
        state_kind = "recording"
    elif set(payload).issubset({"metadata"}):
        state_kind = "empty"
    else:
        state_kind = "labels"

    summary: ProjectSummary | None = None
    invalid_reason = ""
    if state_kind == "labels":
        try:
            summary = summarize_loaded_project(payload, path=state_path)
            validate_loaded_project(payload)
        except (RuntimeError, TypeError, ValueError) as exc:
            invalid_reason = str(exc)

    if summary is not None:
        metadata.setdefault("n_predictions_committed", int(summary.prediction_frames))

    return WorkspaceInspection(
        workspace_root=workspace_root,
        descriptor=descriptor,
        current_state_path=state_path,
        state_kind=state_kind,
        commit_id=current_project_commit_id(workspace_root),
        summary=summary,
        metadata=metadata,
        is_valid=invalid_reason == "",
        invalid_reason=invalid_reason,
    )


__all__ = ["WorkspaceInspection", "inspect_workspace"]
