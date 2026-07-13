"""Project inspection summaries for descriptor, state, metadata, and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xpkg.project.layout import (
    ProjectDescriptor,
    load_project_descriptor,
    resolve_project_root,
)
from xpkg.project.store import current_project_state_path
from xpkg.project.summary import ProjectSummaryIndex, snapshot_project_summary


@dataclass(frozen=True, slots=True)
class ProjectInspection:
    """Validated summary of the current committed state for one project root."""

    project_root: Path
    descriptor: ProjectDescriptor
    current_state_path: Path
    state_kind: str
    commit_id: str | None
    summary: ProjectSummaryIndex
    is_valid: bool
    invalid_reason: str


def inspect_project(path: str | Path) -> ProjectInspection:
    """Inspect one project through the canonical project-state APIs."""
    project_root = resolve_project_root(path)
    if project_root is None:
        raise FileNotFoundError(f"Not an xpkg project: {path}")

    descriptor = load_project_descriptor(project_root)
    state_path = current_project_state_path(project_root)
    summary = snapshot_project_summary(project_root)
    invalid_reason = "; ".join(summary.warnings)

    return ProjectInspection(
        project_root=project_root,
        descriptor=descriptor,
        current_state_path=state_path,
        state_kind=summary.state_kind,
        commit_id=summary.commit_id,
        summary=summary,
        is_valid=summary.state_kind != "unreadable" and invalid_reason == "",
        invalid_reason=invalid_reason,
    )


__all__ = ["ProjectInspection", "inspect_project"]
