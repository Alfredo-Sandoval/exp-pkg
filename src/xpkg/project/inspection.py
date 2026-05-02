"""Project inspection helpers owned by the public project package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xpkg.project.layout import (
    ProjectDescriptor,
    load_project_descriptor,
    resolve_project_root,
)
from xpkg.project.store import (
    current_project_commit_id,
    current_project_state_path,
    load_project_metadata,
    load_project_payload,
)
from xpkg.project.validation import (
    ProjectSummary,
    summarize_loaded_project,
    validate_loaded_project,
)


@dataclass(frozen=True, slots=True)
class ProjectInspection:
    """Normalized inspection result for one project root."""

    project_root: Path
    descriptor: ProjectDescriptor
    current_state_path: Path
    state_kind: str
    commit_id: str | None
    summary: ProjectSummary | None
    metadata: dict[str, Any]
    is_valid: bool
    invalid_reason: str


def inspect_project(path: str | Path) -> ProjectInspection:
    """Inspect one project using canonical project/archive APIs."""
    project_root = resolve_project_root(path)
    if project_root is None:
        raise FileNotFoundError(f"Not an xpkg project: {path}")

    descriptor = load_project_descriptor(project_root)
    state_path = current_project_state_path(project_root)
    metadata = dict(load_project_metadata(project_root) or {})
    payload = load_project_payload(project_root)

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

    return ProjectInspection(
        project_root=project_root,
        descriptor=descriptor,
        current_state_path=state_path,
        state_kind=state_kind,
        commit_id=current_project_commit_id(project_root),
        summary=summary,
        metadata=metadata,
        is_valid=invalid_reason == "",
        invalid_reason=invalid_reason,
    )


__all__ = ["ProjectInspection", "inspect_project"]
