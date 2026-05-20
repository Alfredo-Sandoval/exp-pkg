"""Small helpers shared by the store submodules.

Lives here (rather than in ``store/__init__.py``) so that submodules can
import these names without triggering a circular import on package load.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xpkg.project.artifact import validate_project
from xpkg.project.layout import (
    ProjectDescriptor,
    _candidate_project_root,
    _now_utc_iso,
    load_project_descriptor,
    project_exports_root,
    project_media_root,
    project_store_root,
    resolve_project_root,
    write_project_descriptor,
)

from ..._core.path_registry import ensure_dir, resolve_path


@dataclass(slots=True)
class ProjectStore:
    """Private project store boundary for `.xpkg/`."""

    project_root: Path

    @property
    def store_root(self) -> Path:
        return project_store_root(self.project_root)

    @property
    def staging_root(self) -> Path:
        return self.store_root / "project"

    def has_durable_store(self) -> bool:
        return (self.store_root / "superblock.a.json").exists() or (
            self.store_root / "superblock.b.json"
        ).exists()

    def has_current_state(self) -> bool:
        if not self.has_durable_store():
            return False
        return self.open().has_current_root("state")

    def current_commit_id(self) -> str | None:
        if not self.has_durable_store():
            return None
        return self.open().load_current_commit().commit_id

    def open(self):
        from xpkg.project.durable_store import ProjectDurableStore

        return ProjectDurableStore.open(self.store_root)

    def current_state_path(self) -> Path:
        if not self.has_durable_store():
            raise FileNotFoundError(f"Project has no durable state root: {self.store_root}")
        return self.open().current_root_path("state")

    def commit_state(
        self,
        state_path: str | Path,
        *,
        reason: str,
        created_by: dict[str, Any] | None = None,
    ) -> Path:
        candidate = resolve_path(state_path)
        if not candidate.is_file():
            raise FileNotFoundError(f"Staged project state not found: {candidate}")

        if self.has_durable_store():
            store = self.open()
            store.commit_new_roots({"state": candidate}, reason=reason, created_by=created_by)
            return store.current_root_path("state")

        from xpkg.project.durable_store import ProjectDurableStore

        store = ProjectDurableStore.create_from_roots(
            store_root=self.store_root,
            initial_roots={"state": candidate},
            created_by=created_by,
            reason=reason,
        )
        return store.current_root_path("state")


def _project_store(path: str | Path) -> ProjectStore:
    root = resolve_project_root(path) or _candidate_project_root(path)
    return ProjectStore(project_root=root)


def current_project_commit_id(path: str | Path) -> str | None:
    return _project_store(path).current_commit_id()


def init_project(
    project: str | Path,
    *,
    title: str | None = None,
    project_id: str | None = None,
    force: bool = False,
) -> ProjectDescriptor:
    root = _candidate_project_root(project)
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"Project root is not a directory: {root}")
    if root.exists():
        entries = list(root.iterdir())
        if entries and not force:
            raise FileExistsError(f"Project directory is not empty: {root}")

    descriptor = ProjectDescriptor.new(
        title=(title or root.name or "exp-pkg Project").strip(),
        project_id=project_id,
    )
    descriptor.validate()

    if not root.exists():
        ensure_dir(root)
    ensure_dir(project_store_root(root))
    ensure_dir(project_media_root(root))
    ensure_dir(project_exports_root(root))

    write_project_descriptor(root, descriptor)
    from xpkg.project.summary import refresh_project_summary

    refresh_project_summary(root)
    return descriptor


def _clone_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


def _stage_project_parent(project_root: Path) -> Path:
    return ensure_dir(_project_store(project_root).staging_root)


def _ensure_project_for_import(
    project: str | Path,
    *,
    title: str | None = None,
    force: bool = False,
) -> Path:
    root = resolve_project_root(project)
    if root is None:
        init_project(
            project,
            title=title,
            force=force,
        )
        return _candidate_project_root(project)
    validate_project(root)
    return root


def _touch_descriptor(
    root: Path,
    *,
    state_summary: dict[str, Any] | None = None,
) -> None:
    descriptor = load_project_descriptor(root)
    descriptor.updated_at = _now_utc_iso()
    write_project_descriptor(root, descriptor)
    from xpkg.project.summary import refresh_project_summary

    refresh_project_summary(root, state_summary=state_summary)
