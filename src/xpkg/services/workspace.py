"""Primary workspace lifecycle service for xpkg project operations.

``WorkspaceService`` is the preferred entrypoint for new integrations that
need to create, open, validate, pack, or unpack an xpkg workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from xpkg.formats.project import (
    ProjectDescriptor,
    current_project_state_path,
    init_project,
    load_project_descriptor,
    pack_project,
    project_descriptor_path,
    resolve_workspace_root,
    save_workspace_labels,
    unpack_project,
    validate_workspace,
    workspace_exports_root,
    workspace_media_root,
    workspace_state_root,
    workspace_store_root,
)

if TYPE_CHECKING:
    from xpkg.model import Labels

PackMode = Literal["portable", "snapshot"]


@dataclass(frozen=True, slots=True)
class WorkspaceLayout:
    """Normalized summary of an xpkg workspace and its managed paths."""

    workspace_root: Path
    descriptor: ProjectDescriptor
    descriptor_path: Path
    store_root: Path
    state_root: Path
    media_root: Path
    exports_root: Path
    current_state_path: Path
    has_current_state: bool


@dataclass(slots=True)
class WorkspaceService:
    """Primary public service for workspace-centric project operations."""

    workspace_root: Path

    @classmethod
    def create(
        cls,
        workspace: str | Path,
        *,
        title: str | None = None,
        project_id: str | None = None,
        default_pack_mode: PackMode = "portable",
        force: bool = False,
    ) -> WorkspaceService:
        """Create a workspace with the canonical public layout and open it."""
        init_project(
            workspace,
            title=title,
            project_id=project_id,
            default_pack_mode=default_pack_mode,
            force=force,
        )
        return cls.open(workspace)

    @classmethod
    def open(cls, workspace: str | Path) -> WorkspaceService:
        """Open an existing workspace root or a path inside one."""
        root = resolve_workspace_root(workspace)
        if root is None:
            raise FileNotFoundError(f"Not an xpkg workspace: {workspace}")
        return cls(workspace_root=root)

    @classmethod
    def unpack(
        cls,
        artifact: str | Path,
        out: str | Path,
        *,
        force: bool = False,
        rename_title: str | None = None,
    ) -> WorkspaceService:
        """Unpack a portable `.expkg` artifact into a workspace and open it."""
        unpack_project(
            artifact,
            out,
            force=force,
            rename_title=rename_title,
        )
        return cls.open(out)

    def descriptor(self) -> ProjectDescriptor:
        """Load the current workspace descriptor."""
        return load_project_descriptor(self.workspace_root)

    def describe(self) -> WorkspaceLayout:
        """Return the normalized managed paths for this workspace."""
        descriptor = self.descriptor()
        state_path = current_project_state_path(self.workspace_root)
        return WorkspaceLayout(
            workspace_root=self.workspace_root,
            descriptor=descriptor,
            descriptor_path=project_descriptor_path(self.workspace_root),
            store_root=workspace_store_root(self.workspace_root),
            state_root=workspace_state_root(self.workspace_root),
            media_root=workspace_media_root(self.workspace_root),
            exports_root=workspace_exports_root(self.workspace_root),
            current_state_path=state_path,
            has_current_state=state_path.exists(),
        )

    def validate(self) -> WorkspaceLayout:
        """Validate the workspace and return its normalized layout."""
        validate_workspace(self.workspace_root)
        return self.describe()

    def load_labels(self) -> Labels:
        """Load the current workspace labels through the public workspace root."""
        from xpkg.model import Labels

        return Labels.load_file(self.workspace_root.as_posix())

    def save_labels(
        self,
        labels: Labels,
        *,
        metadata: dict[str, Any] | None = None,
        journal: bool = True,
        regenerate_predictions: bool = False,
    ) -> Path:
        """Commit labels into the workspace-managed durable state."""
        return save_workspace_labels(
            self.workspace_root,
            labels,
            metadata=metadata,
            journal=journal,
            regenerate_predictions=regenerate_predictions,
        )

    def pack(
        self,
        *,
        out: str | Path | None = None,
        mode: PackMode | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Pack the workspace into a portable `.expkg` artifact."""
        return pack_project(
            self.workspace_root,
            out=out,
            mode=mode,
            overwrite=overwrite,
        )


__all__ = ["WorkspaceLayout", "WorkspaceService"]
