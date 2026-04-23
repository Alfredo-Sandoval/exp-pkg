"""Primary workspace-first service facade for downstream xpkg integrations."""

from __future__ import annotations

from xpkg.services.workspace import (
    WorkspaceImports,
    WorkspaceInspection,
    WorkspaceLayout,
    WorkspaceService,
)

__all__ = ["WorkspaceService", "WorkspaceImports", "WorkspaceLayout", "WorkspaceInspection"]
