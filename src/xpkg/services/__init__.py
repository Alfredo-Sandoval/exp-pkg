"""Primary workspace-first service facade for downstream xpkg integrations."""

from __future__ import annotations

from xpkg.services.figures import WorkspaceFigures
from xpkg.services.workspace import (
    WorkspaceImports,
    WorkspaceInspection,
    WorkspaceLayout,
    WorkspaceSegmentation,
    WorkspaceService,
)

__all__ = [
    "WorkspaceService",
    "WorkspaceImports",
    "WorkspaceLayout",
    "WorkspaceInspection",
    "WorkspaceFigures",
    "WorkspaceSegmentation",
]
