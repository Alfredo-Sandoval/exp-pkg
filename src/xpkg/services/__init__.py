"""Primary workspace-first service facade for downstream xpkg integrations."""

from __future__ import annotations

from xpkg.services.artifacts import WorkspaceArtifacts
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
    "WorkspaceArtifacts",
    "WorkspaceFigures",
    "WorkspaceSegmentation",
]
