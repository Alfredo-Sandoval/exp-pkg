"""Consumer-facing service facade for project-first xpkg integrations."""

from __future__ import annotations

from xpkg.services.artifacts import ProjectArtifacts
from xpkg.services.figures import ProjectFigures
from xpkg.services.project import (
    ProjectImports,
    ProjectInspection,
    ProjectLayout,
    ProjectSegmentation,
    ProjectService,
)

__all__ = [
    "ProjectService",
    "ProjectImports",
    "ProjectLayout",
    "ProjectInspection",
    "ProjectArtifacts",
    "ProjectFigures",
    "ProjectSegmentation",
]
