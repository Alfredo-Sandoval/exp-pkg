"""Consumer-facing service facade for project-first xpkg integrations."""

from __future__ import annotations

from xpkg.services.artifacts import ProjectArtifacts
from xpkg.services.calibrations import ProjectCalibrations
from xpkg.services.figures import ProjectFigures
from xpkg.services.project import (
    CalibrationFormat,
    MotionFormat,
    PoseFormat,
    ProjectInspection,
    ProjectLayout,
    ProjectMetadata,
    ProjectSegmentation,
    ProjectService,
)

__all__ = [
    "ProjectService",
    "ProjectLayout",
    "ProjectInspection",
    "ProjectArtifacts",
    "ProjectCalibrations",
    "ProjectFigures",
    "ProjectMetadata",
    "ProjectSegmentation",
    "PoseFormat",
    "CalibrationFormat",
    "MotionFormat",
]
