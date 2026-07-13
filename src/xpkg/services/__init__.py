"""Consumer-facing service facade for project-first xpkg integrations."""

from __future__ import annotations

from xpkg.services.artifacts import ProjectArtifacts
from xpkg.services.calibrations import ProjectCalibrations
from xpkg.services.figures import ProjectFigures
from xpkg.services.project import (
    BehaviorFormat,
    CalibrationFormat,
    EventFormat,
    PoseFormat,
    ProjectInspection,
    ProjectLayout,
    ProjectMetadata,
    ProjectSegmentation,
    ProjectService,
    SignalFormat,
    SynchronizationFormat,
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
    "BehaviorFormat",
    "CalibrationFormat",
    "EventFormat",
    "SignalFormat",
    "SynchronizationFormat",
]
