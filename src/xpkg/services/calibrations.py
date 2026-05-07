"""Service-bound convenience API for project calibrations."""

from __future__ import annotations

import builtins
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xpkg.model.calibration import Calibration, WorldFrame
from xpkg.project.calibration import (
    import_anipose_calibration_project,
    list_project_calibrations,
    load_project_calibration,
    project_calibration_path,
    project_calibration_root,
    project_calibration_source_root,
    project_calibrations_root,
    save_project_calibration,
)


@dataclass(frozen=True, slots=True)
class ProjectCalibrations:
    """Service-bound access to stored calibrations for one project."""

    project_root: Path

    def save(
        self,
        calibration: Calibration,
        *,
        calibration_id: str | None = None,
        force: bool = False,
    ) -> Path:
        """Persist a calibration as ``.xpkg/calibrations/<id>/calibration.json``."""

        return save_project_calibration(
            self.project_root,
            calibration,
            calibration_id=calibration_id,
            force=force,
        )

    def load(self, calibration_id: str) -> Calibration:
        """Load one stored calibration by ID."""

        return load_project_calibration(self.project_root, calibration_id)

    def list(self) -> builtins.list[Path]:
        """Return stored calibration JSON paths in stable order."""

        return list_project_calibrations(self.project_root)

    def path(self, calibration_id: str) -> Path:
        """Return the canonical JSON path for one stored calibration."""

        return project_calibration_path(self.project_root, calibration_id)

    def calibration_root(self, calibration_id: str) -> Path:
        """Return the managed directory for one stored calibration."""

        return project_calibration_root(self.project_root, calibration_id)

    def source_root(self, calibration_id: str) -> Path:
        """Return the optional source-sidecar directory for one stored calibration."""

        return project_calibration_source_root(self.project_root, calibration_id)

    def root(self) -> Path:
        """Return the project-managed calibration directory under ``.xpkg/``."""

        return project_calibrations_root(self.project_root)

    def import_anipose(
        self,
        toml_path: str | Path,
        *,
        calibration_id: str | None = None,
        name: str | None = None,
        units: str = "unknown",
        captured_at: str | None = None,
        world_frame: WorldFrame | Mapping[str, Any] | None = None,
        tool_version: str | None = None,
        force: bool = False,
    ) -> Path:
        """Import an Anipose ``calibration.toml`` into this project."""

        return import_anipose_calibration_project(
            toml_path,
            self.project_root,
            calibration_id=calibration_id,
            name=name,
            units=units,
            captured_at=captured_at,
            world_frame=world_frame,
            tool_version=tool_version,
            force=force,
        )


__all__ = ["ProjectCalibrations"]
