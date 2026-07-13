"""Service-bound convenience API for project calibrations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xpkg.model.calibration import Calibration, WorldFrame
from xpkg.model.session import SessionCalibration
from xpkg.project.calibration import (
    import_anipose_calibration_project,
    import_opencv_stereo_calibration_project,
    list_project_calibrations,
    load_project_calibration,
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
        session_id: str | None = None,
        force: bool = False,
    ) -> Path:
        """Persist a calibration as ``.xpkg/calibrations/<id>/calibration.json``."""

        return save_project_calibration(
            self.project_root,
            calibration,
            calibration_id=calibration_id,
            session_id=session_id,
            force=force,
        )

    def load(
        self, calibration_id: str, *, session_id: str | None = None
    ) -> Calibration:
        """Load one session-owned calibration by ID."""

        return load_project_calibration(
            self.project_root, calibration_id, session_id=session_id
        )

    def list(
        self, *, session_id: str | None = None
    ) -> tuple[SessionCalibration, ...]:
        """Return calibration relationships for one selected session."""

        return list_project_calibrations(self.project_root, session_id=session_id)

    def import_anipose(
        self,
        toml_path: str | Path,
        *,
        calibration_id: str | None = None,
        session_id: str | None = None,
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
            session_id=session_id,
            name=name,
            units=units,
            captured_at=captured_at,
            world_frame=world_frame,
            tool_version=tool_version,
            force=force,
        )

    def import_opencv_stereo(
        self,
        yaml_path: str | Path,
        *,
        calibration_id: str | None = None,
        session_id: str | None = None,
        name: str | None = None,
        camera_names: tuple[str, str] = ("camera_1", "camera_2"),
        units: str = "unknown",
        captured_at: str | None = None,
        tool_version: str | None = None,
        force: bool = False,
    ) -> Path:
        """Import an OpenCV stereo-calibration YAML into this project."""

        return import_opencv_stereo_calibration_project(
            yaml_path,
            self.project_root,
            calibration_id=calibration_id,
            session_id=session_id,
            name=name,
            camera_names=camera_names,
            units=units,
            captured_at=captured_at,
            tool_version=tool_version,
            force=force,
        )


__all__ = ["ProjectCalibrations"]
