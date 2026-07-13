"""Project actions for session-owned camera calibrations."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from xpkg._core.path_registry import ensure_dir, resolve_path, slugify_path_component
from xpkg.model.calibration import Calibration, CalibrationSource, WorldFrame
from xpkg.model.session import SessionCalibration
from xpkg.model.session_actions import add_session_calibration, replace_session_calibration
from xpkg.project.layout import project_media_root, require_project_root


def _calibration_id(value: str | None, *, fallback: str) -> str:
    raw = slugify_path_component(fallback) if value is None else str(value).strip()
    if not raw:
        raise ValueError("calibration_id must be a non-empty string.")
    if raw in {".", ".."} or "/" in raw or "\\" in raw:
        raise ValueError("calibration_id must be a single path component.")
    return raw


def save_project_calibration(
    project: str | Path,
    calibration: Calibration,
    *,
    calibration_id: str | None = None,
    session_id: str | None = None,
    force: bool = False,
) -> Path:
    """Commit one camera calibration as a typed session relationship."""
    if not isinstance(calibration, Calibration):
        raise TypeError(f"calibration must be a Calibration, got {calibration!r}.")
    from xpkg.project.recording import (
        _load_or_create_experiment,
        _select_or_create_session,
        save_project_session,
    )

    target_id = _calibration_id(calibration_id, fallback=calibration.name)
    experiment = _load_or_create_experiment(project)
    session = _select_or_create_session(experiment, requested_session_id=session_id)
    link = SessionCalibration(name=target_id, calibration=calibration)
    existing = {item.name for item in session.calibrations}
    if target_id in existing:
        if not force:
            raise FileExistsError(
                f"Recording session {session.session_id!r} already has calibration "
                f"{target_id!r}."
            )
        session = replace_session_calibration(session, link)
    else:
        session = add_session_calibration(session, link)
    return save_project_session(project, session, reason="project.save.calibration")


def load_project_calibration(
    project: str | Path,
    calibration_id: str,
    *,
    session_id: str | None = None,
) -> Calibration:
    """Load one named calibration from a selected recording session."""
    from xpkg.project.recording import load_project_session

    session = load_project_session(project, session_id=session_id)
    return session.calibration(calibration_id)


def list_project_calibrations(
    project: str | Path, *, session_id: str | None = None
) -> tuple[SessionCalibration, ...]:
    """Return calibration relationships from a selected recording session."""
    from xpkg.project.recording import load_project_session

    return load_project_session(project, session_id=session_id).calibrations


def _calibration_with_imported_source_path(
    calibration: Calibration,
    imported_from: str,
) -> Calibration:
    source = calibration.source or CalibrationSource()
    return replace(calibration, source=replace(source, imported_from=imported_from))


def _copy_calibration_source(
    source: Path, project: str | Path, calibration_id: str
) -> tuple[Path, str]:
    root = require_project_root(project)
    target = project_media_root(root) / "calibrations" / calibration_id / source.name
    ensure_dir(target.parent)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    relative = target.resolve().relative_to(root.resolve()).as_posix()
    return target, relative


def _import_calibration_source_project(
    source: str | Path,
    project: str | Path,
    *,
    reader: Callable[..., Calibration],
    source_label: str,
    calibration_id: str | None = None,
    session_id: str | None = None,
    name: str | None = None,
    units: str = "unknown",
    captured_at: str | None = None,
    tool_version: str | None = None,
    force: bool = False,
    reader_kwargs: Mapping[str, Any] | None = None,
) -> Path:
    source_path = resolve_path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"{source_label} not found: {source_path}")
    target_id = _calibration_id(calibration_id, fallback=name or source_path.stem)
    calibration = reader(
        source_path,
        name=name,
        units=units,
        captured_at=captured_at,
        tool_version=tool_version,
        **dict(reader_kwargs or {}),
    )
    _target, relative_source = _copy_calibration_source(source_path, project, target_id)
    calibration = _calibration_with_imported_source_path(calibration, relative_source)
    return save_project_calibration(
        project,
        calibration,
        calibration_id=target_id,
        session_id=session_id,
        force=force,
    )


def import_anipose_calibration_project(
    toml_path: str | Path,
    project: str | Path,
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
    """Import Anipose calibration into a recording session."""
    from xpkg.io.readers.anipose import read_anipose_calibration

    return _import_calibration_source_project(
        toml_path,
        project,
        reader=read_anipose_calibration,
        source_label="Anipose calibration TOML",
        calibration_id=calibration_id,
        session_id=session_id,
        name=name,
        units=units,
        captured_at=captured_at,
        tool_version=tool_version,
        force=force,
        reader_kwargs={"world_frame": world_frame},
    )


def import_opencv_stereo_calibration_project(
    yaml_path: str | Path,
    project: str | Path,
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
    """Import OpenCV stereo calibration into a recording session."""
    from xpkg.io.readers.opencv_stereo import read_opencv_stereo_calibration

    return _import_calibration_source_project(
        yaml_path,
        project,
        reader=read_opencv_stereo_calibration,
        source_label="OpenCV stereo calibration YAML",
        calibration_id=calibration_id,
        session_id=session_id,
        name=name,
        units=units,
        captured_at=captured_at,
        tool_version=tool_version,
        force=force,
        reader_kwargs={"camera_names": camera_names},
    )


__all__ = [
    "import_anipose_calibration_project",
    "import_opencv_stereo_calibration_project",
    "list_project_calibrations",
    "load_project_calibration",
    "save_project_calibration",
]
