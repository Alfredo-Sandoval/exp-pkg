"""Project-scoped calibration storage helpers."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from xpkg.io.calibration import read_calibration_json, write_calibration_json
from xpkg.model.calibration import Calibration, CalibrationSource, WorldFrame
from xpkg.project.layout import (
    project_store_root,
    resolve_project_root,
)

from .._core.path_registry import ensure_dir, resolve_path, slugify_path_component

CALIBRATIONS_DIRNAME = "calibrations"
CALIBRATION_FILENAME = "calibration.json"
CALIBRATION_SOURCE_DIRNAME = "source"


def _require_project_root(project: str | Path) -> Path:
    root = resolve_project_root(project)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")
    return root


def _calibration_id(value: str | None, *, fallback: str) -> str:
    raw = fallback if value is None else str(value).strip()
    if value is None:
        raw = slugify_path_component(raw)
    if not raw:
        raise ValueError("calibration_id must be a non-empty string.")
    if raw in {".", ".."} or "/" in raw or "\\" in raw:
        raise ValueError("calibration_id must be a single path component.")
    return raw


def project_calibrations_root(project: str | Path) -> Path:
    """Return the project-managed calibration directory under ``.xpkg/``."""

    return project_store_root(_require_project_root(project)) / CALIBRATIONS_DIRNAME


def project_calibration_root(project: str | Path, calibration_id: str) -> Path:
    """Return the managed directory for one project calibration."""

    root = project_calibrations_root(project)
    return root / _calibration_id(calibration_id, fallback=calibration_id)


def project_calibration_path(project: str | Path, calibration_id: str) -> Path:
    """Return the canonical JSON path for one project calibration."""

    return project_calibration_root(project, calibration_id) / CALIBRATION_FILENAME


def project_calibration_source_root(project: str | Path, calibration_id: str) -> Path:
    """Return the optional source-sidecar directory for one project calibration."""

    return project_calibration_root(project, calibration_id) / CALIBRATION_SOURCE_DIRNAME


def save_project_calibration(
    project: str | Path,
    calibration: Calibration,
    *,
    calibration_id: str | None = None,
    force: bool = False,
) -> Path:
    """Write a calibration into ``.xpkg/calibrations/<id>/calibration.json``."""

    if not isinstance(calibration, Calibration):
        raise TypeError(f"calibration must be a Calibration, got {calibration!r}.")
    target_id = _calibration_id(calibration_id, fallback=calibration.name)
    target_root = project_calibration_root(project, target_id)
    target_path = target_root / CALIBRATION_FILENAME
    if target_root.exists() and not force:
        raise FileExistsError(f"Project calibration already exists: {target_root}")
    ensure_dir(target_path.parent)
    write_calibration_json(calibration, target_path)
    return target_path


def load_project_calibration(project: str | Path, calibration_id: str) -> Calibration:
    """Load one project-managed calibration by ID."""

    return read_calibration_json(project_calibration_path(project, calibration_id))


def list_project_calibrations(project: str | Path) -> list[Path]:
    """Return project-managed calibration JSON files in stable order."""

    root = project_calibrations_root(project)
    if not root.exists():
        return []
    return sorted(path for path in root.glob(f"*/{CALIBRATION_FILENAME}") if path.is_file())


def _calibration_with_imported_source_path(
    calibration: Calibration,
    imported_from: str,
) -> Calibration:
    source = calibration.source or CalibrationSource()
    return replace(calibration, source=replace(source, imported_from=imported_from))


def import_anipose_calibration_project(
    toml_path: str | Path,
    project: str | Path,
    *,
    calibration_id: str | None = None,
    name: str | None = None,
    units: str = "unknown",
    captured_at: str | None = None,
    world_frame: WorldFrame | Mapping[str, Any] | None = None,
    tool_version: str | None = None,
    force: bool = False,
) -> Path:
    """Import an Anipose ``calibration.toml`` into the project calibration store."""
    from xpkg.io.readers.anipose import read_anipose_calibration

    source_path = resolve_path(toml_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Anipose calibration TOML not found: {source_path}")
    target_id = _calibration_id(calibration_id, fallback=name or source_path.stem)
    source_relative_path = f"{CALIBRATION_SOURCE_DIRNAME}/{source_path.name}"
    calibration = read_anipose_calibration(
        source_path,
        name=name,
        units=units,
        captured_at=captured_at,
        world_frame=world_frame,
        tool_version=tool_version,
    )
    calibration = _calibration_with_imported_source_path(
        calibration,
        imported_from=source_relative_path,
    )
    calibration_path = save_project_calibration(
        project,
        calibration,
        calibration_id=target_id,
        force=force,
    )
    source_target = project_calibration_source_root(project, target_id) / source_path.name
    ensure_dir(source_target.parent)
    shutil.copy2(source_path, source_target)
    return calibration_path


__all__ = [
    "CALIBRATIONS_DIRNAME",
    "CALIBRATION_FILENAME",
    "CALIBRATION_SOURCE_DIRNAME",
    "import_anipose_calibration_project",
    "list_project_calibrations",
    "load_project_calibration",
    "project_calibration_path",
    "project_calibration_root",
    "project_calibration_source_root",
    "project_calibrations_root",
    "save_project_calibration",
]
