from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, cast

from xpkg.project import (
    import_anipose_calibration_project,
    init_project,
    list_project_calibrations,
    load_project_calibration,
    pack_project,
    project_calibration_path,
    project_calibration_root,
    project_calibration_source_root,
    project_calibrations_root,
    validate_expkg,
)
from xpkg.services import ProjectService


def _write_anipose_toml(path: Path) -> Path:
    path.write_text(
        """
[cam_top]
name = "cam_top"
size = [1920, 1080]
matrix = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
distortions = [0.1, -0.01, 0.001, 0.002, 0.0]
rotation = [0.0, 0.0, 0.0]
translation = [1.0, 2.0, 3.0]
fisheye = false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _read_expkg_manifest(artifact: Path) -> dict[str, Any]:
    with zipfile.ZipFile(artifact) as archive:
        raw = archive.read("EXPKG.json").decode("utf-8")
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    return payload


def test_project_calibration_store_round_trips_and_packs(tmp_path: Path) -> None:
    project = tmp_path / "Calibration Project"
    init_project(project, title="Calibration Project")
    source = _write_anipose_toml(tmp_path / "calibration.toml")

    calibration_path = import_anipose_calibration_project(
        source,
        project,
        calibration_id="rig-2024-03-15",
        name="rig-2024-03-15",
        units="mm",
    )

    assert (
        calibration_path
        == project / ".xpkg" / "calibrations" / "rig-2024-03-15" / "calibration.json"
    )
    assert project_calibrations_root(project) == project / ".xpkg" / "calibrations"
    assert (
        project_calibration_root(project, "rig-2024-03-15")
        == project / ".xpkg" / "calibrations" / "rig-2024-03-15"
    )
    assert project_calibration_path(project, "rig-2024-03-15") == calibration_path
    source_root = project_calibration_source_root(project, "rig-2024-03-15")
    assert source_root == project / ".xpkg" / "calibrations" / "rig-2024-03-15" / "source"
    assert (source_root / "calibration.toml").is_file()
    assert list_project_calibrations(project) == [calibration_path]
    loaded = load_project_calibration(project, "rig-2024-03-15")
    assert loaded.name == "rig-2024-03-15"
    assert loaded.source is not None
    assert loaded.source.imported_from == "source/calibration.toml"
    assert loaded.camera_by_name("cam_top").extrinsics.translation == (1.0, 2.0, 3.0)

    artifact = pack_project(project, out=tmp_path / "Calibration Project.expkg")
    validate_expkg(artifact)
    manifest = _read_expkg_manifest(artifact)
    members = cast("list[dict[str, Any]]", manifest["members"])
    member_paths = {entry["path"] for entry in members}
    assert ".xpkg/calibrations/rig-2024-03-15/calibration.json" in member_paths
    assert ".xpkg/calibrations/rig-2024-03-15/source/calibration.toml" in member_paths


def test_project_service_imports_anipose_calibration(tmp_path: Path) -> None:
    project = ProjectService.create(tmp_path / "Service Calibration Project")
    source = _write_anipose_toml(tmp_path / "calibration.toml")

    calibration_path = project.calibrations.import_anipose(
        source,
        calibration_id="rig",
        name="service-rig",
        units="mm",
    )

    assert (
        calibration_path
        == project.project_root / ".xpkg" / "calibrations" / "rig" / "calibration.json"
    )
    assert project.calibrations.path("rig") == calibration_path
    assert project.calibrations.source_root("rig") == (
        project.project_root / ".xpkg" / "calibrations" / "rig" / "source"
    )
    assert load_project_calibration(project.project_root, "rig").name == "service-rig"

    imported_via_imports = project.imports.anipose_calibration(
        source,
        calibration_id="rig-from-imports",
        name="service-rig-from-imports",
        units="mm",
    )
    assert imported_via_imports == (
        project.project_root
        / ".xpkg"
        / "calibrations"
        / "rig-from-imports"
        / "calibration.json"
    )
