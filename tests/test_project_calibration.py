from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, cast

from tests.calibration_helpers import write_anipose_toml, write_opencv_stereo_yaml
from xpkg.project import (
    init_project,
    list_project_calibrations,
    load_project_calibration,
    pack_project,
    validate_expkg,
)
from xpkg.project.calibration import (
    import_anipose_calibration_project,
    project_calibration_path,
    project_calibration_root,
    project_calibration_source_root,
    project_calibrations_root,
)
from xpkg.services import ProjectService


def _read_expkg_manifest(artifact: Path) -> dict[str, Any]:
    with zipfile.ZipFile(artifact) as archive:
        raw = archive.read("EXPKG.json").decode("utf-8")
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    return payload


def test_project_calibration_store_round_trips_and_packs(tmp_path: Path) -> None:
    project = tmp_path / "Calibration Project"
    init_project(project, title="Calibration Project")
    source = write_anipose_toml(tmp_path / "calibration.toml")

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
    source = write_anipose_toml(tmp_path / "calibration.toml")

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

    imported_via_dispatch = project.import_calibration(
        "anipose",
        path=source,
        calibration_id="rig-from-imports",
        name="service-rig-from-imports",
        units="mm",
    )
    assert imported_via_dispatch == (
        project.project_root
        / ".xpkg"
        / "calibrations"
        / "rig-from-imports"
        / "calibration.json"
    )


def test_project_service_imports_opencv_stereo_calibration(tmp_path: Path) -> None:
    project = ProjectService.create(tmp_path / "OpenCV Calibration Project")
    source = write_opencv_stereo_yaml(tmp_path / "stereo.yml")

    calibration_path = project.calibrations.import_opencv_stereo(
        source,
        calibration_id="stereo-rig",
        name="arena-stereo",
        camera_names=("left", "right"),
        units="mm",
        captured_at="2026-05-20T15:00:00Z",
    )

    assert calibration_path == (
        project.project_root / ".xpkg" / "calibrations" / "stereo-rig" / "calibration.json"
    )
    assert (
        project.project_root / ".xpkg" / "calibrations" / "stereo-rig" / "source" / "stereo.yml"
    ).is_file()
    loaded = load_project_calibration(project.project_root, "stereo-rig")
    assert loaded.name == "arena-stereo"
    assert loaded.captured_at == "2026-05-20T15:00:00Z"
    assert loaded.units == "mm"
    assert loaded.source is not None
    assert loaded.source.tool == "opencv"
    assert loaded.source.imported_from == "source/stereo.yml"
    assert loaded.source.metadata["format"] == "opencv-stereo-yaml"
    assert loaded.world_frame is not None
    assert loaded.world_frame.anchor == "left"
    assert loaded.camera_by_name("right").extrinsics.translation == (10.0, 20.0, 30.0)
    assert (
        loaded.metadata["opencv_stereo_transform"]
        == "R,T transform camera_1 coordinates into camera_2 coordinates."
    )

    imported_via_dispatch = project.import_calibration(
        "opencv-stereo-yaml",
        path=source,
        calibration_id="stereo-rig-from-imports",
        name="arena-stereo-from-imports",
        camera_names=("left", "right"),
        units="mm",
    )
    assert imported_via_dispatch == (
        project.project_root
        / ".xpkg"
        / "calibrations"
        / "stereo-rig-from-imports"
        / "calibration.json"
    )
