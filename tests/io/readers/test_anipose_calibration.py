from __future__ import annotations

import math
import tomllib
from pathlib import Path

from xpkg.io.calibration import read_calibration_json, write_calibration_json
from xpkg.io.readers import read_anipose_calibration, write_anipose_calibration


def _write_anipose_toml(path: Path) -> Path:
    path.write_text(
        """
error = 0.42

[cam_top]
name = "cam_top"
size = [1920, 1080]
matrix = [[1000.0, 0.0, 960.0], [0.0, 1001.0, 540.0], [0.0, 0.0, 1.0]]
distortions = [0.1, -0.01, 0.001, 0.002, 0.0]
rotation = [0.0, 0.0, 0.0]
translation = [1.0, 2.0, 3.0]
fisheye = false

[cam_side]
name = "cam_side"
size = [1280, 720]
matrix = [[900.0, 0.0, 640.0], [0.0, 901.0, 360.0], [0.0, 0.0, 1.0]]
distortions = [0.01, -0.02, 0.03, -0.04]
rotation = [0.0, 0.0, 1.5707963267948966]
translation = [4.0, 5.0, 6.0]
fisheye = true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_read_anipose_calibration_maps_to_generic_calibration(tmp_path: Path) -> None:
    calibration = read_anipose_calibration(
        _write_anipose_toml(tmp_path / "calibration.toml"),
        name="rig",
        units="mm",
        imported_at="2026-05-07T14:00:00Z",
    )

    assert calibration.name == "rig"
    assert calibration.units == "mm"
    assert calibration.source is not None
    assert calibration.source.tool == "anipose"
    assert calibration.source.imported_from == "calibration.toml"
    assert [camera.name for camera in calibration.cameras] == ["cam_top", "cam_side"]

    top = calibration.camera_by_name("cam_top")
    assert top.intrinsics.model == "pinhole"
    assert top.intrinsics.distortion.model == "opencv_5"
    assert top.image_size == (1920, 1080)
    assert top.quality.reprojection_error_px == 0.42

    side = calibration.camera_by_name("cam_side")
    assert side.intrinsics.model == "fisheye"
    assert side.intrinsics.distortion.model == "fisheye_4"
    rotation_matrix = side.extrinsics.rotation.to_matrix()
    assert math.isclose(rotation_matrix[0][1], -1.0, abs_tol=1e-12)
    assert math.isclose(rotation_matrix[1][0], 1.0, abs_tol=1e-12)


def test_calibration_json_and_anipose_toml_round_trip(tmp_path: Path) -> None:
    calibration = read_anipose_calibration(
        _write_anipose_toml(tmp_path / "calibration.toml"),
        name="rig",
        units="mm",
        imported_at="2026-05-07T14:00:00Z",
    )

    json_path = tmp_path / "Calibration.json"
    write_calibration_json(calibration, json_path)
    assert read_calibration_json(json_path) == calibration

    exported_toml = tmp_path / "exported_calibration.toml"
    write_anipose_calibration(calibration, exported_toml)
    exported_payload = tomllib.loads(exported_toml.read_text(encoding="utf-8"))

    assert exported_payload["cam_top"]["name"] == "cam_top"
    assert exported_payload["cam_top"]["size"] == [1920, 1080]
    assert exported_payload["cam_top"]["fisheye"] is False
    assert exported_payload["cam_side"]["fisheye"] is True
    assert exported_payload["cam_side"]["distortions"] == [0.01, -0.02, 0.03, -0.04]
