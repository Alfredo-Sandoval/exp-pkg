from __future__ import annotations

from pathlib import Path

import pytest

from xpkg.io.readers import read_opencv_stereo_calibration


def _write_opencv_stereo_yaml(path: Path, *, distortion_key: str = "D2") -> Path:
    path.write_text(
        f"""
%YAML:1.0
---
image_width: 640
image_height: 480
M1: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 1000.0, 0.0, 320.0, 0.0, 1001.0, 240.0, 0.0, 0.0, 1.0 ]
D1: !!opencv-matrix
   rows: 1
   cols: 5
   dt: d
   data: [ 0.1, -0.01, 0.001, 0.002, 0.0 ]
M2: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 990.0, 0.0, 300.0, 0.0, 991.0, 230.0, 0.0, 0.0, 1.0 ]
{distortion_key}: !!opencv-matrix
   rows: 1
   cols: 4
   dt: d
   data: [ 0.02, -0.03, 0.004, -0.005 ]
R: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0 ]
T: !!opencv-matrix
   rows: 3
   cols: 1
   dt: d
   data: [ 10.0, 20.0, 30.0 ]
E: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 0.0, -30.0, 20.0, 30.0, 0.0, -10.0, -20.0, 10.0, 0.0 ]
F: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 0.0, -0.03, 0.02, 0.03, 0.0, -0.01, -0.02, 0.01, 0.0 ]
""".lstrip(),
        encoding="utf-8",
    )
    return path


def test_read_opencv_stereo_calibration_maps_to_generic_calibration(tmp_path: Path) -> None:
    calibration = read_opencv_stereo_calibration(
        _write_opencv_stereo_yaml(tmp_path / "stereo.yml"),
        name="arena-rig",
        camera_names=("left", "right"),
        units="mm",
        imported_at="2026-05-20T15:00:00Z",
    )

    assert calibration.name == "arena-rig"
    assert calibration.units == "mm"
    assert calibration.world_frame is not None
    assert calibration.world_frame.anchor == "left"
    assert calibration.source is not None
    assert calibration.source.tool == "opencv"
    assert calibration.source.imported_from == "stereo.yml"
    assert calibration.source.metadata["format"] == "opencv-stereo-yaml"
    assert [camera.name for camera in calibration.cameras] == ["left", "right"]

    left = calibration.camera_by_name("left")
    assert left.image_size == (640, 480)
    assert left.intrinsics.model == "pinhole"
    assert left.intrinsics.matrix[0] == (1000.0, 0.0, 320.0)
    assert left.intrinsics.distortion.model == "opencv_5"
    assert left.extrinsics.translation == (0.0, 0.0, 0.0)
    assert left.extrinsics.rotation.to_matrix() == (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    right = calibration.camera_by_name("right")
    assert right.intrinsics.distortion.model == "opencv_5"
    assert right.intrinsics.distortion.coeffs == (0.02, -0.03, 0.004, -0.005, 0.0)
    assert right.metadata["source_distortion_coeff_count"] == 4
    assert right.extrinsics.translation == (10.0, 20.0, 30.0)
    assert right.extrinsics.rotation.to_matrix() == (
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    assert (
        calibration.metadata["opencv_stereo_transform"]
        == "R,T transform camera_1 coordinates into camera_2 coordinates."
    )
    assert calibration.metadata["essential_matrix"][0] == [0.0, -30.0, 20.0]
    assert calibration.metadata["fundamental_matrix"][1] == [0.03, 0.0, -0.01]


def test_read_opencv_stereo_calibration_requires_unrectified_stereo_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing 'D2'"):
        read_opencv_stereo_calibration(
            _write_opencv_stereo_yaml(tmp_path / "rectification_only.yml", distortion_key="P2")
        )


def test_read_opencv_stereo_calibration_rejects_unsupported_distortion_count(
    tmp_path: Path,
) -> None:
    source = _write_opencv_stereo_yaml(tmp_path / "stereo.yml")
    text = source.read_text(encoding="utf-8")
    source.write_text(
        text.replace(
            "cols: 4\n   dt: d\n   data: [ 0.02, -0.03, 0.004, -0.005 ]",
            "cols: 12\n   dt: d\n"
            "   data: [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "
            "0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported distortion coefficient count 12"):
        read_opencv_stereo_calibration(source)
