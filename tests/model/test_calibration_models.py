from __future__ import annotations

import pytest

import xpkg.model as model
from xpkg.model import (
    CALIBRATION_SCHEMA_VERSION,
    Calibration,
    CalibrationQuality,
    CalibrationSource,
    Camera,
    CameraDistortion,
    CameraExtrinsics,
    CameraIntrinsics,
    CameraRotation,
    WorldFrame,
)


def _camera(name: str = "cam_top") -> Camera:
    return Camera(
        name=name,
        image_size=(1920, 1080),
        intrinsics=CameraIntrinsics(
            matrix=((1000.0, 0.0, 960.0), (0.0, 1001.0, 540.0), (0.0, 0.0, 1.0)),
            distortion=CameraDistortion(model="opencv_5", coeffs=(0.1, -0.01, 0.0, 0.0, 0.0)),
        ),
        extrinsics=CameraExtrinsics(
            rotation=CameraRotation(
                representation="matrix",
                value=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ),
            translation=(1.0, 2.0, 3.0),
        ),
        quality=CalibrationQuality(reprojection_error_px=0.42),
    )


def test_calibration_round_trips_json_friendly_payload() -> None:
    calibration = Calibration(
        name="rig-2024-03-15",
        captured_at="2024-03-15T10:30:00Z",
        units="mm",
        world_frame=WorldFrame(
            anchor="charuco_board",
            description="Origin at board corner.",
        ),
        cameras=(_camera(),),
        source=CalibrationSource(
            tool="anipose",
            tool_version="1.0.1",
            imported_from="calibration.toml",
            imported_at="2026-05-07T14:00:00Z",
        ),
    )

    payload = calibration.to_dict()

    assert payload["schema_version"] == CALIBRATION_SCHEMA_VERSION
    assert payload["name"] == "rig-2024-03-15"
    assert payload["cameras"][0]["intrinsics"]["distortion"]["model"] == "opencv_5"
    assert payload["cameras"][0]["quality"]["reprojection_error_px"] == 0.42
    assert Calibration.from_dict(payload) == calibration


def test_calibration_requires_unique_camera_names() -> None:
    camera = _camera("cam_a")

    with pytest.raises(ValueError, match="camera names must be unique"):
        Calibration(name="rig", cameras=(camera, camera))


def test_calibration_validates_distortion_count_and_rotation_shape() -> None:
    with pytest.raises(ValueError, match="requires 5 coeffs"):
        CameraDistortion(model="opencv_5", coeffs=(0.1,))

    with pytest.raises(ValueError, match="rotation matrix"):
        CameraRotation(representation="matrix", value=((1.0, 0.0),))


def test_calibration_models_are_available_from_public_api() -> None:
    assert model.Calibration is Calibration
    assert model.Camera is Camera
    assert model.CameraIntrinsics is CameraIntrinsics
    assert model.CameraExtrinsics is CameraExtrinsics
    assert model.CameraRotation is CameraRotation
    assert model.CameraDistortion is CameraDistortion
    assert model.WorldFrame is WorldFrame
