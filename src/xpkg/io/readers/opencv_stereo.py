"""Read OpenCV stereo-calibration YAML files."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from xpkg._core.time import now_utc_iso
from xpkg.model.calibration import (
    Calibration,
    CalibrationSource,
    Camera,
    CameraDistortion,
    CameraExtrinsics,
    CameraIntrinsics,
    CameraRotation,
    WorldFrame,
)


def _node(fs: cv2.FileStorage, key: str) -> cv2.FileNode:
    node = fs.getNode(key)
    if node.empty():
        raise ValueError(f"OpenCV stereo calibration YAML is missing {key!r}.")
    return node


def _matrix(
    fs: cv2.FileStorage,
    key: str,
    *,
    shape: tuple[int, int],
) -> tuple[tuple[float, ...], ...]:
    value = _node(fs, key).mat()
    if value is None:
        raise ValueError(f"OpenCV stereo calibration key {key!r} must be a matrix.")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(
            f"OpenCV stereo calibration key {key!r} must have shape {shape}, "
            f"got {array.shape}."
        )
    return tuple(tuple(float(item) for item in row) for row in array)


def _vector(fs: cv2.FileStorage, key: str, *, length: int | None = None) -> tuple[float, ...]:
    value = _node(fs, key).mat()
    if value is None:
        raise ValueError(f"OpenCV stereo calibration key {key!r} must be a numeric vector.")
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if length is not None and array.size != length:
        raise ValueError(
            f"OpenCV stereo calibration key {key!r} must contain {length} values, "
            f"got {array.size}."
        )
    return tuple(float(item) for item in array)


def _scalar_int(fs: cv2.FileStorage, key: str) -> int:
    value = int(_node(fs, key).real())
    if value <= 0:
        raise ValueError(f"OpenCV stereo calibration key {key!r} must be positive.")
    return value


def _optional_matrix_payload(fs: cv2.FileStorage, key: str) -> list[list[float]] | None:
    node = fs.getNode(key)
    if node.empty():
        return None
    value = node.mat()
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        return None
    return [[float(item) for item in row] for row in array]


def _distortion_from_opencv(
    coeffs: Sequence[float],
    *,
    camera_name: str,
) -> tuple[CameraDistortion, dict[str, Any]]:
    values = tuple(float(item) for item in coeffs)
    metadata: dict[str, Any] = {"source_distortion_coeff_count": len(values)}
    if not values:
        return CameraDistortion(), metadata
    if len(values) == 4:
        metadata["xpkg_distortion_assumption"] = "OpenCV k3 omitted; stored as 0.0 in opencv_5."
        return CameraDistortion(model="opencv_5", coeffs=(*values, 0.0)), metadata
    if len(values) == 5:
        return CameraDistortion(model="opencv_5", coeffs=values), metadata
    if len(values) == 8:
        return CameraDistortion(model="opencv_8", coeffs=values), metadata
    raise ValueError(
        f"OpenCV stereo camera {camera_name!r} has unsupported distortion coefficient "
        f"count {len(values)}; expected 4, 5, or 8."
    )


def _camera(
    *,
    name: str,
    image_size: tuple[int, int],
    intrinsics_matrix: tuple[tuple[float, ...], ...],
    distortion_coeffs: Sequence[float],
    rotation: tuple[tuple[float, ...], ...],
    translation: tuple[float, float, float],
) -> Camera:
    distortion, metadata = _distortion_from_opencv(distortion_coeffs, camera_name=name)
    return Camera(
        name=name,
        image_size=image_size,
        intrinsics=CameraIntrinsics(
            model="pinhole",
            matrix=intrinsics_matrix,
            distortion=distortion,
        ),
        extrinsics=CameraExtrinsics(
            rotation=CameraRotation(representation="matrix", value=rotation),
            translation=translation,
        ),
        metadata=metadata,
    )


def read_opencv_stereo_calibration(
    path: str | Path,
    *,
    name: str | None = None,
    camera_names: tuple[str, str] = ("camera_1", "camera_2"),
    units: str = "unknown",
    captured_at: str | None = None,
    tool_version: str | None = None,
    imported_at: str | None = None,
) -> Calibration:
    """Read an OpenCV stereo-calibration YAML file into a generic calibration.

    The accepted file is the unrectified stereo-calibration result with ``M1``,
    ``D1``, ``M2``, ``D2``, ``R``, ``T``, ``image_width``, and ``image_height``.
    OpenCV defines ``R`` and ``T`` as the transform from camera 1 coordinates
    into camera 2 coordinates; xpkg stores camera 1 as the world-frame anchor.
    """

    calibration_path = Path(path)
    if len(camera_names) != 2:
        raise ValueError("camera_names must contain exactly two names.")

    fs = cv2.FileStorage(str(calibration_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(
            f"OpenCV stereo calibration YAML not found or unreadable: {calibration_path}"
        )
    try:
        image_size = (_scalar_int(fs, "image_width"), _scalar_int(fs, "image_height"))
        m1 = _matrix(fs, "M1", shape=(3, 3))
        m2 = _matrix(fs, "M2", shape=(3, 3))
        d1 = _vector(fs, "D1")
        d2 = _vector(fs, "D2")
        rotation_2 = _matrix(fs, "R", shape=(3, 3))
        translation_2 = cast("tuple[float, float, float]", _vector(fs, "T", length=3))
        essential = _optional_matrix_payload(fs, "E")
        fundamental = _optional_matrix_payload(fs, "F")
    finally:
        fs.release()

    camera_1 = _camera(
        name=camera_names[0],
        image_size=image_size,
        intrinsics_matrix=m1,
        distortion_coeffs=d1,
        rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        translation=(0.0, 0.0, 0.0),
    )
    camera_2 = _camera(
        name=camera_names[1],
        image_size=image_size,
        intrinsics_matrix=m2,
        distortion_coeffs=d2,
        rotation=rotation_2,
        translation=translation_2,
    )

    metadata: dict[str, Any] = {
        "opencv_stereo_transform": "R,T transform camera_1 coordinates into camera_2 coordinates.",
        "coordinate_frame_assumption": (
            "camera_1 optical frame is the xpkg calibration world frame."
        ),
    }
    if essential is not None:
        metadata["essential_matrix"] = essential
    if fundamental is not None:
        metadata["fundamental_matrix"] = fundamental

    return Calibration(
        name=name or calibration_path.stem,
        captured_at=captured_at,
        units=units,
        world_frame=WorldFrame(
            anchor=camera_names[0],
            description="Camera 1 optical frame; camera 2 pose uses OpenCV stereoCalibrate R,T.",
        ),
        cameras=(camera_1, camera_2),
        source=CalibrationSource(
            tool="opencv",
            tool_version=tool_version,
            imported_from=calibration_path.name,
            imported_at=imported_at or now_utc_iso(drop_microseconds=True),
            metadata={"format": "opencv-stereo-yaml"},
        ),
        metadata=metadata,
    )


__all__ = ["read_opencv_stereo_calibration"]
