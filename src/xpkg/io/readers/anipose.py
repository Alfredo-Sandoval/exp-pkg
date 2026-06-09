"""Read and write Anipose camera calibration TOML files."""

from __future__ import annotations

import math
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from xpkg._core.json_utils import dump_json
from xpkg._core.time import now_utc_iso
from xpkg.model.calibration import (
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

_BARE_TOML_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Anipose calibration must contain a TOML table: {path}")
    return payload


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a TOML table.")
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float_list(value: Any, *, name: str) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        raise TypeError(f"{name} must be a numeric array, not a string.")
    return [float(item) for item in value]


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number >= 0.0:
            return number
    return None


def _top_level_error(payload: Mapping[str, Any]) -> float | None:
    metadata = payload.get("metadata")
    metadata_mapping = metadata if isinstance(metadata, Mapping) else {}
    return _first_float(
        payload.get("reprojection_error_px"),
        payload.get("reprojection_error"),
        payload.get("error"),
        metadata_mapping.get("reprojection_error_px"),
        metadata_mapping.get("reprojection_error"),
        metadata_mapping.get("error"),
    )


def _distortion_from_anipose(
    distortions: Any,
    *,
    fisheye: bool,
    camera_name: str,
) -> CameraDistortion:
    coeffs = _float_list(distortions, name=f"{camera_name} distortions")
    if not coeffs:
        return CameraDistortion()
    if fisheye:
        return CameraDistortion(model="fisheye_4", coeffs=tuple(coeffs))
    if len(coeffs) == 4:
        coeffs.append(0.0)
    if len(coeffs) == 5:
        return CameraDistortion(model="opencv_5", coeffs=tuple(coeffs))
    if len(coeffs) == 8:
        return CameraDistortion(model="opencv_8", coeffs=tuple(coeffs))
    raise ValueError(
        f"Anipose camera {camera_name!r} has unsupported non-fisheye distortion "
        f"coefficient count {len(coeffs)}; expected 4, 5, or 8."
    )


def _camera_quality(
    section: Mapping[str, Any],
    *,
    global_error: float | None,
) -> CalibrationQuality:
    error = _first_float(
        section.get("reprojection_error_px"),
        section.get("reprojection_error"),
        section.get("error"),
        global_error,
    )
    n_views = section.get("n_views", section.get("views", section.get("n_images")))
    return CalibrationQuality(reprojection_error_px=error, n_views=n_views)


def _camera_from_anipose_section(
    section_name: str,
    section: Mapping[str, Any],
    *,
    global_error: float | None,
) -> Camera:
    camera_name = str(section.get("name") or section_name).strip()
    if not camera_name:
        raise ValueError(f"Anipose camera section {section_name!r} has an empty name.")

    fisheye = _as_bool(section.get("fisheye"))
    rotation = CameraRotation(
        representation="rodrigues",
        value=tuple(_float_list(section.get("rotation"), name=f"{camera_name} rotation")),
    )
    intrinsics = CameraIntrinsics(
        model="fisheye" if fisheye else "pinhole",
        matrix=section.get("matrix", ()),
        distortion=_distortion_from_anipose(
            section.get("distortions"),
            fisheye=fisheye,
            camera_name=camera_name,
        ),
    )
    return Camera(
        name=camera_name,
        image_size=section.get("size", section.get("image_size", ())),
        intrinsics=intrinsics,
        extrinsics=CameraExtrinsics(
            rotation=CameraRotation(representation="matrix", value=rotation.to_matrix()),
            translation=cast(
                "tuple[float, float, float]",
                tuple(_float_list(section.get("translation"), name=f"{camera_name} translation")),
            ),
        ),
        quality=_camera_quality(section, global_error=global_error),
    )


def _camera_sections(payload: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    sections: list[tuple[str, Mapping[str, Any]]] = []
    for section_name, section in payload.items():
        if not isinstance(section, Mapping):
            continue
        if "matrix" not in section:
            continue
        sections.append((str(section_name), section))
    if not sections:
        raise ValueError("Anipose calibration TOML contains no camera sections.")
    return sections


def read_anipose_calibration(
    path: str | Path,
    *,
    name: str | None = None,
    units: str = "unknown",
    captured_at: str | None = None,
    world_frame: WorldFrame | Mapping[str, Any] | None = None,
    tool_version: str | None = None,
    imported_at: str | None = None,
) -> Calibration:
    """Read an Anipose ``calibration.toml`` into a generic xpkg calibration."""

    calibration_path = Path(path)
    payload = _load_toml(calibration_path)
    metadata = payload.get("metadata")
    metadata_mapping = metadata if isinstance(metadata, Mapping) else {}
    calibration_name = (
        name
        or payload.get("name")
        or metadata_mapping.get("name")
        or calibration_path.stem
    )
    global_error = _top_level_error(payload)
    cameras = [
        _camera_from_anipose_section(
            section_name,
            _as_mapping(section, name=section_name),
            global_error=global_error,
        )
        for section_name, section in _camera_sections(payload)
    ]
    resolved_world_frame = (
        WorldFrame.from_dict(cast("Mapping[str, Any]", world_frame))
        if isinstance(world_frame, Mapping)
        else world_frame
    )
    return Calibration(
        name=str(calibration_name),
        captured_at=captured_at,
        units=units,
        world_frame=resolved_world_frame,
        cameras=tuple(cameras),
        source=CalibrationSource(
            tool="anipose",
            tool_version=tool_version,
            imported_from=calibration_path.name,
            imported_at=imported_at or now_utc_iso(drop_microseconds=True),
        ),
    )


def _toml_key(value: str) -> str:
    if _BARE_TOML_KEY_RE.match(value):
        return value
    return dump_json(value, ensure_ascii=True, compact=True)


def _toml_string(value: str) -> str:
    return dump_json(value, ensure_ascii=True, compact=True)


def _toml_float(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"TOML numeric values must be finite, got {number}.")
    text = format(number, ".17g")
    if "e" not in text.lower() and "." not in text:
        text += ".0"
    return text


def _toml_float_array(values: Any) -> str:
    return "[" + ", ".join(_toml_float(float(value)) for value in values) + "]"


def _toml_int_array(values: tuple[int, int]) -> str:
    return "[" + ", ".join(str(int(value)) for value in values) + "]"


def _toml_matrix(values: tuple[tuple[float, ...], ...]) -> str:
    rows = [_toml_float_array(row) for row in values]
    return "[" + ", ".join(rows) + "]"


def _anipose_distortion_payload(camera: Camera) -> tuple[bool, tuple[float, ...]]:
    distortion = camera.intrinsics.distortion
    model = distortion.model
    if model == "fisheye_4":
        return True, distortion.coeffs
    if model in {"opencv_5", "opencv_8", "none"}:
        return camera.intrinsics.model == "fisheye", distortion.coeffs
    raise ValueError(f"Unsupported distortion model for Anipose export: {model!r}.")


def write_anipose_calibration(
    calibration: Calibration,
    path: str | Path,
) -> None:
    """Write an xpkg calibration as an Anipose-compatible TOML file."""

    if not isinstance(calibration, Calibration):
        raise TypeError(f"calibration must be a Calibration, got {calibration!r}.")

    lines: list[str] = []
    for camera in calibration.cameras:
        fisheye, distortions = _anipose_distortion_payload(camera)
        lines.extend(
            [
                f"[{_toml_key(camera.name)}]",
                f"name = {_toml_string(camera.name)}",
                f"size = {_toml_int_array(camera.image_size)}",
                f"matrix = {_toml_matrix(camera.intrinsics.matrix)}",
                f"distortions = {_toml_float_array(distortions)}",
                f"rotation = {_toml_float_array(camera.extrinsics.rotation.as_rodrigues())}",
                f"translation = {_toml_float_array(camera.extrinsics.translation)}",
                f"fisheye = {str(fisheye).lower()}",
                "",
            ]
        )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


__all__ = ["read_anipose_calibration", "write_anipose_calibration"]
