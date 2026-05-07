"""Camera calibration primitives for multi-view experiment projects."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

CALIBRATION_SCHEMA_VERSION = 1

IntrinsicsModel = Literal["pinhole", "fisheye", "omnidir"]
DistortionModel = Literal["none", "opencv_5", "opencv_8", "fisheye_4"]
RotationRepresentation = Literal["matrix", "rodrigues", "quaternion"]

_INTRINSICS_MODELS = {"pinhole", "fisheye", "omnidir"}
_DISTORTION_COEFF_COUNTS = {
    "none": 0,
    "opencv_5": 5,
    "opencv_8": 8,
    "fisheye_4": 4,
}
_ROTATION_REPRESENTATIONS = {"matrix", "rodrigues", "quaternion"}


def _required_text(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string.")
    return text


def _optional_text(value: object | None, *, name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, name=name)


def _metadata(value: Mapping[str, Any] | None, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or None.")
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        normalized[_required_text(key, name=f"{name} key")] = item
    return normalized


def _finite_float(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {number}.")
    return number


def _optional_nonnegative_float(value: Any | None, *, name: str) -> float | None:
    if value is None:
        return None
    number = _finite_float(value, name=name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative, got {number}.")
    return number


def _optional_nonnegative_int(value: Any | None, *, name: str) -> int | None:
    if value is None:
        return None
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative, got {number}.")
    return number


def _float_tuple(value: Iterable[Any], *, length: int, name: str) -> tuple[float, ...]:
    if isinstance(value, str):
        raise TypeError(f"{name} must be an iterable of numbers, not a string.")
    items = tuple(_finite_float(item, name=f"{name} item") for item in value)
    if len(items) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(items)}.")
    return items


def _matrix3x3(value: Iterable[Iterable[Any]], *, name: str) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, str):
        raise TypeError(f"{name} must be a 3x3 numeric matrix, not a string.")
    rows = tuple(_float_tuple(row, length=3, name=f"{name} row") for row in value)
    if len(rows) != 3:
        raise ValueError(f"{name} must contain 3 rows, got {len(rows)}.")
    return rows


def _image_size(value: Iterable[Any], *, name: str) -> tuple[int, int]:
    if isinstance(value, str):
        raise TypeError(f"{name} must be a two-item iterable, not a string.")
    items = tuple(value)
    if len(items) != 2:
        raise ValueError(f"{name} must contain width and height.")
    width = int(items[0])
    height = int(items[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} width and height must be positive.")
    return (width, height)


def _world_frame(
    value: WorldFrame | Mapping[str, Any] | None,
) -> WorldFrame | None:
    if value is None:
        return None
    if isinstance(value, WorldFrame):
        return value
    if isinstance(value, Mapping):
        return WorldFrame.from_dict(value)
    raise TypeError("world_frame must be a WorldFrame, mapping, or None.")


def _rotation_matrix_from_rodrigues(
    value: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    rx, ry, rz = _float_tuple(value, length=3, name="Rodrigues rotation")
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta == 0.0:
        return (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )

    kx, ky, kz = rx / theta, ry / theta, rz / theta
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    one_minus_cos = 1.0 - cos_t

    return (
        (
            cos_t + kx * kx * one_minus_cos,
            kx * ky * one_minus_cos - kz * sin_t,
            kx * kz * one_minus_cos + ky * sin_t,
        ),
        (
            ky * kx * one_minus_cos + kz * sin_t,
            cos_t + ky * ky * one_minus_cos,
            ky * kz * one_minus_cos - kx * sin_t,
        ),
        (
            kz * kx * one_minus_cos - ky * sin_t,
            kz * ky * one_minus_cos + kx * sin_t,
            cos_t + kz * kz * one_minus_cos,
        ),
    )


def _rodrigues_from_rotation_matrix(
    value: Iterable[Iterable[Any]],
) -> tuple[float, float, float]:
    matrix = _matrix3x3(value, name="rotation matrix")
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    theta = math.acos(cos_theta)
    if abs(theta) < 1e-12:
        return (0.0, 0.0, 0.0)

    sin_theta = math.sin(theta)
    if abs(sin_theta) < 1e-12:
        # Stable enough for the 180-degree case: choose the largest diagonal axis.
        xx = max(0.0, (matrix[0][0] + 1.0) / 2.0)
        yy = max(0.0, (matrix[1][1] + 1.0) / 2.0)
        zz = max(0.0, (matrix[2][2] + 1.0) / 2.0)
        axis = [math.sqrt(xx), math.sqrt(yy), math.sqrt(zz)]
        if matrix[0][1] < 0.0:
            axis[1] = -axis[1]
        if matrix[0][2] < 0.0:
            axis[2] = -axis[2]
        return (theta * axis[0], theta * axis[1], theta * axis[2])

    scale = theta / (2.0 * sin_theta)
    return (
        scale * (matrix[2][1] - matrix[1][2]),
        scale * (matrix[0][2] - matrix[2][0]),
        scale * (matrix[1][0] - matrix[0][1]),
    )


def _rotation_matrix_from_quaternion(
    value: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    w, x, y, z = _float_tuple(value, length=4, name="quaternion rotation")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("quaternion rotation must not be all zeros.")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


@dataclass(frozen=True, slots=True)
class CameraDistortion:
    """Lens distortion model and coefficients."""

    model: DistortionModel = "none"
    coeffs: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        model = _required_text(self.model, name="distortion model")
        if model not in _DISTORTION_COEFF_COUNTS:
            allowed = ", ".join(sorted(_DISTORTION_COEFF_COUNTS))
            raise ValueError(f"distortion model must be one of: {allowed}.")
        coeffs = tuple(
            _finite_float(value, name="distortion coeff")
            for value in self.coeffs
        )
        expected = _DISTORTION_COEFF_COUNTS[model]
        if len(coeffs) != expected:
            raise ValueError(
                f"distortion model {model!r} requires {expected} coeffs, got {len(coeffs)}."
            )
        object.__setattr__(self, "model", cast("DistortionModel", model))
        object.__setattr__(self, "coeffs", coeffs)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly distortion payload."""
        return {"model": self.model, "coeffs": list(self.coeffs)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CameraDistortion:
        """Hydrate distortion data from a JSON-friendly payload."""
        if not isinstance(payload, Mapping):
            raise TypeError("distortion payload must be a mapping.")
        return cls(
            model=cast("DistortionModel", payload.get("model", "none")),
            coeffs=tuple(payload.get("coeffs") or ()),
        )


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Intrinsic camera matrix and lens model."""

    matrix: tuple[tuple[float, ...], ...]
    model: IntrinsicsModel = "pinhole"
    distortion: CameraDistortion = field(default_factory=CameraDistortion)

    def __post_init__(self) -> None:
        model = _required_text(self.model, name="intrinsics model")
        if model not in _INTRINSICS_MODELS:
            allowed = ", ".join(sorted(_INTRINSICS_MODELS))
            raise ValueError(f"intrinsics model must be one of: {allowed}.")
        distortion = self.distortion
        if isinstance(distortion, Mapping):
            distortion = CameraDistortion.from_dict(cast("Mapping[str, Any]", distortion))
        if not isinstance(distortion, CameraDistortion):
            raise TypeError("intrinsics distortion must be a CameraDistortion.")
        object.__setattr__(self, "model", cast("IntrinsicsModel", model))
        object.__setattr__(
            self,
            "matrix",
            _matrix3x3(self.matrix, name="intrinsics matrix"),
        )
        object.__setattr__(self, "distortion", distortion)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly intrinsics payload."""
        return {
            "model": self.model,
            "matrix": [list(row) for row in self.matrix],
            "distortion": self.distortion.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CameraIntrinsics:
        """Hydrate intrinsics from a JSON-friendly payload."""
        if not isinstance(payload, Mapping):
            raise TypeError("intrinsics payload must be a mapping.")
        return cls(
            model=cast("IntrinsicsModel", payload.get("model", "pinhole")),
            matrix=payload.get("matrix", ()),
            distortion=CameraDistortion.from_dict(payload.get("distortion") or {}),
        )


@dataclass(frozen=True, slots=True)
class CameraRotation:
    """Camera rotation with an explicit representation tag."""

    representation: RotationRepresentation
    value: tuple[tuple[float, ...], ...] | tuple[float, ...]

    def __post_init__(self) -> None:
        representation = _required_text(self.representation, name="rotation representation")
        if representation not in _ROTATION_REPRESENTATIONS:
            allowed = ", ".join(sorted(_ROTATION_REPRESENTATIONS))
            raise ValueError(f"rotation representation must be one of: {allowed}.")
        if representation == "matrix":
            normalized_value = _matrix3x3(
                cast("Iterable[Iterable[Any]]", self.value),
                name="rotation matrix",
            )
        elif representation == "rodrigues":
            normalized_value = _float_tuple(
                cast("Iterable[Any]", self.value),
                length=3,
                name="Rodrigues rotation",
            )
        else:
            normalized_value = _float_tuple(
                cast("Iterable[Any]", self.value),
                length=4,
                name="quaternion rotation",
            )
        object.__setattr__(
            self,
            "representation",
            cast("RotationRepresentation", representation),
        )
        object.__setattr__(self, "value", normalized_value)

    def to_matrix(self) -> tuple[tuple[float, ...], ...]:
        """Return this rotation as a 3x3 matrix."""
        if self.representation == "matrix":
            return cast("tuple[tuple[float, ...], ...]", self.value)
        if self.representation == "rodrigues":
            return _rotation_matrix_from_rodrigues(cast("tuple[float, ...]", self.value))
        return _rotation_matrix_from_quaternion(cast("tuple[float, ...]", self.value))

    def as_rodrigues(self) -> tuple[float, float, float]:
        """Return this rotation as an OpenCV Rodrigues vector."""
        if self.representation == "rodrigues":
            return cast("tuple[float, float, float]", self.value)
        return _rodrigues_from_rotation_matrix(self.to_matrix())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly rotation payload."""
        if self.representation == "matrix":
            value: list[Any] = [
                list(row)
                for row in cast("tuple[tuple[float, ...], ...]", self.value)
            ]
        else:
            value = list(cast("tuple[float, ...]", self.value))
        return {"representation": self.representation, "value": value}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CameraRotation:
        """Hydrate a rotation from a JSON-friendly payload."""
        if not isinstance(payload, Mapping):
            raise TypeError("rotation payload must be a mapping.")
        return cls(
            representation=cast(
                "RotationRepresentation",
                payload.get("representation", "matrix"),
            ),
            value=payload.get("value", ()),
        )


@dataclass(frozen=True, slots=True)
class CameraExtrinsics:
    """Extrinsic camera pose in the calibration world frame."""

    rotation: CameraRotation
    translation: tuple[float, float, float]

    def __post_init__(self) -> None:
        rotation = self.rotation
        if isinstance(rotation, Mapping):
            rotation = CameraRotation.from_dict(cast("Mapping[str, Any]", rotation))
        if not isinstance(rotation, CameraRotation):
            raise TypeError("extrinsics rotation must be a CameraRotation.")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(
            self,
            "translation",
            _float_tuple(self.translation, length=3, name="translation"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly extrinsics payload."""
        return {
            "rotation": self.rotation.to_dict(),
            "translation": list(self.translation),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CameraExtrinsics:
        """Hydrate extrinsics from a JSON-friendly payload."""
        if not isinstance(payload, Mapping):
            raise TypeError("extrinsics payload must be a mapping.")
        return cls(
            rotation=CameraRotation.from_dict(payload.get("rotation") or {}),
            translation=payload.get("translation", ()),
        )


@dataclass(frozen=True, slots=True)
class CalibrationQuality:
    """Calibration quality metrics when the source format provides them."""

    reprojection_error_px: float | None = None
    n_views: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reprojection_error_px",
            _optional_nonnegative_float(
                self.reprojection_error_px,
                name="reprojection_error_px",
            ),
        )
        object.__setattr__(
            self,
            "n_views",
            _optional_nonnegative_int(self.n_views, name="n_views"),
        )
        object.__setattr__(
            self,
            "metadata",
            _metadata(self.metadata, name="calibration quality metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly quality payload."""
        payload: dict[str, Any] = {}
        if self.reprojection_error_px is not None:
            payload["reprojection_error_px"] = self.reprojection_error_px
        if self.n_views is not None:
            payload["n_views"] = self.n_views
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> CalibrationQuality:
        """Hydrate quality metrics from a JSON-friendly payload."""
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("calibration quality payload must be a mapping.")
        return cls(
            reprojection_error_px=payload.get("reprojection_error_px"),
            n_views=payload.get("n_views"),
            metadata=_metadata(payload.get("metadata"), name="calibration quality metadata"),
        )


@dataclass(frozen=True, slots=True)
class Camera:
    """One calibrated camera in a rig."""

    name: str
    image_size: tuple[int, int]
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    quality: CalibrationQuality = field(default_factory=CalibrationQuality)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        intrinsics = self.intrinsics
        if isinstance(intrinsics, Mapping):
            intrinsics = CameraIntrinsics.from_dict(cast("Mapping[str, Any]", intrinsics))
        if not isinstance(intrinsics, CameraIntrinsics):
            raise TypeError("camera intrinsics must be a CameraIntrinsics object.")

        extrinsics = self.extrinsics
        if isinstance(extrinsics, Mapping):
            extrinsics = CameraExtrinsics.from_dict(cast("Mapping[str, Any]", extrinsics))
        if not isinstance(extrinsics, CameraExtrinsics):
            raise TypeError("camera extrinsics must be a CameraExtrinsics object.")

        quality = self.quality
        if isinstance(quality, Mapping):
            quality = CalibrationQuality.from_dict(cast("Mapping[str, Any]", quality))
        if not isinstance(quality, CalibrationQuality):
            raise TypeError("camera quality must be a CalibrationQuality object.")

        object.__setattr__(self, "name", _required_text(self.name, name="camera name"))
        object.__setattr__(self, "image_size", _image_size(self.image_size, name="image_size"))
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "extrinsics", extrinsics)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(
            self,
            "metadata",
            _metadata(self.metadata, name="camera calibration metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly camera calibration payload."""
        payload: dict[str, Any] = {
            "name": self.name,
            "image_size": list(self.image_size),
            "intrinsics": self.intrinsics.to_dict(),
            "extrinsics": self.extrinsics.to_dict(),
        }
        quality = self.quality.to_dict()
        if quality:
            payload["quality"] = quality
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Camera:
        """Hydrate one camera calibration from a JSON-friendly payload."""
        if not isinstance(payload, Mapping):
            raise TypeError("camera calibration payload must be a mapping.")
        return cls(
            name=payload.get("name", ""),
            image_size=payload.get("image_size", ()),
            intrinsics=CameraIntrinsics.from_dict(payload.get("intrinsics") or {}),
            extrinsics=CameraExtrinsics.from_dict(payload.get("extrinsics") or {}),
            quality=CalibrationQuality.from_dict(payload.get("quality")),
            metadata=_metadata(payload.get("metadata"), name="camera calibration metadata"),
        )


@dataclass(frozen=True, slots=True)
class WorldFrame:
    """Human-readable description of the calibration coordinate frame."""

    anchor: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor", _optional_text(self.anchor, name="world anchor"))
        object.__setattr__(
            self,
            "description",
            _optional_text(self.description, name="world frame description"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly world-frame payload."""
        payload: dict[str, Any] = {}
        if self.anchor is not None:
            payload["anchor"] = self.anchor
        if self.description is not None:
            payload["description"] = self.description
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorldFrame:
        """Hydrate a world-frame description from a JSON-friendly payload."""
        if not isinstance(payload, Mapping):
            raise TypeError("world_frame payload must be a mapping.")
        return cls(anchor=payload.get("anchor"), description=payload.get("description"))


@dataclass(frozen=True, slots=True)
class CalibrationSource:
    """Provenance for imported or generated calibration data."""

    tool: str | None = None
    tool_version: str | None = None
    imported_from: str | None = None
    imported_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool", _optional_text(self.tool, name="source tool"))
        object.__setattr__(
            self,
            "tool_version",
            _optional_text(self.tool_version, name="source tool_version"),
        )
        object.__setattr__(
            self,
            "imported_from",
            _optional_text(self.imported_from, name="source imported_from"),
        )
        object.__setattr__(
            self,
            "imported_at",
            _optional_text(self.imported_at, name="source imported_at"),
        )
        object.__setattr__(
            self,
            "metadata",
            _metadata(self.metadata, name="calibration source metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly source payload."""
        payload: dict[str, Any] = {}
        for key in ("tool", "tool_version", "imported_from", "imported_at"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> CalibrationSource | None:
        """Hydrate source provenance from a JSON-friendly payload."""
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise TypeError("calibration source payload must be a mapping.")
        return cls(
            tool=payload.get("tool"),
            tool_version=payload.get("tool_version"),
            imported_from=payload.get("imported_from"),
            imported_at=payload.get("imported_at"),
            metadata=_metadata(payload.get("metadata"), name="calibration source metadata"),
        )


def _camera_tuple(value: Iterable[Camera | Mapping[str, Any]]) -> tuple[Camera, ...]:
    if isinstance(value, str):
        raise TypeError("calibration cameras must be an iterable, not a string.")
    cameras: list[Camera] = []
    for item in value:
        if isinstance(item, Camera):
            cameras.append(item)
        elif isinstance(item, Mapping):
            cameras.append(Camera.from_dict(item))
        else:
            raise TypeError("calibration cameras must contain Camera objects or mappings.")
    if not cameras:
        raise ValueError("calibration must contain at least one camera.")
    names = [camera.name for camera in cameras]
    if len(set(names)) != len(names):
        raise ValueError("calibration camera names must be unique.")
    return tuple(cameras)


@dataclass(frozen=True, slots=True)
class Calibration:
    """Multi-camera calibration for one physical rig or camera setup."""

    name: str
    cameras: tuple[Camera, ...]
    captured_at: str | None = None
    units: str = "unknown"
    world_frame: WorldFrame | None = None
    source: CalibrationSource | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = CALIBRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        version = int(self.schema_version)
        if version != CALIBRATION_SCHEMA_VERSION:
            raise ValueError(
                "calibration schema_version must be "
                f"{CALIBRATION_SCHEMA_VERSION}, got {version}."
            )
        source = self.source
        if isinstance(source, Mapping):
            source = CalibrationSource.from_dict(cast("Mapping[str, Any]", source))
        if source is not None and not isinstance(source, CalibrationSource):
            raise TypeError("calibration source must be a CalibrationSource or None.")

        object.__setattr__(self, "schema_version", version)
        object.__setattr__(self, "name", _required_text(self.name, name="calibration name"))
        object.__setattr__(
            self,
            "captured_at",
            _optional_text(self.captured_at, name="captured_at"),
        )
        object.__setattr__(self, "units", _required_text(self.units, name="units"))
        object.__setattr__(self, "world_frame", _world_frame(self.world_frame))
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "cameras", _camera_tuple(self.cameras))
        object.__setattr__(
            self,
            "metadata",
            _metadata(self.metadata, name="calibration metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly calibration payload."""
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "units": self.units,
        }
        if self.captured_at is not None:
            payload["captured_at"] = self.captured_at
        if self.world_frame is not None:
            world_frame = self.world_frame.to_dict()
            if world_frame:
                payload["world_frame"] = world_frame
        payload["cameras"] = [camera.to_dict() for camera in self.cameras]
        if self.source is not None:
            source = self.source.to_dict()
            if source:
                payload["source"] = source
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Calibration:
        """Hydrate a calibration from a JSON-friendly payload."""
        if not isinstance(payload, Mapping):
            raise TypeError("calibration payload must be a mapping.")
        return cls(
            schema_version=int(payload.get("schema_version", 0)),
            name=payload.get("name", ""),
            captured_at=payload.get("captured_at"),
            units=payload.get("units", "unknown"),
            world_frame=_world_frame(payload.get("world_frame")),
            cameras=tuple(payload.get("cameras") or ()),
            source=CalibrationSource.from_dict(payload.get("source")),
            metadata=_metadata(payload.get("metadata"), name="calibration metadata"),
        )

    def camera_by_name(self, name: str) -> Camera:
        """Return a calibrated camera by exact name."""
        target = _required_text(name, name="camera name")
        for camera in self.cameras:
            if camera.name == target:
                return camera
        raise KeyError(f"calibration {self.name!r} has no camera named {target!r}.")


__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "Calibration",
    "CalibrationQuality",
    "CalibrationSource",
    "Camera",
    "CameraDistortion",
    "CameraExtrinsics",
    "CameraIntrinsics",
    "CameraRotation",
    "DistortionModel",
    "IntrinsicsModel",
    "RotationRepresentation",
    "WorldFrame",
]
