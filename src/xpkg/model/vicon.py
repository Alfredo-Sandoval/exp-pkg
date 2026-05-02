"""Canonical in-memory model for Vicon recordings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from xpkg.pose.naming import (
    lookup_unique_label_or_marker,
    normalize_event_side,
    normalize_event_type,
    normalize_label_name,
    normalize_marker_name,
)


def _tuple_str(items: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(item) for item in items)


def _coerce_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value)


def _float_array(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError(f"{name} must be an array, got scalar {array!r}.")
    return array


def _bool_array(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.ndim == 0:
        raise ValueError(f"{name} must be an array, got scalar {array!r}.")
    return array


def _lookup_unique_index(labels: Sequence[str], name: str, *, kind: str) -> int:
    return lookup_unique_label_or_marker(labels, name, kind=kind)


@dataclass(frozen=True, slots=True)
class ViconAnalogData:
    """Analog channels sampled alongside a Vicon point stream."""

    fps: int
    samples_per_frame: int
    channel_names: tuple[str, ...]
    values: np.ndarray  # (samples, channels) float64
    channel_units: tuple[str, ...] = ()
    channel_descriptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        fps = int(self.fps)
        samples_per_frame = int(self.samples_per_frame)
        channel_names = _tuple_str(self.channel_names)
        values = _float_array(self.values, name="analog.values")
        channel_units = _tuple_str(self.channel_units)
        channel_descriptions = _tuple_str(self.channel_descriptions)

        if fps <= 0:
            raise ValueError(f"analog.fps must be positive, got {fps}.")
        if samples_per_frame <= 0:
            raise ValueError(
                "analog.samples_per_frame must be positive, "
                f"got {samples_per_frame}."
            )
        if values.ndim != 2:
            raise ValueError(
                "analog.values must have shape (samples, channels), "
                f"got {values.shape}."
            )
        if values.shape[1] != len(channel_names):
            raise ValueError(
                "analog.channel_names length does not match values.shape[1]: "
                f"{len(channel_names)} vs {values.shape[1]}."
            )
        if channel_units and len(channel_units) != len(channel_names):
            raise ValueError(
                "analog.channel_units length does not match channel_names: "
                f"{len(channel_units)} vs {len(channel_names)}."
            )
        if channel_descriptions and len(channel_descriptions) != len(channel_names):
            raise ValueError(
                "analog.channel_descriptions length does not match channel_names: "
                f"{len(channel_descriptions)} vs {len(channel_names)}."
            )
        if values.shape[0] % samples_per_frame != 0:
            raise ValueError(
                "analog.values sample count must be divisible by samples_per_frame, "
                f"got {values.shape[0]} and {samples_per_frame}."
            )

        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "samples_per_frame", samples_per_frame)
        object.__setattr__(self, "channel_names", channel_names)
        object.__setattr__(self, "values", values)
        object.__setattr__(
            self,
            "channel_units",
            channel_units or ("",) * len(channel_names),
        )
        object.__setattr__(
            self,
            "channel_descriptions",
            channel_descriptions or ("",) * len(channel_names),
        )

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.values.shape[1])

    @property
    def duration_s(self) -> float:
        return float(self.n_samples / self.fps)

    def channel_index(self, name: str) -> int:
        return _lookup_unique_index(self.channel_names, name, kind="Analog channel")

    def channel_indices_by_unit(self, unit: str) -> tuple[int, ...]:
        normalized_unit = str(unit).strip().lower()
        return tuple(
            index
            for index, channel_unit in enumerate(self.channel_units)
            if str(channel_unit).strip().lower() == normalized_unit
        )

    @property
    def candidate_emg_channel_indices(self) -> tuple[int, ...]:
        voltage_units = {"v", "volt", "volts", "mv", "millivolt", "millivolts"}
        return tuple(
            index
            for index, (name, unit) in enumerate(
                zip(self.channel_names, self.channel_units, strict=True)
            )
            if str(unit).strip().lower() in voltage_units
            or normalize_label_name(name).startswith(("emg", "voltage"))
        )

    @property
    def candidate_emg_channel_names(self) -> tuple[str, ...]:
        return tuple(
            self.channel_names[index] for index in self.candidate_emg_channel_indices
        )


@dataclass(frozen=True, slots=True)
class ViconAdditionalPointData:
    """Additional non-marker C3D point channels preserved with a recording."""

    labels: tuple[str, ...]
    values: np.ndarray  # (frames, points, 5) float64 raw C3D point rows

    def __post_init__(self) -> None:
        labels = _tuple_str(self.labels)
        values = _float_array(self.values, name="additional_points.values")

        if values.ndim != 3 or values.shape[2] != 5:
            raise ValueError(
                "additional_points.values must have shape (frames, points, 5), "
                f"got {values.shape}."
            )
        if values.shape[1] != len(labels):
            raise ValueError(
                "additional_points.labels length does not match values.shape[1]: "
                f"{len(labels)} vs {values.shape[1]}."
            )

        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "values", values)

    @property
    def n_frames(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_points(self) -> int:
        return int(self.values.shape[1])

    @property
    def xyz(self) -> np.ndarray:
        return self.values[:, :, :3]

    @property
    def residual(self) -> np.ndarray:
        return self.values[:, :, 3]

    @property
    def camera_counts(self) -> np.ndarray:
        return self.values[:, :, 4]

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.xyz).all(axis=2) & (self.residual >= 0)

    def point_index(self, name: str) -> int:
        return _lookup_unique_index(self.labels, name, kind="Additional point")


@dataclass(frozen=True, slots=True)
class ViconForcePlatformMetadata:
    """Typed C3D FORCE_PLATFORM metadata needed for force-plate mapping."""

    used: int
    plate_types: tuple[int, ...]
    channels: np.ndarray  # (plates, 6) 1-based analog channel numbers
    corners: np.ndarray  # (plates, 4, 3) C3D FORCE_PLATFORM.CORNERS values
    origins: np.ndarray  # (plates, 3) C3D FORCE_PLATFORM.ORIGIN values
    provenance: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        used = int(self.used)
        plate_types = tuple(int(value) for value in self.plate_types)
        channels = np.asarray(self.channels, dtype=np.int64)
        corners = _float_array(self.corners, name="force_platform.corners")
        origins = _float_array(self.origins, name="force_platform.origins")
        provenance = tuple((str(key), str(value)) for key, value in self.provenance)

        if used <= 0:
            raise ValueError(f"force_platform.used must be positive, got {used}.")
        if len(plate_types) != used:
            raise ValueError(
                "force_platform.plate_types length must match used: "
                f"{len(plate_types)} vs {used}."
            )
        if channels.shape != (used, 6):
            raise ValueError(
                "force_platform.channels must have shape (plates, 6), "
                f"got {channels.shape} for used={used}."
            )
        if np.any(channels <= 0):
            raise ValueError("force_platform.channels must use positive 1-based indices.")
        if corners.shape != (used, 4, 3):
            raise ValueError(
                "force_platform.corners must have shape (plates, 4, 3), "
                f"got {corners.shape} for used={used}."
            )
        if origins.shape != (used, 3):
            raise ValueError(
                "force_platform.origins must have shape (plates, 3), "
                f"got {origins.shape} for used={used}."
            )
        if any(not key for key, _value in provenance):
            raise ValueError("force_platform.provenance keys cannot be empty.")

        object.__setattr__(self, "used", used)
        object.__setattr__(self, "plate_types", plate_types)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "corners", corners)
        object.__setattr__(self, "origins", origins)
        object.__setattr__(self, "provenance", provenance)


@dataclass(frozen=True, slots=True)
class ViconCamera:
    """A single Vicon camera parsed from a sibling XCP file."""

    device_id: int
    user_id: int
    sensor: str
    position: np.ndarray  # (3,) XYZ in mm
    orientation: np.ndarray  # (4,) quaternion (x, y, z, w)
    focal_length: float
    image_error: float
    world_error: float
    sensor_size: tuple[int, int]

    def __post_init__(self) -> None:
        position = _float_array(self.position, name="camera.position")
        orientation = _float_array(self.orientation, name="camera.orientation")
        sensor_size = (int(self.sensor_size[0]), int(self.sensor_size[1]))

        if position.shape != (3,):
            raise ValueError(f"camera.position must have shape (3,), got {position.shape}.")
        if orientation.shape != (4,):
            raise ValueError(
                f"camera.orientation must have shape (4,), got {orientation.shape}."
            )

        object.__setattr__(self, "device_id", int(self.device_id))
        object.__setattr__(self, "user_id", int(self.user_id))
        object.__setattr__(self, "sensor", str(self.sensor))
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "focal_length", float(self.focal_length))
        object.__setattr__(self, "image_error", float(self.image_error))
        object.__setattr__(self, "world_error", float(self.world_error))
        object.__setattr__(self, "sensor_size", sensor_size)

    @property
    def label(self) -> str:
        return f"Cam {self.user_id}"

    def forward_vector(self, *, length: float = 100.0) -> np.ndarray:
        """Return the camera optical axis in world coordinates.

        XCP stores the orientation quaternion as world-to-camera, so this
        rotates the local +Z axis by the conjugate to recover the world-space
        viewing direction.
        """

        x, y, z, w = np.asarray(self.orientation, dtype=np.float64)
        x, y, z = -x, -y, -z
        forward = np.array(
            [
                2.0 * (x * z + w * y),
                2.0 * (y * z - w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(forward))
        if norm > 0.0:
            forward /= norm
        return forward * float(length)


@dataclass(frozen=True, slots=True)
class ViconEvent:
    """A single raw event parsed from C3D ``EVENT`` metadata."""

    context: str
    label: str
    frame: int
    source_frame: int
    time_seconds: float
    event_type: str
    side: str | None = None
    subject_label: str | None = None

    def __post_init__(self) -> None:
        context = str(self.context).strip()
        label = str(self.label).strip()
        frame = int(self.frame)
        source_frame = int(self.source_frame)
        time_seconds = float(self.time_seconds)
        event_type = normalize_event_type(str(self.event_type).strip() or label)
        side = self.side
        subject_label = self.subject_label

        if not context:
            raise ValueError("event.context cannot be empty.")
        if not label:
            raise ValueError("event.label cannot be empty.")
        if frame < 0:
            raise ValueError(f"event.frame must be >= 0, got {frame}.")
        if source_frame < 0:
            raise ValueError(f"event.source_frame must be >= 0, got {source_frame}.")
        if side is not None:
            side = normalize_event_side(side)
            if side is None:
                raise ValueError("event.side must be 'left', 'right', or None.")
        if subject_label is not None:
            subject_label = str(subject_label).strip() or None

        object.__setattr__(self, "context", context)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "source_frame", source_frame)
        object.__setattr__(self, "time_seconds", time_seconds)
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "subject_label", subject_label)


@dataclass(frozen=True, slots=True)
class ViconMarkerModel:
    """Marker-model metadata resolved from a VSK or detected schema."""

    name: str
    display_name: str
    marker_names: tuple[str, ...]
    edges: tuple[tuple[str, str], ...] = ()
    source: str = "detected"

    def __post_init__(self) -> None:
        marker_names = _tuple_str(self.marker_names)
        edges = tuple((str(parent), str(child)) for parent, child in self.edges)
        if not str(self.name).strip():
            raise ValueError("model.name cannot be empty.")
        if not str(self.display_name).strip():
            raise ValueError("model.display_name cannot be empty.")
        if not marker_names:
            raise ValueError("model.marker_names cannot be empty.")
        normalized_markers = {normalize_marker_name(marker_name) for marker_name in marker_names}
        for parent, child in edges:
            if normalize_marker_name(parent) not in normalized_markers:
                raise ValueError(f"model edge references unknown marker {parent!r}.")
            if normalize_marker_name(child) not in normalized_markers:
                raise ValueError(f"model edge references unknown marker {child!r}.")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "display_name", str(self.display_name))
        object.__setattr__(self, "marker_names", marker_names)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "source", str(self.source))

    def marker_index(self, name: str) -> int:
        return _lookup_unique_index(self.marker_names, name, kind="Model marker")


@dataclass(frozen=True, slots=True)
class ViconRecording:
    """Parsed Vicon recording with 3D markers plus optional sidecar data."""

    path: Path
    source_type: str
    fps: int
    marker_names: tuple[str, ...]
    source_marker_labels: tuple[str, ...]
    positions: np.ndarray  # (frames, markers, 3) float64
    marker_valid: np.ndarray  # (frames, markers) bool
    frame_offset: int
    events: tuple[ViconEvent, ...] = ()
    analog: ViconAnalogData | None = None
    additional_points: ViconAdditionalPointData | None = None
    cameras: tuple[ViconCamera, ...] = ()
    model: ViconMarkerModel | None = None
    force_platform: ViconForcePlatformMetadata | None = None
    xcp_path: Path | None = None
    vsk_path: Path | None = None

    def __post_init__(self) -> None:
        path = Path(self.path)
        source_type = str(self.source_type).strip().lower()
        fps = int(self.fps)
        marker_names = _tuple_str(self.marker_names)
        source_marker_labels = _tuple_str(self.source_marker_labels)
        positions = _float_array(self.positions, name="recording.positions")
        marker_valid = _bool_array(self.marker_valid, name="recording.marker_valid")
        events = tuple(self.events)
        cameras = tuple(self.cameras)
        force_platform = self.force_platform
        xcp_path = _coerce_path(self.xcp_path)
        vsk_path = _coerce_path(self.vsk_path)

        if source_type not in {"csv", "c3d"}:
            raise ValueError(f"recording.source_type must be 'csv' or 'c3d', got {source_type!r}.")
        if fps <= 0:
            raise ValueError(f"recording.fps must be positive, got {fps}.")
        if positions.ndim != 3 or positions.shape[2] != 3:
            raise ValueError(
                "recording.positions must have shape (frames, markers, 3), "
                f"got {positions.shape}."
            )
        if marker_valid.shape != positions.shape[:2]:
            raise ValueError(
                "recording.marker_valid shape does not match positions axes: "
                f"{marker_valid.shape} vs {positions.shape[:2]}."
            )
        if len(marker_names) != positions.shape[1]:
            raise ValueError(
                "recording.marker_names length does not match positions.shape[1]: "
                f"{len(marker_names)} vs {positions.shape[1]}."
            )
        if len(source_marker_labels) != positions.shape[1]:
            raise ValueError(
                "recording.source_marker_labels length does not match positions.shape[1]: "
                f"{len(source_marker_labels)} vs {positions.shape[1]}."
            )
        if events and source_type != "c3d":
            raise ValueError("recording.events are only supported for 'c3d' source_type.")
        for event in events:
            if not isinstance(event, ViconEvent):
                raise TypeError(f"recording.events must contain ViconEvent items, got {event!r}.")
            if event.frame >= positions.shape[0]:
                raise ValueError(
                    "recording.event frame falls outside recording frame range: "
                    f"{event.frame} vs {positions.shape[0]} frames."
                )
            expected_source_frame = int(self.frame_offset) + int(event.frame)
            if event.source_frame != expected_source_frame:
                raise ValueError(
                    "recording.event source_frame must equal frame_offset + frame: "
                    f"{event.source_frame} vs {expected_source_frame}."
                )
        if (
            self.additional_points is not None
            and self.additional_points.n_frames != positions.shape[0]
        ):
            raise ValueError(
                "additional_points frame count does not match recording frame count: "
                f"{self.additional_points.n_frames} vs {positions.shape[0]}."
            )
        if self.analog is not None:
            expected_samples = positions.shape[0] * self.analog.samples_per_frame
            if self.analog.n_samples != expected_samples:
                raise ValueError(
                    "analog sample count does not match frame_count * samples_per_frame: "
                    f"{self.analog.n_samples} vs {expected_samples}."
                )
        if self.model is not None:
            observed_markers = {normalize_marker_name(marker_name) for marker_name in marker_names}
            missing_model_markers = tuple(
                marker_name
                for marker_name in self.model.marker_names
                if normalize_marker_name(marker_name) not in observed_markers
            )
            if missing_model_markers:
                raise ValueError(
                    "recording.model references markers missing from recording: "
                    f"{missing_model_markers}."
                )
        if force_platform is not None and not isinstance(
            force_platform,
            ViconForcePlatformMetadata,
        ):
            raise TypeError(
                "recording.force_platform must be ViconForcePlatformMetadata or None."
            )
        if force_platform is not None and source_type != "c3d":
            raise ValueError("recording.force_platform is only supported for 'c3d' source_type.")

        object.__setattr__(self, "path", path)
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "marker_names", marker_names)
        object.__setattr__(self, "source_marker_labels", source_marker_labels)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "marker_valid", marker_valid)
        object.__setattr__(self, "frame_offset", int(self.frame_offset))
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "cameras", cameras)
        object.__setattr__(self, "force_platform", force_platform)
        object.__setattr__(self, "xcp_path", xcp_path)
        object.__setattr__(self, "vsk_path", vsk_path)

    @property
    def n_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_markers(self) -> int:
        return int(self.positions.shape[1])

    @property
    def duration_s(self) -> float:
        return float(self.n_frames / self.fps)

    @property
    def missing_marker_samples(self) -> int:
        return int((~self.marker_valid).sum())

    @property
    def has_analog(self) -> bool:
        return self.analog is not None and self.analog.n_channels > 0

    @property
    def has_additional_points(self) -> bool:
        return self.additional_points is not None and self.additional_points.n_points > 0

    @property
    def has_events(self) -> bool:
        return bool(self.events)

    @property
    def has_cameras(self) -> bool:
        return bool(self.cameras)

    @property
    def gait_events(self) -> tuple[ViconEvent, ...]:
        return tuple(
            event
            for event in self.events
            if event.side is not None and event.event_type in {"foot_strike", "foot_off"}
        )

    def marker_index(self, name: str) -> int:
        return _lookup_unique_index(
            self.source_marker_labels,
            name,
            kind="Marker",
        )


__all__ = [
    "ViconAdditionalPointData",
    "ViconAnalogData",
    "ViconCamera",
    "ViconEvent",
    "ViconForcePlatformMetadata",
    "ViconMarkerModel",
    "ViconRecording",
]
