"""Canonical 2D and 3D pose trajectories with explicit spatial semantics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np


class CoordinateFrameKind(StrEnum):
    """Spatial frame in which pose coordinates are expressed."""

    IMAGE_PIXEL = "image_pixel"
    CAMERA = "camera"
    CALIBRATION_WORLD = "calibration_world"
    MARKER_WORLD = "marker_world"
    LIFTED_MODEL = "lifted_model"


@dataclass(frozen=True, slots=True)
class PoseCoordinateFrame:
    """Coordinate frame and units shared by every point in a trajectory."""

    kind: CoordinateFrameKind
    units: str
    name: str | None = None
    axis_convention: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CoordinateFrameKind):
            raise TypeError("pose coordinate-frame kind must be a CoordinateFrameKind.")
        object.__setattr__(self, "units", _required_text(self.units, name="pose units"))
        for field_name in ("name", "axis_convention", "description"):
            value = getattr(self, field_name)
            object.__setattr__(
                self,
                field_name,
                None if value is None else _required_text(value, name=f"pose {field_name}"),
            )


@dataclass(frozen=True, slots=True)
class PoseTrack:
    """One stable technical identity on a pose trajectory track axis."""

    track_id: str
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _required_text(self.track_id, name="pose track id"))
        if self.name is not None:
            object.__setattr__(self, "name", _required_text(self.name, name="pose track name"))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class PoseTrajectory:
    """Source-neutral multi-track 2D or 3D named-keypoint trajectory."""

    fps: float
    tracks: tuple[PoseTrack, ...]
    keypoint_names: tuple[str, ...]
    positions: np.ndarray
    valid: np.ndarray
    dims: Literal[2, 3]
    coordinate_frame: PoseCoordinateFrame
    confidence: np.ndarray | None = None
    frame_offset: int = 0
    skeleton_edges: tuple[tuple[str, str], ...] = ()
    source_kind: str = ""
    source_path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        fps = float(self.fps)
        tracks = tuple(self.tracks)
        if any(not isinstance(track, PoseTrack) for track in tracks):
            raise TypeError("pose trajectory tracks must contain PoseTrack objects.")
        keypoint_names = tuple(
            _required_text(name, name="pose keypoint name") for name in self.keypoint_names
        )
        positions = np.asarray(self.positions, dtype=np.float64)
        valid = np.asarray(self.valid, dtype=bool)
        dims = int(self.dims)
        coordinate_frame = self.coordinate_frame
        confidence = (
            None
            if self.confidence is None
            else np.asarray(self.confidence, dtype=np.float64)
        )
        frame_offset = int(self.frame_offset)
        skeleton_edges = tuple(
            (str(parent), str(child)) for parent, child in self.skeleton_edges
        )
        source_kind = str(self.source_kind).strip()
        source_path = None if self.source_path is None else Path(self.source_path)
        metadata = MappingProxyType(dict(self.metadata))

        _validate_trajectory(
            fps=fps,
            track_ids=tuple(track.track_id for track in tracks),
            keypoint_names=keypoint_names,
            positions=positions,
            valid=valid,
            dims=dims,
            coordinate_frame=coordinate_frame,
            confidence=confidence,
        )

        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "tracks", tracks)
        object.__setattr__(self, "keypoint_names", keypoint_names)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "coordinate_frame", coordinate_frame)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "frame_offset", frame_offset)
        object.__setattr__(self, "skeleton_edges", skeleton_edges)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "metadata", metadata)

    @property
    def n_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_keypoints(self) -> int:
        return int(self.positions.shape[2])

    @property
    def n_tracks(self) -> int:
        return int(self.positions.shape[1])

    @property
    def track_ids(self) -> tuple[str, ...]:
        """Return stable track identifiers in array-axis order."""
        return tuple(track.track_id for track in self.tracks)

    def track(self, track_id: str) -> PoseTrack:
        """Return one trajectory track by stable identifier."""
        key = _required_text(track_id, name="pose track id")
        for track in self.tracks:
            if track.track_id == key:
                return track
        raise KeyError(f"Pose trajectory has no track {key!r}.")


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _validate_trajectory(
    *,
    fps: float,
    track_ids: tuple[str, ...],
    keypoint_names: tuple[str, ...],
    positions: np.ndarray,
    valid: np.ndarray,
    dims: int,
    coordinate_frame: object,
    confidence: np.ndarray | None,
) -> None:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"pose trajectory fps must be positive, got {fps}.")
    if dims not in {2, 3}:
        raise ValueError(f"pose trajectory dims must be 2 or 3, got {dims}.")
    if positions.ndim != 4 or positions.shape[3] != dims:
        raise ValueError(
            "pose trajectory positions must have shape "
            f"(frames, tracks, keypoints, dims), got {positions.shape} for dims={dims}."
        )
    if valid.shape != positions.shape[:3]:
        raise ValueError(
            "pose trajectory valid shape must match positions frame/track/keypoint axes: "
            f"{valid.shape} vs {positions.shape[:3]}."
        )
    if not track_ids or len(track_ids) != positions.shape[1]:
        raise ValueError("pose trajectory track_ids must name positions axis 1.")
    if len(set(track_ids)) != len(track_ids):
        raise ValueError("pose trajectory track_ids must be unique.")
    if len(keypoint_names) != positions.shape[2]:
        raise ValueError("pose trajectory keypoint_names must name positions axis 2.")
    if len(set(keypoint_names)) != len(keypoint_names):
        raise ValueError("pose trajectory keypoint_names must be unique.")
    if not isinstance(coordinate_frame, PoseCoordinateFrame):
        raise TypeError("pose trajectory coordinate_frame must be a PoseCoordinateFrame.")
    if dims == 2 and coordinate_frame.kind is not CoordinateFrameKind.IMAGE_PIXEL:
        raise ValueError("2D pose trajectories must use the image_pixel coordinate frame.")
    if dims == 3 and coordinate_frame.kind is CoordinateFrameKind.IMAGE_PIXEL:
        raise ValueError("3D pose trajectories cannot use the image_pixel coordinate frame.")
    if confidence is not None and confidence.shape != valid.shape:
        raise ValueError("pose trajectory confidence shape must match valid shape.")
    if confidence is not None and not np.all(np.isfinite(confidence)):
        raise ValueError("pose trajectory confidence values must be finite.")


__all__ = ["CoordinateFrameKind", "PoseCoordinateFrame", "PoseTrack", "PoseTrajectory"]
