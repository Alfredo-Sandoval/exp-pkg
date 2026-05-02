"""Neutral pose trajectory adapter data.

``PoseTrajectory`` is a read-only downstream adapter for algorithms that only
need point trajectories. Workspace storage remains source-native, currently
``labels`` or ``vicon``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class PoseTrajectory:
    """Read-only pose/marker trajectory view shared across source formats."""

    fps: int
    keypoint_names: tuple[str, ...]
    positions: np.ndarray
    valid: np.ndarray
    dims: Literal[2, 3]
    frame_offset: int = 0
    skeleton_edges: tuple[tuple[str, str], ...] = ()
    source_kind: str = ""
    source_path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        fps = int(self.fps)
        keypoint_names = tuple(str(name) for name in self.keypoint_names)
        positions = np.asarray(self.positions, dtype=np.float64)
        valid = np.asarray(self.valid, dtype=bool)
        dims = int(self.dims)
        frame_offset = int(self.frame_offset)
        skeleton_edges = tuple(
            (str(parent), str(child)) for parent, child in self.skeleton_edges
        )
        source_kind = str(self.source_kind)
        source_path = None if self.source_path is None else Path(self.source_path)
        metadata = dict(self.metadata)

        if fps <= 0:
            raise ValueError(f"pose trajectory fps must be positive, got {fps}.")
        if dims not in {2, 3}:
            raise ValueError(f"pose trajectory dims must be 2 or 3, got {dims}.")
        if positions.ndim != 3 or positions.shape[2] != dims:
            raise ValueError(
                "pose trajectory positions must have shape (frames, keypoints, dims), "
                f"got {positions.shape} for dims={dims}."
            )
        if valid.shape != positions.shape[:2]:
            raise ValueError(
                "pose trajectory valid shape must match positions frame/keypoint axes: "
                f"{valid.shape} vs {positions.shape[:2]}."
            )
        if len(keypoint_names) != positions.shape[1]:
            raise ValueError(
                "pose trajectory keypoint_names length does not match positions axis 1: "
                f"{len(keypoint_names)} vs {positions.shape[1]}."
            )

        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "keypoint_names", keypoint_names)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "dims", dims)
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
        return int(self.positions.shape[1])


def pose_trajectory_from_vicon_recording(recording: Any) -> PoseTrajectory:
    """Adapt a native ``ViconRecording`` into a neutral 3D trajectory view."""
    skeleton_edges: Sequence[tuple[str, str]] = ()
    if recording.model is not None:
        skeleton_edges = recording.model.edges
    return PoseTrajectory(
        fps=recording.fps,
        keypoint_names=recording.marker_names,
        positions=recording.positions,
        valid=recording.marker_valid,
        dims=3,
        frame_offset=recording.frame_offset,
        skeleton_edges=tuple(skeleton_edges),
        source_kind="vicon",
        source_path=recording.path,
        metadata={"source_type": recording.source_type},
    )


__all__ = ["PoseTrajectory", "pose_trajectory_from_vicon_recording"]
