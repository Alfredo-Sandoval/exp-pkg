from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from xpkg.model import (
    PoseTrajectory,
    ViconMarkerModel,
    ViconRecording,
    pose_trajectory_from_vicon_recording,
)


def test_pose_trajectory_validates_shapes() -> None:
    with pytest.raises(ValueError, match="positions"):
        PoseTrajectory(
            fps=100,
            keypoint_names=("hip",),
            positions=np.zeros((2, 1, 2), dtype=np.float64),
            valid=np.ones((2, 1), dtype=bool),
            dims=3,
        )

    with pytest.raises(ValueError, match="valid shape"):
        PoseTrajectory(
            fps=100,
            keypoint_names=("hip",),
            positions=np.zeros((2, 1, 3), dtype=np.float64),
            valid=np.ones((2, 2), dtype=bool),
            dims=3,
        )

    with pytest.raises(ValueError, match="keypoint_names"):
        PoseTrajectory(
            fps=100,
            keypoint_names=("hip", "knee"),
            positions=np.zeros((2, 1, 3), dtype=np.float64),
            valid=np.ones((2, 1), dtype=bool),
            dims=3,
        )

    invalid_dims_kwargs: dict[str, Any] = {
        "fps": 100,
        "keypoint_names": ("hip",),
        "positions": np.zeros((2, 1, 4), dtype=np.float64),
        "valid": np.ones((2, 1), dtype=bool),
        "dims": 4,
    }
    with pytest.raises(ValueError, match="dims"):
        PoseTrajectory(**invalid_dims_kwargs)


def test_pose_trajectory_from_vicon_recording_maps_native_fields() -> None:
    positions = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    valid = np.array([[True, True], [True, False]])
    recording = ViconRecording(
        path=Path("trial.c3d"),
        source_type="c3d",
        fps=120,
        marker_names=("L_Hip", "L_Knee"),
        source_marker_labels=("Mouse1:L_Hip", "Mouse1:L_Knee"),
        positions=positions,
        marker_valid=valid,
        frame_offset=10,
        model=ViconMarkerModel(
            name="mouse",
            display_name="Mouse",
            marker_names=("L_Hip", "L_Knee"),
            edges=(("L_Hip", "L_Knee"),),
            source="vsk",
        ),
    )

    trajectory = pose_trajectory_from_vicon_recording(recording)

    assert trajectory.fps == 120
    assert trajectory.keypoint_names == ("L_Hip", "L_Knee")
    assert trajectory.positions is positions
    assert trajectory.valid is valid
    assert trajectory.dims == 3
    assert trajectory.frame_offset == 10
    assert trajectory.skeleton_edges == (("L_Hip", "L_Knee"),)
    assert trajectory.source_kind == "vicon"
    assert trajectory.source_path == Path("trial.c3d")
    assert trajectory.metadata == {"source_type": "c3d"}
