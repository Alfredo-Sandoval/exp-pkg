from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from xpkg.model import PoseTrajectory


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

