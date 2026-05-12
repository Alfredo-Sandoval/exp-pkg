"""Per-keypoint coordinate-and-score → :class:`PredictedPoint` conversion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xpkg.pose.annotations import PredictedPoint
    from xpkg.pose.skeleton import Keypoint


def points_from_coords_scores(
    node_names: Sequence[str | Keypoint],
    coords: np.ndarray,
    scores: np.ndarray,
    *,
    likelihood_threshold: float,
) -> dict[str | Keypoint, PredictedPoint]:
    """Build predicted point objects from parallel coordinate and score arrays.

    Per-keypoint confidence is preserved on the returned ``PredictedPoint``
    so downstream callers can construct ``PredictedInstance`` records that
    carry calibrated confidence end to end.
    """
    from xpkg.pose.annotations import PredictedPoint

    coords_array = np.asarray(coords, dtype=np.float64)
    scores_array = np.asarray(scores, dtype=np.float64)
    node_count = len(node_names)

    if coords_array.shape != (node_count, 2):
        raise ValueError(
            "coords must have shape "
            f"({node_count}, 2), got {coords_array.shape}."
        )
    if scores_array.shape != (node_count,):
        raise ValueError(
            "scores must have shape "
            f"({node_count},), got {scores_array.shape}."
        )

    points: dict[str | Keypoint, PredictedPoint] = {}
    for node_idx, node_name in enumerate(node_names):
        score = float(scores_array[node_idx])
        if not np.isfinite(score) or score < likelihood_threshold:
            continue
        x_val = float(coords_array[node_idx, 0])
        y_val = float(coords_array[node_idx, 1])
        if np.isnan(x_val) or np.isnan(y_val):
            continue
        points[node_name] = PredictedPoint(
            x=x_val, y=y_val, score=score, visible=True, complete=True
        )
    return points


__all__ = ["points_from_coords_scores"]
