"""Prediction payload extraction and coercion for project state documents."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class PredictionLabelsView(Protocol):
    @property
    def videos(self) -> Sequence[object]: ...

    @property
    def labeled_frames(self) -> Sequence[Any]: ...


class PredictionAppendItem:
    """Linear prediction row used by project state-document serialization."""

    __slots__ = ("frame_index", "heatmaps", "instances", "video_index")

    def __init__(
        self,
        video_index: int,
        frame_index: int,
        instances: Sequence[Any],
        *,
        heatmaps: Any | None = None,
    ) -> None:
        self.video_index = int(video_index)
        self.frame_index = int(frame_index)
        self.instances = list(instances)
        self.heatmaps = heatmaps


def coerce_predictions_from_labels(labels: PredictionLabelsView) -> list[PredictionAppendItem]:
    """Extract predicted instances from a Labels-like object."""

    items: list[PredictionAppendItem] = []
    video_to_index = {video: idx for idx, video in enumerate(labels.videos)}

    for labeled_frame in labels.labeled_frames:
        predicted_instances = list(labeled_frame.predicted_instances)
        if not predicted_instances:
            continue
        if labeled_frame.video not in video_to_index:
            continue
        items.append(
            PredictionAppendItem(
                video_index=video_to_index[labeled_frame.video],
                frame_index=labeled_frame.frame_idx,
                instances=predicted_instances,
                heatmaps=labeled_frame.heatmaps,
            )
        )
    return items


__all__ = [
    "PredictionAppendItem",
    "PredictionLabelsView",
    "coerce_predictions_from_labels",
]
