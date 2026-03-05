"""Prediction coercion helpers for `.siesta` writer paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import h5py
import numpy as np

from posetta.io.labels.model import Labels as LabelsModel
from posetta.io.siesta_format.predictions_datasets import (
    PredictionAppendItem,
    _instance_keypoint_length,
    predicted_instance_types,
)
from posetta.io.siesta_format.shared import _skeleton_keypoint_count


def coerce_predictions_from_labels(labels: LabelsModel) -> list[PredictionAppendItem]:
    """Extract predictions from a Labels object into a linear list."""
    items: list[PredictionAppendItem] = []

    video_to_index = {v: i for i, v in enumerate(labels.videos)}

    for labeled_frame in labels.labeled_frames:
        predicted_instances = [
            inst
            for inst in labeled_frame.instances
            if isinstance(inst, predicted_instance_types())
        ]
        if not predicted_instances:
            continue

        if labeled_frame.video not in video_to_index:
            continue

        video_index = video_to_index[labeled_frame.video]

        items.append(
            PredictionAppendItem(
                video_index=video_index,
                frame_index=labeled_frame.frame_idx,
                instances=predicted_instances,
                heatmaps=labeled_frame.heatmaps,
            )
        )
    return items


def _prediction_lookup_map(
    video_index_lookup: Mapping[object, int] | Sequence[object] | None,
) -> dict[object, int]:
    if video_index_lookup is None:
        return {}
    if isinstance(video_index_lookup, Mapping):
        lookup_map: dict[object, int] = {}
        for key, value in video_index_lookup.items():
            if not isinstance(value, int | np.integer):
                raise TypeError("video_index_lookup values must be integers")
            lookup_map[key] = int(value)
        return lookup_map
    return {video: idx for idx, video in enumerate(video_index_lookup)}


def _coerce_prediction_items(
    predictions: Any,
    *,
    video_index_lookup: Mapping[object, int] | Sequence[object] | None,
) -> list[PredictionAppendItem]:
    lookup_map = _prediction_lookup_map(video_index_lookup)

    def _resolve_video_idx(candidate: Any) -> int:
        if isinstance(candidate, int | np.integer):
            return int(candidate)
        if candidate in lookup_map:
            return int(lookup_map[candidate])
        raise ValueError("Video object not found in provided video_index_lookup")

    def _resolve_frame_idx(candidate: Any) -> int:
        return int(candidate)

    def _coerce_from_sequence(seq: Sequence[Any]) -> list[PredictionAppendItem]:
        items: list[PredictionAppendItem] = []
        for entry in seq:
            if isinstance(entry, PredictionAppendItem):
                items.append(entry)
                continue
            if isinstance(entry, tuple) and len(entry) >= 3:
                video_idx = _resolve_video_idx(entry[0])
                frame_idx = _resolve_frame_idx(entry[1])
                instances_val = entry[2]
                heatmaps_val = entry[3] if len(entry) >= 4 else None
                items.append(
                    PredictionAppendItem(
                        video_index=video_idx,
                        frame_index=frame_idx,
                        instances=list(instances_val or []),
                        heatmaps=heatmaps_val,
                    )
                )
                continue
            raise TypeError(
                "Predictions must be PredictionAppendItem or (video, frame, instances[, heatmaps])"
            )
        return items

    def _coerce_from_mapping(pred_map: Mapping[Any, Any]) -> list[PredictionAppendItem]:
        items: list[PredictionAppendItem] = []
        for key, value in pred_map.items():
            if not isinstance(key, tuple) or len(key) < 2:
                raise TypeError("Prediction mapping keys must be (video_index, frame_index)")
            video_idx = _resolve_video_idx(key[0])
            frame_idx = _resolve_frame_idx(key[1])
            heatmaps_val = None
            instances_val = value
            if isinstance(value, tuple) and len(value) >= 2:
                instances_val = value[0]
                heatmaps_val = value[1]
            items.append(
                PredictionAppendItem(
                    video_index=video_idx,
                    frame_index=frame_idx,
                    instances=list(instances_val or []),
                    heatmaps=heatmaps_val,
                )
            )
        return items

    if predictions is None:
        return []
    if isinstance(predictions, Mapping):
        return _coerce_from_mapping(predictions)
    if isinstance(predictions, Sequence) and not isinstance(predictions, bytes | str):
        return _coerce_from_sequence(predictions)
    raise TypeError("Unsupported predictions data structure")


def _infer_prediction_keypoint_count(
    file: h5py.File,
    prediction_items: Sequence[PredictionAppendItem],
) -> int:
    keypoint_count = max(
        (
            _instance_keypoint_length(inst)
            for item in prediction_items
            for inst in item.instances or []
        ),
        default=0,
    )
    if keypoint_count == 0:
        keypoint_count = _skeleton_keypoint_count(file, default=0)
    return max(int(keypoint_count), 1)


__all__ = [
    "_coerce_prediction_items",
    "_infer_prediction_keypoint_count",
    "_prediction_lookup_map",
    "coerce_predictions_from_labels",
]
