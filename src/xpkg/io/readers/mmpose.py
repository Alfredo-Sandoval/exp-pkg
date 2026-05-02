"""Low-level readers for official MMPose demo saved-predictions JSON exports."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from xpkg._core.json_utils import load_json_dict
from xpkg.io.readers._common import (
    PoseTrack,
    build_pose_track,
    resolve_node_indices_from_names,
)


@dataclass(frozen=True)
class _MMPoseSequence:
    dataset_name: str
    node_names: tuple[str, ...]
    skeleton_links: tuple[tuple[int, int], ...]
    frame_instances: tuple[tuple[dict[str, Any], ...], ...]


def _require_mapping(value: object, *, field_name: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"MMPose {field_name} must be a JSON object in {path}.")
    return {str(key): item for key, item in value.items()}


def _require_list(value: object, *, field_name: str, path: Path) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"MMPose {field_name} must be a JSON array in {path}.")
    return list(value)


def _load_sequence(path: Path) -> _MMPoseSequence:
    payload = load_json_dict(path)
    meta_info = _require_mapping(payload.get("meta_info"), field_name="meta_info", path=path)
    instance_info = _require_list(
        payload.get("instance_info"),
        field_name="instance_info",
        path=path,
    )

    dataset_name = str(meta_info.get("dataset_name", "")).strip()
    if not dataset_name:
        raise ValueError(f"MMPose meta_info.dataset_name is missing in {path}.")
    node_names = tuple(_parse_node_names(meta_info, path=path))
    skeleton_links = tuple(_parse_skeleton_links(meta_info, path=path, node_count=len(node_names)))
    frame_instances = tuple(_parse_frame_instances(instance_info, path=path))
    return _MMPoseSequence(
        dataset_name=dataset_name,
        node_names=node_names,
        skeleton_links=skeleton_links,
        frame_instances=frame_instances,
    )


def _parse_node_names(meta_info: dict[str, Any], *, path: Path) -> list[str]:
    raw_names = _require_mapping(
        meta_info.get("keypoint_id2name"),
        field_name="meta_info.keypoint_id2name",
        path=path,
    )
    if not raw_names:
        raise ValueError(f"MMPose meta_info.keypoint_id2name is empty in {path}.")

    indexed_names: list[tuple[int, str]] = []
    for raw_idx, raw_name in raw_names.items():
        try:
            node_idx = int(raw_idx)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"MMPose keypoint_id2name key {raw_idx!r} is not int-like in {path}."
            ) from exc
        node_name = str(raw_name).strip()
        if not node_name:
            raise ValueError(f"MMPose keypoint_id2name[{raw_idx!r}] is empty in {path}.")
        indexed_names.append((node_idx, node_name))

    indexed_names.sort(key=lambda item: item[0])
    expected_indices = list(range(len(indexed_names)))
    actual_indices = [node_idx for node_idx, _node_name in indexed_names]
    if actual_indices != expected_indices:
        raise ValueError(
            "MMPose keypoint_id2name indices must be contiguous and zero-based in "
            f"{path}; found {actual_indices!r}."
        )
    return [node_name for _node_idx, node_name in indexed_names]


def _parse_skeleton_links(
    meta_info: dict[str, Any],
    *,
    path: Path,
    node_count: int,
) -> list[tuple[int, int]]:
    raw_links = _require_list(
        meta_info.get("skeleton_links"),
        field_name="meta_info.skeleton_links",
        path=path,
    )
    normalized_links: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for link in raw_links:
        if not isinstance(link, list | tuple) or len(link) != 2:
            raise ValueError(
                "Each MMPose skeleton link must be a two-element array in "
                f"{path}; got {link!r}."
            )
        start = int(link[0])
        end = int(link[1])
        if not 0 <= start < node_count or not 0 <= end < node_count:
            raise ValueError(
                "MMPose skeleton link indices must be within the node range in "
                f"{path}; got {(start, end)!r} for {node_count} nodes."
            )
        if start == end:
            raise ValueError(f"MMPose skeleton links cannot self-reference in {path}.")
        ordered = (start, end) if start < end else (end, start)
        if ordered in seen:
            continue
        seen.add(ordered)
        normalized_links.append(ordered)
    return normalized_links


def _parse_frame_instances(
    instance_info: list[Any],
    *,
    path: Path,
) -> list[tuple[dict[str, Any], ...]]:
    if not instance_info:
        raise ValueError(f"MMPose instance_info contains no frames in {path}.")

    frame_instances_by_index: dict[int, tuple[dict[str, Any], ...]] = {}
    for item in instance_info:
        if not isinstance(item, dict):
            raise TypeError(f"Each MMPose instance_info entry must be an object in {path}.")
        if "frame_id" not in item or "instances" not in item:
            raise ValueError(
                "Only video-style MMPose demo JSON is supported. Expected each "
                f"instance_info entry in {path} to contain frame_id and instances."
            )

        frame_id = int(item["frame_id"])
        if frame_id <= 0:
            raise ValueError(f"MMPose frame_id must be >= 1 in {path}, got {frame_id}.")
        frame_index = frame_id - 1
        if frame_index in frame_instances_by_index:
            raise ValueError(f"Duplicate MMPose frame_id={frame_id} found in {path}.")

        raw_instances = _require_list(
            item["instances"],
            field_name=f"instance_info[{frame_id}].instances",
            path=path,
        )
        parsed_instances: list[dict[str, Any]] = []
        for instance in raw_instances:
            if not isinstance(instance, dict):
                raise TypeError(
                    f"Each MMPose frame instance must be an object in {path}; got {instance!r}."
                )
            parsed_instances.append(instance)
        frame_instances_by_index[frame_index] = tuple(parsed_instances)

    frame_count = max(frame_instances_by_index) + 1
    return [frame_instances_by_index.get(frame_idx, tuple()) for frame_idx in range(frame_count)]


def _max_instance_count(sequence: _MMPoseSequence) -> int:
    return max(len(instances) for instances in sequence.frame_instances)


def _coerce_instance_array(
    value: object,
    *,
    expected_shape: tuple[int, ...],
    field_name: str,
    path: Path,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != expected_shape:
        raise ValueError(
            f"MMPose {field_name} in {path} must have shape {expected_shape}, got {array.shape}."
        )
    return array


def _read_track_from_sequence(
    sequence: _MMPoseSequence,
    *,
    path: Path,
    track_index: int,
) -> PoseTrack:
    idx = int(track_index)
    if idx < 0:
        raise ValueError(f"track_index must be >= 0, got {track_index!r}.")

    max_instances = _max_instance_count(sequence)
    if max_instances <= idx:
        raise IndexError(
            f"track_index={idx} out of range for MMPose predictions with "
            f"max_instances={max_instances} in {path}."
        )

    frame_count = len(sequence.frame_instances)
    node_count = len(sequence.node_names)
    coords = np.full((frame_count, node_count, 2), np.nan, dtype=np.float64)
    scores = np.full((frame_count, node_count), np.nan, dtype=np.float64)

    for frame_idx, instances in enumerate(sequence.frame_instances):
        if len(instances) <= idx:
            continue
        instance = instances[idx]
        coords[frame_idx] = _coerce_instance_array(
            instance.get("keypoints"),
            expected_shape=(node_count, 2),
            field_name=f"instance_info[{frame_idx + 1}].instances[{idx}].keypoints",
            path=path,
        )
        scores[frame_idx] = _coerce_instance_array(
            instance.get("keypoint_scores"),
            expected_shape=(node_count,),
            field_name=f"instance_info[{frame_idx + 1}].instances[{idx}].keypoint_scores",
            path=path,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        instance_score = np.nanmean(scores, axis=1)

    return build_pose_track(
        coords=coords,
        scores=scores,
        node_names=sequence.node_names,
        instance_score=instance_score,
        source_label=f"MMPose file {path}",
    )


def read_sequence_dataset_name(path: Path) -> str:
    """Return the dataset name stored in an official MMPose demo JSON export."""

    return _load_sequence(path).dataset_name


def read_skeleton_links(path: Path) -> list[tuple[int, int]]:
    """Return skeleton links stored in an official MMPose demo JSON export."""

    return list(_load_sequence(path).skeleton_links)


def read_node_names(path: Path) -> list[str]:
    """Return node names from an official MMPose demo JSON export."""

    return list(_load_sequence(path).node_names)


def read_track(path: Path, *, track_index: int) -> PoseTrack:
    """Read one instance-indexed pose track from an official MMPose demo JSON export."""

    resolved_path = Path(path)
    sequence = _load_sequence(resolved_path)
    return _read_track_from_sequence(sequence, path=resolved_path, track_index=track_index)


def resolve_node_indices(path: Path, target_names: Sequence[str]) -> list[int]:
    """Map target node names to their indices in an MMPose demo JSON export."""

    return resolve_node_indices_from_names(read_node_names(path), target_names)


__all__ = [
    "PoseTrack",
    "read_node_names",
    "read_skeleton_links",
    "read_sequence_dataset_name",
    "read_track",
    "resolve_node_indices",
]
