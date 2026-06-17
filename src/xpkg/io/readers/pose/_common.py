"""Shared contracts for low-level external pose readers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PoseTrack:
    """Arrays for one pose track from an external tracking export."""

    coords: np.ndarray
    scores: np.ndarray
    node_names: tuple[str, ...]
    instance_score: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def build_pose_track(
    *,
    coords: np.ndarray,
    scores: np.ndarray,
    node_names: Sequence[str],
    instance_score: np.ndarray,
    source_label: str,
    metadata: Mapping[str, Any] | None = None,
) -> PoseTrack:
    """Coerce arrays into the stable PoseTrack contract."""

    coords_array = np.asarray(coords, dtype=np.float64)
    scores_array = np.asarray(scores, dtype=np.float64)
    instance_score_array = np.asarray(instance_score, dtype=np.float64)
    node_name_tuple = tuple(str(name) for name in node_names)

    if coords_array.ndim != 3 or coords_array.shape[2] != 2:
        raise ValueError(
            f"{source_label} coords must have shape (frames, nodes, 2), got {coords_array.shape}."
        )

    frames = int(coords_array.shape[0])
    nodes = int(coords_array.shape[1])
    if scores_array.shape != (frames, nodes):
        raise ValueError(
            f"{source_label} scores shape {scores_array.shape} does not match "
            f"(frames, nodes)=({frames}, {nodes})."
        )
    if instance_score_array.shape != (frames,):
        raise ValueError(
            f"{source_label} instance_score shape {instance_score_array.shape} does not "
            f"match (frames,)={frames}."
        )
    if len(node_name_tuple) != nodes:
        raise ValueError(
            f"{source_label} node_names length {len(node_name_tuple)} does not match nodes={nodes}."
        )
    return PoseTrack(
        coords=coords_array,
        scores=scores_array,
        node_names=node_name_tuple,
        instance_score=instance_score_array,
        metadata={} if metadata is None else metadata,
    )


def with_pose_track_metadata(
    track: PoseTrack,
    metadata: Mapping[str, Any],
) -> PoseTrack:
    """Return ``track`` with metadata merged over its existing metadata."""

    merged = dict(track.metadata)
    merged.update(dict(metadata))
    return PoseTrack(
        coords=track.coords,
        scores=track.scores,
        node_names=track.node_names,
        instance_score=track.instance_score,
        metadata=merged,
    )


def resolve_node_indices_from_names(
    node_names: Sequence[str],
    target_names: Sequence[str],
) -> list[int]:
    """Map requested node names to unique indices in a node-name sequence."""

    if not target_names:
        raise ValueError("resolve_node_indices requires non-empty target_names.")

    targets = [str(name) for name in target_names]
    index_by_name = {str(name): idx for idx, name in enumerate(node_names)}
    missing = [name for name in targets if name not in index_by_name]
    if missing:
        raise KeyError(f"Target node names not present in file: {missing}")

    indices: list[int] = []
    seen: set[int] = set()
    for name in targets:
        idx = index_by_name[name]
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
    return indices


__all__ = [
    "PoseTrack",
    "build_pose_track",
    "resolve_node_indices_from_names",
    "with_pose_track_metadata",
]
