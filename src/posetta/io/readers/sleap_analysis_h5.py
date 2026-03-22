"""Low-level readers for SLEAP analysis H5 tracking exports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _decode_node_name(name: Any) -> str:
    if isinstance(name, bytes):
        return name.decode("utf-8")
    return str(name)


@dataclass(frozen=True)
class SleapTrack:
    """Arrays for one track from a SLEAP analysis H5 file."""

    coords: np.ndarray
    scores: np.ndarray
    node_names: tuple[str, ...]
    instance_score: np.ndarray


def read_node_names(path: Path) -> list[str]:
    """Return decoded node names from a SLEAP analysis H5."""
    with h5py.File(path, "r") as handle:
        names = np.asarray(handle["node_names"][...])
    return [_decode_node_name(name) for name in names]


def read_track(path: Path, *, track_index: int) -> SleapTrack:
    """Read one track from a SLEAP analysis H5 export."""
    idx = int(track_index)
    if idx < 0:
        raise ValueError(f"track_index must be >= 0, got {track_index!r}.")

    path = Path(path)
    with h5py.File(path, "r") as handle:
        tracks = handle["tracks"]
        point_scores = handle["point_scores"]
        instance_scores = handle["instance_scores"]
        node_names = tuple(
            _decode_node_name(name) for name in np.asarray(handle["node_names"][...])
        )

        if tracks.shape[0] <= idx:
            raise IndexError(
                f"track_index={idx} out of range for tracks with shape {tracks.shape}."
            )
        if point_scores.shape[0] <= idx:
            raise IndexError(
                "track_index="
                f"{idx} out of range for point_scores with shape {point_scores.shape}."
            )
        if instance_scores.shape[0] <= idx:
            raise IndexError(
                "track_index="
                f"{idx} out of range for instance_scores with shape "
                f"{instance_scores.shape}."
            )

        # SLEAP layout: tracks[idx] -> (2, nodes, frames)
        coords_raw = np.asarray(tracks[idx], dtype=float)
        scores_raw = np.asarray(point_scores[idx], dtype=float)  # (nodes, frames)
        instance_score = np.asarray(instance_scores[idx], dtype=float)  # (frames,)

    if coords_raw.shape[0] != 2:
        raise ValueError(
            "Expected tracks[idx] first dim=2 (x,y), "
            f"got shape {coords_raw.shape} for {path}."
        )
    nodes = int(coords_raw.shape[1])
    frames = int(coords_raw.shape[2])
    if scores_raw.shape != (nodes, frames):
        raise ValueError(
            f"point_scores shape {scores_raw.shape} does not match "
            f"(nodes, frames)=({nodes}, {frames})."
        )
    if instance_score.shape != (frames,):
        raise ValueError(
            f"instance_scores shape {instance_score.shape} "
            f"does not match (frames,)={frames}."
        )
    if len(node_names) != nodes:
        raise ValueError(
            f"node_names length {len(node_names)} does not match nodes={nodes} for {path}."
        )

    coords = np.stack([coords_raw[0].T, coords_raw[1].T], axis=-1)  # (frames, nodes, 2)
    scores = scores_raw.T  # (frames, nodes)

    coords = np.asarray(coords, dtype=float)
    scores = np.asarray(scores, dtype=float)
    instance_score = np.asarray(instance_score, dtype=float)

    return SleapTrack(
        coords=coords,
        scores=scores,
        node_names=node_names,
        instance_score=instance_score,
    )


def resolve_node_indices(path: Path, target_names: Sequence[str]) -> list[int]:
    """Map target node names to their indices in the H5 file."""
    if not target_names:
        raise ValueError("resolve_node_indices requires non-empty target_names.")

    targets = [str(name) for name in target_names]
    names = read_node_names(path)
    index_by_name = {name: idx for idx, name in enumerate(names)}
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
    "SleapTrack",
    "read_node_names",
    "read_track",
    "resolve_node_indices",
]
