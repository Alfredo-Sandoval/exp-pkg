"""Low-level readers for SLEAP analysis H5 tracking exports."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from xpkg.io.readers.pose._common import (
    PoseTrack,
    build_pose_track,
    resolve_node_indices_from_names,
)


def _decode_node_name(name: Any) -> str:
    if isinstance(name, bytes):
        return name.decode("utf-8")
    return str(name)


def read_node_names(path: Path) -> list[str]:
    """Return decoded node names from a SLEAP analysis H5."""
    with h5py.File(path, "r") as handle:
        names = np.asarray(handle["node_names"][...])
    return [_decode_node_name(name) for name in names]


def read_track_count(path: Path) -> int:
    """Return the number of tracked instances stored in a SLEAP analysis H5."""
    with h5py.File(path, "r") as handle:
        tracks = handle["tracks"]
        return int(tracks.shape[0])


def read_track_names(path: Path) -> list[str]:
    """Return decoded track names, reconciled to match ``read_track_count``.

    SLEAP writes an empty ``track_names`` array (an empty ``float64`` dataset,
    not bytes) for exports whose instances were never assigned named tracks,
    while ``tracks`` still carries a placeholder instance. The stored names are
    therefore shorter than the track count; synthesize ``track-{i}`` names for
    the unnamed tail so callers that zip names against tracks (e.g. the SLEAP
    project importer) stay aligned. Only decode ``track_names`` when it is a
    non-empty dataset, since the empty placeholder is a float array, not bytes.
    """
    with h5py.File(path, "r") as handle:
        track_count = int(handle["tracks"].shape[0])
        track_names_ds = handle.get("track_names")
        names: list[str] = []
        if isinstance(track_names_ds, h5py.Dataset) and track_names_ds.shape[0] > 0:
            names = [_decode_node_name(name) for name in np.asarray(track_names_ds[...])]
    if len(names) < track_count:
        names = [names[idx] if idx < len(names) else f"track-{idx}" for idx in range(track_count)]
    return names


def read_track(path: Path, *, track_index: int) -> PoseTrack:
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
                f"track_index={idx} out of range for point_scores with shape {point_scores.shape}."
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
            f"Expected tracks[idx] first dim=2 (x,y), got shape {coords_raw.shape} for {path}."
        )
    coords = np.stack([coords_raw[0].T, coords_raw[1].T], axis=-1)  # (frames, nodes, 2)
    scores = scores_raw.T  # (frames, nodes)

    return build_pose_track(
        coords=coords,
        scores=scores,
        node_names=node_names,
        instance_score=instance_score,
        source_label=f"SLEAP file {path}",
    )


def resolve_node_indices(path: Path, target_names: Sequence[str]) -> list[int]:
    """Map target node names to their indices in the H5 file."""
    return resolve_node_indices_from_names(read_node_names(path), target_names)


__all__ = [
    "PoseTrack",
    "read_track_count",
    "read_track_names",
    "read_node_names",
    "read_track",
    "resolve_node_indices",
]
