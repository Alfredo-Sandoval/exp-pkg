"""Low-level readers for SLEAP analysis H5 tracking exports."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from xpkg.io.hdf5 import require_dataset
from xpkg.io.readers.pose._common import (
    PoseTrack,
    build_pose_track,
    resolve_node_indices_from_names,
)


def _decode_node_name(name: Any) -> str:
    if isinstance(name, bytes):
        return name.decode("utf-8")
    return str(name)


def _sleap_track_layout(tracks: h5py.Dataset, node_count: int) -> str:
    if tracks.ndim != 4:
        raise ValueError(f"SLEAP tracks must be four-dimensional, got shape {tracks.shape}.")
    legacy = tracks.shape[1] == 2 and tracks.shape[2] == node_count
    standard = tracks.shape[1] == node_count and tracks.shape[2] == 2
    if legacy and standard:
        raise ValueError(
            "SLEAP tracks layout is ambiguous because both coordinate and node dimensions "
            f"have length 2: {tracks.shape}."
        )
    if legacy:
        return "track_xy_node_frame"
    if standard:
        return "frame_node_xy_track"
    raise ValueError(
        "Unsupported SLEAP tracks layout. Expected (tracks, 2, nodes, frames) or "
        f"(frames, nodes, 2, tracks), got {tracks.shape} with {node_count} nodes."
    )


def _sleap_track_count(tracks: h5py.Dataset, node_count: int) -> int:
    layout = _sleap_track_layout(tracks, node_count)
    return int(tracks.shape[0] if layout == "track_xy_node_frame" else tracks.shape[3])


def read_node_names(path: Path) -> list[str]:
    """Return decoded node names from a SLEAP analysis H5."""
    with h5py.File(str(path), "r") as handle:
        names = np.asarray(require_dataset(handle, "node_names")[...])
    return [_decode_node_name(name) for name in names]


def read_track_count(path: Path) -> int:
    """Return the number of tracked instances stored in a SLEAP analysis H5."""
    with h5py.File(str(path), "r") as handle:
        tracks = require_dataset(handle, "tracks")
        node_count = int(require_dataset(handle, "node_names").shape[0])
        return _sleap_track_count(tracks, node_count)


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
    with h5py.File(str(path), "r") as handle:
        tracks = require_dataset(handle, "tracks")
        node_count = int(require_dataset(handle, "node_names").shape[0])
        track_count = _sleap_track_count(tracks, node_count)
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
    with h5py.File(str(path), "r") as handle:
        tracks = require_dataset(handle, "tracks")
        point_scores = require_dataset(handle, "point_scores")
        instance_scores = require_dataset(handle, "instance_scores")
        node_names = tuple(
            _decode_node_name(name)
            for name in np.asarray(require_dataset(handle, "node_names")[...])
        )
        layout = _sleap_track_layout(tracks, len(node_names))
        track_count = _sleap_track_count(tracks, len(node_names))

        if track_count <= idx:
            raise IndexError(
                f"track_index={idx} out of range for {track_count} tracks with shape "
                f"{tracks.shape}."
            )
        if layout == "track_xy_node_frame":
            if point_scores.shape[0] <= idx or instance_scores.shape[0] <= idx:
                raise IndexError(f"SLEAP score datasets do not contain track_index={idx}.")
            coords_raw = np.asarray(tracks[idx], dtype=float)
            scores_raw = np.asarray(point_scores[idx], dtype=float)
            instance_score = np.asarray(instance_scores[idx], dtype=float)
            coords = np.stack([coords_raw[0].T, coords_raw[1].T], axis=-1)
            scores = scores_raw.T
        else:
            if point_scores.shape[-1] <= idx or instance_scores.shape[-1] <= idx:
                raise IndexError(f"SLEAP score datasets do not contain track_index={idx}.")
            coords = np.asarray(tracks[:, :, :, idx], dtype=float)
            scores = np.asarray(point_scores[:, :, idx], dtype=float)
            instance_score = np.asarray(instance_scores[:, idx], dtype=float)

    return build_pose_track(
        coords=coords,
        scores=scores,
        node_names=node_names,
        instance_score=instance_score,
        source_label=f"SLEAP file {path}",
        metadata={
            "source": {"type": "sleap_analysis_h5", "path": str(path)},
            "software": "SLEAP",
            "file_type": "h5",
            "track_index": idx,
            "tracks_layout": layout,
        },
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
