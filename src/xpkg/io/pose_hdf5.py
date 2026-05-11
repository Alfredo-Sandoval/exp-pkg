# pyright: reportMissingImports=false
# Justification: h5py lacks bundled type stubs in this environment.

"""HDF5 export helpers for xpkg pose labels."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

from xpkg._core.logging_utils import get_logger
from xpkg.pose.annotations.instances import Instance, PredictedInstance
from xpkg.pose.annotations.points import PredictedPoint

if TYPE_CHECKING:
    from xpkg.io.labels.model import Labels

logger = get_logger(__name__)


def video_index_map(labels: Labels) -> dict[object, int]:
    """Map each labels.video object to its stable index in labels.videos."""
    return {video: idx for idx, video in enumerate(labels.videos)}


def track_id(inst: Instance, *, missing: int) -> int:
    """Return the integer track id for an instance, or `missing` when absent."""
    track = inst.track
    if track is None:
        return missing
    return int(track.id)


def ensure_matching_skeleton(
    actual_names: Sequence[str], expected_names: Sequence[str], *, label: str
) -> None:
    """Raise when an instance skeleton layout does not match the export layout."""
    if actual_names != expected_names:
        raise ValueError(label)


def export_pose_h5(
    labels: Labels,
    output_path: str | Path,
    *,
    include_predictions: bool = True,
    include_confidence: bool = True,
    video_index: int | None = None,
    transpose_for_column_major: bool = True,
) -> Path:
    """Export xpkg pose labels to an HDF5 analysis matrix."""
    labels.validate()
    output_path = Path(output_path)

    if not labels.skeletons:
        raise ValueError("No skeleton found in labels; cannot export HDF5.")

    skeleton = labels.skeletons[0]
    node_names = skeleton.keypoint_names
    n_nodes = len(node_names)
    video_indices = video_index_map(labels)

    labeled_frames = list(labels.labeled_frames)
    if video_index is not None:
        labeled_frames = [
            lf for lf in labeled_frames if int(video_indices[lf.video]) == int(video_index)
        ]
    if not labeled_frames:
        if video_index is None:
            raise ValueError("No labeled frames found in labels; cannot export HDF5.")
        raise ValueError(f"No labeled frames found for video_index={int(video_index)}")

    tracks: dict[int, str] = {}
    for lf in labeled_frames:
        for inst in lf.instances:
            if inst.track:
                tid = track_id(inst, missing=0)
                if tid not in tracks:
                    tracks[tid] = inst.track.name

    if not tracks:
        tracks = {0: "default"}

    track_ids = sorted(tracks.keys())
    track_names = [tracks[tid] for tid in track_ids]
    n_tracks = len(track_ids)

    all_frames = [lf.frame_idx for lf in labeled_frames]
    max_frame = max(all_frames) + 1

    pose_data = np.full((max_frame, n_nodes, 2, n_tracks), np.nan, dtype=np.float32)
    occupancy = np.zeros((max_frame, n_tracks), dtype=bool)
    confidence_data = np.zeros((max_frame, n_nodes, n_tracks), dtype=np.float32)
    frame_indices = []

    for lf in labeled_frames:
        frame_idx = lf.frame_idx
        frame_indices.append(frame_idx)

        for inst in lf.instances:
            if not include_predictions and isinstance(inst, PredictedInstance):
                continue
            ensure_matching_skeleton(
                inst.skeleton.keypoint_names,
                node_names,
                label="Mixed skeleton layouts detected during HDF5 export.",
            )

            tid = track_id(inst, missing=0)
            if tid not in track_ids:
                continue
            track_idx = track_ids.index(tid)

            occupancy[frame_idx, track_idx] = True

            for node_idx, kp in enumerate(skeleton.keypoints):
                pt = inst[kp]
                if pt.x is not None and pt.y is not None:
                    pose_data[frame_idx, node_idx, 0, track_idx] = float(pt.x)
                    pose_data[frame_idx, node_idx, 1, track_idx] = float(pt.y)

                    if include_confidence:
                        confidence = pt.score if isinstance(pt, PredictedPoint) else 1.0
                        confidence_data[frame_idx, node_idx, track_idx] = float(confidence)

    if transpose_for_column_major:
        pose_data = np.asfortranarray(pose_data)
        occupancy = np.asfortranarray(occupancy)
        confidence_data = np.asfortranarray(confidence_data)

    with h5py.File(str(output_path), "w") as f:
        f.attrs["format"] = "xpkg_pose_analysis_v1"
        f.attrs["n_frames"] = max_frame
        f.attrs["n_nodes"] = n_nodes
        f.attrs["n_tracks"] = n_tracks

        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("node_names", data=node_names, dtype=dt)
        f.create_dataset("track_names", data=track_names, dtype=dt)

        f.create_dataset(
            "tracks",
            data=pose_data,
            compression="gzip",
            compression_opts=1,
        )

        f.create_dataset(
            "track_occupancy",
            data=occupancy,
            compression="gzip",
            compression_opts=1,
        )

        if include_confidence:
            f.create_dataset(
                "confidence",
                data=confidence_data,
                compression="gzip",
                compression_opts=1,
            )

        f.create_dataset("frame_indices", data=np.array(frame_indices, dtype=np.int32))

    logger.info(
        "Exported HDF5 analysis file to %s (%d frames, %d nodes, %d tracks)",
        output_path,
        max_frame,
        n_nodes,
        n_tracks,
    )
    return output_path


__all__ = [
    "ensure_matching_skeleton",
    "export_pose_h5",
    "track_id",
    "video_index_map",
]
