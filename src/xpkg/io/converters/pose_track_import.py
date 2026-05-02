"""Shared helpers for converting low-level PoseTrack arrays into Labels."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from xpkg.io.converters.converter_helpers import points_from_coords_scores
from xpkg.io.readers._common import PoseTrack
from xpkg.pose.skeleton import build_keypoint_skeleton

if TYPE_CHECKING:
    from xpkg.model import Labels as _Labels
    from xpkg.pose.skeleton import Skeleton as _Skeleton


def build_pose_track_skeleton(
    node_names: Sequence[str],
    *,
    skeleton_name: str,
    skeleton_links: Sequence[tuple[int, int]] | None = None,
) -> _Skeleton:
    """Build a skeleton from ordered node names plus optional link indices."""

    skeleton = build_keypoint_skeleton(list(node_names), name=skeleton_name)
    if not skeleton_links:
        return skeleton

    node_count = len(node_names)
    normalized_links: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for raw_start, raw_end in skeleton_links:
        start = int(raw_start)
        end = int(raw_end)
        if not 0 <= start < node_count or not 0 <= end < node_count:
            raise ValueError(
                "Pose-track skeleton link indices must be within the node range; "
                f"got {(start, end)!r} for {node_count} nodes."
            )
        if start == end:
            raise ValueError("Pose-track skeleton links cannot self-reference.")
        ordered = (start, end) if start < end else (end, start)
        if ordered in seen:
            continue
        seen.add(ordered)
        normalized_links.append(ordered)

    skeleton.links_ids = normalized_links
    return skeleton


def validate_pose_tracks_consistency(
    tracks: Sequence[PoseTrack],
    *,
    source_label: str,
) -> tuple[int, list[str]]:
    """Validate that multiple PoseTrack inputs share frame and node axes."""

    if not tracks:
        raise ValueError(f"{source_label} contains no tracks.")

    reference = tracks[0]
    frame_count = int(reference.coords.shape[0])
    node_names = list(reference.node_names)

    for track in tracks[1:]:
        if list(track.node_names) != node_names:
            raise ValueError(
                f"{source_label} track node names do not agree across tracks."
            )
        if int(track.coords.shape[0]) != frame_count:
            raise ValueError(
                f"{source_label} track frame counts do not agree across tracks."
            )

    return frame_count, node_names


def _normalize_track_names(track_names: Sequence[str], *, track_count: int) -> list[str]:
    normalized = [
        name.strip() if isinstance(name, str) and name.strip() else f"track-{track_idx}"
        for track_idx, name in enumerate(track_names)
    ]
    if len(normalized) < track_count:
        normalized.extend(
            f"track-{track_idx}" for track_idx in range(len(normalized), track_count)
        )
    return normalized[:track_count]


def labels_from_pose_tracks(
    tracks: Sequence[PoseTrack],
    *,
    skeleton_name: str,
    video: Any,
    likelihood_threshold: float,
    track_names: Sequence[str] | None = None,
    skeleton_links: Sequence[tuple[int, int]] | None = None,
) -> _Labels:
    """Convert PoseTrack arrays into the canonical `xpkg.model.Labels` object."""

    from xpkg.model import Labels
    from xpkg.pose.annotations import Instance, LabeledFrame, Track

    frame_count, node_names = validate_pose_tracks_consistency(
        tracks,
        source_label="PoseTrack import",
    )
    skeleton = build_pose_track_skeleton(
        node_names,
        skeleton_name=skeleton_name,
        skeleton_links=skeleton_links,
    )
    labels = Labels(skeletons=[skeleton], videos=[video])

    track_defs: list[Track] | None = None
    if track_names is not None:
        normalized_track_names = _normalize_track_names(track_names, track_count=len(tracks))
        track_defs = [
            Track(spawned_on=track_idx, name=normalized_track_names[track_idx])
            for track_idx in range(len(tracks))
        ]

    for frame_idx in range(frame_count):
        instances: list[Instance] = []
        for track_idx, track in enumerate(tracks):
            points = points_from_coords_scores(
                node_names,
                track.coords[frame_idx],
                track.scores[frame_idx],
                likelihood_threshold=likelihood_threshold,
            )
            if not points:
                continue

            instance_kwargs: dict[str, Any] = {
                "skeleton": skeleton,
                "tracking_score": float(track.instance_score[frame_idx]),
                "init_points": points,
            }
            if track_defs is not None:
                instance_kwargs["track"] = track_defs[track_idx]
            instances.append(Instance(**instance_kwargs))

        if instances:
            labels.append(LabeledFrame(video=video, frame_idx=frame_idx, instances=instances))

    labels.update_cache()
    return labels


__all__ = [
    "build_pose_track_skeleton",
    "labels_from_pose_tracks",
    "validate_pose_tracks_consistency",
]
