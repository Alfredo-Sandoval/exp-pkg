"""Lightweight track utilities that operate on `Labels` containers."""

from __future__ import annotations

from typing import Any as _Any

from posetta.core.annotations import Instance, LabeledFrame, Track
from posetta.io.labels.video_types import VideoProtocol


def get_track_occupancy(labels: _Any, video: VideoProtocol) -> dict[Track | None, set[int]]:
    """Return per-track frame indices for `video`."""
    return labels._cache.get_video_track_occupancy(video=video)


def add_track(labels: _Any, video: VideoProtocol, track: Track) -> None:
    """Register `track` for `video` and update the cache."""
    labels.tracks.append(track)
    labels._cache.add_track(video, track)


def remove_track(labels: _Any, track: Track) -> None:
    """Remove all references to `track` from instances."""
    for inst in labels.instances():
        if inst.track == track:
            inst.track = None
    labels.tracks.remove(track)


def remove_all_tracks(labels: _Any) -> None:
    """Clear every track reference."""
    for inst in labels.instances():
        inst.track = None
    labels.tracks = []


def remove_unused_tracks(labels: _Any) -> None:
    """Eliminate tracks that lack any instances."""
    if len(labels.tracks) == 0:
        return
    all_tracks = set(labels.tracks)
    used_tracks = {inst.track for inst in labels.instances()}
    for track in all_tracks - used_tracks:
        labels.tracks.remove(track)


def find_track_occupancy(
    labels: _Any,
    video: VideoProtocol,
    track: Track | int | None,
    frame_range: tuple | range | None = None,
) -> list[Instance]:
    """Return instances for `track` within `frame_range` of `video`."""
    frame_range = range(*frame_range) if isinstance(frame_range, tuple) else frame_range

    def _index_by_identity(items: list, target: object) -> int:
        for i, item in enumerate(items):
            if item is target:
                return i
        return -1

    def does_track_match(
        inst: Instance, tr: Track | int | None, labeled_frame: LabeledFrame
    ) -> bool:
        if tr is None:
            return inst.track is None
        if isinstance(tr, Track) and inst.track is tr:
            return True
        if (
            isinstance(tr, int)
            and _index_by_identity(labeled_frame.instances, inst) == tr
            and inst.track is None
        ):
            return True
        return False

    return [
        instance
        for lf in labels.query.find(video)
        for instance in lf.instances
        if does_track_match(instance, track, lf)
        and (frame_range is None or lf.frame_idx in frame_range)
    ]


def track_swap(
    labels: _Any,
    video: VideoProtocol,
    new_track: Track,
    old_track: Track | None,
    frame_range: tuple,
) -> None:
    """Swap instance assignments between `old_track` and `new_track` in `frame_range`."""
    labels._cache.track_swap(video, new_track, old_track, frame_range)
    old_track_instances = find_track_occupancy(labels, video, old_track, frame_range)
    new_track_instances = find_track_occupancy(labels, video, new_track, frame_range)
    for instance in old_track_instances:
        instance.track = new_track
    if isinstance(old_track, Track):
        for instance in new_track_instances:
            instance.track = old_track


def track_set_instance(
    labels: _Any, frame: LabeledFrame, instance: Instance, new_track: Track
) -> None:
    """Move `instance` within `frame` to `new_track`, keeping cache updated."""
    track_swap(
        labels, frame.video, new_track, instance.track, (frame.frame_idx, frame.frame_idx + 1)
    )
    if instance.track is None:
        labels._cache.remove_instance(frame, instance)
    instance.track = new_track


__all__ = [
    "add_track",
    "find_track_occupancy",
    "get_track_occupancy",
    "remove_all_tracks",
    "remove_track",
    "remove_unused_tracks",
    "track_set_instance",
    "track_swap",
]
