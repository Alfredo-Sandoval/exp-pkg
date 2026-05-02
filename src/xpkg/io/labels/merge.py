"""Merge operations for label sets that may carry independent object references."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from xpkg.io.labels.video_types import VideoProtocol
from xpkg.media.video import Video
from xpkg.pose.annotations import LabeledFrame, Track
from xpkg.pose.skeleton import Skeleton

if TYPE_CHECKING:
    from xpkg.io.labels.model import Labels


def unify_video_references(labels: Labels) -> None:
    """Ensure all LabeledFrame.video references point to the shared Video object.

    Strategy:
    - Build a map of absolute filename -> Video object from labels.videos.
    - For each labeled frame, retarget its video to the shared object when a
      clear absolute match is found.
    - Update labels cache afterwards.
    """
    video_by_abs: dict[str, VideoProtocol] = {}
    for video in labels.videos:
        if not video.filename:
            raise ValueError("Video missing filename during reference unification")
        key = str(Path(str(video.filename)).resolve())
        video_by_abs[key] = video

    changed = False
    for lf in list(labels.labeled_frames):
        video = lf.video
        if video is None:
            raise ValueError("LabeledFrame missing video reference during unification")
        if not video.filename:
            raise ValueError("Video filename missing during reference unification")
        key = str(Path(str(video.filename)).resolve())
        target = video_by_abs.get(key)
        if target is None:
            raise KeyError(f"Video {key} not found in labels.videos")
        if lf.video is not target:
            lf.video = cast(Video, target)
            changed = True
    if changed:
        labels.update_cache()


def _unify_against_base(base_labels: Labels, new_labels: Labels) -> None:
    """Unify object references in `new_labels` to objects from `base_labels`.

    Avoids JSON round-trip (from_json removed). We match by:
      - Video: absolute path only (no basename fallbacks)
      - Skeleton: structural equality via Skeleton.matches
      - Track: name + spawned_on
    """
    base_vid_by_abs: dict[str, VideoProtocol] = {}
    for video in base_labels.videos:
        if not video.filename:
            raise ValueError("Base labels contain a video without a filename")
        abspath = str(Path(str(video.filename)).resolve())
        base_vid_by_abs[abspath] = video

    def _map_video(video: VideoProtocol) -> VideoProtocol:
        if not video.filename:
            raise ValueError("Video missing filename while aligning labels")
        abspath = str(Path(str(video.filename)).resolve())
        target = base_vid_by_abs.get(abspath)
        if target is not None:
            return target
        raise ValueError(
            "Unable to align videos between labels sets. "
            "Provide absolute filenames in both labels sets (no basename matching allowed)."
        )

    for lf in list(new_labels.labeled_frames):
        if lf.video is None:
            raise ValueError("LabeledFrame missing video while aligning labels")
        mapped = _map_video(lf.video)
        if mapped is not lf.video:
            lf.video = cast(Video, mapped)

    for sug in list(new_labels.suggestions):
        if sug.video is None:
            raise ValueError("Suggestion missing video while aligning labels")
        mapped = _map_video(sug.video)
        if mapped is not sug.video:
            sug.video = mapped

    base_skeletons = list(base_labels.skeletons)

    def _find_skel(skeleton: Skeleton) -> Skeleton:
        for base_skeleton in base_skeletons:
            if base_skeleton.matches(skeleton):
                return base_skeleton
        return skeleton

    for lf in list(new_labels.labeled_frames):
        for inst in list(lf.instances):
            target = _find_skel(inst.skeleton)
            if target is not inst.skeleton:
                inst.skeleton = target
                inst.realign_points()

    skels = list(new_labels.skeletons)
    merged: list[Skeleton] = []
    for skeleton in skels:
        target = _find_skel(skeleton)
        if target not in merged:
            merged.append(target)
    new_labels.skeletons = merged

    base_tracks = list(base_labels.tracks)

    def _track_key(track: Track) -> tuple[str, int]:
        return (track.name, int(track.spawned_on))

    base_track_map = {_track_key(track): track for track in base_tracks}

    for lf in list(new_labels.labeled_frames):
        for inst in list(lf.instances):
            track = inst.track
            if track is None:
                continue
            target = base_track_map.get(_track_key(track))
            if target is not None and target is not track:
                inst.track = target


def complex_merge_between(base_labels: Labels, new_labels: Labels, unify: bool = True):
    """Merge `new_labels` into `base_labels`, optionally reconciling shared objects."""
    if unify:
        _unify_against_base(base_labels, new_labels)

    merged, extra_base, extra_new = LabeledFrame.complex_merge_between(
        base_labels=base_labels, new_frames=new_labels.labeled_frames
    )

    if not extra_base and not extra_new:
        base_labels._update_from_labels(merge=True)
        base_labels.update_cache()

    base_labels.suggestions.extend(new_labels.suggestions)
    merge_container_dicts(base_labels.negative_anchors, new_labels.negative_anchors)
    return merged, extra_base, extra_new


def finish_complex_merge(base_labels: Labels, resolved_frames: list[LabeledFrame]):
    """Finalize a multi-frame merge by committing resolved frames and recalculating caches."""
    base_labels.labeled_frames.extend(resolved_frames)
    merge_matching_frames(base_labels)
    base_labels._update_from_labels(merge=True)
    base_labels.update_cache()


def merge_container_dicts(dict_a: dict, dict_b: dict) -> None:
    """Extend `dict_a` with entries from `dict_b`, deduplicating shared lists."""
    for key in dict_b.keys():
        if key in dict_a:
            dict_a[key].extend(dict_b[key])
            from xpkg._core.path_registry import uniquify as _uniq

            dict_a[key][:] = _uniq(dict_a[key])
        else:
            dict_a[key] = dict_b[key]


def merge_matching_frames(labels: Labels, video: VideoProtocol | None = None):
    """Merge frames that refer to the same video into a merged list."""
    if video is None:
        for vid in {lf.video for lf in labels.labeled_frames}:
            labels.labeled_frames = LabeledFrame.merge_frames(labels.labeled_frames, video=vid)
        return
    labels.labeled_frames = LabeledFrame.merge_frames(
        labels.labeled_frames, video=cast(Video, video)
    )


__all__ = [
    "complex_merge_between",
    "finish_complex_merge",
    "merge_container_dicts",
    "merge_matching_frames",
    "unify_video_references",
]
