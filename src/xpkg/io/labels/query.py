"""Utilities for querying labels and building fast lookup tables."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from xpkg.core.annotations import LabeledFrame
from xpkg.io.labels.video_types import VideoProtocol
from xpkg.io.video import Video

if TYPE_CHECKING:
    from xpkg.io.labels.model import Labels


def group_labeled_frames_by_video(
    labeled_frames: Iterable[LabeledFrame], videos: Iterable[VideoProtocol]
) -> dict[VideoProtocol, list[LabeledFrame]]:
    """Group labeled frames by their associated video for fast lookups."""

    by_vid: dict[VideoProtocol, list[LabeledFrame]] = {v: [] for v in videos}
    for lf in labeled_frames:
        if lf.video in by_vid:
            by_vid[lf.video].append(lf)
        else:
            by_vid.setdefault(lf.video, []).append(lf)
    return by_vid


def build_frame_index_map(
    by_video: dict[VideoProtocol, list[LabeledFrame]],
) -> dict[VideoProtocol, dict[int, LabeledFrame]]:
    """Create per-video frame_idx -> LabeledFrame maps."""
    idx_map: dict[VideoProtocol, dict[int, LabeledFrame]] = {}
    for vid, lfs in by_video.items():
        idx_map[vid] = {lf.frame_idx: lf for lf in lfs}
    return idx_map


def find_frames(
    frame_idx_map: dict[VideoProtocol, dict[int, LabeledFrame]],
    video: VideoProtocol,
    frame_idx: int | Iterable[int] | None = None,
) -> list[LabeledFrame] | None:
    """Return frames for `video`, optionally limited to `frame_idx`."""
    if frame_idx is None:
        if video not in frame_idx_map:
            return None
        return list(frame_idx_map[video].values())
    if video not in frame_idx_map:
        return None
    fmap = frame_idx_map[video]
    if isinstance(frame_idx, Iterable) and not isinstance(frame_idx, str | bytes):
        return [fmap[int(cast(Any, idx))] for idx in frame_idx if int(cast(Any, idx)) in fmap]
    if int(frame_idx) not in fmap:
        return None
    return [fmap[int(frame_idx)]]


def fancy_frame_indices(
    frame_idx_map_for_video: dict[int, LabeledFrame],
    from_frame_idx: int,
    reverse: bool,
) -> list[int]:
    """Choose frame indices near `from_frame_idx`, optionally reversed."""
    frame_idxs = sorted(frame_idx_map_for_video.keys())
    if not frame_idxs:
        return []
    if reverse:
        nxt = max((x for x in frame_idxs if x < from_frame_idx), default=frame_idxs[-1])
    else:
        nxt = min((x for x in frame_idxs if x > from_frame_idx), default=frame_idxs[0])
    cut = frame_idxs.index(nxt)
    return frame_idxs[cut:] + frame_idxs[:cut]


class LabelsQuery:
    """Provide list-like/sliceable access to a labels archive."""

    def __init__(self, labels: Labels) -> None:
        self._labels = labels

    def __getitem__(self, key, *secondary_key) -> LabeledFrame | list[LabeledFrame] | None:
        """Proxy to `get` for convenient indexing."""
        return self.get(key, *secondary_key)

    def get(
        self,
        key: Any,
        *secondary_key: Any,
        use_cache: bool = False,
        raise_errors: bool = False,
    ) -> LabeledFrame | list[LabeledFrame] | None:
        """Resolve `key` to labeled frames, supporting videos, frames, and ranges."""
        if len(secondary_key) > 0:
            if not isinstance(key, tuple):
                key = (key,)
            key = tuple(key) + tuple(secondary_key)

        if isinstance(key, slice):
            start, stop, step = key.indices(len(self._labels))
            key = range(start, stop, step)
        elif isinstance(key, np.integer):
            key = int(key)
        elif isinstance(key, np.ndarray):
            key = list(key.flat)

        if isinstance(key, int):
            return self._labels.labels.__getitem__(key)
        if isinstance(key, VideoProtocol):
            if key not in self._labels.videos:
                if raise_errors:
                    raise KeyError("Video not found in labels.")
                return None
            return self.find(video=key)
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], VideoProtocol):
            video = key[0]
            frame_selector: object = key[1]
            if video not in self._labels.videos:
                if raise_errors:
                    raise KeyError("Video not found in labels.")
                return None
            if isinstance(frame_selector, np.integer):
                frame_selector = int(frame_selector)
            elif isinstance(frame_selector, np.ndarray):
                frame_selector = list(frame_selector.flat)
            if isinstance(frame_selector, int):
                _hit = self.find_first(video=video, frame_idx=frame_selector, use_cache=use_cache)
                if _hit is None:
                    if raise_errors:
                        raise KeyError(
                            f"No label found for specified video at frame {frame_selector}."
                        )
                    return None
                return _hit
            if isinstance(frame_selector, list):
                frame_indices = [int(idx) for idx in frame_selector]
                return self.find(video=video, frame_idx=frame_indices)
            if isinstance(frame_selector, range):
                return self.find(video=video, frame_idx=frame_selector)
            if raise_errors:
                raise KeyError("Invalid label indexing arguments.")
            return None
        if isinstance(key, list | range):
            return cast(Any, [self.__getitem__(i) for i in key])
        if raise_errors:
            raise KeyError("Invalid label indexing arguments.")
        return None

    def find(
        self,
        video: VideoProtocol,
        frame_idx: int | Iterable[int] | None = None,
        return_new: bool = False,
    ) -> list[LabeledFrame]:
        """Return frames for `video`, optionally generating placeholders."""
        fi = int(frame_idx) if isinstance(frame_idx, int | np.integer) else 0
        null_result = [LabeledFrame(video=cast(Video, video), frame_idx=fi)] if return_new else []
        result = self._labels._cache.find_frames(video, frame_idx)
        return null_result if result is None else result

    def frames(self, video: VideoProtocol, from_frame_idx: int = -1, reverse: bool = False):
        """Yield frames before/after `from_frame_idx` for `video`."""
        frame_idxs = self._labels._cache.find_fancy_frame_idxs(video, from_frame_idx, reverse)
        for idx in frame_idxs:
            yield self._labels._cache._frame_idx_map[video][idx]

    def find_first(
        self, video: VideoProtocol, frame_idx: int | None = None, use_cache: bool = False
    ) -> LabeledFrame | None:
        """Return the first matching frame, optionally using the cache."""
        if use_cache:
            label = self.find(video=video, frame_idx=frame_idx)
            return None if len(label) == 0 else label[0]
        if video in self._labels.videos:
            for label in self._labels.labels:
                if label.video == video and (frame_idx is None or (label.frame_idx == frame_idx)):
                    return label
        return None

    def find_last(self, video: VideoProtocol, frame_idx: int | None = None) -> LabeledFrame | None:
        """Return the latest labeled frame for `video` (optionally matching `frame_idx`)."""
        if video in self._labels.videos:
            for label in reversed(self._labels.labels):
                if label.video == video and (frame_idx is None or (label.frame_idx == frame_idx)):
                    return label
        return None

    def instance_count(self, video: VideoProtocol, frame_idx: int) -> int:
        """Count instances for the requested video/frame combination."""
        labeled_frame = self.find_first(video, frame_idx)
        if labeled_frame is None:
            return 0
        return len(labeled_frame.instances)


__all__ = [
    "LabelsQuery",
    "build_frame_index_map",
    "fancy_frame_indices",
    "find_frames",
    "group_labeled_frames_by_video",
]
