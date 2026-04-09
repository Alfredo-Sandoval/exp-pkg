"""Runtime cache of labeled frames, track occupancy, and frame indexing helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from xpkg.core.annotations import Instance, LabeledFrame, Track
from xpkg.core.logging_utils import get_logger
from xpkg.io.labels.query import (
    build_frame_index_map,
    fancy_frame_indices,
    find_frames,
    group_labeled_frames_by_video,
)
from xpkg.io.labels.video_types import VideoProtocol

if TYPE_CHECKING:
    from xpkg.io.labels.model import Labels


logger = get_logger(__name__)


@dataclass
class LabelsDataCache:
    """Cache of labels data tuned for fast per-video/frame lookups."""

    labels: Labels

    def __post_init__(self):
        self.update()

    def update(self, new_frame: LabeledFrame | None = None):
        """Refresh internal mappings; optionally append a single frame."""
        if new_frame is None:
            self._lf_by_video = group_labeled_frames_by_video(
                self.labels.labeled_frames, self.labels.videos
            )
            self._frame_idx_map = build_frame_index_map(self._lf_by_video)
            self._track_occupancy: dict[VideoProtocol, dict[Track | None, set[int]]] = {}
            self._frame_count_cache: dict[
                VideoProtocol | None, dict[str, set[tuple[int, int]]]
            ] = {}
            for video in self.labels.videos:
                self._track_occupancy[video] = self._make_track_occupancy(video)
        else:
            new_vid = new_frame.video
            if new_vid not in self._lf_by_video:
                self._lf_by_video[new_vid] = []
            if new_vid not in self._frame_idx_map:
                self._frame_idx_map[new_vid] = {}
            self._lf_by_video[new_vid].append(new_frame)
            self._frame_idx_map[new_vid][new_frame.frame_idx] = new_frame

    def find_frames(
        self, video: VideoProtocol, frame_idx: int | Iterable[int] | None = None
    ) -> list[LabeledFrame] | None:
        """Return cached frames for a given video index or list of indices."""
        return find_frames(self._frame_idx_map, video, frame_idx)

    def find_fancy_frame_idxs(
        self, video: VideoProtocol, from_frame_idx: int, reverse: bool
    ) -> list[int]:
        """Return nearby frame indices using the `fancy_frame_indices` helper."""
        if video not in self._frame_idx_map:
            return []
        return fancy_frame_indices(self._frame_idx_map[video], from_frame_idx, reverse)

    def _make_track_occupancy(self, video: VideoProtocol) -> dict[Track | None, set[int]]:
        """Build per-track frame indices for a video."""
        frame_idx_map = self._frame_idx_map[video]
        tracks: dict[Track | None, set[int]] = {}
        for frame_idx in sorted(frame_idx_map.keys()):
            instances = frame_idx_map[frame_idx]
            for instance in instances:
                if instance.track not in tracks:
                    tracks[instance.track] = set()
                tracks[instance.track].add(frame_idx)
        return tracks

    def get_track_occupancy(self, video: VideoProtocol, track: Track | None) -> set[int]:
        """Return the frame indices for the provided track (two-way caching)."""
        if track not in self.get_video_track_occupancy(video=video):
            self._track_occupancy[video][track] = set()
        return self._track_occupancy[video][track]

    def get_video_track_occupancy(self, video: VideoProtocol) -> dict[Track | None, set[int]]:
        """Return the occupancy mapping for every track in `video`."""
        if video not in self._track_occupancy:
            self._track_occupancy[video] = {}
        return self._track_occupancy[video]

    def remove_frame(self, frame: LabeledFrame):
        """Remove a frame from the cache, cleaning indexes and occupancy."""
        self._lf_by_video[frame.video].remove(frame)
        if (
            frame.video in self._frame_idx_map
            and frame.frame_idx in self._frame_idx_map[frame.video]
        ):
            del self._frame_idx_map[frame.video][frame.frame_idx]

    def remove_video(self, video: VideoProtocol):
        """Evict the frame/track data for a full video."""
        if video in self._lf_by_video:
            del self._lf_by_video[video]
        if video in self._frame_idx_map:
            del self._frame_idx_map[video]
        if video in self._track_occupancy:
            del self._track_occupancy[video]
        if video in self._frame_count_cache:
            del self._frame_count_cache[video]

    def reset_video(self, video: VideoProtocol):
        """Clear cached state for `video` without rebuilding all caches."""
        if video not in self.labels.videos:
            return

        video_idx = self.labels.videos.index(video)
        self._lf_by_video[video] = []
        self._frame_idx_map[video] = {}
        self._track_occupancy[video] = {}

        if None in self._frame_count_cache:
            for key, entries in list(self._frame_count_cache[None].items()):
                pruned = {pair for pair in entries if pair[0] != video_idx}
                if pruned:
                    self._frame_count_cache[None][key] = pruned
                else:
                    del self._frame_count_cache[None][key]

        if video in self._frame_count_cache:
            del self._frame_count_cache[video]

    def track_swap(
        self,
        video: VideoProtocol,
        new_track: Track,
        old_track: Track | None,
        frame_range: tuple,
    ):
        """Move frame range membership between tracks to keep occupancy coherent."""
        start, end = frame_range
        range_frames = set(range(start, end))

        old_occ = self.get_track_occupancy(video, old_track)
        new_occ = self.get_track_occupancy(video, new_track)

        within_old = old_occ & range_frames
        within_new = new_occ & range_frames

        if old_track is not None:
            old_occ -= range_frames
            old_occ |= within_new

        new_occ -= range_frames
        new_occ |= within_old

    def add_track(self, video: VideoProtocol, track: Track):
        """Ensure a track entry exists for the given video (no-op otherwise)."""
        self.get_track_occupancy(video=video, track=track)

    def add_instance(self, frame: LabeledFrame, instance: Instance):
        """Register a new instance in the cache, updating track occupancy."""
        occ = self.get_track_occupancy(video=frame.video, track=instance.track)
        occ.add(frame.frame_idx)
        self.update_counts_for_frame(frame)

    def remove_instance(self, frame: LabeledFrame, instance: Instance):
        """Remove an instance from occupancy when its track drops frames."""
        video = frame.video
        if video not in self._track_occupancy:
            return
        track = instance.track
        if track not in self._track_occupancy[video]:
            self.update_counts_for_frame(frame)
            return

        remaining = [i for i in frame.instances if i.track == track]
        if len(remaining) == 0:
            self._track_occupancy[video][track].discard(frame.frame_idx)
        self.update_counts_for_frame(frame)

    def get_labeled_frame_count(self, video: VideoProtocol | None = None, filter: str = "") -> int:
        """Return labeled-frame count (optionally filtered by user/predicted)."""
        if filter not in ("", "user", "predicted"):
            raise ValueError(f"LabelsDataCache.get_labeled_frame_count() invalid filter: {filter}")
        if video not in self._frame_count_cache:
            self._frame_count_cache[video] = {}
        if self._frame_count_cache[video].get(filter) is None:
            self._frame_count_cache[video][filter] = self.get_filtered_frame_idxs(video, filter)
        return len(self._frame_count_cache[video][filter])

    def get_filtered_frame_idxs(
        self, video: VideoProtocol | None = None, filter: str = ""
    ) -> set[tuple[int, int]]:
        """Return (video_idx, frame_idx) tuples matching the requested filter."""
        if video not in self.labels.videos:
            video = None

        if filter == "":

            def _pred(lf: LabeledFrame) -> bool:
                return video is None or lf.video == video
        elif filter == "user":

            def _pred(lf: LabeledFrame) -> bool:
                return (video is None or lf.video == video) and lf.has_user_instances
        elif filter == "predicted":

            def _pred(lf: LabeledFrame) -> bool:
                return (video is None or lf.video == video) and lf.has_predicted_instances
        else:
            raise ValueError(f"Invalid filter: {filter}")

        index_by_obj: dict[VideoProtocol, int] = {v: i for i, v in enumerate(self.labels.videos)}
        index_by_path: dict[str, int] = {}
        for v, i in index_by_obj.items():
            filename = v.filename
            if filename:
                key = str(Path(str(filename)).resolve())
                index_by_path[key] = i

        def _idx_for_video(v: VideoProtocol) -> int | None:
            if v in index_by_obj:
                return index_by_obj[v]
            filename = v.filename
            if filename:
                key = str(Path(str(filename)).resolve())
                if key in index_by_path:
                    return index_by_path[key]
            return None

        out: set[tuple[int, int]] = set()
        if video is not None:
            v_idx = index_by_obj.get(video)
            if v_idx is None:
                v_idx = _idx_for_video(video)
            if v_idx is None:
                return set()
            for lf in self.labels:
                if _pred(lf):
                    out.add((v_idx, lf.frame_idx))
            return out

        for lf in self.labels:
            if not _pred(lf):
                continue
            v = lf.video
            idx = index_by_obj.get(v)
            if idx is None:
                idx = _idx_for_video(v)
                if idx is None:
                    continue
                lf.video = self.labels.videos[idx]
            out.add((idx, lf.frame_idx))
        return out

    def update_counts_for_frame(self, frame: LabeledFrame):
        """Invalidate cached stats when a frame changes."""
        video = frame.video
        if video is None:
            raise ValueError("LabeledFrame is missing its video reference")
        if video not in self.labels.videos:
            matches = [v for v in self.labels.videos if v.filename == video.filename]
            if not matches:
                raise KeyError("Video not found in labels for frame count update")
            video = matches[0]

        frame_idx = frame.frame_idx
        if video not in self.labels.videos:
            return
        video_idx = self.labels.videos.index(video)
        if video not in self._frame_count_cache:
            self._frame_count_cache[video] = {}

        if frame.has_user_instances:
            self._add_count_cache(video, video_idx, frame_idx, "user")
        else:
            self._del_count_cache(video, video_idx, frame_idx, "user")

        if frame.has_predicted_instances:
            self._add_count_cache(video, video_idx, frame_idx, "predicted")
        else:
            self._del_count_cache(video, video_idx, frame_idx, "predicted")

        if len(frame.instances):
            self._add_count_cache(video, video_idx, frame_idx, "")
        else:
            self._del_count_cache(video, video_idx, frame_idx, "")

    def _add_count_cache(
        self, video: VideoProtocol | None, video_idx: int, frame_idx: int, type_key: str
    ):
        idx_pair = (video_idx, frame_idx)
        if type_key in self._frame_count_cache.get(video, {}):
            self._frame_count_cache[video][type_key].add(idx_pair)
        if None in self._frame_count_cache and type_key in self._frame_count_cache[None]:
            self._frame_count_cache[None][type_key].add(idx_pair)

    def _del_count_cache(
        self, video: VideoProtocol | None, video_idx: int, frame_idx: int, type_key: str
    ):
        idx_pair = (video_idx, frame_idx)
        if type_key in self._frame_count_cache.get(video, {}):
            self._frame_count_cache[video][type_key].discard(idx_pair)
        if None in self._frame_count_cache and type_key in self._frame_count_cache[None]:
            self._frame_count_cache[None][type_key].discard(idx_pair)


__all__ = ["LabelsDataCache"]
