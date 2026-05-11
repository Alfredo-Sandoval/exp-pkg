"""Internal implementation of the canonical `xpkg.model` labels container."""

from __future__ import annotations

import itertools
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast, overload, runtime_checkable

import numpy as np

import xpkg.io.labels.export_ops as export_ops
import xpkg.io.labels.serialization as serialization
from xpkg.io.labels.cache import LabelsDataCache
from xpkg.io.labels.merge import complex_merge_between as _complex_merge_between
from xpkg.io.labels.merge import finish_complex_merge as _finish_complex_merge
from xpkg.io.labels.merge import merge_container_dicts as _merge_container_dicts
from xpkg.io.labels.merge import merge_matching_frames as _merge_matching_frames
from xpkg.io.labels.query import LabelsQuery
from xpkg.io.labels.tracks import add_track as _add_track
from xpkg.io.labels.video_types import VideoProtocol
from xpkg.pose.annotations import (
    Instance,
    LabeledFrame,
    PredictedInstance,
    Track,
)
from xpkg.pose.skeleton import (
    Keypoint,
    Skeleton,
)

from ..._core.logging_utils import get_logger

logger = get_logger(__name__)

LABELS_JSON_FILE_VERSION = "2.0.0"


def _build_default_template_points(keypoint_count: int, *, spacing: float = 50.0) -> np.ndarray:
    """Build a deterministic radial layout for template point fallbacks."""
    if keypoint_count == 0:
        return np.empty((0, 2), dtype=float)

    radius = max(spacing, (spacing * keypoint_count) / (2 * np.pi))
    angles = np.linspace(0, 2 * np.pi, num=keypoint_count, endpoint=False)
    coords = [
        np.array([radius * np.cos(theta), radius * np.sin(theta)], dtype=float)
        for theta in angles
    ]
    return np.stack(coords)


def _template_points_from_instances(
    instances: Iterable[Instance],
    *,
    keypoint_count: int,
) -> np.ndarray:
    """Compute mean keypoint positions without app-only helpers."""
    sum_xy = np.zeros((keypoint_count, 2), dtype=float)
    cnt_xy = np.zeros((keypoint_count, 2), dtype=float)
    any_seen = False

    for instance in instances:
        points = instance.get_points_array(copy=True, invisible_as_nan=True, full=False)
        if points.shape != (keypoint_count, 2):
            raise ValueError(
                "Instance points shape does not match skeleton keypoint count: "
                f"expected {(keypoint_count, 2)}, got {points.shape}"
            )
        any_seen = True
        valid_mask = ~np.isnan(points)
        sum_xy += np.where(valid_mask, points, 0.0)
        cnt_xy += valid_mask.astype(float)

    if not any_seen:
        return _build_default_template_points(keypoint_count)

    with np.errstate(invalid="ignore", divide="ignore"):
        template = sum_xy / cnt_xy

    missing_rows = np.all(cnt_xy == 0.0, axis=1)
    if missing_rows.any():
        defaults = _build_default_template_points(keypoint_count)
        template[missing_rows] = defaults[missing_rows]

    return template


@runtime_checkable
class Materializable(Protocol):
    """Objects that can materialize themselves into plain data."""

    def materialize(self) -> Any: ...


@dataclass
class SuggestionFrame:
    """Lightweight suggestion item identifying a video frame to review/label.

    Attributes:
        video: Video object or identifier.
        frame_idx: Index of the frame in the video.
        group: Optional group identifier for the suggestion.
        score: Optional confidence or priority score.
    """

    video: Any
    frame_idx: int
    group: int | None = None
    score: float | None = None


@dataclass(repr=False)
class Labels:
    """Canonical container for labeled frames, videos, skeletons, and tracks."""

    labeled_frames: list[LabeledFrame] = field(default_factory=list)
    videos: list[VideoProtocol] = field(default_factory=list)
    skeletons: list[Skeleton] = field(default_factory=list)
    keypoints: list[Keypoint] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    suggestions: list[SuggestionFrame] = field(default_factory=list)
    negative_anchors: dict[VideoProtocol, list] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    preferences: dict[str, Any] = field(default_factory=dict)
    session: dict[str, Any] = field(default_factory=dict)
    _path: Path | None = field(default=None, init=False, repr=False)
    query: LabelsQuery = field(init=False, repr=False)

    def __post_init__(self):
        self._update_from_labels()

        self._cache = LabelsDataCache(self)

        self.__temp_dir = None

        self._template_instance_points: dict[int, dict[str, Any]] = {}

        self.query = LabelsQuery(self)

    @property
    def path(self) -> Path | None:
        """Return the source labels path for this object, if known."""
        return self._path

    @path.setter
    def path(self, value: str | Path | None) -> None:
        if value is None:
            self._path = None
            return
        if isinstance(value, str) and not value.strip():
            raise ValueError("Labels.path must be a non-empty path")
        self._path = Path(value)

    def _update_from_labels(self, merge: bool = False):
        if merge or len(self.videos) == 0:
            existing = set(self.videos)
            sources = ({label.video for label in self.labels}) | (
                {sug.video for sug in self.suggestions}
            )
            for vid in sources:
                if vid not in existing:
                    self.videos.append(vid)
                    existing.add(vid)

        if merge or len(self.skeletons) == 0:
            seen_skel_ids = {id(sk) for sk in self.skeletons}
            for sk in (inst.skeleton for label in self.labels for inst in label.instances):
                if id(sk) not in seen_skel_ids:
                    self.skeletons.append(sk)
                    seen_skel_ids.add(id(sk))

        if merge or len(self.keypoints) == 0:
            seen_kp_ids = {id(kp) for kp in self.keypoints}
            for skeleton in self.skeletons:
                for kp in skeleton.keypoints:
                    if id(kp) not in seen_kp_ids:
                        self.keypoints.append(kp)
                        seen_kp_ids.add(id(kp))

        if merge or len(self.tracks) == 0:
            other_tracks = {
                instance.track
                for frame in self.labels
                for instance in frame.instances
                if instance.track
            }

            other_tracks = other_tracks.union(
                {
                    instance.from_predicted.track
                    for frame in self.labels
                    for instance in frame.instances
                    if instance.from_predicted and instance.from_predicted.track
                }
            )

            new_tracks = list(other_tracks - set(self.tracks))

            new_tracks.sort(key=lambda t: (t.spawned_on, t.name))

            self.tracks.extend(new_tracks)

    def _update_containers(self, new_label: LabeledFrame):
        if new_label.video not in self.videos:
            self.videos.append(new_label.video)

        seen_skeleton_ids = {id(skeleton) for skeleton in self.skeletons}
        seen_keypoint_ids = {id(kp) for kp in self.keypoints}
        new_skeletons: list[Skeleton] = []
        for instance in new_label:
            sk = instance.skeleton
            sk_id = id(sk)
            if sk_id in seen_skeleton_ids:
                continue
            seen_skeleton_ids.add(sk_id)
            new_skeletons.append(sk)
        for skeleton in new_skeletons:
            self.skeletons.append(skeleton)
            for kp in skeleton.keypoints:
                kp_id = id(kp)
                if kp_id in seen_keypoint_ids:
                    continue
                seen_keypoint_ids.add(kp_id)
                self.keypoints.append(kp)

        for instance in new_label.instances:
            tr = instance.track
            if tr and tr not in self.tracks:
                self.tracks.append(tr)

        self.tracks.sort(key=lambda t: (t.spawned_on, t.name))

        self._cache.update(new_label)

    def update_cache(self):
        """Ensure cache structures match the current set of labeled frames."""
        self._cache.update()

    def validate(self) -> None:
        """Enforce structural invariants for frames, instances, and tracks."""

        seen_instance_lists: set[int] = set()
        for lf in self.labeled_frames:
            if not isinstance(lf, LabeledFrame):
                raise TypeError(
                    "Labels.labeled_frames entries must be LabeledFrame objects; "
                    f"got {type(lf).__name__}"
                )
            inst_list = lf.instances
            if inst_list.labeled_frame is not lf:
                raise ValueError("InstancesList.labeled_frame must reference its owner")
            inst_list_id = id(inst_list)
            if inst_list_id in seen_instance_lists:
                raise ValueError("InstancesList cannot be shared across frames")
            seen_instance_lists.add(inst_list_id)

            frame_tracks: set[tuple[int | None, str]] = set()
            for inst in inst_list:
                if inst.frame is not lf:
                    raise ValueError("Instance.frame must reference owning LabeledFrame")

                inst._assert_points_synced()

                track = inst.track
                if track is None:
                    continue
                track_key = (track.spawned_on, track.name)
                if track_key in frame_tracks:
                    raise ValueError(
                        f"Duplicate track assignment for frame {lf.frame_idx}: {track.name}"
                    )
                frame_tracks.add(track_key)

    @property
    def labels(self):
        """Return the mutable list of labeled frames."""
        return self.labeled_frames

    @property
    def skeleton(self) -> Skeleton:
        """Return the single skeleton when only one exists (raise otherwise)."""
        if len(self.skeletons) == 1:
            return self.skeletons[0]
        else:
            raise ValueError(
                "Labels.skeleton can only be used when there is only a single skeleton "
                "saved in the labels. Use Labels.skeletons instead."
            )

    @property
    def video(self) -> VideoProtocol:
        """Return the single video when only one exists (raise otherwise)."""
        if len(self.videos) == 0:
            raise ValueError("There are no videos in the labels.")
        elif len(self.videos) == 1:
            return self.videos[0]
        else:
            raise ValueError(
                "Labels.video can only be used when there is only a single video saved "
                "in the labels. Use Labels.videos instead."
            )

    @property
    def has_missing_videos(self) -> bool:
        """Return True when any referenced video file is missing on disk."""

        def _missing(v: VideoProtocol) -> bool:
            filename = v.filename
            return bool(filename) and not os.path.exists(str(filename))

        return any(_missing(video) for video in self.videos)

    def __len__(self) -> int:
        return len(self.labeled_frames)

    def __iter__(self):
        """Iterate over the backing list to tolerate mid-iteration mutations."""
        return iter(self.labeled_frames)

    def close(self):
        """Release resources associated with videos."""
        for video in self.videos:
            video.close()

    def index(self, value: Any, start: int = 0, stop: int = 9223372036854775807) -> int:
        """Return the position of `value` among the labeled frames."""
        return self.labeled_frames.index(value, start, stop)

    def __repr__(self) -> str:
        return (
            "Labels("
            f"labeled_frames={len(self.labeled_frames)}, "
            f"videos={len(self.videos)}, "
            f"skeletons={len(self.skeletons)}, "
            f"tracks={len(self.tracks)}"
            ")"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __contains__(self, value: object) -> bool:
        type_checks = {
            LabeledFrame: lambda x: x in self.labeled_frames,
            VideoProtocol: lambda x: x in self.videos,
            Skeleton: lambda x: x in self.skeletons,
            Keypoint: lambda x: x in self.keypoints,
        }
        for cls, checker in type_checks.items():
            if isinstance(value, cls):
                return checker(value)

        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], VideoProtocol):
            frame_selector = value[1]
            if isinstance(frame_selector, np.integer):
                frame_selector = frame_selector.tolist()
            if isinstance(frame_selector, int):
                return self.query.find_first(value[0], frame_selector) is not None

        return False

    def __getitem__(self, key: Any) -> LabeledFrame | list[LabeledFrame]:
        """Return labeled frames matching the flexible selection `key`."""
        res = self.query.get(key)
        if res is None:
            raise IndexError(f"No label found for key: {key}")
        return res

    def get(
        self,
        key: (
            int
            | slice
            | np.integer
            | np.ndarray
            | list
            | range
            | VideoProtocol
            | tuple[VideoProtocol, np.integer | np.ndarray | int | list | range]
        ),
        *secondary_key: int | slice | np.integer | np.ndarray | list | range,
        use_cache: bool = False,
        raise_errors: bool = False,
    ) -> LabeledFrame | list[LabeledFrame] | None:
        """Retrieve labeled frames using the query helper with optional caching."""
        return self.query.get(key, *secondary_key, use_cache=use_cache, raise_errors=raise_errors)

    def extract(self, inds, copy: bool = False) -> Labels:
        """Return a subset of labels defined by `inds`, optionally copying them."""
        selected = self.__getitem__(inds)
        if isinstance(selected, list):
            lfs = list(selected)
        else:
            lfs = [selected]

        videos_used = set([lf.video for lf in lfs])
        videos = list(videos_used)

        suggestions = [
            suggestion for suggestion in self.suggestions if suggestion.video in videos_used
        ]

        new_labels = Labels(
            labeled_frames=lfs,
            videos=videos,
            skeletons=self.skeletons,
            tracks=self.tracks,
            suggestions=suggestions,
            provenance=self.provenance,
        )
        new_labels.path = self.path
        if copy:
            new_labels = new_labels.copy()
        return new_labels

    def copy(self) -> Labels:
        """Return a deep copy of this labels set."""

        clone = Labels(
            labeled_frames=[lf.copy() for lf in self.labeled_frames],
            videos=list(self.videos),
            skeletons=list(self.skeletons),
            tracks=list(self.tracks),
            suggestions=list(self.suggestions),
            provenance=dict(self.provenance),
            session=dict(self.session),
        )
        clone.path = self.path
        return clone

    def split(self, n: float | int, copy: bool = True) -> tuple[Labels, Labels]:
        """Divide the labels into two groups by count or proportion."""
        if len(self) == 1:
            if copy:
                return self.copy(), self.copy()
            else:
                return self, self

        if not isinstance(n, int):
            n = round(len(self) * n)
        n = max(min(n, len(self) - 1), 1)
        all_indices = np.random.permutation(len(self))
        idx_a = all_indices[:n].tolist()
        idx_b = all_indices[n:].tolist()

        return self.extract(idx_a, copy=copy), self.extract(idx_b, copy=copy)

    @overload
    def __setitem__(self, index: int, value: LabeledFrame) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[LabeledFrame]) -> None: ...

    def __setitem__(self, index: int | slice, value: LabeledFrame | Iterable[LabeledFrame]) -> None:
        raise NotImplementedError(
            "Labels does not support item replacement; use explicit APIs (insert/remove/update)"
        )

    def insert(self, index, value: LabeledFrame):
        """Insert a labeled frame unless it already exists in this labels set."""
        if value in self or (value.video, value.frame_idx) in self:
            return

        self.labeled_frames.insert(index, value)
        self._update_containers(value)

    def append(self, value: LabeledFrame):
        """Append `value` after the existing frames list."""
        self.insert(len(self) + 1, value)

    def __delitem__(self, index: int | slice | Any) -> None:
        if isinstance(index, slice):
            for item in self.labeled_frames[index]:
                self.labeled_frames.remove(item)
        else:
            self.labeled_frames.remove(self.labeled_frames[index])

    def remove(self, value: LabeledFrame):
        """Remove `value` from the labels set."""
        self.remove_frame(value)

    def remove_frame(self, lf: LabeledFrame, update_cache: bool = True):
        """Remove a labeled frame and optionally refresh caches."""
        self.labeled_frames.remove(lf)
        if update_cache:
            self._cache.remove_frame(lf)

    def remove_frames(self, lfs: list[LabeledFrame]):
        """Bulk-remove labeled frames and refresh the cache."""
        to_remove = set(lfs)
        self.labeled_frames = [lf for lf in self.labeled_frames if lf not in to_remove]
        self.update_cache()

    def remove_empty_instances(self, keep_empty_frames: bool = True):
        """Prune empty instances and optionally drop empty frames."""
        for lf in self.labeled_frames:
            lf.remove_empty_instances()
        self.update_cache()
        if not keep_empty_frames:
            self.remove_empty_frames()

    def remove_empty_frames(self):
        """Drop any frames that no longer contain instances."""
        self.labeled_frames = [lf for lf in self.labeled_frames if len(lf.instances) > 0]
        self.update_cache()

    @property
    def user_labeled_frames(self) -> list[LabeledFrame]:
        """Return all frames that include user-labeled instances."""
        return [lf for lf in self.labeled_frames if lf.has_user_instances]

    @property
    def user_labeled_frame_inds(self) -> list[int]:
        """Return indices of frames that contain user instances."""
        return [i for i, lf in enumerate(self.labeled_frames) if lf.has_user_instances]

    @property
    def all_instances(self) -> list[Instance]:
        """Return every instance across all frames."""
        return list(self.instances())

    @property
    def user_instances(self) -> list[Instance]:
        """Return every non-predicted instance."""
        return [
            inst
            for inst in self.all_instances
            if isinstance(inst, Instance) and not isinstance(inst, PredictedInstance)
        ]

    @property
    def predicted_instances(self) -> list[PredictedInstance]:
        """Return every predicted instance referenced by these labels."""
        return [inst for inst in self.all_instances if isinstance(inst, PredictedInstance)]

    @property
    def has_user_instances(self) -> bool:
        """Return True when any frame contains user-labeled instances."""
        return any(lf.has_user_instances for lf in self.labeled_frames)

    @property
    def has_predicted_instances(self) -> bool:
        """Return True when any frame contains predicted instances."""
        return any(lf.has_predicted_instances for lf in self.labeled_frames)

    @property
    def max_user_instances(self) -> int:
        """Return the maximum number of user instances across frames."""
        max_count = 0
        for lf in self.labeled_frames:
            max_count = max(max_count, lf.user_instance_count)
        return max_count

    def describe(self):
        """Accumulate basic stats about user and predicted annotations."""
        n_user = 0
        n_pred = 0
        n_user_inst = 0
        n_pred_inst = 0
        for lf in self.labeled_frames:
            if lf.has_user_instances:
                n_user += 1
                n_user_inst += len(lf.user_instances)
            if lf.has_predicted_instances:
                n_pred += 1
                n_pred_inst += len(lf.predicted_instances)

    def instances(self, video: VideoProtocol | None = None, skeleton: Skeleton | None = None):
        """Yield instances filtered by optional video and skeleton."""
        for label in self.labels:
            if video is None or label.video == video:
                for instance in label.instances:
                    if skeleton is None or instance.skeleton == skeleton:
                        yield instance

    def get_template_instance_points(self, skeleton: Skeleton):
        """Return cached template point arrays for `skeleton`."""
        sk_key = id(skeleton)

        rebuild_template = False
        if len(self.labeled_frames) < 100:
            rebuild_template = True
        elif sk_key not in self._template_instance_points:
            rebuild_template = True
        elif skeleton.keypoints != self._template_instance_points[sk_key]["keypoints"]:
            rebuild_template = True

        if rebuild_template:
            if self.labeled_frames and any(self.instances()):
                first_n_instances = itertools.islice(self.instances(skeleton=skeleton), 1000)
                template_points = _template_points_from_instances(
                    first_n_instances,
                    keypoint_count=len(skeleton.keypoints),
                )
                self._template_instance_points[sk_key] = {
                    "points": template_points,
                    "keypoints": skeleton.keypoints,
                }
            else:
                template_points = _build_default_template_points(len(skeleton.keypoints))
                self._template_instance_points[sk_key] = {
                    "points": template_points,
                    "keypoints": skeleton.keypoints,
                }

        return self._template_instance_points[sk_key]["points"]

    def remove_instance(
        self, frame: LabeledFrame, instance: Instance, in_transaction: bool = False
    ):
        """Remove `instance` and update caches unless part of a larger transaction."""
        for i, inst in enumerate(frame.instances):
            if inst is instance:
                del frame.instances[i]
                break
        if not in_transaction:
            self._cache.remove_instance(frame, instance)

    def add_instance(self, frame: LabeledFrame, instance: Instance):
        """Add `instance` to `frame` and keep track occupancy updated."""
        tracks_in_frame = [
            inst.track for inst in frame if isinstance(inst, Instance) and inst.track is not None
        ]
        if instance.from_predicted is None and instance.track in tracks_in_frame:
            logger.debug(
                "add_instance: clearing duplicate track %s for instance without from_predicted",
                instance.track,
            )
            instance.track = None

        frame.instances.append(instance)
        if (instance.track is not None) and (instance.track not in self.tracks):
            _add_track(self, video=frame.video, track=instance.track)

        self._cache.add_instance(frame, instance)

    def add_predicted_instances(
        self, video: VideoProtocol, frame_idx: int, pred_list: list[Instance]
    ) -> LabeledFrame:
        """Insert predicted instances into the frame at `frame_idx` for `video`."""
        lf = self.query.find_first(video, frame_idx, use_cache=True)
        if lf is None:
            lf = LabeledFrame(video=cast(Any, video), frame_idx=int(frame_idx))
            self.append(lf)
        for inst in pred_list or []:
            self.add_instance(lf, inst)
        return lf

    def add_video(self, video: VideoProtocol):
        """Ensure `video` is tracked by this labels set."""
        if video not in self.videos:
            self.videos.append(video)

    def remove_video(self, video: VideoProtocol):
        """Purge `video` and all associated data from this labels set."""
        if video not in self.videos:
            raise KeyError("Video is not in labels.")

        for label in reversed(self.labeled_frames):
            if label.video == video:
                self.labeled_frames.remove(label)

        if video in self.negative_anchors:
            del self.negative_anchors[video]

        self.videos.remove(video)
        self._cache.remove_video(video)

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any] | None,
        *,
        suggestions_payload: dict[str, Any] | None = None,
        video_builder: serialization.VideoBuilder | None = None,
        video_finalizer: serialization.HydratedVideoFinalizer | None = None,
    ) -> Labels:
        """Construct `Labels` from a labels payload dictionary.

        Args:
            payload: The main labels payload dictionary.
            suggestions_payload: Optional suggestions data dictionary.
            video_builder: Optional video construction hook for product-specific media policy.
            video_finalizer: Optional post-hydration hook for releasing media resources.

        Returns:
            Labels: The hydrated Labels instance.
        """
        return serialization.labels_from_payload(
            cls,
            payload,
            suggestions_payload=suggestions_payload,
            video_builder=video_builder,
            video_finalizer=video_finalizer,
        )

    def extend_from(self, new_frames: Labels | list[LabeledFrame], unify: bool = False):
        """Extend the labels set with new frames, optionally unifying shared objects."""
        if isinstance(new_frames, Labels):
            new_frames = new_frames.labeled_frames

        if not isinstance(new_frames, list) or len(new_frames) == 0:
            return False
        if not isinstance(new_frames[0], LabeledFrame):
            return False

        if unify:
            temp = Labels(labeled_frames=new_frames)
            _merge_matching_frames(temp)
            new_frames = temp.labeled_frames

        self.labeled_frames.extend(new_frames)

        self.merge_matching_frames()

        self._update_from_labels(merge=True)
        self._cache.update()

        return True

    def has_frame(
        self,
        lf: LabeledFrame | None = None,
        video: VideoProtocol | None = None,
        frame_idx: int | None = None,
        use_cache: bool = True,
    ) -> bool:
        """Return True if a labeled frame exists for the given identifier."""
        if lf is not None:
            video = lf.video
            frame_idx = lf.frame_idx
        if video is None or frame_idx is None:
            raise ValueError("Either lf or video and frame_idx must be provided.")

        if use_cache:
            return len(self.query.find(video, frame_idx=frame_idx, return_new=False)) > 0

        else:
            if video not in self.videos:
                return False
            for lf in self.labeled_frames:
                if lf.video == video and lf.frame_idx == frame_idx:
                    return True
            return False

    def remove_user_instances(self, new_labels: Labels | None = None):
        """Remove all user instances, optionally keeping frames referenced by `new_labels`."""
        keep_lfs = []
        for lf in self.labeled_frames:
            if new_labels is not None:
                if not new_labels.has_frame(lf):
                    keep_lfs.append(lf)
                    continue

            if lf.has_predicted_instances:
                lf.instances = lf.predicted_instances
                keep_lfs.append(lf)

        self.labeled_frames = keep_lfs

    def remove_predictions(self, new_labels: Labels | None = None):
        """Remove predicted instances, optionally preserving frames that exist in `new_labels`."""
        keep_lfs = []
        for lf in self.labeled_frames:
            if new_labels is not None:
                if not new_labels.has_frame(lf):
                    keep_lfs.append(lf)
                    continue

            if lf.has_user_instances:
                lf.instances = lf.user_instances
                keep_lfs.append(lf)

        self.labeled_frames = keep_lfs

    def remove_untracked_instances(self, remove_empty_frames: bool = True):
        """Drop instances whose tracks are marked as untracked."""
        for lf in self.labeled_frames:
            lf.remove_untracked()
        if remove_empty_frames:
            self.remove_empty_frames()

    @classmethod
    def complex_merge_between(
        cls, base_labels: Labels, new_labels: Labels, unify: bool = True
    ) -> tuple:
        """Delegate complex label merges to the helper while optionally unifying."""
        return _complex_merge_between(base_labels, new_labels, unify)

    @staticmethod
    def finish_complex_merge(base_labels: Labels, resolved_frames: list[LabeledFrame]):
        """Finalize a complex merge with resolved frames."""
        _finish_complex_merge(base_labels, resolved_frames)

    @staticmethod
    def merge_container_dicts(dict_a: dict, dict_b: dict) -> None:
        """Merge dictionary-based containers, keeping unique items."""
        _merge_container_dicts(dict_a, dict_b)

    def merge_matching_frames(self, video: VideoProtocol | None = None):
        """Coalesce frames that refer to the same video into merged positions."""
        _merge_matching_frames(self, video)

    @classmethod
    def load_file(cls, filename: str, *args, **kwargs):
        """Load labels from disk."""
        return serialization.labels_load_file(cls, filename, *args, **kwargs)

    @classmethod
    def save_file(cls, labels: Labels, filename: str, _default_suffix: str = "", *args, **kwargs):
        """Save labels to disk."""
        return serialization.labels_save_file(
            labels,
            filename,
            default_suffix=_default_suffix,
            **kwargs,
        )

    def save(self, filename: str, *args, **kwargs):
        """Save labels to a project or JSON file."""
        return self.save_file(self, filename)

    def numpy(
        self,
        video: VideoProtocol | int | None = None,
        all_frames: bool = True,
        untracked: bool = False,
        return_confidence: bool = False,
    ) -> np.ndarray:
        """Return `np.ndarray` data for the selected `video`/frames.

        Args:
            video: Video object or index to export. Defaults to the first video.
            all_frames: If True, return a matrix covering all frames in the video.
            untracked: If True, return instances in the order they appear in the frame.
            return_confidence: If True, return (x, y, confidence) instead of (x, y).

        Returns:
            np.ndarray: Array of shape (frames, tracks, keypoints, dims).
        """
        return export_ops.labels_numpy(
            self,
            video=video,
            all_frames=all_frames,
            untracked=untracked,
            return_confidence=return_confidence,
        )

    def to_dataframe(
        self,
        video: VideoProtocol | int | None = None,
        scorer: str = "xpkg",
    ):
        """Convert labels for a video to a DeepLabCut-style MultiIndex DataFrame.

        Args:
            video: Video object or index to export. Defaults to the first video.
            scorer: Scorer name to use in the MultiIndex.

        Returns:
            pd.DataFrame with MultiIndex columns (scorer, bodypart, coords)
        """
        return export_ops.labels_to_dataframe(self, video=video, scorer=scorer)

    def merge_keypoints(self, base_keypoint: str, merge_keypoint: str):
        """Merge `merge_keypoint` into `base_keypoint` across the dataset."""
        export_ops.merge_keypoints(self, base_keypoint=base_keypoint, merge_keypoint=merge_keypoint)

    def rename_or_merge_keypoint(
        self,
        *,
        skeleton: Skeleton,
        old_name: str,
        new_name: str,
    ) -> None:
        """Rename keypoint, or merge into an existing one for single-skeleton projects."""
        if new_name in skeleton.keypoint_names and len(self.skeletons) == 1:
            self.merge_keypoints(base_keypoint=new_name, merge_keypoint=old_name)
            return
        skeleton.rename_keypoint(old_name, new_name)

    def drop_keypoint_heatmaps(self, keypoint_index: int) -> None:
        """Remove a keypoint channel from per-frame heatmaps."""
        export_ops.drop_keypoint_heatmaps(self, keypoint_index=keypoint_index)


__all__ = [
    "Labels",
    "Materializable",
    "SuggestionFrame",
]
