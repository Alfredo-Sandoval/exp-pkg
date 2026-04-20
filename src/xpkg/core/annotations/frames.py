"""Frame-level annotation structures and helpers."""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable, Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, SupportsIndex

import numpy as np

from xpkg.core.annotations.instances import (
    Instance,
    InstanceLike,
    PredictedInstance,
    Track,
    is_predicted_instance,
)
from xpkg.core.annotations.regions import ROI, SegmentationMask
from xpkg.core.logging_utils import get_logger

logger = get_logger("xpkg.core.annotations")

if TYPE_CHECKING:
    from xpkg.io.video import Video

    class _LabelsQueryLike(Protocol):
        def find(
            self,
            video: Video,
            frame_idx: int | Iterable[int] | None = None,
            return_new: bool = False,
        ) -> list[LabeledFrame]: ...

    class LabelsLike(Protocol):
        @property
        def query(self) -> _LabelsQueryLike: ...

        @property
        def labeled_frames(self) -> list[LabeledFrame]: ...
else:
    from typing import Any as Video


class InstancesList(list[InstanceLike]):
    """List-like wrapper that keeps track of associated labeled frame metadata."""

    def __init__(
        self,
        iterable: Iterable[InstanceLike] | None = None,
        *,
        labeled_frame: LabeledFrame | None = None,
    ):
        """Initialize with optional iterable and bind a owning labeled frame."""
        initial = iterable if iterable is not None else ()
        super().__init__(initial)
        self._labeled_frame: LabeledFrame | None = None
        if labeled_frame is not None:
            self.labeled_frame = labeled_frame

    @property
    def labeled_frame(self) -> LabeledFrame | None:
        """Return the labeled frame that owns these instances."""
        return self._labeled_frame

    @labeled_frame.setter
    def labeled_frame(self, labeled_frame: LabeledFrame | None):
        if self._labeled_frame == labeled_frame:
            return
        self._labeled_frame = labeled_frame
        for instance in self:
            instance.frame = labeled_frame

    def append(self, value: InstanceLike) -> None:
        """Append an Instance and bind it to the owning frame."""
        value.frame = self.labeled_frame
        super().append(value)

    def extend(self, values: Iterable[InstanceLike]) -> None:
        """Extend the list with multiple instances, preserving frame binding."""
        for instance in values:
            self.append(instance)

    def __delitem__(self, index: SupportsIndex | slice):
        if isinstance(index, slice):
            removed = list(self[index])
            super().__delitem__(index)
            for instance in removed:
                instance.frame = None
            return
        idx = operator.index(index)
        instance = self[idx]
        super().__delitem__(idx)
        instance.frame = None

    def insert(self, index: SupportsIndex, value: InstanceLike) -> None:
        """Insert an instance at the requested index, keeping frame binding."""
        idx = operator.index(index)
        value.frame = self.labeled_frame
        super().insert(idx, value)

    def __setitem__(
        self,
        index: SupportsIndex | slice,
        value: InstanceLike | Iterable[InstanceLike],
    ) -> None:
        """Replace entries while ensuring only Instance-like items are stored."""
        if isinstance(index, slice):
            if not isinstance(value, Iterable):
                raise TypeError("Slice assignment requires an iterable of instances.")
            values = list(value)
            for instance in values:
                instance.frame = self.labeled_frame
            super().__setitem__(index, values)
            return
        idx = operator.index(index)
        if not isinstance(value, Instance):
            raise TypeError("InstancesList items must be Instance-like objects.")
        value.frame = self.labeled_frame
        super().__setitem__(idx, value)

    def pop(self, index: SupportsIndex = -1) -> InstanceLike:
        """Pop an instance and detach it from the owning labeled frame."""
        idx = operator.index(index)
        instance = super().pop(idx)
        instance.frame = None
        return instance

    def remove(self, value: InstanceLike) -> None:
        """Remove an instance and clear its frame reference."""
        if value in self:
            super().remove(value)
            value.frame = None

    def clear(self) -> None:
        """Clear all instances and unlink them from the frame."""
        for instance in self:
            instance.frame = None
        super().clear()

    def copy(self) -> list[InstanceLike]:
        """Return a shallow list copy of the instances."""
        return list(self)


@dataclass(eq=False)
class LabeledFrame:
    """Frame-level wrapper holding video/frame metadata plus instances."""

    video: Video
    frame_idx: int
    _instances: InstancesList = field(default_factory=InstancesList)
    heatmaps: np.ndarray | None = None
    _masks: list[SegmentationMask] = field(default_factory=list)
    _rois: list[ROI] = field(default_factory=list)

    def __post_init__(self):
        """Hook to ensure the internal InstancesList is bound to the frame."""
        if self._instances is not None:
            self.instances = self._instances

    def __init__(
        self,
        video: Video,
        frame_idx: int,
        instances: InstancesList | Sequence[InstanceLike] | None = None,
        masks: list[SegmentationMask] | None = None,
        rois: list[ROI] | None = None,
    ):
        """Initialize the labeled frame, optionally binding instances."""
        self.video = video
        self.frame_idx = frame_idx

        if instances is None:
            self._instances = InstancesList()
        elif isinstance(instances, InstancesList):
            self._instances = instances
        else:
            self._instances = InstancesList(instances)

        self.heatmaps = None
        self._masks = list(masks) if masks is not None else []
        self._rois = list(rois) if rois is not None else []
        self.instances = self._instances

    def __len__(self) -> int:
        return len(self._instances)

    def __iter__(self):
        return iter(self._instances)

    def __getitem__(self, index) -> Instance | PredictedInstance:
        return self._instances.__getitem__(index)

    def index(self, value: Instance) -> int:
        """Return the index of `value` within this list (by identity)."""
        for i, inst in enumerate(self._instances):
            if inst is value:
                return i
        raise ValueError(f"{value!r} is not in instances")

    def __delitem__(self, index):
        self._instances.__delitem__(index)

    def __repr__(self) -> str:
        vname = self.video.filename
        vdesc = f"'{vname}'" if vname else "<images>"
        return (
            f"LabeledFrame(video={vdesc}, "
            f"frame_idx={self.frame_idx}, "
            f"instances={len(self._instances)})"
        )

    def insert(self, index: int, value: Instance):
        """Insert an instance at `index`, keeping frame binding intact."""
        self._instances.insert(index, value)

    def __setitem__(self, index, value: Instance):
        self._instances.__setitem__(index, value)

    def find(self, track: Track | int | None = -1, user: bool = False) -> list[Instance]:
        """Return instances filtered by track index and optional user-only flag.

        Args:
            track: Track object, track ID, or None to filter by. Use -1 to skip track filtering.
            user: If True, only return user-provided (non-predicted) instances.

        Returns:
            list[Instance]: A list of matching instances.
        """
        instances = self.instances
        if user:
            instances = list(
                filter(
                    lambda inst: (
                        isinstance(inst, Instance) and not isinstance(inst, PredictedInstance)
                    ),
                    instances,
                )
            )
        if track != -1:

            def _track_matches(inst: Instance) -> bool:
                if track is None:
                    return inst.track is None
                if isinstance(track, int):
                    return inst.track is not None and inst.track.id == track
                # track is Track
                return inst.track is not None and inst.track.matches(track)

            instances = list(filter(_track_matches, instances))
        return instances

    @property
    def instances(self) -> InstancesList:
        """Return the InstancesList storing this frame's instances."""
        return self._instances

    @instances.setter
    def instances(self, instances: InstancesList | Sequence[InstanceLike]):
        """Bind a new InstancesList (or raw list) to this frame."""
        if isinstance(instances, InstancesList):
            instances.labeled_frame = self
        else:
            instances = InstancesList(instances, labeled_frame=self)
        self._instances = instances

    @property
    def masks(self) -> list[SegmentationMask]:
        """Return the segmentation masks attached to this frame."""
        return self._masks

    @masks.setter
    def masks(self, masks: list[SegmentationMask] | Sequence[SegmentationMask]):
        self._masks = list(masks)

    @property
    def rois(self) -> list[ROI]:
        """Return the regions of interest attached to this frame."""
        return self._rois

    @rois.setter
    def rois(self, rois: list[ROI] | Sequence[ROI]):
        self._rois = list(rois)

    @property
    def has_masks(self) -> bool:
        """Return True if any segmentation masks are attached."""
        return len(self._masks) > 0

    @property
    def has_rois(self) -> bool:
        """Return True if any ROIs are attached."""
        return len(self._rois) > 0

    @property
    def user_masks(self) -> list[SegmentationMask]:
        """Return non-predicted segmentation masks."""
        return [m for m in self._masks if not m.is_predicted]

    @property
    def predicted_masks(self) -> list[SegmentationMask]:
        """Return predicted segmentation masks."""
        return [m for m in self._masks if m.is_predicted]

    def copy(self) -> LabeledFrame:
        """Return a deep copy of this frame (instances + heatmaps + masks + rois).

        Returns:
            LabeledFrame: A new LabeledFrame instance with copied data.
        """
        from xpkg.core.annotations.serde import make_instance_cattr

        converter = make_instance_cattr()
        instances_payload = converter.unstructure(self._instances)
        instances_copy = converter.structure(instances_payload, InstancesList)
        masks_copy = [
            SegmentationMask.from_dict(m.to_dict(), track=m.track) for m in self._masks
        ]
        rois_copy = [ROI.from_dict(r.to_dict(), track=r.track) for r in self._rois]
        clone = LabeledFrame(
            video=self.video,
            frame_idx=self.frame_idx,
            instances=instances_copy,
            masks=masks_copy,
            rois=rois_copy,
        )
        if self.heatmaps is not None:
            clone.heatmaps = np.array(self.heatmaps, copy=True)
        return clone

    @property
    def user_instances(self) -> list[Instance]:
        """Return all user-provided (non-predicted) instances."""
        return [
            inst
            for inst in self._instances
            if isinstance(inst, Instance) and not isinstance(inst, PredictedInstance)
        ]

    @property
    def training_instances(self) -> list[Instance]:
        """Return user instances that have visible points for training."""
        return [
            inst
            for inst in self._instances
            if not isinstance(inst, PredictedInstance) and inst.visible_point_count > 0
        ]

    @property
    def predicted_instances(self) -> list[PredictedInstance]:
        """Return the list of predicted instances for this frame."""
        return [inst for inst in self._instances if isinstance(inst, PredictedInstance)]

    @property
    def tracked_instances(self) -> list[PredictedInstance]:
        """Return predicted instances with associated tracks."""
        return [
            inst
            for inst in self._instances
            if isinstance(inst, PredictedInstance) and inst.track is not None
        ]

    def remove_untracked(self):
        """Delete instances that lack track assignments."""
        self.instances = [inst for inst in self.instances if inst.track is not None]

    @property
    def has_user_instances(self) -> bool:
        """Return True if at least one user (non-predicted) instance exists."""
        return any(
            isinstance(inst, Instance) and not isinstance(inst, PredictedInstance)
            for inst in self._instances
        )

    @property
    def has_predicted_instances(self) -> bool:
        """Return True when any predicted instances are present."""
        return any(isinstance(inst, PredictedInstance) for inst in self._instances)

    @property
    def has_tracked_instances(self) -> bool:
        """Return True if any predicted instance has an associated track."""
        return any(
            isinstance(inst, PredictedInstance) and inst.track is not None
            for inst in self._instances
        )

    @property
    def user_instance_count(self) -> int:
        """Return the count of user instances (non-predicted)."""
        return sum(
            1
            for inst in self._instances
            if isinstance(inst, Instance) and not isinstance(inst, PredictedInstance)
        )

    @property
    def predicted_instance_count(self) -> int:
        """Return the count of predicted instances."""
        return sum(1 for inst in self._instances if isinstance(inst, PredictedInstance))

    @property
    def tracked_instance_count(self) -> int:
        """Return the count of tracked predicted instances."""
        return sum(
            1
            for inst in self._instances
            if isinstance(inst, PredictedInstance) and inst.track is not None
        )

    def remove_empty_instances(self):
        """Drop instances that report zero visible points."""
        self.instances = [inst for inst in self.instances if inst.visible_point_count > 0]

    @property
    def unused_predictions(self) -> list[PredictedInstance]:
        """Return predicted instances that were never displayed (unused)."""
        predictions = [inst for inst in self._instances if isinstance(inst, PredictedInstance)]
        used_prediction_ids: set[int] = {
            id(inst.from_predicted)
            for inst in self._instances
            if isinstance(inst, Instance)
            and not isinstance(inst, PredictedInstance)
            and inst.from_predicted is not None
        }
        used_track_keys: set[tuple[int, str]] = {
            (inst.track.spawned_on, inst.track.name)
            for inst in self._instances
            if isinstance(inst, Instance)
            and not isinstance(inst, PredictedInstance)
            and inst.track is not None
        }
        user_instances = [
            inst
            for inst in self._instances
            if isinstance(inst, Instance) and not isinstance(inst, PredictedInstance)
        ]
        geom_candidates = [
            pred
            for pred in predictions
            if id(pred) not in used_prediction_ids
            and (
                pred.track is None
                or (pred.track.spawned_on, pred.track.name) not in used_track_keys
            )
        ]

        geom_used_ids: set[int] = set()
        if user_instances and geom_candidates:
            skeleton_keys: dict[int, str] = {}

            def _skeleton_key(inst: Instance) -> str:
                skel = inst.skeleton
                skel_id = id(skel)
                cached = skeleton_keys.get(skel_id)
                if cached is not None:
                    return cached
                key = skel.content_hash()
                skeleton_keys[skel_id] = key
                return key

            user_by_skeleton: dict[str, list[Instance]] = {}
            for user in user_instances:
                user_by_skeleton.setdefault(_skeleton_key(user), []).append(user)

            preds_by_skeleton: dict[str, list[PredictedInstance]] = {}
            for pred in geom_candidates:
                preds_by_skeleton.setdefault(_skeleton_key(pred), []).append(pred)

            points_cache: dict[int, np.ndarray] = {}

            def _points_array(inst: Instance) -> np.ndarray:
                inst_id = id(inst)
                cached = points_cache.get(inst_id)
                if cached is not None:
                    return cached
                pts = np.asarray(inst.points_array, dtype=float)
                if pts.size:
                    pts[~np.isfinite(pts)] = math.nan
                points_cache[inst_id] = pts
                return pts

            def _stack_points(instances: Sequence[Instance]) -> np.ndarray:
                return np.stack([_points_array(inst) for inst in instances], axis=0)

            def _centroids(points_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                count = points_stack.shape[0]
                centroids = np.full((count, 2), math.nan, dtype=float)
                valid = np.zeros(count, dtype=bool)
                for idx in range(count):
                    pts = points_stack[idx]
                    valid_mask = ~np.isnan(pts).any(axis=1)
                    if not valid_mask.any():
                        continue
                    valid_points = pts[valid_mask]
                    centroid = np.median(valid_points, axis=0)
                    centroids[idx] = centroid
                    valid[idx] = np.all(np.isfinite(centroid))
                return centroids, valid

            tol = 6.0
            tol_sq = tol * tol

            # Vectorized centroid/keypoint matching within each skeleton group.
            for skel_key, pred_group in preds_by_skeleton.items():
                user_group = user_by_skeleton.get(skel_key)
                if not user_group:
                    continue

                pred_points = _stack_points(pred_group)
                user_points = _stack_points(user_group)

                diffs = np.abs(user_points[:, None, :, :] - pred_points[None, :, :, :])
                kp_matches = np.all(diffs <= tol, axis=-1).any(axis=-1)

                user_centroids, user_valid = _centroids(user_points)
                pred_centroids, pred_valid = _centroids(pred_points)
                if user_valid.any() and pred_valid.any():
                    centroid_diffs = user_centroids[:, None, :] - pred_centroids[None, :, :]
                    centroid_dist_sq = np.sum(centroid_diffs**2, axis=-1)
                    centroid_matches = (centroid_dist_sq <= tol_sq) & (
                        user_valid[:, None] & pred_valid[None, :]
                    )
                else:
                    centroid_matches = np.zeros(kp_matches.shape, dtype=bool)

                pair_matches = kp_matches | centroid_matches
                matched_preds = pair_matches.any(axis=0)
                for pred, matched in zip(pred_group, matched_preds, strict=False):
                    if matched:
                        geom_used_ids.add(id(pred))

        def _is_unused(pred: PredictedInstance) -> bool:
            if id(pred) in used_prediction_ids:
                return False

            if pred.track is not None:
                track_key = (pred.track.spawned_on, pred.track.name)
                if track_key in used_track_keys:
                    return False
            if id(pred) in geom_used_ids:
                return False
            return True

        return [pred for pred in predictions if _is_unused(pred)]

    @property
    def instances_to_show(self) -> list[Instance]:
        """Return the instances to show, preferring predictions when available."""
        unused_predictions = self.unused_predictions
        unused_pred_ids = {id(pred) for pred in unused_predictions}
        inst_to_show = [
            inst
            for inst in self._instances
            if (isinstance(inst, Instance) and not isinstance(inst, PredictedInstance))
            or id(inst) in unused_pred_ids
        ]
        inst_to_show.sort(
            key=lambda inst: inst.track.spawned_on if isinstance(inst.track, Track) else math.inf
        )
        return inst_to_show

    @staticmethod
    def merge_frames(
        labeled_frames: list[LabeledFrame], video: Video, remove_redundant=True
    ) -> list[LabeledFrame]:
        """Deduplicate labeled frames for a video and optionally drop duplicates."""
        redundant_count = 0
        frames_found = dict()
        for idx, lf in enumerate(labeled_frames):
            if lf.video == video:
                if lf.frame_idx in frames_found.keys():
                    dst_idx = frames_found[lf.frame_idx]
                    if remove_redundant:
                        for new_inst in lf.instances:
                            redundant = False
                            for old_inst in labeled_frames[dst_idx].instances:
                                if new_inst.matches(old_inst):
                                    redundant = True
                                    if not is_predicted_instance(new_inst):
                                        redundant_count += 1
                                    break
                            if not redundant:
                                labeled_frames[dst_idx].instances.append(new_inst)
                    else:
                        labeled_frames[dst_idx].instances.extend(lf.instances)
                    lf.instances = []
                else:
                    frames_found[lf.frame_idx] = idx
        labeled_frames = list(filter(lambda lf: len(lf.instances), labeled_frames))
        if redundant_count:
            logger.info("Skipped %d redundant instances", redundant_count)
        return labeled_frames

    @classmethod
    def complex_merge_between(
        cls, base_labels: LabelsLike, new_frames: list[LabeledFrame]
    ) -> tuple[dict[Video, dict[int, list[Instance]]], list[Any], list[Any]]:
        """Compare base labels and new frames to produce merge diagnostics."""
        merged = dict()
        extra_base = []
        extra_new = []
        for new_frame in new_frames:
            base_lfs = base_labels.query.find(new_frame.video, new_frame.frame_idx)
            merged_instances = None
            if not base_lfs:
                base_labels.labeled_frames.append(new_frame)
                merged_instances = new_frame.instances
            else:
                (
                    merged_instances,
                    extra_base_frame,
                    extra_new_frame,
                ) = cls.complex_frame_merge(base_lfs[0], new_frame)
                if extra_base_frame:
                    extra_base.append(extra_base_frame)
                if extra_new_frame:
                    extra_new.append(extra_new_frame)
            if merged_instances:
                if new_frame.video not in merged:
                    merged[new_frame.video] = dict()
                merged[new_frame.video][new_frame.frame_idx] = merged_instances
        return merged, extra_base, extra_new

    @classmethod
    def complex_frame_merge(
        cls, base_frame: LabeledFrame, new_frame: LabeledFrame
    ) -> tuple[list[Instance], Any, Any]:
        """Merge two frames and return merged instances with conflict info."""
        merged_instances: list[Instance] = []
        redundant_instances: list[Instance] = []
        extra_base_instances: list[Instance] = list(base_frame.instances)
        extra_new_instances: list[Instance] = []
        for new_inst in new_frame:
            redundant = False
            for base_inst in base_frame.instances:
                if new_inst.matches(base_inst):
                    extra_base_instances.remove(base_inst)
                    redundant_instances.append(base_inst)
                    redundant = True
                    continue
            if not redundant:
                extra_new_instances.append(new_inst)
        conflict = False
        if extra_base_instances and extra_new_instances:
            base_predictions = [
                inst for inst in extra_base_instances if is_predicted_instance(inst)
            ]
            new_predictions = [inst for inst in extra_new_instances if is_predicted_instance(inst)]
            base_has_nonpred = len(extra_base_instances) - len(base_predictions)
            new_has_nonpred = len(extra_new_instances) - len(new_predictions)
            if base_predictions and new_predictions:
                conflict = True
            elif base_has_nonpred and new_has_nonpred:
                conflict = True
        if conflict:
            base_frame.instances.clear()
            base_frame.instances.extend(redundant_instances)
            base_frame.instances.extend(extra_new_instances)
            merged_instances = copy(extra_new_instances)
            extra_base_instances = []
            extra_new_instances = []
        else:
            base_frame.instances.extend(extra_new_instances)
            merged_instances = copy(extra_new_instances)
            extra_base_instances = []
            extra_new_instances = []
        extra_base = (
            cls(
                video=base_frame.video,
                frame_idx=base_frame.frame_idx,
                instances=extra_base_instances,
            )
            if extra_base_instances
            else None
        )
        extra_new = (
            cls(
                video=new_frame.video,
                frame_idx=new_frame.frame_idx,
                instances=extra_new_instances,
            )
            if extra_new_instances
            else None
        )
        return merged_instances, extra_base, extra_new

    @property
    def image(self) -> np.ndarray:
        """Return the image frame corresponding to this labeled frame."""
        idx = 0 if self.frame_idx is None else int(self.frame_idx)
        return self.video.get_frame(idx)

    def numpy(self) -> np.ndarray:
        """Return stacked point arrays or an empty array padded with NaNs.

        Returns:
            np.ndarray: Array of shape (instances, keypoints, 2).
        """
        if len(self.instances) > 0:
            return np.stack([inst.numpy() for inst in self.instances], axis=0)
        else:
            return np.full((0, 0, 2), np.nan)


__all__ = ["InstancesList", "LabeledFrame", "logger"]
