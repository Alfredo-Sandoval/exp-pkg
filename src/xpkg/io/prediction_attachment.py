"""Public prediction attachment and hydration helpers for xpkg labels."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Any, cast

import numpy as np

from xpkg.model.labels import Labels
from xpkg.model.labels_tracks import add_track
from xpkg.pose.annotations import (
    Instance,
    LabeledFrame,
    Point,
    PointArray,
    PredictedInstance,
    PredictedPointArray,
    Track,
)


@dataclass(frozen=True, slots=True)
class PredictionAttachResult:
    attached_frame_indices: tuple[int, ...]
    attached_instance_count: int
    updated_track_map: dict[int, Track]


@dataclass(frozen=True, slots=True)
class PredictionClearResult:
    cleared_frame_indices: tuple[int, ...]
    removed_instance_count: int
    user_annotation_unchanged: bool


@dataclass(frozen=True, slots=True)
class PreparedPredictionAttachment:
    frame_payloads: dict[int, dict[str, Any]]
    frame_indices: tuple[int, ...]
    clear_result: PredictionClearResult
    replace_existing_predictions: bool = False
    replace_track_ids: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class PredictionPayloadPartition:
    attachable_payloads: dict[int, dict[str, Any]]
    failed_frame_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PredictionFrameSnapshot:
    video: Any
    frame_states: tuple[tuple[int, LabeledFrame | None], ...]
    tracks: tuple[Track, ...]


@dataclass(slots=True)
class PredictionFrameSnapshotBuilder:
    label_tracks: tuple[Track, ...]
    frame_indices: tuple[int, ...]
    next_track_offset: int = 0
    next_frame_offset: int = 0
    tracks_by_id: dict[int, Track] = field(default_factory=dict)
    ordered_tracks: list[Track] = field(default_factory=list)
    frame_states: list[tuple[int, LabeledFrame | None]] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.next_track_offset >= len(self.label_tracks) and self.next_frame_offset >= len(
            self.frame_indices
        )


@dataclass(frozen=True, slots=True)
class PredictionUserInstanceResult:
    frame_idx: int
    created_user_instances: tuple[Instance, ...]
    created_user_instance_count: int
    removed_predicted_count: int


@dataclass(frozen=True, slots=True)
class PredictionUserPayloadApplyResult:
    applied_frame_indices: tuple[int, ...]
    created_user_instance_count: int
    updated_track_map: dict[int, Track]
    next_frame_offset: int
    total_frames: int

    @property
    def done(self) -> bool:
        return self.next_frame_offset >= self.total_frames


@dataclass(frozen=True, slots=True)
class PredictionApplyResult:
    clear_result: PredictionClearResult
    attach_result: PredictionAttachResult
    next_frame_offset: int
    total_frames: int

    @property
    def done(self) -> bool:
        return self.next_frame_offset >= self.total_frames


@dataclass(slots=True)
class PredictionAttachmentTransaction:
    frame_payloads: dict[int, dict[str, Any]]
    frame_indices: tuple[int, ...]
    track_map_before_attachment: dict[int, Track]
    current_track_map: dict[int, Track]
    snapshot_builder: PredictionFrameSnapshotBuilder | None
    rollback_snapshot: PredictionFrameSnapshot | None = None
    prepared_attachment: PreparedPredictionAttachment | None = None
    next_frame_offset: int = 0
    total_added: int = 0
    restored: bool = False

    @property
    def total_frames(self) -> int:
        return len(self.frame_indices)


def resolved_video_cache_key(video: Any) -> str | None:
    video_path = str(video.filename or "").strip()
    if not video_path:
        return None
    return Path(video_path).resolve().as_posix()


def missing_prediction_frames(
    *,
    requested_frames: Iterable[int],
    frame_payloads: Mapping[int, dict[str, Any]],
) -> list[int]:
    """Return requested frames missing from prediction payload output."""
    return [
        int(frame_idx) for frame_idx in requested_frames if int(frame_idx) not in frame_payloads
    ]


def validate_prediction_payloads(
    *,
    requested_frames: Iterable[int],
    frame_payloads: Mapping[int, dict[str, Any]],
    allow_empty_frames: bool = False,
) -> None:
    missing_frames = missing_prediction_frames(
        requested_frames=requested_frames,
        frame_payloads=frame_payloads,
    )
    if missing_frames:
        raise RuntimeError(
            "Prediction export aborted: model returned no predictions for frames "
            f"{missing_frames[:5]}" + ("..." if len(missing_frames) > 5 else "")
        )
    if allow_empty_frames:
        return
    empty_frames = frames_without_keypoints(frame_payloads)
    if empty_frames:
        raise RuntimeError(
            "Prediction export aborted: model returned no keypoints for frames "
            f"{empty_frames[:5]}" + ("..." if len(empty_frames) > 5 else "")
        )


def frames_without_keypoints(frame_payloads: Mapping[int, dict[str, Any]]) -> list[int]:
    """Return frame indices whose payload has no finite keypoints."""
    missing_frames: list[int] = []
    for frame_idx, payload in frame_payloads.items():
        insts = payload.get("instances") or []
        if not _frame_has_points(insts):
            missing_frames.append(int(frame_idx))
    return missing_frames


def coerce_track_map(track_map: object) -> dict[int, Track]:
    if track_map is None:
        return {}
    if not isinstance(track_map, Mapping):
        raise TypeError("track_map must be a mapping[int, Track] | None")
    normalized: dict[int, Track] = {}
    for raw_key, raw_track in track_map.items():
        if isinstance(raw_key, bool) or not isinstance(raw_key, int):
            raise TypeError("track_map keys must be integers")
        if not isinstance(raw_track, Track):
            raise TypeError("track_map values must be Track objects")
        normalized[int(raw_key)] = raw_track
    return normalized


def partition_attachable_payloads(
    *,
    requested_frames: Sequence[int],
    frame_payloads: Mapping[int, dict[str, Any]],
) -> PredictionPayloadPartition:
    missing_frames = missing_prediction_frames(
        requested_frames=requested_frames,
        frame_payloads=frame_payloads,
    )
    if missing_frames:
        raise RuntimeError(
            "Propagation aborted: tracker returned no payload for frames "
            f"{missing_frames[:5]}" + ("..." if len(missing_frames) > 5 else "")
        )

    attachable_payloads = {
        int(frame_idx): dict(payload)
        for frame_idx, payload in frame_payloads.items()
        if payload.get("instances")
    }
    if not attachable_payloads:
        raise RuntimeError(
            "Propagation failed on all target frames. Try a shorter range or reseed from "
            "a closer labeled frame."
        )

    failed_frame_indices = tuple(
        sorted(set(int(frame_idx) for frame_idx in requested_frames) - set(attachable_payloads))
    )
    return PredictionPayloadPartition(
        attachable_payloads=attachable_payloads,
        failed_frame_indices=failed_frame_indices,
    )


def clear_predictions(
    *,
    labels: Labels,
    video: Any | None,
    frame_indices: Sequence[int] | None = None,
    track_ids: Sequence[int] | None = None,
) -> PredictionClearResult:
    if video is None and track_ids is not None:
        raise ValueError("track_ids requires a video target")
    target_track_ids = None if track_ids is None else {int(track_id) for track_id in track_ids}
    target_frames = resolve_clear_target_frames(
        labels=labels,
        video=video,
        frame_indices=frame_indices,
    )
    if not target_frames:
        return PredictionClearResult(
            cleared_frame_indices=(),
            removed_instance_count=0,
            user_annotation_unchanged=True,
        )

    cleared_frame_indices: list[int] = []
    removed_instance_count = 0
    for labeled_frame in target_frames:
        removed_count = clear_matching_predictions_from_frame(
            labels=labels,
            labeled_frame=labeled_frame,
            target_track_ids=target_track_ids,
        )
        if removed_count == 0:
            continue
        cleared_frame_indices.append(int(labeled_frame.frame_idx))
        removed_instance_count += removed_count

    if not cleared_frame_indices:
        return PredictionClearResult(
            cleared_frame_indices=(),
            removed_instance_count=0,
            user_annotation_unchanged=True,
        )
    return PredictionClearResult(
        cleared_frame_indices=tuple(sorted(set(cleared_frame_indices))),
        removed_instance_count=removed_instance_count,
        user_annotation_unchanged=True,
    )


def snapshot_prediction_frames(
    *,
    labels: Labels,
    video: Any,
    frame_indices: Sequence[int],
) -> PredictionFrameSnapshot:
    snapshot_builder = begin_prediction_frame_snapshot(
        labels=labels,
        frame_indices=frame_indices,
    )
    total_items = len(snapshot_builder.label_tracks) + len(snapshot_builder.frame_indices)
    advance_prediction_frame_snapshot(
        labels=labels,
        video=video,
        snapshot_builder=snapshot_builder,
        max_items=max(total_items, 1),
    )
    return finish_prediction_frame_snapshot(
        video=video,
        snapshot_builder=snapshot_builder,
    )


def begin_prediction_frame_snapshot(
    *,
    labels: Labels,
    frame_indices: Sequence[int],
) -> PredictionFrameSnapshotBuilder:
    return PredictionFrameSnapshotBuilder(
        label_tracks=tuple(track for track in labels.tracks if isinstance(track, Track)),
        frame_indices=resolve_snapshot_frame_indices(frame_indices),
    )


def advance_prediction_frame_snapshot(
    *,
    labels: Labels,
    video: Any,
    snapshot_builder: PredictionFrameSnapshotBuilder,
    max_items: int,
) -> bool:
    if max_items < 1:
        raise ValueError("max_items must be >= 1")

    remaining = max_items
    while remaining > 0 and snapshot_builder.next_track_offset < len(snapshot_builder.label_tracks):
        track = snapshot_builder.label_tracks[snapshot_builder.next_track_offset]
        append_snapshot_track(snapshot_builder=snapshot_builder, track=track)
        snapshot_builder.next_track_offset += 1
        remaining -= 1

    while remaining > 0 and snapshot_builder.next_frame_offset < len(
        snapshot_builder.frame_indices
    ):
        frame_idx = snapshot_builder.frame_indices[snapshot_builder.next_frame_offset]
        labeled_frame = labels.query.find_first(video, frame_idx, use_cache=True)
        snapshot_builder.frame_states.append(
            (
                frame_idx,
                None if labeled_frame is None else clone_labeled_frame_for_snapshot(labeled_frame),
            )
        )
        if labeled_frame is not None:
            for inst in labeled_frame.instances:
                track = inst.track
                if isinstance(track, Track):
                    append_snapshot_track(snapshot_builder=snapshot_builder, track=track)
        snapshot_builder.next_frame_offset += 1
        remaining -= 1
    return snapshot_builder.done


def finish_prediction_frame_snapshot(
    *,
    video: Any,
    snapshot_builder: PredictionFrameSnapshotBuilder,
) -> PredictionFrameSnapshot:
    if not snapshot_builder.done:
        raise RuntimeError("Prediction frame snapshot builder must be fully drained.")
    return PredictionFrameSnapshot(
        video=video,
        frame_states=tuple(snapshot_builder.frame_states),
        tracks=tuple(snapshot_builder.ordered_tracks),
    )


def restore_prediction_frames(
    *,
    labels: Labels,
    snapshot: PredictionFrameSnapshot,
) -> None:
    if not snapshot.frame_states:
        return

    canonical_video = canonical_video_for_labels(labels=labels, video=snapshot.video)
    canonical_tracks = list(snapshot.tracks)
    canonical_by_id = {int(track.id): track for track in canonical_tracks}
    for frame_idx, frame_state in snapshot.frame_states:
        existing_frame = labels.query.find_first(canonical_video, int(frame_idx), use_cache=True)
        restored_frame = materialize_restored_frame(
            frame_state=frame_state,
            canonical_video=canonical_video,
            canonical_tracks=canonical_tracks,
            canonical_by_id=canonical_by_id,
        )
        if existing_frame is None:
            if restored_frame is not None:
                append_restored_frame(labels=labels, restored_frame=restored_frame)
            continue
        if restored_frame is None:
            drop_frame(labels=labels, labeled_frame=existing_frame)
            continue
        restore_frame_contents(
            labels=labels,
            target_frame=existing_frame,
            restored_frame=restored_frame,
        )
    labels.tracks = canonical_tracks


def normalize_prediction_tracks(
    *,
    labels: Labels,
    labeled_frame: LabeledFrame,
    preds: list[PredictedInstance],
    track_map: dict[int, Track] | None = None,
) -> dict[int, Track]:
    """Map predicted integer track ids to canonical Track objects and return the updated map."""
    normalized_map = dict(track_map or {})
    for pred in preds:
        track = pred.track
        if not isinstance(track, Track):
            continue
        track_id = int(track.id)
        if track_id < 0:
            continue
        canonical = normalized_map.get(track_id)
        if canonical is None:
            add_track(labels, video=labeled_frame.video, track=track)
            canonical = track
            normalized_map[track_id] = canonical
        pred.track = canonical
    return normalized_map


def make_predicted_instances(
    skeleton: Any,
    frame_obj: LabeledFrame,
    instances_payload: list[dict[str, Any]],
    *,
    conf_thresh: float = 0.0,
) -> list[PredictedInstance]:
    """Convert raw inference payloads into predicted instances."""
    keypoints = skeleton.keypoints
    out: list[PredictedInstance] = []
    for inst in instances_payload:
        pts = inst.get("keypoints")
        if pts is None:
            raise ValueError("Prediction instance missing keypoints")
        score = float(inst.get("score", 0.0))
        track_id = inst.get("track_id")
        point_array = _predicted_points_from_payload(pts, conf_thresh=conf_thresh)
        if len(point_array) != len(keypoints):
            raise ValueError(
                "Prediction instance contains "
                f"{len(point_array)} keypoints but skeleton expects {len(keypoints)}. "
                "Check model configuration."
            )
        pred = PredictedInstance(skeleton=skeleton, frame=frame_obj, init_points=point_array)
        pred.score = score
        if isinstance(track_id, int) and track_id >= 0:
            pred.track = Track(spawned_on=int(track_id), name=f"track_{track_id}")
        out.append(pred)
    return out


def create_user_instances_from_predictions(
    *,
    labels: Labels,
    labeled_frame: LabeledFrame,
) -> PredictionUserInstanceResult:
    predictions = tuple(labeled_frame.unused_predictions)
    created_user_instances = tuple(
        build_user_instance_from_prediction(
            predicted_instance=predicted_instance,
            labeled_frame=labeled_frame,
            force_visible=False,
        )
        for predicted_instance in predictions
    )
    for user_instance in created_user_instances:
        labels.add_instance(labeled_frame, user_instance)
    return PredictionUserInstanceResult(
        frame_idx=int(labeled_frame.frame_idx),
        created_user_instances=created_user_instances,
        created_user_instance_count=len(created_user_instances),
        removed_predicted_count=0,
    )


def apply_user_instances_from_payload_slice(
    *,
    labels: Labels,
    video: Any,
    frame_payloads: Mapping[int, dict[str, Any]],
    skeleton: Any,
    track_map: Mapping[int, Track] | None = None,
    frame_indices: Sequence[int] | None = None,
    start_idx: int = 0,
    max_frames: int | None = None,
) -> PredictionUserPayloadApplyResult:
    resolved_frame_indices = resolve_attach_frame_indices(
        frame_payloads=frame_payloads,
        frame_indices=frame_indices,
    )
    total_frames = len(resolved_frame_indices)
    end_idx = resolve_apply_end_idx(
        total_frames=total_frames,
        start_idx=start_idx,
        max_frames=max_frames,
    )
    updated_track_map = dict(coerce_track_map(track_map))
    created_frame_indices: list[int] = []
    created_user_instance_count = 0
    for frame_idx in resolved_frame_indices[start_idx:end_idx]:
        labeled_frame = ensure_labeled_frame(labels=labels, video=video, frame_idx=frame_idx)
        assert_frame_has_no_user_instances(labeled_frame=labeled_frame)
        created_users, updated_track_map = create_user_instances_from_payload(
            labels=labels,
            labeled_frame=labeled_frame,
            skeleton=skeleton,
            payload=frame_payloads.get(frame_idx),
            track_map=updated_track_map,
        )
        created_frame_indices.append(frame_idx)
        created_user_instance_count += len(created_users)
    return PredictionUserPayloadApplyResult(
        applied_frame_indices=tuple(created_frame_indices),
        created_user_instance_count=created_user_instance_count,
        updated_track_map=updated_track_map,
        next_frame_offset=end_idx,
        total_frames=total_frames,
    )


def accept_prediction(
    *,
    labels: Labels,
    predicted_instance: PredictedInstance,
    labeled_frame: LabeledFrame | None = None,
) -> PredictionUserInstanceResult:
    frame = resolve_prediction_frame(
        predicted_instance=predicted_instance,
        labeled_frame=labeled_frame,
    )
    user_instance = build_user_instance_from_prediction(
        predicted_instance=predicted_instance,
        labeled_frame=frame,
        force_visible=True,
    )
    labels.add_instance(frame, user_instance)
    labels.remove_instance(frame, predicted_instance)
    return PredictionUserInstanceResult(
        frame_idx=int(frame.frame_idx),
        created_user_instances=(user_instance,),
        created_user_instance_count=1,
        removed_predicted_count=1,
    )


def begin_replacing_transaction(
    *,
    labels: Labels,
    frame_payloads: Mapping[int, dict[str, Any]],
    frame_indices: Sequence[int] | None = None,
    track_map: object | None = None,
) -> PredictionAttachmentTransaction:
    resolved_frame_indices = tuple(
        resolve_attach_frame_indices(
            frame_payloads=frame_payloads,
            frame_indices=frame_indices,
        )
    )
    normalized_payloads = {
        int(frame_idx): dict(payload) for frame_idx, payload in frame_payloads.items()
    }
    track_map_before_attachment = coerce_track_map(track_map)
    return PredictionAttachmentTransaction(
        frame_payloads=normalized_payloads,
        frame_indices=resolved_frame_indices,
        track_map_before_attachment=dict(track_map_before_attachment),
        current_track_map=dict(track_map_before_attachment),
        snapshot_builder=begin_prediction_frame_snapshot(
            labels=labels,
            frame_indices=resolved_frame_indices,
        ),
    )


def advance_replacing_transaction_snapshot(
    *,
    labels: Labels,
    video: Any,
    transaction: PredictionAttachmentTransaction,
    max_items: int,
) -> bool:
    if transaction.snapshot_builder is None:
        raise RuntimeError(
            "Prediction attachment transaction must own a snapshot builder before snapshot capture."
        )
    done = advance_prediction_frame_snapshot(
        labels=labels,
        video=video,
        snapshot_builder=transaction.snapshot_builder,
        max_items=max_items,
    )
    if not done:
        return False
    transaction.rollback_snapshot = finish_prediction_frame_snapshot(
        video=video,
        snapshot_builder=transaction.snapshot_builder,
    )
    transaction.snapshot_builder = None
    return True


def prepare_replacing_transaction(
    *,
    transaction: PredictionAttachmentTransaction,
) -> PredictionAttachmentTransaction:
    if transaction.rollback_snapshot is None:
        raise RuntimeError("Prediction rollback snapshot must exist before attachment planning.")
    transaction.prepared_attachment = prepare_replacing_attachment(
        requested_frames=transaction.frame_indices,
        frame_payloads=transaction.frame_payloads,
        frame_indices=transaction.frame_indices,
    )
    transaction.current_track_map = dict(transaction.track_map_before_attachment)
    transaction.next_frame_offset = 0
    transaction.total_added = 0
    transaction.restored = False
    return transaction


def apply_replacing_transaction_slice(
    *,
    labels: Labels,
    video: Any,
    transaction: PredictionAttachmentTransaction,
    skeleton: Any,
    conf_thresh: float,
    max_frames: int,
) -> PredictionApplyResult:
    if transaction.prepared_attachment is None:
        raise RuntimeError(
            "Prediction attachment transaction must be prepared before applying slices."
        )
    result = apply_prepared_attachment(
        labels=labels,
        video=video,
        attachment=transaction.prepared_attachment,
        skeleton=skeleton,
        conf_thresh=conf_thresh,
        track_map=transaction.current_track_map,
        start_idx=transaction.next_frame_offset,
        max_frames=max_frames,
    )
    transaction.current_track_map = result.attach_result.updated_track_map
    transaction.next_frame_offset = result.next_frame_offset
    transaction.total_added += result.attach_result.attached_instance_count
    return result


def restore_replacing_transaction(
    *,
    labels: Labels,
    transaction: PredictionAttachmentTransaction,
) -> dict[int, Track]:
    if transaction.rollback_snapshot is None:
        raise RuntimeError("Prediction rollback snapshot must exist before restore.")
    if not transaction.restored:
        restore_prediction_frames(labels=labels, snapshot=transaction.rollback_snapshot)
        transaction.restored = True
    transaction.current_track_map = dict(transaction.track_map_before_attachment)
    transaction.next_frame_offset = 0
    transaction.total_added = 0
    return dict(transaction.current_track_map)


def prepare_attachment(
    *,
    labels: Labels,
    video: Any,
    requested_frames: Iterable[int],
    frame_payloads: Mapping[int, dict[str, Any]],
    frame_indices: Sequence[int] | None = None,
    clear_frame_indices: Sequence[int] | None = None,
    track_ids: Sequence[int] | None = None,
) -> PreparedPredictionAttachment:
    resolved_frame_indices = tuple(
        resolve_attach_frame_indices(
            frame_payloads=frame_payloads,
            frame_indices=frame_indices,
        )
    )
    resolved_clear_frame_indices = resolved_frame_indices
    if clear_frame_indices is not None:
        resolved_clear_frame_indices = tuple(int(frame_idx) for frame_idx in clear_frame_indices)
    validate_prediction_payloads(
        requested_frames=requested_frames,
        frame_payloads=frame_payloads,
    )
    clear_result = clear_predictions(
        labels=labels,
        video=video,
        frame_indices=resolved_clear_frame_indices,
        track_ids=track_ids,
    )
    if not clear_result.user_annotation_unchanged:
        raise RuntimeError(
            "Prediction attach aborted: clearing predicted instances modified user annotations."
        )
    return PreparedPredictionAttachment(
        frame_payloads={
            int(frame_idx): dict(payload) for frame_idx, payload in frame_payloads.items()
        },
        frame_indices=resolved_frame_indices,
        clear_result=clear_result,
    )


def prepare_replacing_attachment(
    *,
    requested_frames: Iterable[int],
    frame_payloads: Mapping[int, dict[str, Any]],
    frame_indices: Sequence[int] | None = None,
    track_ids: Sequence[int] | None = None,
) -> PreparedPredictionAttachment:
    resolved_frame_indices = tuple(
        resolve_attach_frame_indices(
            frame_payloads=frame_payloads,
            frame_indices=frame_indices,
        )
    )
    validate_prediction_payloads(
        requested_frames=requested_frames,
        frame_payloads=frame_payloads,
    )
    replace_track_ids = None
    if track_ids is not None:
        replace_track_ids = tuple(int(track_id) for track_id in track_ids)
    return PreparedPredictionAttachment(
        frame_payloads={
            int(frame_idx): dict(payload) for frame_idx, payload in frame_payloads.items()
        },
        frame_indices=resolved_frame_indices,
        clear_result=PredictionClearResult(
            cleared_frame_indices=(),
            removed_instance_count=0,
            user_annotation_unchanged=True,
        ),
        replace_existing_predictions=True,
        replace_track_ids=replace_track_ids,
    )


def attach_predictions(
    *,
    labels: Labels,
    video: Any,
    frame_payloads: Mapping[int, dict[str, Any]],
    skeleton: Any,
    conf_thresh: float,
    track_map: Mapping[int, Track] | None = None,
    frame_indices: Sequence[int] | None = None,
    replace_existing_predictions: bool = False,
    replace_track_ids: Sequence[int] | None = None,
) -> PredictionAttachResult:
    sorted_frames = resolve_attach_frame_indices(
        frame_payloads=frame_payloads,
        frame_indices=frame_indices,
    )
    updated_track_map = dict(coerce_track_map(track_map))
    total_added = 0
    normalized_replace_track_ids = (
        None if replace_track_ids is None else {int(track_id) for track_id in replace_track_ids}
    )
    for frame_idx in sorted_frames:
        labeled_frame = ensure_labeled_frame(labels=labels, video=video, frame_idx=frame_idx)
        if replace_existing_predictions:
            remove_matching_predictions_from_frame(
                labels=labels,
                labeled_frame=labeled_frame,
                replace_track_ids=normalized_replace_track_ids,
            )
        payload = frame_payloads.get(frame_idx) or {}
        heatmaps = payload.get("heatmaps")
        if heatmaps is not None:
            labeled_frame.heatmaps = heatmaps
        preds = make_predicted_instances(
            skeleton,
            labeled_frame,
            list(payload.get("instances") or []),
            conf_thresh=conf_thresh,
        )
        updated_track_map = normalize_prediction_tracks(
            labels=labels,
            labeled_frame=labeled_frame,
            preds=preds,
            track_map=updated_track_map,
        )
        pred_instances: list[Instance] = list(preds)
        labels.add_predicted_instances(video, int(frame_idx), pred_instances)
        total_added += len(preds)
    return PredictionAttachResult(
        attached_frame_indices=tuple(sorted_frames),
        attached_instance_count=total_added,
        updated_track_map=updated_track_map,
    )


def apply_prepared_attachment(
    *,
    labels: Labels,
    video: Any,
    attachment: PreparedPredictionAttachment,
    skeleton: Any,
    conf_thresh: float,
    track_map: Mapping[int, Track] | None = None,
    start_idx: int = 0,
    max_frames: int | None = None,
) -> PredictionApplyResult:
    total_frames = len(attachment.frame_indices)
    end_idx = resolve_apply_end_idx(
        total_frames=total_frames,
        start_idx=start_idx,
        max_frames=max_frames,
    )
    frame_slice = attachment.frame_indices[start_idx:end_idx]
    attach_result = attach_predictions(
        labels=labels,
        video=video,
        frame_payloads=attachment.frame_payloads,
        skeleton=skeleton,
        conf_thresh=conf_thresh,
        track_map=track_map,
        frame_indices=frame_slice,
        replace_existing_predictions=attachment.replace_existing_predictions,
        replace_track_ids=attachment.replace_track_ids,
    )
    return PredictionApplyResult(
        clear_result=attachment.clear_result,
        attach_result=attach_result,
        next_frame_offset=end_idx,
        total_frames=total_frames,
    )


def apply_attachment(
    *,
    labels: Labels,
    video: Any,
    requested_frames: Iterable[int],
    frame_payloads: Mapping[int, dict[str, Any]],
    skeleton: Any,
    conf_thresh: float,
    track_map: Mapping[int, Track] | None = None,
    frame_indices: Sequence[int] | None = None,
    clear_frame_indices: Sequence[int] | None = None,
    track_ids: Sequence[int] | None = None,
) -> PredictionApplyResult:
    attachment = prepare_attachment(
        labels=labels,
        video=video,
        requested_frames=requested_frames,
        frame_payloads=frame_payloads,
        frame_indices=frame_indices,
        clear_frame_indices=clear_frame_indices,
        track_ids=track_ids,
    )
    return apply_prepared_attachment(
        labels=labels,
        video=video,
        attachment=attachment,
        skeleton=skeleton,
        conf_thresh=conf_thresh,
        track_map=track_map,
    )


def prediction_frame_payloads_by_video_index(
    *,
    predictions_payload: Mapping[str, object],
) -> dict[int, dict[int, dict[str, Any]]]:
    """Convert serialized prediction arrays into video-indexed frame payloads."""
    frames_info, data_info, attrs = _payload_sections(predictions_payload)
    video_indices = _require_vector(
        _required_array(
            frames_info,
            "video_index",
            dtype=np.int32,
            label="predictions.frames.video_index",
        ),
        label="predictions.frames.video_index",
    )
    frame_indices = _require_vector(
        _required_array(
            frames_info,
            "frame_index",
            dtype=np.int32,
            label="predictions.frames.frame_index",
        ),
        label="predictions.frames.frame_index",
    )
    num_instances = _require_vector(
        _required_array(
            frames_info,
            "num_instances",
            dtype=np.int32,
            label="predictions.frames.num_instances",
        ),
        label="predictions.frames.num_instances",
    )
    keypoints_arr = _normalize_keypoints_array(data_info)
    if len(video_indices) == 0 or keypoints_arr.size == 0:
        return {}

    dataset_length = _validate_prediction_rows(
        video_indices=video_indices,
        frame_indices=frame_indices,
        num_instances=num_instances,
        keypoints_arr=keypoints_arr,
    )
    instance_scores = _optional_array(data_info, "instance_score", dtype=np.float32)
    track_ids = _optional_array(data_info, "track_id", dtype=np.int32)
    _validate_optional_row_count(
        instance_scores,
        label="predictions.data.instance_score",
        row_count=dataset_length,
    )
    _validate_optional_row_count(
        track_ids,
        label="predictions.data.track_id",
        row_count=dataset_length,
    )
    heatmaps_arr = data_info.get("heatmaps")
    heatmaps = None if heatmaps_arr is None else np.asarray(heatmaps_arr)
    committed_length = _coerce_committed_length(attrs, dataset_length=dataset_length)
    payloads_by_video: dict[int, dict[int, dict[str, Any]]] = {}
    for row_idx in range(committed_length):
        vid_idx = int(video_indices[row_idx])
        frame_idx = int(frame_indices[row_idx])
        frame_payload = payloads_by_video.setdefault(vid_idx, {}).setdefault(
            frame_idx,
            {"instances": [], "heatmaps": None},
        )
        frame_payload["heatmaps"] = _row_heatmaps(heatmaps, row_idx)
        instance_count = int(num_instances[row_idx])
        if instance_count < 0:
            raise ValueError("predictions.frames.num_instances cannot contain negative values.")
        if instance_count > int(keypoints_arr.shape[1]):
            raise ValueError(
                f"Prediction row {row_idx} declares {instance_count} instances but "
                f"keypoints stores {int(keypoints_arr.shape[1])}."
            )
        for inst_idx in range(instance_count):
            frame_payload["instances"].append(
                _instance_payload(
                    keypoints_arr=keypoints_arr,
                    instance_scores=instance_scores,
                    track_ids=track_ids,
                    row_idx=row_idx,
                    inst_idx=inst_idx,
                )
            )
    return payloads_by_video


def prediction_payloads_by_video(
    *,
    labels: Labels,
    predictions_payload: Mapping[str, object],
) -> dict[Any, dict[int, dict[str, Any]]]:
    """Convert serialized prediction arrays into video/frame attachment payloads."""
    indexed_payloads = prediction_frame_payloads_by_video_index(
        predictions_payload=predictions_payload,
    )
    videos = list(labels.videos)
    payloads_by_video: dict[Any, dict[int, dict[str, Any]]] = {}
    for vid_idx, frame_payloads in sorted(indexed_payloads.items()):
        if vid_idx < 0 or vid_idx >= len(videos):
            raise ValueError(
                f"Prediction payload references unknown video index {vid_idx}; "
                f"labels contain {len(videos)} videos."
            )
        payloads_by_video[videos[vid_idx]] = frame_payloads
    return payloads_by_video


def hydrate_predictions(
    labels: Labels,
    predictions_payload: Mapping[str, object] | None,
    *,
    logger: Logger,
) -> None:
    """Hydrate predictions from a serialized payload into a ``Labels`` object."""
    if not predictions_payload:
        return
    skeleton = labels.skeleton
    if skeleton is None:
        raise ValueError("Prediction hydration requires labels.skeleton.")
    payloads_by_video = prediction_payloads_by_video(
        labels=labels,
        predictions_payload=predictions_payload,
    )
    if not payloads_by_video:
        return

    track_map = _initial_track_map(labels)
    total_frames = 0
    for video, frame_payloads in payloads_by_video.items():
        attach_result = attach_predictions(
            labels=labels,
            video=video,
            frame_payloads=frame_payloads,
            skeleton=skeleton,
            conf_thresh=0.0,
            track_map=track_map,
        )
        track_map = attach_result.updated_track_map
        total_frames += len(attach_result.attached_frame_indices)

    logger.debug("Hydrated %d prediction frames into Labels", total_frames)


def ensure_labeled_frame(*, labels: Labels, video: Any, frame_idx: int) -> LabeledFrame:
    labeled_frame = labels.query.find_first(video, int(frame_idx), use_cache=True)
    if labeled_frame is not None:
        return labeled_frame
    labeled_frame = LabeledFrame(video=video, frame_idx=int(frame_idx))
    labels.append(labeled_frame)
    return labeled_frame


def remove_matching_predictions_from_frame(
    *,
    labels: Labels,
    labeled_frame: LabeledFrame,
    replace_track_ids: set[int] | None,
) -> None:
    removable_predictions: list[PredictedInstance] = []
    for inst in tuple(labeled_frame.instances):
        if not isinstance(inst, PredictedInstance):
            continue
        if replace_track_ids is not None:
            track = inst.track
            track_id = int(track.id) if isinstance(track, Track) else None
            if track_id not in replace_track_ids:
                continue
        removable_predictions.append(inst)

    for prediction in removable_predictions:
        labels.remove_instance(labeled_frame, prediction)


def canonical_video_for_labels(*, labels: Labels, video: Any) -> Any:
    if video in labels.videos:
        return video
    requested_key = resolved_video_cache_key(video)
    if requested_key is None:
        return video
    for candidate in labels.videos:
        if resolved_video_cache_key(candidate) == requested_key:
            return candidate
    return video


def resolve_clear_target_frames(
    *,
    labels: Labels,
    video: Any | None,
    frame_indices: Sequence[int] | None,
) -> tuple[LabeledFrame, ...]:
    if video is None:
        if frame_indices is None:
            return tuple(labels.labeled_frames)
        target_frame_indices = {int(frame_idx) for frame_idx in frame_indices}
        return tuple(
            labeled_frame
            for labeled_frame in labels.labeled_frames
            if int(labeled_frame.frame_idx) in target_frame_indices
        )

    canonical_video = canonical_video_for_labels(labels=labels, video=video)
    if frame_indices is None:
        return tuple(labels.query.find(canonical_video))

    seen_frame_indices: set[int] = set()
    target_frames: list[LabeledFrame] = []
    for raw_frame_idx in frame_indices:
        frame_idx = int(raw_frame_idx)
        if frame_idx in seen_frame_indices:
            continue
        seen_frame_indices.add(frame_idx)
        labeled_frame = labels.query.find_first(canonical_video, frame_idx, use_cache=True)
        if labeled_frame is not None:
            target_frames.append(labeled_frame)
    return tuple(target_frames)


def clear_matching_predictions_from_frame(
    *,
    labels: Labels,
    labeled_frame: LabeledFrame,
    target_track_ids: set[int] | None,
) -> int:
    removable_predictions: list[PredictedInstance] = []
    for inst in tuple(labeled_frame.instances):
        if not isinstance(inst, PredictedInstance):
            continue
        if target_track_ids is not None:
            track = inst.track
            track_id = int(track.id) if isinstance(track, Track) else None
            if track_id not in target_track_ids:
                continue
        removable_predictions.append(inst)

    if not removable_predictions:
        return 0

    for prediction in removable_predictions:
        labels.remove_instance(labeled_frame, prediction)
    if not labeled_frame.instances:
        labels.remove_frame(labeled_frame)
    return len(removable_predictions)


def resolve_snapshot_frame_indices(frame_indices: Sequence[int]) -> tuple[int, ...]:
    seen_frame_indices: set[int] = set()
    resolved_frame_indices: list[int] = []
    for raw_frame_idx in frame_indices:
        frame_idx = int(raw_frame_idx)
        if frame_idx in seen_frame_indices:
            continue
        seen_frame_indices.add(frame_idx)
        resolved_frame_indices.append(frame_idx)
    return tuple(resolved_frame_indices)


def append_snapshot_track(
    *,
    snapshot_builder: PredictionFrameSnapshotBuilder,
    track: Track,
) -> None:
    track_id = int(track.id)
    if track_id in snapshot_builder.tracks_by_id:
        return
    snapshot_builder.tracks_by_id[track_id] = track
    snapshot_builder.ordered_tracks.append(track)


def materialize_restored_frame(
    *,
    frame_state: LabeledFrame | None,
    canonical_video: Any,
    canonical_tracks: list[Track],
    canonical_by_id: dict[int, Track],
) -> LabeledFrame | None:
    if frame_state is None:
        return None
    restored_frame = clone_labeled_frame_for_snapshot(frame_state)
    restored_frame.video = canonical_video
    for inst in restored_frame.instances:
        track = inst.track
        if not isinstance(track, Track):
            continue
        track_id = int(track.id)
        canonical = canonical_by_id.get(track_id)
        if canonical is None:
            canonical = track
            canonical_by_id[track_id] = canonical
            canonical_tracks.append(canonical)
        inst.track = canonical
    validate_labeled_frame_contract(restored_frame)
    return restored_frame


def restore_frame_contents(
    *,
    labels: Labels,
    target_frame: LabeledFrame,
    restored_frame: LabeledFrame,
) -> None:
    for inst in tuple(target_frame.instances):
        labels.remove_instance(target_frame, inst)
    target_frame.heatmaps = None
    if restored_frame.heatmaps is not None:
        target_frame.heatmaps = np.array(restored_frame.heatmaps, copy=True)
    for inst in restored_frame.instances:
        labels.add_instance(target_frame, inst)
    validate_labeled_frame_contract(target_frame)


def append_restored_frame(
    *,
    labels: Labels,
    restored_frame: LabeledFrame,
) -> None:
    labels.labeled_frames.append(restored_frame)
    labels._update_containers(restored_frame)
    validate_labeled_frame_contract(restored_frame)


def drop_frame(
    *,
    labels: Labels,
    labeled_frame: LabeledFrame,
) -> None:
    for inst in tuple(labeled_frame.instances):
        labels.remove_instance(labeled_frame, inst)
    labels.remove_frame(labeled_frame)


def validate_labeled_frame_contract(labeled_frame: LabeledFrame) -> None:
    if not isinstance(labeled_frame, LabeledFrame):
        raise TypeError(
            "Labels.labeled_frames entries must be LabeledFrame objects; "
            f"got {type(labeled_frame).__name__}"
        )
    instances = labeled_frame.instances
    if instances.labeled_frame is not labeled_frame:
        raise ValueError("InstancesList.labeled_frame must reference its owner")
    frame_tracks: set[tuple[int | None, str]] = set()
    for inst in instances:
        if inst.frame is not labeled_frame:
            raise ValueError("Instance.frame must reference owning LabeledFrame")
        inst._assert_points_synced()
        track = inst.track
        if track is None:
            continue
        track_key = (track.spawned_on, track.name)
        if track_key in frame_tracks:
            raise ValueError(
                f"Duplicate track assignment for frame {labeled_frame.frame_idx}: {track.name}"
            )
        frame_tracks.add(track_key)


def clone_labeled_frame_for_snapshot(labeled_frame: LabeledFrame) -> LabeledFrame:
    """Clone a frame for rollback without depending on serializer round-trips."""
    clone = LabeledFrame(
        video=labeled_frame.video,
        frame_idx=int(labeled_frame.frame_idx),
        instances=[clone_instance_for_snapshot(inst) for inst in labeled_frame.instances],
        masks=list(labeled_frame.masks),
        rois=list(labeled_frame.rois),
    )
    if labeled_frame.heatmaps is not None:
        clone.heatmaps = np.array(labeled_frame.heatmaps, copy=True)
    validate_labeled_frame_contract(clone)
    return clone


def clone_instance_for_snapshot(instance: Instance) -> Instance:
    if isinstance(instance, PredictedInstance):
        init_points = _copy_predicted_point_array(instance.point_records(copy=False))
        return PredictedInstance(
            skeleton=instance.skeleton,
            track=instance.track,
            init_points=init_points,
            tracking_score=float(instance.tracking_score),
            score=float(instance.score),
        )

    init_points = _copy_point_array(instance.point_records(copy=False))
    return Instance(
        skeleton=instance.skeleton,
        track=instance.track,
        from_predicted=instance.from_predicted,
        init_points=init_points,
        tracking_score=float(instance.tracking_score),
    )


def _copy_predicted_point_array(points: np.ndarray) -> PredictedPointArray:
    copied = cast(PredictedPointArray, PredictedPointArray.make_default(len(points)))
    copied["x"] = points["x"]
    copied["y"] = points["y"]
    copied["visible"] = points["visible"]
    copied["complete"] = points["complete"]
    copied["score"] = points["score"]
    copied["flags"] = points["flags"]
    return copied


def _copy_point_array(points: np.ndarray) -> PointArray:
    copied = PointArray.make_default(len(points))
    copied["x"] = points["x"]
    copied["y"] = points["y"]
    copied["visible"] = points["visible"]
    copied["complete"] = points["complete"]
    copied["flags"] = points["flags"]
    return copied


def resolve_attach_frame_indices(
    *,
    frame_payloads: Mapping[int, dict[str, Any]],
    frame_indices: Sequence[int] | None,
) -> list[int]:
    if frame_indices is None:
        return sorted(int(frame_idx) for frame_idx in frame_payloads)
    return [int(frame_idx) for frame_idx in frame_indices]


def resolve_apply_end_idx(*, total_frames: int, start_idx: int, max_frames: int | None) -> int:
    if start_idx < 0:
        raise ValueError("start_idx must be >= 0")
    if max_frames is None:
        return total_frames
    if max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    return min(total_frames, start_idx + max_frames)


def resolve_prediction_frame(
    *,
    predicted_instance: PredictedInstance,
    labeled_frame: LabeledFrame | None,
) -> LabeledFrame:
    frame = predicted_instance.frame if predicted_instance.frame is not None else labeled_frame
    if frame is None:
        raise ValueError("Prediction acceptance requires a labeled frame.")
    return frame


def build_user_instance_from_prediction(
    *,
    predicted_instance: PredictedInstance,
    labeled_frame: LabeledFrame,
    force_visible: bool,
) -> Instance:
    user_instance = Instance(
        skeleton=predicted_instance.skeleton,
        from_predicted=predicted_instance,
        frame=labeled_frame,
    )
    for keypoint_name in user_instance.skeleton.keypoint_names:
        if keypoint_name not in predicted_instance:
            continue
        point = predicted_instance[keypoint_name]
        user_instance[keypoint_name] = Point(
            x=float(point.x),
            y=float(point.y),
            visible=True if force_visible else bool(point.visible),
            complete=False,
        )
    user_instance.track = predicted_instance.track
    return user_instance


def assert_frame_has_no_user_instances(*, labeled_frame: LabeledFrame) -> None:
    user_instances = tuple(labeled_frame.user_instances)
    if not user_instances:
        return
    raise ValueError(
        "User-label apply requires empty target frames; "
        f"frame {int(labeled_frame.frame_idx)} already has {len(user_instances)} user instances."
    )


def create_user_instances_from_payload(
    *,
    labels: Labels,
    labeled_frame: LabeledFrame,
    skeleton: Any,
    payload: dict[str, Any] | None,
    track_map: dict[int, Track],
) -> tuple[tuple[Instance, ...], dict[int, Track]]:
    if payload is None:
        raise RuntimeError(
            f"User-label apply missing payload for frame {int(labeled_frame.frame_idx)}."
        )
    preds = make_predicted_instances(
        skeleton,
        labeled_frame,
        list(payload.get("instances") or []),
        conf_thresh=0.0,
    )
    if not preds:
        raise ValueError(
            "User-label apply requires at least one instance for frame "
            f"{int(labeled_frame.frame_idx)}."
        )
    updated_track_map = normalize_prediction_tracks(
        labels=labels,
        labeled_frame=labeled_frame,
        preds=preds,
        track_map=track_map,
    )
    created_users = tuple(
        build_user_instance_from_prediction(
            predicted_instance=predicted_instance,
            labeled_frame=labeled_frame,
            force_visible=False,
        )
        for predicted_instance in preds
    )
    for user_instance in created_users:
        labels.add_instance(labeled_frame, user_instance)
    return created_users, updated_track_map


def _predicted_points_from_payload(pts: object, *, conf_thresh: float) -> PredictedPointArray:
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Prediction keypoints must be a 2D array, got shape {arr.shape}.")

    point_count = int(arr.shape[0])
    points = cast(PredictedPointArray, PredictedPointArray.make_default(point_count))
    x = arr[:, 0] if arr.shape[1] >= 1 else np.full(point_count, np.nan, dtype=np.float64)
    y = arr[:, 1] if arr.shape[1] >= 2 else np.full(point_count, np.nan, dtype=np.float64)
    score = arr[:, 2] if arr.shape[1] >= 3 else np.zeros(point_count, dtype=np.float64)
    points["x"] = x
    points["y"] = y
    points["score"] = score
    points["visible"] = (score >= float(conf_thresh)) & np.isfinite(x) & np.isfinite(y)
    points["complete"] = False
    points["flags"] = 0
    return points


def _frame_has_points(insts: object) -> bool:
    if not isinstance(insts, list):
        return False
    for raw in insts:
        if not isinstance(raw, Mapping):
            continue
        raw_mapping = cast(Mapping[str, object], raw)
        arr = np.asarray(raw_mapping.get("keypoints", []), dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2 and np.isfinite(arr[:, :2]).any():
            return True
    return False


def _payload_sections(
    predictions_payload: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    frames_raw = _required_mapping(predictions_payload, "frames", label="predictions.frames")
    data_raw = _required_mapping(predictions_payload, "data", label="predictions.data")
    attrs_raw = predictions_payload.get("attrs")
    if attrs_raw is None:
        attrs: dict[str, object] = {}
    elif isinstance(attrs_raw, Mapping):
        attrs = dict(attrs_raw)
    else:
        raise TypeError("predictions.attrs must be a mapping when provided.")
    frames_info = dict(frames_raw)
    data_info = dict(data_raw)
    return frames_info, data_info, attrs


def _required_mapping(data: Mapping[str, object], key: str, *, label: str) -> Mapping[str, object]:
    raw = data.get(key)
    if not isinstance(raw, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    return cast(Mapping[str, object], raw)


def _required_array(data: Mapping[str, object], key: str, *, dtype: Any, label: str) -> np.ndarray:
    if key not in data:
        raise ValueError(f"{label} is required.")
    return np.asarray(data[key], dtype=dtype)


def _optional_array(data: Mapping[str, object], key: str, *, dtype: Any) -> np.ndarray:
    if key not in data:
        return np.asarray([], dtype=dtype)
    return np.asarray(data[key], dtype=dtype)


def _require_vector(arr: np.ndarray, *, label: str) -> np.ndarray:
    if arr.ndim != 1:
        raise ValueError(f"{label} must be a 1D array, got shape {arr.shape}.")
    return arr


def _coerce_committed_length_value(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("attrs['committed_length'] must be a non-negative integer.")
    if isinstance(value, int):
        committed_length = value
    elif isinstance(value, str):
        text = value.strip()
        if not text.isdigit():
            raise ValueError("attrs['committed_length'] must be a non-negative integer.")
        committed_length = int(text)
    else:
        raise TypeError("attrs['committed_length'] must be a non-negative integer.")
    if committed_length < 0:
        raise ValueError("attrs['committed_length'] must be a non-negative integer.")
    return committed_length


def _coerce_committed_length(attrs: Mapping[str, object], *, dataset_length: int) -> int:
    committed_length_raw = attrs.get("committed_length")
    if committed_length_raw is None:
        committed_length = dataset_length
    else:
        committed_length = _coerce_committed_length_value(committed_length_raw)
    if committed_length > dataset_length:
        raise ValueError(
            f"Committed length {committed_length} exceeds dataset length {dataset_length}"
        )
    return int(committed_length)


def _normalize_keypoints_array(data_info: Mapping[str, object]) -> np.ndarray:
    keypoints_arr = _required_array(
        data_info,
        "keypoints",
        dtype=np.float32,
        label="predictions.data.keypoints",
    )
    if keypoints_arr.size == 0:
        return keypoints_arr
    if keypoints_arr.ndim == 4:
        return keypoints_arr
    if keypoints_arr.ndim == 3:
        return keypoints_arr[:, np.newaxis, :, :]
    if keypoints_arr.ndim == 2:
        return keypoints_arr[np.newaxis, np.newaxis, :, :]
    raise ValueError(f"predictions.data.keypoints must have ndim >= 2, got {keypoints_arr.ndim}")


def _validate_prediction_rows(
    *,
    video_indices: np.ndarray,
    frame_indices: np.ndarray,
    num_instances: np.ndarray,
    keypoints_arr: np.ndarray,
) -> int:
    row_count = int(video_indices.shape[0])
    expected = {
        "predictions.frames.frame_index": int(frame_indices.shape[0]),
        "predictions.frames.num_instances": int(num_instances.shape[0]),
        "predictions.data.keypoints": int(keypoints_arr.shape[0]),
    }
    mismatched = [f"{label}={count}" for label, count in expected.items() if count != row_count]
    if mismatched:
        joined = ", ".join(mismatched)
        raise ValueError(
            "Prediction payload row counts must match "
            f"predictions.frames.video_index={row_count}; got {joined}."
        )
    return row_count


def _validate_optional_row_count(arr: np.ndarray, *, label: str, row_count: int) -> None:
    if arr.size == 0:
        return
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 2D array when provided, got shape {arr.shape}.")
    if int(arr.shape[0]) != row_count:
        raise ValueError(f"{label} must have {row_count} rows, got {int(arr.shape[0])}.")


def _row_heatmaps(heatmaps: np.ndarray | None, row_idx: int) -> Any:
    if heatmaps is None:
        return None
    if heatmaps.ndim == 4 and row_idx < int(heatmaps.shape[0]):
        return heatmaps[row_idx]
    if heatmaps.ndim == 3 and row_idx == 0:
        return heatmaps
    return None


def _instance_payload(
    *,
    keypoints_arr: np.ndarray,
    instance_scores: np.ndarray,
    track_ids: np.ndarray,
    row_idx: int,
    inst_idx: int,
) -> dict[str, Any]:
    keypoints = [
        [
            float(row[0]) if len(row) >= 1 else float("nan"),
            float(row[1]) if len(row) >= 2 else float("nan"),
            float(row[2]) if len(row) >= 3 else 0.0,
        ]
        for row in np.asarray(keypoints_arr[row_idx, inst_idx], dtype=np.float32).tolist()
    ]
    payload: dict[str, Any] = {"keypoints": keypoints}
    if instance_scores.size > 0 and row_idx < int(instance_scores.shape[0]):
        if instance_scores.ndim == 2 and inst_idx < int(instance_scores.shape[1]):
            payload["score"] = float(instance_scores[row_idx, inst_idx])
    if track_ids.size > 0 and row_idx < int(track_ids.shape[0]):
        if track_ids.ndim == 2 and inst_idx < int(track_ids.shape[1]):
            payload["track_id"] = int(track_ids[row_idx, inst_idx])
    return payload


def _initial_track_map(labels: Labels) -> dict[int, Track]:
    track_map: dict[int, Track] = {}
    for track in labels.tracks:
        track_id = int(track.id)
        if track_id >= 0:
            track_map[track_id] = track
    return track_map


__all__ = [
    "PredictionApplyResult",
    "PredictionAttachResult",
    "PredictionAttachmentTransaction",
    "PredictionClearResult",
    "PredictionFrameSnapshot",
    "PredictionFrameSnapshotBuilder",
    "PredictionPayloadPartition",
    "PredictionUserInstanceResult",
    "PredictionUserPayloadApplyResult",
    "PreparedPredictionAttachment",
    "accept_prediction",
    "advance_prediction_frame_snapshot",
    "advance_replacing_transaction_snapshot",
    "apply_attachment",
    "apply_prepared_attachment",
    "apply_replacing_transaction_slice",
    "apply_user_instances_from_payload_slice",
    "attach_predictions",
    "begin_prediction_frame_snapshot",
    "begin_replacing_transaction",
    "clear_predictions",
    "coerce_track_map",
    "create_user_instances_from_predictions",
    "finish_prediction_frame_snapshot",
    "frames_without_keypoints",
    "hydrate_predictions",
    "make_predicted_instances",
    "missing_prediction_frames",
    "normalize_prediction_tracks",
    "partition_attachable_payloads",
    "prediction_frame_payloads_by_video_index",
    "prediction_payloads_by_video",
    "prepare_attachment",
    "prepare_replacing_attachment",
    "prepare_replacing_transaction",
    "resolved_video_cache_key",
    "restore_prediction_frames",
    "restore_replacing_transaction",
    "snapshot_prediction_frames",
    "validate_prediction_payloads",
]
