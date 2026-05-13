from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from xpkg.io.prediction_attachment import (
    accept_prediction,
    advance_replacing_transaction_snapshot,
    apply_attachment,
    apply_prepared_attachment,
    apply_replacing_transaction_slice,
    apply_user_instances_from_payload_slice,
    attach_predictions,
    begin_replacing_transaction,
    clear_predictions,
    coerce_track_map,
    create_user_instances_from_predictions,
    frames_without_keypoints,
    hydrate_predictions,
    make_predicted_instances,
    missing_prediction_frames,
    normalize_prediction_tracks,
    partition_attachable_payloads,
    prediction_payloads_by_video,
    prepare_attachment,
    prepare_replacing_attachment,
    prepare_replacing_transaction,
    resolved_video_cache_key,
    restore_prediction_frames,
    restore_replacing_transaction,
    snapshot_prediction_frames,
    validate_prediction_payloads,
)
from xpkg.model import (
    Instance,
    LabeledFrame,
    Labels,
    Point,
    PredictedInstance,
    PredictedPoint,
    Track,
    VideoStub,
    build_keypoint_skeleton,
)


class _NoIterList(list[Any]):
    def __iter__(self):
        raise AssertionError("This test guards against iterating labels.labeled_frames.")


def _labels() -> Labels:
    video = VideoStub(filename="movie.mp4", frames=100, height=64, width=64)
    skeleton = build_keypoint_skeleton(["nose", "tail"], name="subject")
    return Labels(videos=[video], skeletons=[skeleton])


def _payload(*, x: float = 10.0, y: float = 20.0, score: float = 0.9) -> dict[str, object]:
    return {
        "instances": [
            {
                "keypoints": [[x, y, score], [x + 5.0, y + 5.0, score - 0.1]],
                "score": score,
                "track_id": 7,
            }
        ]
    }


def _user_instance(labels: Labels, *, x: float = 1.0, y: float = 2.0) -> Instance:
    return Instance(
        skeleton=labels.skeleton,
        init_points={
            "nose": Point(x=x, y=y, visible=True, complete=True),
            "tail": Point(x=x + 1.0, y=y + 1.0, visible=True, complete=True),
        },
    )


def _prediction_instance(labels: Labels, *, visible_tail: bool = True) -> PredictedInstance:
    prediction = PredictedInstance(skeleton=labels.skeleton)
    nose, tail = labels.skeleton.keypoints
    prediction[nose] = PredictedPoint(x=1.0, y=2.0, visible=True, score=0.9)
    prediction[tail] = PredictedPoint(x=3.0, y=4.0, visible=visible_tail, score=0.2)
    prediction.track = Track(spawned_on=1, name="track_1")
    return prediction


def test_prediction_payload_gap_helpers_report_missing_and_empty_frames() -> None:
    payloads = {
        0: {"instances": [{"keypoints": [[1.0, 2.0, 0.9], [3.0, 4.0, 0.8]]}]},
        1: {"instances": [{"keypoints": [[float("nan"), float("nan"), 0.0]]}]},
        3: {"instances": []},
    }

    assert missing_prediction_frames(requested_frames=(0, 1, 2, 3), frame_payloads=payloads) == [2]
    assert frames_without_keypoints(payloads) == [1, 3]


def test_normalize_prediction_tracks_reuses_canonical_label_tracks() -> None:
    labels = _labels()
    existing_track = Track(spawned_on=2, name="existing")
    labels.tracks.append(existing_track)
    frame = LabeledFrame(video=cast(Any, labels.video), frame_idx=0)
    prediction = _prediction_instance(labels)
    prediction.track = Track(spawned_on=2, name="incoming")

    track_map = normalize_prediction_tracks(
        labels=labels,
        labeled_frame=frame,
        preds=[prediction],
        track_map={2: existing_track},
    )

    assert labels.tracks == [existing_track]
    assert prediction.track is existing_track
    assert track_map[2] is existing_track


def test_resolved_video_cache_key_returns_resolved_posix_path_or_none(tmp_path: Path) -> None:
    video = VideoStub(
        filename=str(tmp_path / ".." / tmp_path.name / "clip.mp4"),
        frames=1,
        height=64,
        width=64,
    )
    blank_video = VideoStub(filename="   ", frames=1, height=64, width=64)

    assert resolved_video_cache_key(video) == (tmp_path / "clip.mp4").resolve().as_posix()
    assert resolved_video_cache_key(blank_video) is None


def test_attach_predictions_adds_predicted_instances_and_canonical_tracks() -> None:
    labels = _labels()
    video = labels.video

    result = attach_predictions(
        labels=labels,
        video=video,
        frame_payloads={3: _payload()},
        skeleton=labels.skeleton,
        conf_thresh=0.5,
    )

    frame = labels.query.find_first(video, 3, use_cache=True)
    assert frame is not None
    assert result.attached_frame_indices == (3,)
    assert result.attached_instance_count == 1
    assert tuple(result.updated_track_map) == (7,)
    assert len(labels.tracks) == 1
    assert len(frame.predicted_instances) == 1
    prediction = frame.predicted_instances[0]
    assert isinstance(prediction, PredictedInstance)
    assert prediction.track is result.updated_track_map[7]
    assert prediction.score == pytest.approx(0.9)
    assert np.allclose(prediction.xy_score_array(), [[10.0, 20.0, 0.9], [15.0, 25.0, 0.8]])


def test_clear_predictions_removes_only_predictions_and_keeps_user_annotations() -> None:
    labels = _labels()
    video = labels.video
    frame = LabeledFrame(video=cast(Any, video), frame_idx=5, instances=[_user_instance(labels)])
    labels.append(frame)
    attach_predictions(
        labels=labels,
        video=video,
        frame_payloads={5: _payload()},
        skeleton=labels.skeleton,
        conf_thresh=0.0,
    )

    result = clear_predictions(labels=labels, video=video, frame_indices=[5])

    remaining_frame = labels.query.find_first(video, 5, use_cache=True)
    assert remaining_frame is frame
    assert result.cleared_frame_indices == (5,)
    assert result.removed_instance_count == 1
    assert result.user_annotation_unchanged is True
    assert len(frame.user_instances) == 1
    assert frame.predicted_instances == []


def test_accept_prediction_converts_prediction_to_user_instance_and_removes_prediction() -> None:
    labels = _labels()
    video = labels.video
    attach_predictions(
        labels=labels,
        video=video,
        frame_payloads={8: _payload(x=30.0, y=40.0)},
        skeleton=labels.skeleton,
        conf_thresh=0.0,
    )
    frame = labels.query.find_first(video, 8, use_cache=True)
    assert frame is not None
    prediction = frame.predicted_instances[0]

    result = accept_prediction(labels=labels, predicted_instance=prediction)

    assert result.frame_idx == 8
    assert result.created_user_instance_count == 1
    assert result.removed_predicted_count == 1
    assert frame.predicted_instances == []
    assert len(frame.user_instances) == 1
    user_instance = frame.user_instances[0]
    assert user_instance.from_predicted is prediction
    assert user_instance.track is prediction.track
    assert np.allclose(user_instance.xy_array(), [[30.0, 40.0], [35.0, 45.0]])


def test_prepare_and_apply_attachment_supports_slice_progression() -> None:
    labels = _labels()
    video = labels.video

    attachment = prepare_attachment(
        labels=labels,
        video=video,
        requested_frames=(1, 3),
        frame_payloads={1: _payload(x=10.0), 3: _payload(x=30.0)},
        frame_indices=(1, 3),
    )
    first_slice = apply_prepared_attachment(
        labels=labels,
        video=video,
        attachment=attachment,
        skeleton=labels.skeleton,
        conf_thresh=0.0,
        track_map={},
        start_idx=0,
        max_frames=1,
    )
    second_slice = apply_prepared_attachment(
        labels=labels,
        video=video,
        attachment=attachment,
        skeleton=labels.skeleton,
        conf_thresh=0.0,
        track_map=first_slice.attach_result.updated_track_map,
        start_idx=first_slice.next_frame_offset,
        max_frames=1,
    )

    assert first_slice.attach_result.attached_frame_indices == (1,)
    assert first_slice.next_frame_offset == 1
    assert first_slice.done is False
    assert second_slice.attach_result.attached_frame_indices == (3,)
    assert second_slice.done is True
    assert labels.query.find_first(video, 1, use_cache=True) is not None
    assert labels.query.find_first(video, 3, use_cache=True) is not None


def test_prepare_replacing_attachment_replaces_existing_predictions_during_apply() -> None:
    labels = _labels()
    video = labels.video
    initial_result = attach_predictions(
        labels=labels,
        video=video,
        frame_payloads={2: _payload(x=10.0)},
        skeleton=labels.skeleton,
        conf_thresh=0.0,
    )

    attachment = prepare_replacing_attachment(
        requested_frames=(2,),
        frame_payloads={2: _payload(x=90.0)},
        frame_indices=(2,),
    )
    result = apply_prepared_attachment(
        labels=labels,
        video=video,
        attachment=attachment,
        skeleton=labels.skeleton,
        conf_thresh=0.0,
        track_map=initial_result.updated_track_map,
        start_idx=0,
        max_frames=1,
    )

    frame = labels.query.find_first(video, 2, use_cache=True)
    assert result.attach_result.attached_frame_indices == (2,)
    assert frame is not None
    assert len(frame.predicted_instances) == 1
    assert np.allclose(frame.predicted_instances[0].xy_array(), [[90.0, 20.0], [95.0, 25.0]])


def test_replacing_transaction_lifecycle_restores_snapshot_and_track_map() -> None:
    labels = _labels()
    video = labels.video
    original_track = Track(spawned_on=1, name="original")
    original_prediction = _prediction_instance(labels)
    original_prediction.track = original_track
    frame = LabeledFrame(video=cast(Any, video), frame_idx=2, instances=[original_prediction])
    labels.append(frame)
    labels.tracks.append(original_track)

    transaction = begin_replacing_transaction(
        labels=labels,
        frame_payloads={2: _payload(x=90.0)},
        frame_indices=(2,),
        track_map={1: original_track},
    )

    assert transaction.current_track_map is not transaction.track_map_before_attachment
    assert advance_replacing_transaction_snapshot(
        labels=labels,
        video=video,
        transaction=transaction,
        max_items=8,
    )
    prepare_replacing_transaction(transaction=transaction)
    result = apply_replacing_transaction_slice(
        labels=labels,
        video=video,
        transaction=transaction,
        skeleton=labels.skeleton,
        conf_thresh=0.0,
        max_frames=1,
    )
    assert result.done is True
    assert transaction.total_added == 1

    restored_track_map = restore_replacing_transaction(labels=labels, transaction=transaction)
    restored = labels.query.find_first(video, 2, use_cache=True)
    assert restored is not None
    assert restored.predicted_instances[0].track is original_track
    assert restored_track_map[1] is original_track
    assert transaction.total_added == 0


def test_apply_attachment_uses_threshold_and_preserves_user_instances() -> None:
    labels = _labels()
    video = labels.video
    frame = LabeledFrame(video=cast(Any, video), frame_idx=2, instances=[_user_instance(labels)])
    labels.append(frame)
    attach_predictions(
        labels=labels,
        video=video,
        frame_payloads={2: _payload(x=1.0, score=0.9)},
        skeleton=labels.skeleton,
        conf_thresh=0.0,
    )

    result = apply_attachment(
        labels=labels,
        video=video,
        requested_frames=(2,),
        frame_payloads={2: _payload(x=5.0, score=0.4)},
        skeleton=labels.skeleton,
        conf_thresh=0.5,
        track_map={},
        frame_indices=(2,),
    )

    attached = labels.query.find_first(video, 2, use_cache=True)
    assert result.clear_result.cleared_frame_indices == (2,)
    assert result.clear_result.user_annotation_unchanged is True
    assert result.attach_result.attached_frame_indices == (2,)
    assert attached is not None
    assert len(attached.user_instances) == 1
    assert bool(attached.predicted_instances[0]["nose"].visible) is False


def test_snapshot_and_restore_roll_back_prediction_attachment_mutations() -> None:
    labels = _labels()
    video = labels.video
    original_track = Track(spawned_on=11, name="manual")
    original_user = _user_instance(labels)
    original_user.track = original_track
    frame = LabeledFrame(video=cast(Any, video), frame_idx=1, instances=[original_user])
    labels.append(frame)

    snapshot = snapshot_prediction_frames(labels=labels, video=video, frame_indices=[1, 9])
    attach_predictions(
        labels=labels,
        video=video,
        frame_payloads={1: _payload(x=50.0), 9: _payload(x=70.0)},
        skeleton=labels.skeleton,
        conf_thresh=0.0,
    )
    new_frame = labels.query.find_first(video, 9, use_cache=True)
    mutated_frame = labels.query.find_first(video, 1, use_cache=True)
    assert new_frame is not None
    assert mutated_frame is not None
    assert len(mutated_frame.predicted_instances) == 1

    restore_prediction_frames(labels=labels, snapshot=snapshot)

    restored_frame = labels.query.find_first(video, 1, use_cache=True)
    assert restored_frame is not None
    assert labels.query.find_first(video, 9, use_cache=True) is None
    assert len(restored_frame.user_instances) == 1
    assert restored_frame.predicted_instances == []
    assert np.allclose(restored_frame.user_instances[0].xy_array(), [[1.0, 2.0], [2.0, 3.0]])
    assert labels.tracks == [original_track]


def test_restore_prediction_frames_uses_targeted_cache_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = _labels()
    video = labels.video
    frame = LabeledFrame(video=cast(Any, video), frame_idx=1, instances=[_user_instance(labels)])
    labels.append(frame)
    snapshot = snapshot_prediction_frames(labels=labels, video=video, frame_indices=[1])
    attach_predictions(
        labels=labels,
        video=video,
        frame_payloads={1: _payload(x=50.0)},
        skeleton=labels.skeleton,
        conf_thresh=0.0,
    )
    monkeypatch.setattr(
        labels,
        "update_cache",
        lambda: pytest.fail("Restore should not rebuild the full labels cache."),
    )

    restore_prediction_frames(labels=labels, snapshot=snapshot)

    restored_frame = labels.query.find_first(video, 1, use_cache=True)
    assert restored_frame is not None
    assert len(restored_frame.user_instances) == 1
    assert restored_frame.predicted_instances == []


def test_restore_prediction_frames_does_not_trigger_whole_label_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = _labels()
    video = labels.video
    original_track = Track(spawned_on=1, name="manual")
    original_prediction = _prediction_instance(labels)
    original_prediction.track = original_track
    frame = LabeledFrame(video=cast(Any, video), frame_idx=2, instances=[original_prediction])
    labels.append(frame)
    labels.tracks.append(original_track)
    snapshot = snapshot_prediction_frames(labels=labels, video=video, frame_indices=[2])
    attach_predictions(
        labels=labels,
        video=video,
        frame_payloads={2: _payload(x=90.0)},
        skeleton=labels.skeleton,
        conf_thresh=0.0,
    )
    labels.labeled_frames = _NoIterList(labels.labeled_frames)
    monkeypatch.setattr(
        labels,
        "update_cache",
        lambda: pytest.fail("Targeted restore should not rebuild the full labels cache."),
    )
    monkeypatch.setattr(
        labels,
        "validate",
        lambda: pytest.fail("Targeted restore should not trigger whole-label validation."),
    )

    restore_prediction_frames(labels=labels, snapshot=snapshot)

    restored_frame = labels.query.find_first(video, 2, use_cache=True)
    assert restored_frame is not None
    assert restored_frame.predicted_instances[0].track is original_track


def test_clear_predictions_supports_track_frame_video_and_global_scopes() -> None:
    labels = _labels()
    video = labels.video
    other_video = VideoStub(filename="other.mp4", frames=10, height=64, width=64)
    labels.videos.append(other_video)

    attach_predictions(
        labels=labels,
        video=video,
        frame_payloads={0: _payload(x=1.0), 1: _payload(x=5.0)},
        skeleton=labels.skeleton,
        conf_thresh=0.0,
    )
    attach_predictions(
        labels=labels,
        video=other_video,
        frame_payloads={2: _payload(x=9.0)},
        skeleton=labels.skeleton,
        conf_thresh=0.0,
    )

    track_clear = clear_predictions(
        labels=labels,
        video=video,
        frame_indices=(0, 1),
        track_ids=(7,),
    )
    other_clear = clear_predictions(labels=labels, video=other_video)
    assert track_clear.cleared_frame_indices == (0, 1)
    assert track_clear.removed_instance_count == 2
    assert other_clear.cleared_frame_indices == (2,)
    assert other_clear.removed_instance_count == 1

    attach_predictions(
        labels=labels,
        video=video,
        frame_payloads={3: _payload(x=15.0)},
        skeleton=labels.skeleton,
        conf_thresh=0.0,
    )
    result = clear_predictions(labels=labels, video=None)

    assert result.cleared_frame_indices == (3,)
    assert result.removed_instance_count == 1
    assert labels.query.find_first(video, 3, use_cache=True) is None


def test_payload_helpers_fail_fast_on_invalid_contracts() -> None:
    labels = _labels()

    with pytest.raises(RuntimeError, match="no predictions for frames"):
        validate_prediction_payloads(requested_frames=[1], frame_payloads={})

    with pytest.raises(RuntimeError, match="no keypoints for frames"):
        validate_prediction_payloads(
            requested_frames=[1],
            frame_payloads={1: {"instances": [{"keypoints": [[np.nan, np.nan, 0.0]]}]}},
        )

    with pytest.raises(TypeError, match="track_map keys must be integers"):
        coerce_track_map({True: Track(spawned_on=1, name="bad")})

    with pytest.raises(RuntimeError, match="failed on all target frames"):
        partition_attachable_payloads(requested_frames=[1], frame_payloads={1: {"instances": []}})

    payload = {
        "frames": {"video_index": [0, 0], "frame_index": [1], "num_instances": [1, 1]},
        "data": {"keypoints": np.zeros((2, 1, 2, 3), dtype=np.float32).tolist()},
    }
    with pytest.raises(ValueError, match="row counts must match"):
        prediction_payloads_by_video(labels=labels, predictions_payload=payload)


def test_hydrate_predictions_rejects_invalid_payload_shapes() -> None:
    labels = _labels()

    with pytest.raises(ValueError, match=r"predictions\.data\.keypoints"):
        hydrate_predictions(
            labels,
            {"frames": {"video_index": [0], "frame_index": [0], "num_instances": [1]}, "data": {}},
            logger=logging.getLogger("xpkg.test"),
        )

    with pytest.raises(ValueError, match="unknown video index 1"):
        hydrate_predictions(
            labels,
            {
                "frames": {"video_index": [1], "frame_index": [0], "num_instances": [1]},
                "data": {"keypoints": np.zeros((1, 1, 2, 3), dtype=np.float32)},
            },
            logger=logging.getLogger("xpkg.test"),
        )

    with pytest.raises(ValueError, match="declares 2 instances"):
        hydrate_predictions(
            labels,
            {
                "frames": {"video_index": [0], "frame_index": [0], "num_instances": [2]},
                "data": {"keypoints": np.zeros((1, 1, 2, 3), dtype=np.float32)},
            },
            logger=logging.getLogger("xpkg.test"),
        )


def test_hydrate_predictions_attaches_serialized_prediction_payload() -> None:
    labels = _labels()
    payload = {
        "frames": {"video_index": [0], "frame_index": [12], "num_instances": [1]},
        "data": {
            "keypoints": [[[[4.0, 5.0, 0.75], [6.0, 7.0, 0.5]]]],
            "instance_score": [[0.8]],
            "track_id": [[2]],
        },
    }

    hydrate_predictions(labels, payload, logger=logging.getLogger("xpkg.test"))

    frame = labels.query.find_first(labels.video, 12, use_cache=True)
    assert frame is not None
    assert len(frame.predicted_instances) == 1
    prediction = frame.predicted_instances[0]
    assert prediction.track is not None
    assert prediction.track.id == 2
    assert np.allclose(prediction.xy_score_array(), [[4.0, 5.0, 0.75], [6.0, 7.0, 0.5]])


def test_make_predicted_instances_rejects_keypoint_count_contract_mismatch() -> None:
    labels = _labels()

    with pytest.raises(
        ValueError,
        match="Prediction instance contains 1 keypoints but skeleton expects 2",
    ):
        make_predicted_instances(
            labels.skeleton,
            cast(Any, object()),
            [{"keypoints": [[1.0, 2.0, 0.9]]}],
        )


def test_create_user_instances_from_predictions_preserves_prediction_visibility() -> None:
    labels = _labels()
    video = labels.video
    frame = LabeledFrame(video=cast(Any, video), frame_idx=4)
    prediction = _prediction_instance(labels, visible_tail=False)
    frame.instances.append(prediction)
    labels.append(frame)

    result = create_user_instances_from_predictions(labels=labels, labeled_frame=frame)

    assert result.frame_idx == 4
    assert result.created_user_instance_count == 1
    assert result.removed_predicted_count == 0
    user_instance = result.created_user_instances[0]
    assert user_instance.from_predicted is prediction
    assert user_instance.track is prediction.track
    assert bool(user_instance["nose"].visible) is True
    assert bool(user_instance["tail"].visible) is False
    assert prediction in frame.predicted_instances
    assert frame.unused_predictions == []


def test_apply_user_instances_from_payload_slice_rejects_existing_user_labels() -> None:
    labels = _labels()
    video = labels.video
    frame = LabeledFrame(video=cast(Any, video), frame_idx=5, instances=[_user_instance(labels)])
    labels.append(frame)

    with pytest.raises(ValueError, match=r"frame 5 already has 1 user instances"):
        apply_user_instances_from_payload_slice(
            labels=labels,
            video=video,
            frame_payloads={5: _payload()},
            skeleton=labels.skeleton,
            track_map={},
            frame_indices=(5,),
            start_idx=0,
            max_frames=1,
        )
