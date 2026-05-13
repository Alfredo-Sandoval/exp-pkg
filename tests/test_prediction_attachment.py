from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pytest

from xpkg.io.prediction_attachment import (
    accept_prediction,
    attach_predictions,
    clear_predictions,
    coerce_track_map,
    hydrate_predictions,
    partition_attachable_payloads,
    prediction_payloads_by_video,
    restore_prediction_frames,
    snapshot_prediction_frames,
    validate_prediction_payloads,
)
from xpkg.model import (
    Instance,
    LabeledFrame,
    Labels,
    Point,
    PredictedInstance,
    Track,
    VideoStub,
    build_keypoint_skeleton,
)


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
