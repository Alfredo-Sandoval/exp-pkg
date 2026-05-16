from __future__ import annotations

from typing import Any, cast

import pytest

from xpkg.io.labels.tracks import add_track, get_track_occupancy, remove_track, track_swap
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


def _labels(*, frames: int = 10) -> Labels:
    video = VideoStub(filename="movie.mp4", frames=frames, height=64, width=64)
    skeleton = build_keypoint_skeleton(["nose", "tail"], name="subject")
    return Labels(videos=[video], skeletons=[skeleton])


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
    prediction[nose] = PredictedPoint(x=10.0, y=20.0, visible=True, score=0.8)
    prediction[tail] = PredictedPoint(x=10.0, y=20.0, visible=visible_tail, score=0.8)
    return prediction


def test_instance_realign_points_after_keypoint_removal_drops_removed_point() -> None:
    labels = _labels()
    skeleton = labels.skeleton
    paw = skeleton.add_keypoint("paw")
    instance = _user_instance(labels)
    instance.realign_points()
    instance[paw] = Point(x=7.0, y=8.0, visible=True, complete=True)

    skeleton.remove_keypoint("tail")
    instance.realign_points()

    assert skeleton.keypoint_names == ["nose", "paw"]
    assert instance["nose"].x == pytest.approx(1.0)
    assert instance["nose"].y == pytest.approx(2.0)
    assert instance["paw"].x == pytest.approx(7.0)
    assert instance["paw"].y == pytest.approx(8.0)


def test_validate_allows_well_formed_graph() -> None:
    labels = _labels()
    frame = LabeledFrame(video=cast(Any, labels.video), frame_idx=0)
    instance = _user_instance(labels)
    frame.instances.append(instance)
    labels.append(frame)

    labels.validate()

    assert instance.frame is frame
    assert [skeleton.name for skeleton in labels.skeletons] == ["subject"]
    assert [video.filename for video in labels.videos] == ["movie.mp4"]


def test_validate_rejects_invalid_frame_graphs_and_foreign_entries() -> None:
    labels = _labels()
    frame = LabeledFrame(video=cast(Any, labels.video), frame_idx=0)
    instance = _user_instance(labels)
    frame.instances.append(instance)
    labels.append(frame)

    instance.frame = LabeledFrame(video=cast(Any, labels.video), frame_idx=1)
    with pytest.raises(ValueError, match="Instance.frame"):
        labels.validate()

    instance.frame = frame
    cast(Any, labels.labeled_frames).append(object())
    with pytest.raises(TypeError, match="LabeledFrame objects"):
        labels.validate()


def test_validate_rejects_shared_instance_lists_duplicate_tracks_and_point_drift() -> None:
    shared_labels = _labels()
    shared_frame = LabeledFrame(video=cast(Any, shared_labels.video), frame_idx=0)
    shared_frame.instances.append(_user_instance(shared_labels))
    shared_labels.append(shared_frame)
    other = LabeledFrame(video=cast(Any, shared_labels.video), frame_idx=1)
    other.instances = shared_frame.instances
    shared_labels.labeled_frames.append(other)
    with pytest.raises(ValueError, match="InstancesList"):
        shared_labels.validate()

    labels = _labels()
    frame = LabeledFrame(video=cast(Any, labels.video), frame_idx=0)
    first = _user_instance(labels)
    frame.instances.append(first)
    labels.append(frame)
    track = Track(spawned_on=0, name="t0")
    first.track = track
    duplicate = _user_instance(labels, x=3.0, y=4.0)
    duplicate.track = track
    frame.instances.append(duplicate)
    with pytest.raises(ValueError, match="Duplicate track assignment"):
        labels.validate()

    frame.instances.pop()
    first_any = cast(Any, first)
    first_any._points = first_any._points[:1]
    with pytest.raises(ValueError):
        labels.validate()


def test_track_helpers_register_remove_and_swap_assignments() -> None:
    labels = _labels(frames=4)
    video = labels.video
    old_track = Track(spawned_on=0, name="old")
    new_track = Track(spawned_on=0, name="new")
    add_track(labels, video, old_track)
    add_track(labels, video, new_track)
    frame = LabeledFrame(video=cast(Any, video), frame_idx=0)
    instance = _user_instance(labels)
    instance.track = old_track
    frame.instances.append(instance)
    labels.append(frame)

    occupancy = get_track_occupancy(labels, video)
    assert old_track in labels.tracks
    assert new_track in labels.tracks
    assert old_track in occupancy

    track_swap(labels, video, new_track, old_track, (0, 1))
    assert instance.track is new_track

    remove_track(labels, new_track)
    assert new_track not in labels.tracks
    assert instance.track is None


def test_prediction_visibility_hides_predictions_shadowed_by_user_instances() -> None:
    labels = _labels()
    video = labels.video
    frame = LabeledFrame(video=cast(Any, video), frame_idx=3)
    labels.append(frame)
    prediction = _prediction_instance(labels)
    frame.instances.append(prediction)
    user_instance = _user_instance(labels, x=10.0, y=20.0)
    frame.instances.append(user_instance)

    assert len(frame.instances) == 2
    assert prediction not in frame.unused_predictions
    assert frame.instances_to_show == [user_instance]


def test_duplicate_track_clear_keeps_hidden_prediction_linked_to_user_instance() -> None:
    labels = _labels()
    video = labels.video
    frame = LabeledFrame(video=cast(Any, video), frame_idx=1)
    labels.append(frame)
    track = Track(spawned_on=1, name="t1")
    first = _user_instance(labels)
    first.track = track
    labels.add_instance(frame, first)
    duplicate = _user_instance(labels, x=3.0, y=4.0)
    duplicate.track = track
    labels.add_instance(frame, duplicate)

    assert duplicate.track is None
    assert {instance.track for instance in frame.instances} == {track, None}

    prediction = _prediction_instance(labels)
    prediction.track = Track(spawned_on=2, name="pred")
    user_from_prediction = Instance(skeleton=labels.skeleton, from_predicted=prediction)
    for keypoint in labels.skeleton.keypoints:
        point = prediction[keypoint]
        user_from_prediction[keypoint] = Point(x=float(point.x), y=float(point.y), visible=True)
    user_from_prediction.track = None
    prediction_frame = LabeledFrame(
        video=cast(Any, video),
        frame_idx=2,
        instances=[prediction, user_from_prediction],
    )
    labels.append(prediction_frame)

    assert prediction not in prediction_frame.unused_predictions
    assert prediction_frame.instances_to_show == [user_from_prediction]
