from __future__ import annotations

import numpy as np

from xpkg.model import (
    Instance,
    KPFlag,
    Point,
    PredictedInstance,
    PredictedPoint,
    build_keypoint_skeleton,
)


def _skeleton():
    return build_keypoint_skeleton(["nose", "tail"], name="subject")


def test_point_xy_or_none_uses_nan_missing_contract() -> None:
    assert Point().xy_or_none() is None
    assert Point(1.5, 2.5).xy_or_none() == (1.5, 2.5)


def test_no_train_flag_is_native_and_independent_of_visibility() -> None:
    point = Point(1.0, 2.0, visible=True)

    assert point.include_in_training is True

    point.set_flag(KPFlag.NO_TRAIN, True)

    assert point.has(KPFlag.NO_TRAIN)
    assert bool(point["visible"]) is True
    assert point.include_in_training is False


def test_instance_xy_array_returns_plain_masked_coordinates() -> None:
    instance = Instance(
        skeleton=_skeleton(),
        init_points={
            "nose": Point(1.0, 2.0, visible=False),
            "tail": Point(3.0, 4.0, visible=True),
        },
    )

    masked = instance.xy_array()

    assert masked.dtype == np.dtype("float64")
    assert masked.shape == (2, 2)
    assert np.isnan(masked[0]).all()
    np.testing.assert_allclose(masked[1], [3.0, 4.0])
    np.testing.assert_allclose(
        instance.xy_array(invisible_as_nan=False),
        [[1.0, 2.0], [3.0, 4.0]],
    )
    records = instance.point_records(copy=False)
    assert records.dtype.names == ("x", "y", "visible", "complete", "flags")
    assert bool(records["visible"][0]) is False


def test_instance_point_records_copy_contract() -> None:
    instance = Instance(
        skeleton=_skeleton(),
        init_points={"nose": Point(1.0, 2.0), "tail": Point(3.0, 4.0)},
    )

    detached = instance.point_records(copy=True)
    borrowed = instance.point_records(copy=False)
    detached["x"][0] = 99.0
    borrowed["x"][1] = 88.0

    np.testing.assert_allclose(instance.xy_array(invisible_as_nan=False), [[1.0, 2.0], [88.0, 4.0]])


def test_instance_xy_score_array_uses_nan_score_for_user_labels() -> None:
    instance = Instance(
        skeleton=_skeleton(),
        init_points={
            "nose": Point(1.0, 2.0),
            "tail": Point(3.0, 4.0),
        },
    )

    xy_score = instance.xy_score_array()

    np.testing.assert_allclose(xy_score[:, :2], [[1.0, 2.0], [3.0, 4.0]])
    assert np.isnan(xy_score[:, 2]).all()
    np.testing.assert_allclose(instance.xy_score_array(missing_score=1.0)[:, 2], [1.0, 1.0])


def test_predicted_instance_xy_score_array_preserves_native_scores() -> None:
    instance = PredictedInstance(
        skeleton=_skeleton(),
        init_points={
            "nose": PredictedPoint(1.0, 2.0, score=0.9),
            "tail": PredictedPoint(3.0, 4.0, visible=False, score=0.2),
        },
    )

    xy_score = instance.xy_score_array()

    assert np.isnan(xy_score[1, :2]).all()
    np.testing.assert_allclose(xy_score[:, 2], [0.9, np.nan])
    np.testing.assert_allclose(
        instance.xy_score_array(invisible_as_nan=False)[:, 2],
        [0.9, 0.2],
    )
    assert "score" in (instance.point_records(copy=False).dtype.names or ())
