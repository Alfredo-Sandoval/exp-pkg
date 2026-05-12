from __future__ import annotations

import numpy as np

from xpkg.io.predictions import PredictionAppendItem, SerializerPredictedInstance


def test_serializer_predicted_instance_exposes_state_writer_shape() -> None:
    instance = SerializerPredictedInstance(
        [[10.0, 20.0], [30.0, 40.0]],
        score=0.75,
        track_id=9,
        deleted=True,
        keypoint_scores=[0.9, 0.8],
    )

    points = instance.get_points_array(copy=False, full=True)

    assert instance.keypoints.shape == (2, 2)
    assert instance.score == 0.75
    assert instance.track_id == 9
    assert instance.track is not None
    assert instance.track.id == 9
    assert instance.deleted is True
    assert np.allclose(instance.keypoint_scores, [0.9, 0.8])
    assert points.dtype.names == ("x", "y", "visible", "complete", "score", "flags")
    assert np.allclose(points["x"], [10.0, 30.0])
    assert np.allclose(points["score"], [0.9, 0.8])


def test_prediction_append_item_import_contract() -> None:
    instance = SerializerPredictedInstance([[1.0, 2.0, 0.4]], score=0.5)
    item = PredictionAppendItem(video_index=1, frame_index=2, instances=[instance])

    assert item.video_index == 1
    assert item.frame_index == 2
    assert item.instances == [instance]
    assert item.heatmaps is None
