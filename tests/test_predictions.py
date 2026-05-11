from __future__ import annotations

import numpy as np
import pytest

from xpkg.io.predictions import prediction_frame_payloads_from_payload


def test_prediction_frame_payloads_returns_empty_for_empty_payload() -> None:
    assert prediction_frame_payloads_from_payload({}) == {}


def test_prediction_frame_payloads_returns_empty_for_no_keypoints() -> None:
    payload = {"frames": {"frame_index": [0, 1]}, "data": {}}

    assert prediction_frame_payloads_from_payload(payload) == {}


def test_prediction_frame_payloads_converts_basic_payload() -> None:
    keypoints = np.array([[[[10.0, 20.0, 1.0]]], [[[30.0, 40.0, 1.0]]]])
    payload = {
        "frames": {"frame_index": [0, 5], "num_instances": [1, 1]},
        "data": {"keypoints": keypoints.tolist()},
    }

    result = prediction_frame_payloads_from_payload(payload)

    assert 0 in result
    assert 5 in result
    assert len(result[0]) == 1
    assert len(result[5]) == 1


def test_prediction_frame_payloads_includes_instance_scores() -> None:
    keypoints = np.array([[[[10.0, 20.0, 1.0]]]])
    scores = np.array([[0.95]])
    payload = {
        "frames": {"frame_index": [0], "num_instances": [1]},
        "data": {"keypoints": keypoints.tolist(), "instance_score": scores.tolist()},
    }

    result = prediction_frame_payloads_from_payload(payload)

    assert result[0][0]["score"] == pytest.approx(0.95, abs=1e-5)


def test_prediction_frame_payloads_includes_track_ids() -> None:
    keypoints = np.array([[[[10.0, 20.0, 1.0]]]])
    track_ids = np.array([[42]])
    payload = {
        "frames": {"frame_index": [0], "num_instances": [1]},
        "data": {"keypoints": keypoints.tolist(), "track_id": track_ids.tolist()},
    }

    result = prediction_frame_payloads_from_payload(payload)

    assert result[0][0]["track_id"] == 42


def test_prediction_frame_payloads_handles_multiple_keypoints() -> None:
    keypoints = np.array(
        [[[[10.0, 20.0, 1.0], [30.0, 40.0, 0.9], [50.0, 60.0, 0.8]]]]
    )
    payload = {
        "frames": {"frame_index": [0], "num_instances": [1]},
        "data": {"keypoints": keypoints.tolist()},
    }

    result = prediction_frame_payloads_from_payload(payload)

    assert len(result[0][0]["keypoints"]) == 3


def test_prediction_frame_payloads_includes_deleted_flag() -> None:
    keypoints = np.array([[[[10.0, 20.0, 1.0]]]])
    deleted = np.array([[1]], dtype=np.uint8)
    payload = {
        "frames": {"frame_index": [0], "num_instances": [1]},
        "data": {"keypoints": keypoints.tolist(), "deleted": deleted.tolist()},
    }

    result = prediction_frame_payloads_from_payload(payload)

    assert result[0][0]["deleted"] is True
