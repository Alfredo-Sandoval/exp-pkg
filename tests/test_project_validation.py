"""Contract tests for materialized project payload validation and summaries."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from xpkg.project.validation import (
    ProjectSummary,
    summarize_loaded_project,
    validate_loaded_project,
)


def _valid_payload() -> dict[str, Any]:
    return {
        "labels": {
            "videos": {
                "filenames": ["a.mp4", "b.mp4"],
                "shapes": [[120, 64, 64, 3], [90, 32, 32, 3]],
            },
            "frames": {
                "video_index": [0, 0, 1],
                "frame_index": [4, 9, 2],
                "num_instances": [1, 2, 1],
            },
        },
        "predictions": {
            "frames": {
                "video_index": [1],
                "frame_index": [7],
                "num_instances": [1],
            },
        },
        "metadata": {},
    }


def test_validate_accepts_consistent_payload() -> None:
    # The validator's pass contract is "return None without raising".
    assert validate_loaded_project(_valid_payload()) is None


def test_validate_rejects_non_dict_payload() -> None:
    with pytest.raises(TypeError, match="not a dict"):
        validate_loaded_project(["labels"])  # type: ignore[arg-type]


def test_validate_reports_all_missing_required_keys() -> None:
    with pytest.raises(RuntimeError, match="Missing required keys: labels, predictions"):
        validate_loaded_project({"metadata": {}})


def test_validate_rejects_non_dict_labels_group() -> None:
    payload = _valid_payload()
    payload["labels"] = "not a group"
    with pytest.raises(RuntimeError, match="missing labels group"):
        validate_loaded_project(payload)


def test_validate_rejects_missing_videos_group() -> None:
    payload = _valid_payload()
    del payload["labels"]["videos"]
    with pytest.raises(RuntimeError, match="labels group missing videos payload"):
        validate_loaded_project(payload)


def test_validate_rejects_missing_filenames() -> None:
    payload = _valid_payload()
    del payload["labels"]["videos"]["filenames"]
    with pytest.raises(RuntimeError, match="videos group missing filenames"):
        validate_loaded_project(payload)


def test_validate_rejects_missing_shapes() -> None:
    payload = _valid_payload()
    del payload["labels"]["videos"]["shapes"]
    with pytest.raises(RuntimeError, match="videos group missing shapes"):
        validate_loaded_project(payload)


def test_validate_rejects_unshaped_shapes_value() -> None:
    payload = _valid_payload()
    payload["labels"]["videos"]["shapes"] = 5
    with pytest.raises(RuntimeError, match="videos group missing shapes"):
        validate_loaded_project(payload)


def test_validate_rejects_shape_row_count_mismatch() -> None:
    payload = _valid_payload()
    payload["labels"]["videos"]["shapes"] = [[120, 64, 64, 3]]
    with pytest.raises(
        RuntimeError,
        match=r"videos\.shapes row count does not match filenames length",
    ):
        validate_loaded_project(payload)


def test_validate_accepts_numpy_video_shapes() -> None:
    payload = _valid_payload()
    payload["labels"]["videos"]["shapes"] = np.zeros((2, 4), dtype=np.int64)
    assert validate_loaded_project(payload) is None


def test_validate_rejects_inconsistent_label_frame_lengths() -> None:
    payload = _valid_payload()
    payload["labels"]["frames"]["frame_index"] = [4]
    with pytest.raises(
        RuntimeError,
        match=(
            "labels.frames datasets have inconsistent lengths: "
            "video_index=3, frame_index=1, num_instances=3"
        ),
    ):
        validate_loaded_project(payload)


def test_validate_rejects_inconsistent_prediction_frame_lengths() -> None:
    payload = _valid_payload()
    payload["predictions"]["frames"]["num_instances"] = []
    with pytest.raises(
        RuntimeError,
        match=r"predictions\.frames datasets have inconsistent lengths",
    ):
        validate_loaded_project(payload)


def test_validate_accepts_numpy_frame_arrays_with_equal_lengths() -> None:
    payload = _valid_payload()
    payload["labels"]["frames"] = {
        "video_index": np.zeros(3, dtype=np.int64),
        "frame_index": np.arange(3, dtype=np.int64),
        "num_instances": np.ones(3, dtype=np.int64),
    }
    assert validate_loaded_project(payload) is None


def test_validate_ignores_empty_frame_group() -> None:
    payload = _valid_payload()
    payload["labels"]["frames"] = {"other_dataset": [1, 2]}
    assert validate_loaded_project(payload) is None


def test_summarize_prefers_metadata_counts() -> None:
    payload = _valid_payload()
    payload["metadata"] = {
        "n_labels": "12",
        "n_predictions_committed": 5,
        "schema_version": "2.1",
        "created": "2026-01-01T00:00:00Z",
        "modified": "2026-02-01T00:00:00Z",
    }

    summary = summarize_loaded_project(payload, path=Path("proj/state.json"))

    assert summary.n_videos == 2
    assert summary.video_filenames == ["a.mp4", "b.mp4"]
    assert summary.video_shapes == (2, 4)
    assert summary.label_frames == 12
    assert summary.prediction_frames == 5
    assert summary.schema_version == "2.1"
    assert summary.created == "2026-01-01T00:00:00Z"
    assert summary.modified == "2026-02-01T00:00:00Z"


def test_summarize_falls_back_to_frame_row_counts() -> None:
    payload = _valid_payload()
    payload["metadata"] = {"n_labels": 0, "n_predictions_committed": "not-a-number"}

    summary = summarize_loaded_project(payload, path=Path("proj/state.json"))

    assert summary.label_frames == 3
    assert summary.prediction_frames == 1


def test_summarize_uses_version_when_schema_version_missing() -> None:
    payload = _valid_payload()
    payload["metadata"] = {"version": "1.0"}

    summary = summarize_loaded_project(payload, path=Path("proj/state.json"))

    assert summary.schema_version == "1.0"


def test_project_summary_print_renders_expected_report() -> None:
    summary = ProjectSummary(
        path=Path("proj/state.json"),
        video_filenames=["a.mp4"],
        video_shapes=(1, 4),
        label_frames=3,
        prediction_frames=1,
        schema_version="2.1",
        created="2026-01-01T00:00:00Z",
        modified="2026-02-01T00:00:00Z",
    )
    stream = io.StringIO()

    summary.print(stream=stream)

    assert stream.getvalue() == (
        "File: proj/state.json\n"
        " videos: 1\n"
        "  shapes: (1, 4)\n"
        " labels frames: 3\n"
        " predictions frames: 1\n"
        " schema: 2.1\n"
        " created: 2026-01-01T00:00:00Z  modified: 2026-02-01T00:00:00Z\n"
    )


def test_project_summary_print_omits_optional_lines() -> None:
    summary = ProjectSummary(path=Path("proj/state.json"))
    stream = io.StringIO()

    summary.print(stream=stream)

    assert stream.getvalue() == (
        "File: proj/state.json\n"
        " videos: 0\n"
        " labels frames: 0\n"
        " predictions frames: 0\n"
    )
