from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xpkg.inspection import InspectionKind, InspectionReport, inspect_path


def _write_dlc_csv(path: Path) -> None:
    columns = pd.MultiIndex.from_tuples(
        [
            ("model", "snout", "x"),
            ("model", "snout", "y"),
            ("model", "snout", "likelihood"),
            ("model", "tail", "x"),
            ("model", "tail", "y"),
            ("model", "tail", "likelihood"),
        ],
        names=["scorer", "bodyparts", "coords"],
    )
    frame = pd.DataFrame(
        [
            [1.0, 2.0, 0.9, 5.0, 6.0, 0.8],
            [1.5, 2.5, 0.4, 5.5, 6.5, 0.3],
        ],
        columns=columns,
    )
    frame.to_csv(path)


def _write_dlc_csv_with_low_run(path: Path) -> None:
    columns = pd.MultiIndex.from_tuples(
        [
            ("model", "snout", "x"),
            ("model", "snout", "y"),
            ("model", "snout", "likelihood"),
            ("model", "tail", "x"),
            ("model", "tail", "y"),
            ("model", "tail", "likelihood"),
        ],
        names=["scorer", "bodyparts", "coords"],
    )
    rows = [
        [1.0, 2.0, 0.95, 5.0, 6.0, 0.95],
        [1.1, 2.1, 0.10, 5.1, 6.1, 0.92],
        [1.2, 2.2, 0.15, 5.2, 6.2, 0.91],
        [1.3, 2.3, 0.20, 5.3, 6.3, 0.90],
        [1.4, 2.4, 0.95, 5.4, 6.4, 0.93],
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path)


def test_inspect_path_summarizes_dlc_csv_pose_qc(tmp_path: Path) -> None:
    csv_path = tmp_path / "tracking.csv"
    _write_dlc_csv(csv_path)

    report = inspect_path(csv_path, confidence_threshold=0.5)

    assert isinstance(report, InspectionReport)
    assert report.status == "inspected"
    assert report.kind is InspectionKind.POSE_PREDICTIONS
    assert report.likely_importers == ("dlc_csv", "lightning_pose_csv")
    assert report.summary["frames"] == 2
    assert report.summary["keypoints"] == 2
    assert report.summary["node_names"] == ["snout", "tail"]
    confidence = report.summary["confidence"]
    assert confidence["below_threshold"] == 2
    assert confidence["longest_low_run_frames"] == 1
    assert confidence["worst_keypoint"] in {"snout", "tail"}
    per_keypoint = {entry["name"]: entry for entry in confidence["per_keypoint"]}
    assert set(per_keypoint) == {"snout", "tail"}
    assert per_keypoint["snout"]["below_threshold"] == 1
    assert per_keypoint["snout"]["longest_low_run"] == 1
    assert per_keypoint["tail"]["below_threshold"] == 1
    assert report.warnings == ()


def test_inspect_path_reports_per_keypoint_low_run(tmp_path: Path) -> None:
    csv_path = tmp_path / "tracking_run.csv"
    _write_dlc_csv_with_low_run(csv_path)

    report = inspect_path(csv_path, confidence_threshold=0.5)
    confidence = report.summary["confidence"]

    per_keypoint = {entry["name"]: entry for entry in confidence["per_keypoint"]}
    assert per_keypoint["snout"]["longest_low_run"] == 3
    assert per_keypoint["snout"]["below_threshold"] == 3
    assert per_keypoint["tail"]["longest_low_run"] == 0
    assert per_keypoint["tail"]["below_threshold"] == 0
    assert confidence["worst_keypoint"] == "snout"
    assert confidence["longest_low_run_frames"] == 3


def test_inspect_path_reports_events_csv_importer(tmp_path: Path) -> None:
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("time,label,duration\n0.1,cue,0.2\n", encoding="utf-8")

    report = inspect_path(csv_path)

    assert report.kind is InspectionKind.EVENTS_TABLE
    assert report.likely_importers == ("events_csv",)
    assert report.summary["timestamp_available"] is True
    assert report.summary["columns"] == ["time", "label", "duration"]


def _write_synthetic_mp4(path: Path, *, frames: int, fps: int) -> None:
    av = pytest.importorskip("av")
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        rng = np.random.default_rng(42)
        for _ in range(frames):
            data = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(data, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def test_inspect_path_reports_video_timing_qc(tmp_path: Path) -> None:
    pytest.importorskip("av")
    video_path = tmp_path / "synth.mp4"
    _write_synthetic_mp4(video_path, frames=30, fps=30)

    report = inspect_path(video_path)

    assert report.kind is InspectionKind.VIDEO
    timing = report.summary.get("timing")
    assert timing is not None, "expected PyAV-based timing payload"
    assert timing["packet_count"] == 30
    assert timing["dropped_frame_suspects"] == 0
    assert timing["measured_fps"] == pytest.approx(30.0, rel=0.05)
    assert timing["fps_drift_pct"] < 1.0
    assert report.warnings == ()


def test_inspect_path_reports_empty_project(tmp_path: Path) -> None:
    from xpkg.services import ProjectService

    project = ProjectService.create(tmp_path / "Empty Project", title="Empty Project")

    report = inspect_path(project.project_root)

    assert report.kind is InspectionKind.XPKG_PROJECT
    assert report.summary["title"] == "Empty Project"
    assert report.summary["state_kind"] == "empty"
    assert report.summary["has_current_state"] is False


def test_inspect_path_summarizes_project_state_without_payload_load(tmp_path: Path) -> None:
    from xpkg.project import current_project_state_path
    from xpkg.services import ProjectService

    project = ProjectService.create(tmp_path / "Large Project", title="Large Project")
    state_path = current_project_state_path(project.project_root)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        '{"format":"xpkg.labels-json","version":1,"payload":{"metadata":{},"predictions":'
        + ("0" * 16_384),
        encoding="utf-8",
    )

    report = inspect_path(project.project_root)

    assert report.kind is InspectionKind.XPKG_PROJECT
    assert report.summary["title"] == "Large Project"
    assert report.summary["state_kind"] == "labels"
    assert report.summary["has_current_state"] is True
    assert report.summary["state_bytes"] == state_path.stat().st_size


def test_inspection_report_to_dict_round_trips_wire_format(tmp_path: Path) -> None:
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("time,label,duration\n0.1,cue,0.2\n", encoding="utf-8")

    payload = inspect_path(csv_path).to_dict()

    assert payload["status"] == "inspected"
    assert payload["kind"] == "events_table"
    assert payload["description"] == "events table"
    assert payload["likely_importers"] == ["events_csv"]
    assert isinstance(payload["summary"], dict)
    assert isinstance(payload["warnings"], list)
