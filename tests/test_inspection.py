from __future__ import annotations

import json
import zipfile
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
    assert report.warnings == ()


def test_inspect_path_project_json_shape_is_stable(tmp_path: Path) -> None:
    from xpkg.services import ProjectService

    project = ProjectService.create(tmp_path / "Shape Project", title="Shape Project")

    payload = inspect_path(project.project_root).to_dict()

    assert list(payload) == [
        "status",
        "path",
        "name",
        "suffix",
        "exists",
        "is_dir",
        "size_bytes",
        "kind",
        "description",
        "likely_importers",
        "summary",
        "warnings",
        "warning_records",
    ]
    assert payload["kind"] == "xpkg_project"
    assert payload["likely_importers"] == []
    assert payload["warnings"] == []
    assert payload["warning_records"] == []
    summary = payload["summary"]
    assert list(summary) == [
        "project_id",
        "title",
        "state_kind",
        "has_current_state",
        "state_bytes",
        "commit_id",
        "modalities",
        "summary_path",
        "metadata_slots",
        "media",
    ]
    assert summary["title"] == "Shape Project"
    assert summary["state_kind"] == "empty"
    assert summary["has_current_state"] is False
    assert summary["state_bytes"] is None
    assert summary["commit_id"] is None
    assert summary["modalities"] == []
    assert summary["media"] == []

    slots = summary["metadata_slots"]
    assert list(slots) == [
        "acquisition",
        "dataset_share",
        "datasheet",
        "model_card",
        "pose_provenance",
    ]
    assert slots == {
        "acquisition": {
            "path": str(project.project_root / ".xpkg" / "metadata" / "acquisition.json"),
            "present": False,
            "valid": None,
        },
        "dataset_share": {
            "path": str(project.project_root / ".xpkg" / "metadata" / "dataset_share.json"),
            "present": False,
            "valid": None,
        },
        "datasheet": {
            "path": str(project.project_root / ".xpkg" / "metadata" / "datasheet.json"),
            "present": False,
            "valid": None,
        },
        "model_card": {
            "path": str(project.project_root / ".xpkg" / "metadata" / "model_card.json"),
            "present": False,
            "valid": None,
        },
        "pose_provenance": {
            "path": str(project.project_root / ".xpkg" / "metadata" / "pose_provenance.json"),
            "present": False,
            "valid": None,
        },
    }


def test_inspect_path_does_not_write_project_summary(tmp_path: Path) -> None:
    from xpkg.project import project_summary_path
    from xpkg.services import ProjectService

    project = ProjectService.create(tmp_path / "Read Only Project", title="Read Only Project")
    summary_path = project_summary_path(project.project_root)
    if summary_path.exists():
        summary_path.unlink()

    report = inspect_path(project.project_root)

    assert report.kind is InspectionKind.XPKG_PROJECT
    assert report.summary["title"] == "Read Only Project"
    assert not summary_path.exists()


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


def test_inspect_path_reports_project_metadata_slots_without_payload_load(tmp_path: Path) -> None:
    from xpkg.project import (
        current_project_state_path,
        project_datasheet_path,
        project_model_card_path,
        project_summary_path,
    )
    from xpkg.services import ProjectService

    project = ProjectService.create(tmp_path / "Metadata Project", title="Metadata Project")
    summary_path = project_summary_path(project.project_root)
    if summary_path.exists():
        summary_path.unlink()
    state_path = current_project_state_path(project.project_root)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        '{"format":"xpkg.labels-json","version":1,"payload":{"metadata":{},"predictions":'
        + ("0" * 16_384),
        encoding="utf-8",
    )
    datasheet_path = project_datasheet_path(project.project_root)
    datasheet_path.parent.mkdir(parents=True, exist_ok=True)
    datasheet_path.write_text(
        json.dumps({"title": "Metadata Project", "summary": "Hand-authored project datasheet."}),
        encoding="utf-8",
    )
    model_card_path = project_model_card_path(project.project_root)
    model_card_path.write_text(json.dumps({"intended_use": {}}), encoding="utf-8")

    report = inspect_path(project.project_root)

    assert report.kind is InspectionKind.XPKG_PROJECT
    assert report.summary["state_bytes"] == state_path.stat().st_size
    assert not summary_path.exists()
    slots = report.summary["metadata_slots"]
    assert list(slots) == [
        "acquisition",
        "dataset_share",
        "datasheet",
        "model_card",
        "pose_provenance",
    ]
    assert slots["datasheet"] == {
        "path": str(datasheet_path),
        "present": True,
        "valid": True,
    }
    assert slots["model_card"]["path"] == str(model_card_path)
    assert slots["model_card"]["present"] is True
    assert slots["model_card"]["valid"] is False
    assert "details" in slots["model_card"]["error"]
    assert slots["acquisition"] == {
        "path": str(project.project_root / ".xpkg" / "metadata" / "acquisition.json"),
        "present": False,
        "valid": None,
    }
    assert slots["dataset_share"]["present"] is False
    assert slots["pose_provenance"]["present"] is False
    assert any(
        "metadata slot 'model_card' is invalid" in warning and "details" in warning
        for warning in report.warnings
    )
    warning_records = report.to_dict()["warning_records"]
    assert any(
        record == {
            "code": "project_metadata_invalid",
            "message": warning,
            "path": str(model_card_path),
            "severity": "warning",
        }
        for warning in report.warnings
        for record in warning_records
        if "metadata slot 'model_card' is invalid" in warning
    )


def test_inspect_path_reports_project_media_from_summary_without_payload_load(
    tmp_path: Path,
) -> None:
    from tests.test_project_contract import _make_media_labels, _write_test_video
    from xpkg.project import current_project_state_path, load_project_summary
    from xpkg.services import ProjectService

    source_video = tmp_path / "source.avi"
    _write_test_video(source_video)
    project = ProjectService.create(tmp_path / "Media Project", title="Media Project")
    project.save_labels(_make_media_labels(source_video, x=3.0, y=4.0))
    summary = load_project_summary(project.project_root)
    managed_media = project.project_root / str(summary.media[0]["path"])
    managed_media.unlink()
    for superblock in (project.project_root / ".xpkg").glob("superblock.*.json"):
        superblock.unlink()
    state_path = current_project_state_path(project.project_root)
    state_path.write_text(
        '{"format":"xpkg.labels-json","version":1,"payload":{"metadata":{},"predictions":'
        + ("0" * 16_384),
        encoding="utf-8",
    )

    report = inspect_path(project.project_root)

    assert report.kind is InspectionKind.XPKG_PROJECT
    assert report.summary["state_kind"] == "labels"
    assert report.summary["state_bytes"] == state_path.stat().st_size
    assert report.summary["media"] == [
        {
            "index": 0,
            "kind": "video_file",
            "path": "Media/source.avi",
            "backend": "opencv",
            "video_id": "video_0",
            "label": "source.avi",
            "frame_count": 3,
            "height": 12,
            "width": 16,
            "channels": 3,
            "image_count": 0,
            "label_frame_count": 1,
            "max_label_frame_index": 0,
            "prediction_frame_count": 0,
            "max_prediction_frame_index": None,
            "exists": False,
        }
    ]
    assert any(
        "Project media item 'source.avi' is missing: Media/source.avi" == warning
        for warning in report.warnings
    )
    assert {
        "code": "project_media_missing",
        "message": "Project media item 'source.avi' is missing: Media/source.avi",
        "path": "Media/source.avi",
        "severity": "warning",
    } in report.to_dict()["warning_records"]


def test_inspect_path_project_image_sequence_media_reports_current_count(
    tmp_path: Path,
) -> None:
    from tests.test_project_contract import _make_labels
    from xpkg.services import ProjectService

    project = ProjectService.create(
        tmp_path / "Image Sequence Project",
        title="Image Sequence Project",
    )
    project.save_labels(_make_labels(tmp_path, x=3.0, y=4.0))

    report = inspect_path(project.project_root)

    assert report.kind is InspectionKind.XPKG_PROJECT
    assert len(report.summary["media"]) == 1
    media = report.summary["media"][0]
    assert media["kind"] == "image_sequence"
    assert media["path"].startswith("Media/")
    assert media["exists"] is True
    assert media["image_count"] == 1
    assert media["current_image_count"] == 1
    assert report.warnings == ()


def test_inspect_path_warns_when_project_media_inventory_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_project_contract import _make_media_labels, _write_test_video
    from xpkg.project import project_summary_path
    from xpkg.services import ProjectService

    source_video = tmp_path / "source.avi"
    _write_test_video(source_video)
    project = ProjectService.create(tmp_path / "Old Media Project", title="Old Media Project")
    project.save_labels(_make_media_labels(source_video, x=3.0, y=4.0))

    summary_path = project_summary_path(project.project_root)
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["state"]["summary"]["video_count"] == 1
    assert summary_payload["media"]["items"]
    summary_payload["media"]["items"] = []
    summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")

    def fail_load_payload(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("inspect must not materialize project payload")

    monkeypatch.setattr("xpkg.project.store.load_project_payload", fail_load_payload)

    report = inspect_path(project.project_root)

    assert report.kind is InspectionKind.XPKG_PROJECT
    assert report.summary["media"] == []
    assert any(
        warning
        == "Project media inventory is unavailable for labels state with 1 recorded video(s)."
        for warning in report.warnings
    )
    assert {
        "code": "project_media_inventory_unavailable",
        "message": "Project media inventory is unavailable for labels state with "
        "1 recorded video(s).",
        "path": str(summary_path),
        "severity": "warning",
    } in report.to_dict()["warning_records"]


def test_inspect_path_reports_expkg_metadata_slots_without_unpacking(tmp_path: Path) -> None:
    artifact = tmp_path / "metadata.expkg"
    with zipfile.ZipFile(artifact, mode="w") as archive:
        archive.writestr(
            "EXPKG.json",
            json.dumps(
                {
                    "format": "xpkg-packed-project",
                    "artifact_schema_version": 1,
                    "media": {"mode": "manifest"},
                }
            ),
        )
        archive.writestr(
            ".xpkg/metadata/datasheet.json",
            json.dumps(
                {
                    "title": "Packed Metadata Project",
                    "summary": "Hand-authored packed project datasheet.",
                }
            ),
        )
        archive.writestr(".xpkg/metadata/model_card.json", json.dumps({"intended_use": {}}))

    report = inspect_path(artifact)

    assert report.kind is InspectionKind.EXPKG_ARTIFACT
    slots = report.summary["metadata_slots"]
    assert list(slots) == [
        "acquisition",
        "dataset_share",
        "datasheet",
        "model_card",
        "pose_provenance",
    ]
    assert slots["datasheet"] == {
        "path": ".xpkg/metadata/datasheet.json",
        "present": True,
        "valid": True,
    }
    assert slots["model_card"]["path"] == ".xpkg/metadata/model_card.json"
    assert slots["model_card"]["present"] is True
    assert slots["model_card"]["valid"] is False
    assert "details" in slots["model_card"]["error"]
    assert slots["acquisition"] == {
        "path": ".xpkg/metadata/acquisition.json",
        "present": False,
        "valid": None,
    }
    assert slots["dataset_share"]["present"] is False
    assert slots["pose_provenance"]["present"] is False
    assert any(
        "Packed project metadata slot 'model_card' is invalid" in warning
        and "details" in warning
        for warning in report.warnings
    )
    assert any(
        record["code"] == "packed_project_metadata_invalid"
        and record["path"] == ".xpkg/metadata/model_card.json"
        and "details" in record["message"]
        for record in report.to_dict()["warning_records"]
    )


def test_inspect_path_reports_malformed_expkg_metadata_warning_record(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "malformed-metadata.expkg"
    with zipfile.ZipFile(artifact, mode="w") as archive:
        archive.writestr(
            "EXPKG.json",
            json.dumps(
                {
                    "format": "xpkg-packed-project",
                    "artifact_schema_version": 1,
                    "media": {"mode": "manifest"},
                }
            ),
        )
        archive.writestr(".xpkg/metadata/datasheet.json", '{"title":')

    report = inspect_path(artifact)

    assert report.kind is InspectionKind.EXPKG_ARTIFACT
    assert any(
        "Packed project metadata slot 'datasheet' is invalid" in warning
        for warning in report.warnings
    )
    assert any(
        record["code"] == "packed_project_metadata_invalid"
        and record["path"] == ".xpkg/metadata/datasheet.json"
        and "datasheet" in record["message"]
        for record in report.to_dict()["warning_records"]
    )


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
    assert isinstance(payload["warning_records"], list)
