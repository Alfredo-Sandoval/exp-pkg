from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_dlc_import import _write_dummy_video, _write_sample_dlc_csv
from tests.test_project_contract import _make_labels
from xpkg.model import AcquisitionMetadata, Labels
from xpkg.project import (
    current_project_state_path,
    load_project_summary,
    save_project_acquisition_metadata,
    validate_expkg,
)
from xpkg.project.layout import project_summary_path
from xpkg.services import ProjectService


def _assert_sample_subject_labels(labels: Labels) -> None:
    assert len(labels.videos) == 1
    assert labels.videos[0].backend == "opencv"
    assert len(labels.skeletons) == 1
    assert len(labels.labeled_frames) == 2
    assert labels.skeletons[0].keypoint_names == ["nose", "tail"]


def test_getting_started_project_service_lifecycle_example_roundtrips(
    tmp_path: Path,
) -> None:
    project = ProjectService.create(tmp_path / "My Project", title="My Project")

    layout = project.validate()
    assert layout.project_root == (tmp_path / "My Project").resolve()
    assert layout.descriptor.title == "My Project"
    assert not layout.has_current_state

    loaded = project.load_labels()
    assert loaded.labeled_frames == []

    artifact = project.pack(out=tmp_path / "My Project.expkg")
    unpacked = ProjectService.unpack(artifact, tmp_path / "Unpacked Project")
    unpacked_layout = unpacked.validate()

    assert unpacked_layout.project_root == (tmp_path / "Unpacked Project").resolve()
    assert unpacked_layout.descriptor.title == "My Project"


def test_readme_recommended_project_service_flow_roundtrips_imported_artifact(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "tracking.csv"
    video_path = tmp_path / "session.avi"
    _write_sample_dlc_csv(csv_path)
    _write_dummy_video(video_path)

    project = ProjectService.create(tmp_path / "Imported Project", title="Imported Project")
    state_path = project.import_pose(
        "dlc-csv",
        path=csv_path,
        video=video_path,
        skeleton_name="subject",
    )

    layout = project.validate()
    assert state_path == current_project_state_path(project.project_root)
    assert layout.current_state_path == current_project_state_path(project.project_root)
    assert layout.current_state_path.suffix == ".json"
    assert layout.has_current_state

    loaded = project.load_labels()
    _assert_sample_subject_labels(loaded)

    artifact = project.pack(out=tmp_path / "Imported Project.expkg")
    validate_expkg(artifact)

    unpacked = ProjectService.unpack(artifact, tmp_path / "Restored Project")
    unpacked_layout = unpacked.validate()
    assert unpacked_layout.current_state_path.suffix == ".json"
    assert unpacked_layout.has_current_state

    restored = unpacked.load_labels()
    _assert_sample_subject_labels(restored)


def test_project_service_open_reuses_existing_project_from_nested_path(
    tmp_path: Path,
) -> None:
    project = ProjectService.create(tmp_path / "Nested Project", title="Nested Project")

    reopened = ProjectService.open(project.project_root / "Media")

    assert reopened.project_root == project.project_root


def test_project_service_describe_uses_shallow_summary_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = ProjectService.create(tmp_path / "Summary Project", title="Summary Project")
    project.save_labels(
        _make_labels(tmp_path, x=3.0, y=4.0),
        metadata={"source": "test"},
    )

    summary = load_project_summary(project.project_root)
    assert summary.state_kind == "labels"
    assert summary.state_summary["label_frame_count"] == 1
    assert summary.state_summary["prediction_frame_count"] == 0
    assert summary.modalities == ("labels",)

    def fail_load_payload(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("describe must not materialize project payload")

    def fail_refresh_summary(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("describe must reuse the matching summary index")

    monkeypatch.setattr("xpkg.services.project.load_project_payload", fail_load_payload)
    monkeypatch.setattr("xpkg.project.store.load_project_payload", fail_load_payload)
    monkeypatch.setattr("xpkg.services.project.refresh_project_summary", fail_refresh_summary)

    summary_text = project_summary_path(project.project_root).read_text(encoding="utf-8")
    layout = ProjectService.open(project.project_root).describe()

    assert layout.summary_path == project_summary_path(project.project_root)
    assert layout.summary.state_kind == "labels"
    assert layout.summary.state_summary["label_frame_count"] == 1
    assert layout.has_current_state is True
    assert project_summary_path(project.project_root).read_text(encoding="utf-8") == summary_text


def test_project_summary_tracks_media_frame_inventory(tmp_path: Path) -> None:
    from tests.test_project_contract import _make_media_labels, _write_test_video

    source_video = tmp_path / "source.avi"
    _write_test_video(source_video)
    project = ProjectService.create(tmp_path / "Media Summary", title="Media Summary")

    project.save_labels(_make_media_labels(source_video, x=3.0, y=4.0))

    summary = load_project_summary(project.project_root)
    assert len(summary.media) == 1
    media = summary.media[0]
    assert media["kind"] == "video_file"
    assert media["path"] == "Media/source.avi"
    assert media["frame_count"] == 3
    assert media["fps"] == pytest.approx(5.0)
    assert media["duration_s"] == pytest.approx(0.6)
    assert media["timebase"] == "frame_index"
    assert media["height"] == 12
    assert media["width"] == 16
    assert media["label_frame_count"] == 1
    assert media["max_label_frame_index"] == 0
    assert media["prediction_frame_count"] == 0
    assert media["max_prediction_frame_index"] is None


def test_project_summary_tracks_project_metadata_slots(tmp_path: Path) -> None:
    project = ProjectService.create(tmp_path / "Metadata Summary", title="Metadata Summary")

    save_project_acquisition_metadata(
        project.project_root,
        AcquisitionMetadata(acquisition_id="acq-001"),
    )

    summary = load_project_summary(project.project_root)
    assert summary.metadata_slots == ("acquisition",)
    assert "project_metadata" in summary.modalities


def test_project_summary_preserves_state_counts_after_metadata_only_commit(
    tmp_path: Path,
) -> None:
    project = ProjectService.create(tmp_path / "State Metadata Summary")
    project.save_labels(_make_labels(tmp_path, x=3.0, y=4.0))

    project.save_state_metadata({"session_json": {"active_frame_idx": 4}})

    summary = load_project_summary(project.project_root)
    assert summary.state_summary["label_frame_count"] == 1
    assert summary.state_summary["prediction_frame_count"] == 0


def test_project_service_metadata_field_roundtrip_uses_current_head(tmp_path: Path) -> None:
    project = ProjectService.create(
        tmp_path / "Service Field Metadata Project",
        title="Service Field Metadata Project",
    )
    project.save_labels(_make_labels(tmp_path, x=1.0, y=2.0))

    saved_path = project.save_state_metadata_field(
        "session_json",
        {"active_frame_idx": 7},
        reason="test.project_service_metadata_field",
    )

    metadata = project.load_state_metadata()

    assert saved_path.is_file()
    assert project.load_state_metadata_field("session_json") == {"active_frame_idx": 7}
    assert metadata is not None
    assert metadata["session_json"] == {"active_frame_idx": 7}


def test_project_service_scoped_metadata_roundtrip_without_current_head(tmp_path: Path) -> None:
    project = ProjectService.create(
        tmp_path / "Service Scoped Metadata Project",
        title="Service Scoped Metadata Project",
    )

    written = project.metadata.update(
        acquisition={
            "acquisition_id": "acq-service",
            "cameras": [{"camera_id": "cam-top", "frame_rate_hz": 120.0}],
        },
        dataset_share={
            "title": "Service metadata dataset",
            "creators": ["Sandoval Lab"],
            "doi": "10.0000/service",
            "license": "BSD-3-Clause",
        },
    )

    acquisition = project.metadata.acquisition
    dataset_share = project.metadata.dataset_share

    assert written["acquisition"].is_file()
    assert written["dataset_share"].is_file()
    assert acquisition is not None
    assert acquisition.acquisition_id == "acq-service"
    assert dataset_share is not None
    assert dataset_share.doi == "10.0000/service"



def test_pose_importer_registry_covers_all_pose_formats() -> None:
    """Adding a PoseFormat literal without a registry entry (or vice versa) must fail loudly."""
    from typing import get_args

    from xpkg.services.project import _POSE_IMPORTERS, PoseFormat

    expected = set(get_args(PoseFormat))
    actual = set(_POSE_IMPORTERS.keys())
    assert expected == actual, (
        f"PoseFormat literal and _POSE_IMPORTERS registry are out of sync: "
        f"missing from registry {expected - actual!r}, "
        f"missing from literal {actual - expected!r}"
    )


def test_calibration_importer_registry_covers_all_calibration_formats() -> None:
    from typing import get_args

    from xpkg.services.project import _CALIBRATION_IMPORTERS, CalibrationFormat

    expected = set(get_args(CalibrationFormat))
    actual = set(_CALIBRATION_IMPORTERS.keys())
    assert expected == actual
