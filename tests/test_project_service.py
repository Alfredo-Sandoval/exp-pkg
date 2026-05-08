from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_dlc_import import _write_dummy_video, _write_sample_dlc_csv
from tests.test_project_contract import _make_labels
from tests.vicon_helpers import (
    write_sample_vicon_c3d,
    write_sample_vicon_csv,
    write_sample_vsk,
    write_sample_xcp,
)
from xpkg.model import Labels
from xpkg.project import current_project_state_path, validate_expkg
from xpkg.services import ProjectService


def _assert_sample_subject_labels(labels: Labels) -> None:
    assert len(labels.videos) == 1
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


def test_project_service_imports_and_loads_vicon_recording(tmp_path: Path) -> None:
    c3d_path = tmp_path / "trial.c3d"
    write_sample_vicon_c3d(c3d_path)
    write_sample_vsk(c3d_path.with_suffix(".vsk"))
    write_sample_xcp(c3d_path.with_suffix(".xcp"))

    project = ProjectService.create(tmp_path / "Vicon Project", title="Vicon Project")
    state_path = project.import_motion("vicon", path=c3d_path)

    assert state_path == current_project_state_path(project.project_root)
    loaded = project.load_vicon_recording()
    assert loaded.source_type == "c3d"
    assert loaded.marker_names == ("center", "R_foot")
    assert loaded.has_analog


def test_project_service_imports_vicon_csv_recording(tmp_path: Path) -> None:
    csv_path = tmp_path / "trial.csv"
    write_sample_vicon_csv(csv_path)
    write_sample_vsk(csv_path.with_suffix(".vsk"))
    write_sample_xcp(csv_path.with_suffix(".xcp"))

    project = ProjectService.create(tmp_path / "Vicon CSV Project", title="Vicon CSV Project")
    state_path = project.import_motion("vicon-csv", path=csv_path)

    assert state_path == current_project_state_path(project.project_root)
    loaded = project.load_vicon_recording()
    assert loaded.source_type == "csv"
    assert loaded.path.is_file()
    assert project.project_root in loaded.path.parents


def test_project_service_load_labels_guides_to_vicon_loader_when_state_cache_rebuilds(
    tmp_path: Path,
) -> None:
    c3d_path = tmp_path / "trial.c3d"
    write_sample_vicon_c3d(c3d_path)

    project = ProjectService.create(
        tmp_path / "Guided Vicon Project",
        title="Guided Vicon Project",
    )
    state_path = project.import_motion("vicon", path=c3d_path)
    state_path.unlink()

    with pytest.raises(ValueError, match="load_vicon_recording"):
        project.load_labels()


def test_project_service_metadata_field_roundtrip_uses_current_head(tmp_path: Path) -> None:
    project = ProjectService.create(
        tmp_path / "Service Field Metadata Project",
        title="Service Field Metadata Project",
    )
    project.save_labels(_make_labels(tmp_path, x=1.0, y=2.0))

    saved_path = project.save_metadata_field(
        "session_json",
        {"active_frame_idx": 7},
        reason="test.project_service_metadata_field",
    )

    metadata = project.load_metadata()

    assert saved_path.is_file()
    assert project.load_metadata_field("session_json") == {"active_frame_idx": 7}
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
