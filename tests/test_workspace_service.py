from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_dlc_import import _write_dummy_video, _write_sample_dlc_csv
from tests.vicon_helpers import (
    write_sample_vicon_c3d,
    write_sample_vicon_csv,
    write_sample_vsk,
    write_sample_xcp,
)
from xpkg.formats import current_project_snapshot_path, current_project_state_path, validate_expkg
from xpkg.model import Labels
from xpkg.services import WorkspaceService


def _assert_sample_subject_labels(labels: Labels) -> None:
    assert len(labels.videos) == 1
    assert len(labels.skeletons) == 1
    assert len(labels.labeled_frames) == 2
    assert labels.skeletons[0].keypoint_names == ["nose", "tail"]


def test_getting_started_workspace_service_lifecycle_example_roundtrips(
    tmp_path: Path,
) -> None:
    workspace = WorkspaceService.create(tmp_path / "My Project", title="My Project")

    layout = workspace.validate()
    assert layout.workspace_root == (tmp_path / "My Project").resolve()
    assert layout.descriptor.title == "My Project"
    assert not layout.has_current_state

    loaded = workspace.load_labels()
    assert loaded.labeled_frames == []

    artifact = workspace.pack(out=tmp_path / "My Project.expkg")
    unpacked = WorkspaceService.unpack(artifact, tmp_path / "Unpacked Project")
    unpacked_layout = unpacked.validate()

    assert unpacked_layout.workspace_root == (tmp_path / "Unpacked Project").resolve()
    assert unpacked_layout.descriptor.title == "My Project"


def test_readme_recommended_workspace_service_flow_roundtrips_imported_artifact(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "tracking.csv"
    video_path = tmp_path / "session.avi"
    _write_sample_dlc_csv(csv_path)
    _write_dummy_video(video_path)

    workspace = WorkspaceService.create(tmp_path / "Imported Project", title="Imported Project")
    snapshot_path = workspace.imports.dlc_csv(
        csv_path,
        video_path,
        skeleton_name="subject",
    )

    layout = workspace.validate()
    assert snapshot_path == current_project_snapshot_path(workspace.workspace_root)
    assert layout.current_state_path == current_project_state_path(workspace.workspace_root)
    assert layout.current_state_path.suffix == ".json"
    assert layout.has_current_state

    loaded = workspace.load_labels()
    _assert_sample_subject_labels(loaded)

    artifact = workspace.pack(out=tmp_path / "Imported Project.expkg")
    validate_expkg(artifact)

    unpacked = WorkspaceService.unpack(artifact, tmp_path / "Restored Project")
    unpacked_layout = unpacked.validate()
    assert unpacked_layout.current_state_path.suffix == ".json"
    assert unpacked_layout.has_current_state

    restored = unpacked.load_labels()
    _assert_sample_subject_labels(restored)


def test_workspace_service_open_reuses_existing_workspace_from_nested_path(
    tmp_path: Path,
) -> None:
    workspace = WorkspaceService.create(tmp_path / "Nested Project", title="Nested Project")

    reopened = WorkspaceService.open(workspace.workspace_root / "Media")

    assert reopened.workspace_root == workspace.workspace_root


def test_workspace_service_imports_and_loads_vicon_recording(tmp_path: Path) -> None:
    c3d_path = tmp_path / "trial.c3d"
    write_sample_vicon_c3d(c3d_path)
    write_sample_vsk(c3d_path.with_suffix(".vsk"))
    write_sample_xcp(c3d_path.with_suffix(".xcp"))

    workspace = WorkspaceService.create(tmp_path / "Vicon Project", title="Vicon Project")
    snapshot_path = workspace.imports.vicon(c3d_path)

    assert snapshot_path == current_project_snapshot_path(workspace.workspace_root)
    loaded = workspace.load_vicon_recording()
    assert loaded.source_type == "c3d"
    assert loaded.marker_names == ("center", "R_foot")
    assert loaded.has_analog


def test_workspace_service_imports_vicon_csv_recording(tmp_path: Path) -> None:
    csv_path = tmp_path / "trial.csv"
    write_sample_vicon_csv(csv_path)
    write_sample_vsk(csv_path.with_suffix(".vsk"))
    write_sample_xcp(csv_path.with_suffix(".xcp"))

    workspace = WorkspaceService.create(tmp_path / "Vicon CSV Project", title="Vicon CSV Project")
    snapshot_path = workspace.imports.vicon_csv(csv_path)

    assert snapshot_path == current_project_snapshot_path(workspace.workspace_root)
    loaded = workspace.load_vicon_recording()
    assert loaded.source_type == "csv"
    assert loaded.path.is_file()
    assert workspace.workspace_root in loaded.path.parents


def test_workspace_service_load_labels_guides_to_vicon_loader_when_snapshot_cache_rebuilds(
    tmp_path: Path,
) -> None:
    c3d_path = tmp_path / "trial.c3d"
    write_sample_vicon_c3d(c3d_path)

    workspace = WorkspaceService.create(
        tmp_path / "Guided Vicon Project",
        title="Guided Vicon Project",
    )
    snapshot_path = workspace.imports.vicon(c3d_path)
    snapshot_path.unlink()

    with pytest.raises(ValueError, match="load_vicon_recording"):
        workspace.load_labels()
