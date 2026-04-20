from __future__ import annotations

from pathlib import Path

from tests.test_dlc_import import _write_dummy_video, _write_sample_dlc_csv
from xpkg.formats import current_project_snapshot_path, current_project_state_path, validate_expkg
from xpkg.model import Labels
from xpkg.services import WorkspaceService


def _assert_sample_mouse_labels(labels: Labels) -> None:
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
        skeleton_name="mouse",
    )

    layout = workspace.validate()
    assert snapshot_path == current_project_snapshot_path(workspace.workspace_root)
    assert layout.current_state_path == current_project_state_path(workspace.workspace_root)
    assert layout.current_state_path.suffix == ".json"
    assert layout.has_current_state

    loaded = workspace.load_labels()
    _assert_sample_mouse_labels(loaded)

    artifact = workspace.pack(out=tmp_path / "Imported Project.expkg")
    validate_expkg(artifact)

    unpacked = WorkspaceService.unpack(artifact, tmp_path / "Restored Project")
    unpacked_layout = unpacked.validate()
    assert unpacked_layout.current_state_path.suffix == ".json"
    assert unpacked_layout.has_current_state

    restored = unpacked.load_labels()
    _assert_sample_mouse_labels(restored)


def test_workspace_service_open_reuses_existing_workspace_from_nested_path(
    tmp_path: Path,
) -> None:
    workspace = WorkspaceService.create(tmp_path / "Nested Project", title="Nested Project")

    reopened = WorkspaceService.open(workspace.workspace_root / "Media")

    assert reopened.workspace_root == workspace.workspace_root
