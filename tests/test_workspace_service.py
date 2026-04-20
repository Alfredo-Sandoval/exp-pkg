from __future__ import annotations

from pathlib import Path

from tests.test_dlc_import import _write_dummy_video, _write_sample_dlc_csv
from xpkg.formats import current_project_snapshot_path, current_project_state_path, validate_expkg
from xpkg.services import WorkspaceService


def test_workspace_service_create_validate_and_pack_roundtrip(tmp_path: Path) -> None:
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


def test_workspace_service_imports_dlc_csv_and_roundtrips_workspace_artifact(
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
    assert len(loaded.videos) == 1
    assert len(loaded.skeletons) == 1
    assert len(loaded.labeled_frames) == 2
    assert loaded.skeletons[0].keypoint_names == ["nose", "tail"]

    artifact = workspace.pack(out=tmp_path / "Imported Project.expkg")
    validate_expkg(artifact)

    unpacked = WorkspaceService.unpack(artifact, tmp_path / "Restored Project")
    unpacked_layout = unpacked.validate()
    assert unpacked_layout.current_state_path.suffix == ".json"
    assert unpacked_layout.has_current_state

    restored = unpacked.load_labels()
    assert len(restored.videos) == 1
    assert len(restored.skeletons) == 1
    assert len(restored.labeled_frames) == 2
    assert restored.skeletons[0].keypoint_names == ["nose", "tail"]
