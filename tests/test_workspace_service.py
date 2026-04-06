from __future__ import annotations

from pathlib import Path

from posetta.services import WorkspaceService


def test_workspace_service_create_validate_and_pack_roundtrip(tmp_path: Path) -> None:
    workspace = WorkspaceService.create(tmp_path / "My Project", title="My Project")

    layout = workspace.validate()
    assert layout.workspace_root == (tmp_path / "My Project").resolve()
    assert layout.descriptor.title == "My Project"
    assert not layout.has_current_archive

    loaded = workspace.load_labels()
    assert loaded.labeled_frames == []

    artifact = workspace.pack(out=tmp_path / "My Project.expkg")
    unpacked = WorkspaceService.unpack(artifact, tmp_path / "Unpacked Project")
    unpacked_layout = unpacked.validate()

    assert unpacked_layout.workspace_root == (tmp_path / "Unpacked Project").resolve()
    assert unpacked_layout.descriptor.title == "My Project"
