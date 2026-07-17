from __future__ import annotations

from pathlib import Path

import pytest

from xpkg.io.manifest import (
    AssetEntry,
    AssetType,
    ProjectManifest,
    coerce_manifest,
    resolve_asset_path,
)


def test_manifest_roundtrip_preserves_project_relative_asset_identity(tmp_path: Path) -> None:
    video_path = tmp_path / "Media" / "session.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"video")

    manifest = ProjectManifest()
    entry = manifest.register(
        video_path,
        AssetType.VIDEO,
        project_root=tmp_path,
        metadata={"role": "source"},
    )

    assert entry.path == "Media/session.mp4"
    assert entry.exists is True
    assert (
        manifest.get_by_path(
            "Media/session.mp4",
            AssetType.VIDEO,
            project_root=tmp_path,
        )
        == entry
    )

    restored = ProjectManifest.from_dict(manifest.to_dict())
    restored_entry = restored.get(entry.id)
    assert restored_entry == entry
    assert (
        resolve_asset_path(
            "Media/session.mp4",
            asset_type=AssetType.VIDEO,
            manifest=restored,
            project_root=tmp_path,
        )
        == video_path
    )


def test_manifest_rejects_relative_path_that_escapes_project_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Relative path escapes project root"):
        AssetEntry.from_path("../outside.mp4", AssetType.VIDEO, project_root=tmp_path)


def test_coerce_manifest_accepts_serialized_mapping() -> None:
    restored = coerce_manifest({"version": 1, "entries": []})

    assert isinstance(restored, ProjectManifest)
    assert len(restored) == 0
