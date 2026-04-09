from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest


def _load_manifest_entries(bundle_path: Path) -> list[dict[str, object]]:
    with h5py.File(str(bundle_path), "r") as handle:
        raw = handle["project_metadata"].attrs["manifest_json"]
        if isinstance(raw, bytes | bytearray):
            raw = raw.decode("utf-8")
        payload = json.loads(str(raw))
    entries = payload.get("entries")
    assert isinstance(entries, list)
    return entries


def _find_manifest_entry(
    entries: list[dict[str, object]],
    *,
    asset_type: str,
    path: Path,
    project_root: Path | None = None,
) -> dict[str, object]:
    expected_paths = {str(path.resolve())}
    if project_root is not None:
        expected_paths.add(path.resolve().relative_to(project_root.resolve()).as_posix())
    for entry in entries:
        if entry.get("asset_type") != asset_type:
            continue
        if entry.get("path") in expected_paths:
            return entry
    raise AssertionError(
        f"Manifest entry not found for {asset_type}: expected one of {sorted(expected_paths)}"
    )


def test_write_siesta_manifest_tracks_bundle_only_by_default(tmp_path: Path) -> None:
    from xpkg.formats import write_siesta
    from xpkg.model import Labels

    project_root = tmp_path / "proj"
    project_root.mkdir(parents=True)

    bundle_path = project_root / "proj.siesta"
    labels = Labels()
    write_siesta(bundle_path, labels)

    with h5py.File(str(bundle_path), "r") as handle:
        raw = handle["project_metadata"].attrs["manifest_json"]
        if isinstance(raw, bytes | bytearray):
            raw = raw.decode("utf-8")
        payload = json.loads(str(raw))

    entries = payload["entries"]
    expected_bundle = "proj.siesta"

    def _has(asset_type: str, path: str, role: str) -> bool:
        for entry in entries:
            if entry.get("asset_type") != asset_type:
                continue
            if entry.get("path") != path:
                continue
            metadata = entry.get("metadata") or {}
            if isinstance(metadata, dict) and metadata.get("role") == role:
                return True
        return False

    assert len(entries) == 1
    assert _has("predictions", expected_bundle, "archive")


def test_write_siesta_persists_preferences_payload(tmp_path: Path) -> None:
    from xpkg.formats import write_siesta
    from xpkg.model import Labels

    bundle_path = tmp_path / "prefs.siesta"
    labels = Labels(preferences={"theme": "paper", "show_scores": True})
    write_siesta(bundle_path, labels)

    with h5py.File(str(bundle_path), "r") as handle:
        raw = handle["project_metadata"].attrs["preferences_json"]
        if isinstance(raw, bytes | bytearray):
            raw = raw.decode("utf-8")
        payload = json.loads(str(raw))

    assert payload == {"show_scores": True, "theme": "paper"}


def test_write_siesta_registers_image_sequence_video_directory(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    from xpkg.formats import write_siesta
    from xpkg.model import Labels, Video

    project_root = tmp_path / "proj"
    frames_dir = project_root / "videos" / "sequence_a"
    frames_dir.mkdir(parents=True)

    frame_paths: list[str] = []
    for idx in range(3):
        frame = np.full((8, 10, 3), idx * 25, dtype=np.uint8)
        frame_path = frames_dir / f"frame_{idx:04d}.png"
        ok = cv2.imwrite(frame_path.as_posix(), frame)
        assert ok
        frame_paths.append(frame_path.as_posix())

    bundle_path = project_root / "proj.sta"
    labels = Labels(videos=[Video.from_image_filenames(frame_paths)])
    write_siesta(bundle_path, labels)

    entries = _load_manifest_entries(bundle_path)
    entry = _find_manifest_entry(
        entries,
        asset_type="video",
        path=frames_dir,
        project_root=project_root,
    )
    metadata = entry.get("metadata")
    assert isinstance(metadata, dict)
    assert metadata == {
        "backend": "images",
        "frame_count": 3,
        "index": 0,
        "role": "image_sequence",
    }


def test_write_siesta_rejects_image_sequence_with_multiple_parent_dirs(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    from xpkg.formats import write_siesta
    from xpkg.model import Labels, Video

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    frame = np.full((8, 10, 3), 10, dtype=np.uint8)
    first_path = first_dir / "frame_0000.png"
    second_path = second_dir / "frame_0001.png"
    assert cv2.imwrite(first_path.as_posix(), frame)
    assert cv2.imwrite(second_path.as_posix(), frame)

    bundle_path = tmp_path / "mixed" / "mixed.sta"
    bundle_path.parent.mkdir()
    labels = Labels(
        videos=[Video.from_image_filenames([first_path.as_posix(), second_path.as_posix()])]
    )

    with pytest.raises(ValueError, match="share exactly one parent directory"):
        write_siesta(bundle_path, labels)
