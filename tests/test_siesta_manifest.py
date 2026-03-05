from __future__ import annotations

import json
from pathlib import Path

import h5py


def test_write_siesta_manifest_tracks_bundle_only_by_default(tmp_path: Path) -> None:
    from posetta.io.labels import Labels
    from posetta.io.siesta_format import write_siesta

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
    expected_bundle = str(bundle_path.resolve())

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
    assert _has("predictions", expected_bundle, "bundle")
