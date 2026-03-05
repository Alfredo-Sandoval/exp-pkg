from __future__ import annotations

import json
from pathlib import Path

import h5py


def test_write_siesta_registers_project_structure_assets(tmp_path: Path) -> None:
    from posetta.io.labels import Labels
    from posetta.io.siesta_format import write_siesta

    project_root = tmp_path / "proj"
    (project_root / "models" / "pose").mkdir(parents=True)
    (project_root / "exports").mkdir(parents=True)
    (project_root / "suggestions").mkdir(parents=True)
    (project_root / "logs").mkdir(parents=True)

    bundle_path = project_root / "proj.siesta"
    labels = Labels()
    write_siesta(bundle_path, labels)

    with h5py.File(str(bundle_path), "r") as handle:
        raw = handle["project_metadata"].attrs["manifest_json"]
        if isinstance(raw, bytes | bytearray):
            raw = raw.decode("utf-8")
        payload = json.loads(str(raw))

    entries = payload["entries"]
    expected_train_out = str((project_root / "models" / "pose").resolve())
    expected_exports = str((project_root / "exports").resolve())
    expected_suggestions = str((project_root / "suggestions").resolve())
    expected_logs = str((project_root / "logs").resolve())
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

    assert _has("model", expected_train_out, "train_output")
    assert _has("other", expected_exports, "exports")
    assert _has("other", expected_suggestions, "suggestions")
    assert _has("other", expected_logs, "logs")
    assert _has("predictions", expected_bundle, "bundle")
