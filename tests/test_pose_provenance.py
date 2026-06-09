"""Round-trip tests for PoseModelProvenance across model, project, and CLI."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from xpkg.model import PoseModelProvenance
from xpkg.project import (
    init_project,
    load_project_pose_provenance,
    pack_project,
    save_project_pose_provenance,
    unpack_project,
)
from xpkg.project.metadata import project_pose_provenance_path
from xpkg.project.store.imports import import_dlc_csv_project
from xpkg.services import ProjectService


def _capture_json(capsys: pytest.CaptureFixture[str]) -> dict:
    out = capsys.readouterr().out.strip()
    assert out, "expected JSON envelope on stdout"
    envelope = json.loads(out)
    assert envelope["ok"] is True
    return envelope["data"]


def _write_dlc_csv(path: Path, frames: int = 10) -> Path:
    cols = pd.MultiIndex.from_product(
        [["demo"], ["nose", "tail"], ["x", "y", "likelihood"]]
    )
    rows = [[float(i), float(i + 1), 0.99, float(i + 5), float(i + 6), 0.95]
            for i in range(frames)]
    pd.DataFrame(rows, columns=cols).to_csv(path)
    return path


def _write_video(path: Path, frames: int = 10) -> Path:
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (32, 32))
    for _ in range(frames):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    return path


def test_pose_model_provenance_round_trips_json_friendly_payload() -> None:
    record = PoseModelProvenance(
        tool="deeplabcut",
        tool_version="2.3.4",
        model_name="ResNet50",
        model_config={"iteration": 5, "shuffle": 1},
        training_set_reference="https://doi.org/10.1234/training-set",
        checkpoint_id="snapshot-200000",
        imported_from="/data/dlc/predictions.csv",
        imported_at="2026-05-07T12:00:00Z",
    )
    assert PoseModelProvenance.from_dict(record.to_dict()) == record


def test_pose_model_provenance_rejects_empty_tool() -> None:
    with pytest.raises(ValueError, match="tool"):
        PoseModelProvenance(tool="")


def test_save_and_load_project_pose_provenance(tmp_path: Path) -> None:
    project = tmp_path / "Provenance Project"
    init_project(project, title="Provenance Project")

    record = PoseModelProvenance(
        tool="sleap",
        tool_version="1.4",
        model_name="topdown_id",
        checkpoint_id="abc123",
    )
    saved = save_project_pose_provenance(project, record)
    assert saved == project_pose_provenance_path(project)
    assert saved.is_file()
    assert load_project_pose_provenance(project) == record


def test_load_project_pose_provenance_returns_none_when_unset(tmp_path: Path) -> None:
    project = tmp_path / "No Provenance Project"
    init_project(project, title="No Provenance Project")
    assert load_project_pose_provenance(project) is None


def test_dlc_import_persists_and_auto_fills_provenance(tmp_path: Path) -> None:
    project = tmp_path / "DLC Provenance Project"
    init_project(project, title="DLC Provenance Project")
    csv = _write_dlc_csv(tmp_path / "tracking.csv")
    video = _write_video(tmp_path / "vid.mp4")

    import_dlc_csv_project(
        csv,
        video,
        project,
        provenance=PoseModelProvenance(
            tool="deeplabcut",
            tool_version="2.3.4",
            model_name="ResNet50",
        ),
    )
    record = load_project_pose_provenance(project)
    assert record is not None
    assert record.tool == "deeplabcut"
    assert record.tool_version == "2.3.4"
    assert record.model_name == "ResNet50"
    assert record.imported_from is not None and record.imported_from.endswith(
        "tracking.csv"
    )
    assert record.imported_at is not None


def test_dlc_import_uses_default_tool_when_mapping_omits_tool(tmp_path: Path) -> None:
    project = tmp_path / "Default Tool Project"
    init_project(project, title="Default Tool Project")
    csv = _write_dlc_csv(tmp_path / "tracking.csv")
    video = _write_video(tmp_path / "vid.mp4")

    import_dlc_csv_project(
        csv,
        video,
        project,
        provenance={"tool_version": "2.4", "checkpoint_id": "snap-100000"},
    )
    record = load_project_pose_provenance(project)
    assert record is not None
    assert record.tool == "deeplabcut"
    assert record.tool_version == "2.4"
    assert record.checkpoint_id == "snap-100000"


def test_pose_provenance_round_trips_through_pack_unpack(tmp_path: Path) -> None:
    project = tmp_path / "Pack Provenance Project"
    init_project(project, title="Pack Provenance Project")
    save_project_pose_provenance(
        project,
        PoseModelProvenance(
            tool="deeplabcut",
            tool_version="2.3.4",
            checkpoint_id="snapshot-200000",
        ),
    )
    expkg = pack_project(project)

    with zipfile.ZipFile(expkg) as zf:
        manifest = json.loads(zf.read("EXPKG.json"))
    assert manifest["pose_provenance"]["tool"] == "deeplabcut"
    assert manifest["pose_provenance"]["checkpoint_id"] == "snapshot-200000"

    restored = unpack_project(expkg, tmp_path / "Restored")
    record = load_project_pose_provenance(restored)
    assert record is not None
    assert record.tool == "deeplabcut"
    assert record.tool_version == "2.3.4"


def test_project_service_save_and_load_pose_provenance(tmp_path: Path) -> None:
    project = ProjectService.create(tmp_path / "Service Provenance Project")
    project.metadata.update(
        pose_provenance=PoseModelProvenance(
            tool="mmpose",
            tool_version="1.3",
            model_name="rtmpose-l",
        )
    )
    record = project.metadata.pose_provenance
    assert record is not None
    assert record.tool == "mmpose"
    assert record.model_name == "rtmpose-l"


def test_cli_dlc_csv_with_model_provenance_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from xpkg.cli import main

    project = tmp_path / "CLI Provenance Project"
    init_project(project, title="CLI Provenance Project")
    csv = _write_dlc_csv(tmp_path / "tracking.csv")
    video = _write_video(tmp_path / "vid.mp4")
    provenance_json = tmp_path / "model_prov.json"
    provenance_json.write_text(
        json.dumps(
            {
                "tool_version": "2.3.4",
                "model_name": "ResNet50_iter5",
                "checkpoint_id": "snapshot-200000",
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "import",
            "pose",
            "dlc-csv",
            "--path",
            str(csv),
            "--video",
            str(video),
            "--out",
            str(project),
            "--provenance-json",
            str(provenance_json),
            "--json",
        ]
    )
    assert code == 0
    payload = _capture_json(capsys)
    assert payload["status"] == "imported"

    record = load_project_pose_provenance(project)
    assert record is not None
    assert record.tool == "deeplabcut"
    assert record.tool_version == "2.3.4"
    assert record.model_name == "ResNet50_iter5"
    assert record.checkpoint_id == "snapshot-200000"
    assert record.imported_from is not None and record.imported_from.endswith(
        "tracking.csv"
    )


def test_normalized_prediction_provenance_maps_known_importer_and_source_inputs() -> None:
    from xpkg.project.store.provenance import _normalized_prediction_provenance

    metadata = {
        "source": "sleap_h5_import",
        "project_name": "demo",
        "source_path": "preds.h5",
        "source_frame_count": 10,
        "video_path": "session.mp4",
    }

    provenance = _normalized_prediction_provenance(metadata, None)

    assert provenance == {
        "schema_version": 1,
        "importer": "sleap_h5_import",
        "tool": {"name": "SLEAP"},
        "source_format": "h5",
        "inputs": {"source_path": "preds.h5", "source_frame_count": 10},
    }


def test_normalized_prediction_provenance_defaults_unknown_importer() -> None:
    from xpkg.project.store.provenance import _normalized_prediction_provenance

    provenance = _normalized_prediction_provenance({}, None)

    assert provenance["importer"] == "unknown_pose_import"
    assert provenance["tool"] == {"name": "unknown"}
    assert provenance["source_format"] == "unknown"
    assert provenance["inputs"] == {}


def test_normalized_prediction_provenance_nests_flat_tool_and_model_keys() -> None:
    from xpkg.project.store.provenance import _normalized_prediction_provenance

    provenance = _normalized_prediction_provenance(
        {"source": "dlc_csv_import"},
        {
            "tool_name": "DLC-custom",
            "tool_version": "3.0",
            "framework_version": "2.16",
            "model_name": "resnet50",
            "model_version": "v2",
            "training_set": "iteration-5",
            "notes": "run-1",
        },
    )

    assert provenance["tool"] == {
        "name": "DLC-custom",
        "version": "3.0",
        "framework_version": "2.16",
    }
    assert provenance["model"] == {
        "name": "resnet50",
        "version": "v2",
        "training_set": "iteration-5",
    }
    assert provenance["notes"] == "run-1"
    assert "tool_name" not in provenance
    assert "model_version" not in provenance


def test_normalized_prediction_provenance_merges_nested_payloads() -> None:
    from xpkg.project.store.provenance import _normalized_prediction_provenance

    provenance = _normalized_prediction_provenance(
        {"source": "dlc_csv_import"},
        {
            "tool": {"name": "DeepLabCut", "build": "nightly"},
            "model": {"name": "resnet"},
            "metadata": {"shuffle": 1},
        },
    )

    assert provenance["tool"] == {"name": "DeepLabCut", "build": "nightly"}
    assert provenance["model"] == {"name": "resnet"}
    assert provenance["metadata"] == {"shuffle": 1}


def test_normalized_prediction_provenance_snapshots_config_file(tmp_path: Path) -> None:
    import hashlib

    from xpkg.project.store.provenance import _normalized_prediction_provenance

    config = tmp_path / "config.yaml"
    config.write_text("iteration: 5\n", encoding="utf-8")

    provenance = _normalized_prediction_provenance(
        {"source": "dlc_csv_import"},
        {"config_snapshot_path": config},
    )

    snapshot = provenance["config_snapshot"]
    assert Path(snapshot["path"]).name == "config.yaml"
    assert snapshot["sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()


def test_normalized_prediction_provenance_rejects_missing_config_snapshot(
    tmp_path: Path,
) -> None:
    from xpkg.project.store.provenance import _normalized_prediction_provenance

    with pytest.raises(FileNotFoundError, match="config snapshot not found"):
        _normalized_prediction_provenance(
            {"source": "dlc_csv_import"},
            {"config_snapshot_path": tmp_path / "missing.yaml"},
        )


def test_persist_pose_provenance_fills_tool_and_import_fields(tmp_path: Path) -> None:
    from xpkg.project.store.provenance import _persist_pose_provenance

    project = tmp_path / "Persist Defaults"
    init_project(project, title="Persist Defaults")
    source = tmp_path / "predictions.csv"
    source.write_text("frame,x\n", encoding="utf-8")

    _persist_pose_provenance(
        project,
        {"model_name": "resnet50"},
        default_tool="deeplabcut",
        source_path=source,
    )

    record = load_project_pose_provenance(project)
    assert record is not None
    assert record.tool == "deeplabcut"
    assert record.model_name == "resnet50"
    assert record.imported_from == "predictions.csv"
    assert record.imported_at is not None
    assert record.imported_at.endswith("Z")


def test_persist_pose_provenance_keeps_existing_import_fields(tmp_path: Path) -> None:
    from xpkg.project.store.provenance import _persist_pose_provenance

    project = tmp_path / "Persist Existing"
    init_project(project, title="Persist Existing")
    record = PoseModelProvenance(
        tool="sleap",
        imported_from="original.h5",
        imported_at="2026-01-01T00:00:00Z",
    )

    _persist_pose_provenance(
        project,
        record,
        default_tool="ignored",
        source_path=tmp_path / "other.h5",
    )

    loaded = load_project_pose_provenance(project)
    assert loaded == record


def test_persist_pose_provenance_none_is_a_no_op(tmp_path: Path) -> None:
    from xpkg.project.store.provenance import _persist_pose_provenance

    project = tmp_path / "Persist None"
    init_project(project, title="Persist None")

    _persist_pose_provenance(
        project,
        None,
        default_tool="deeplabcut",
        source_path=tmp_path / "predictions.csv",
    )

    assert load_project_pose_provenance(project) is None


def test_persist_pose_provenance_rejects_unsupported_payload_type(tmp_path: Path) -> None:
    from xpkg.project.store.provenance import _persist_pose_provenance

    project = tmp_path / "Persist Bad Type"
    init_project(project, title="Persist Bad Type")

    with pytest.raises(TypeError, match="must be PoseModelProvenance or mapping"):
        _persist_pose_provenance(
            project,
            42,  # type: ignore[arg-type]
            default_tool="deeplabcut",
            source_path=tmp_path / "predictions.csv",
        )
