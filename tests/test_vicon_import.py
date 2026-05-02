from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.vicon_helpers import (
    write_sample_vicon_c3d,
    write_sample_vicon_csv,
    write_sample_vsk,
    write_sample_xcp,
)


def test_import_vicon_workspace_roundtrips_through_workspace_and_expkg(tmp_path: Path) -> None:
    from xpkg.adapters import read_vicon_json_payload
    from xpkg.workspace import (
        current_project_snapshot_path,
        import_vicon_workspace,
        load_workspace_vicon_recording,
        pack_project,
        unpack_project,
        validate_expkg,
    )

    c3d_path = tmp_path / "trial.c3d"
    write_sample_vicon_c3d(c3d_path)
    write_sample_vsk(c3d_path.with_suffix(".vsk"))
    write_sample_xcp(c3d_path.with_suffix(".xcp"))
    workspace = tmp_path / "Imported Vicon Project"

    snapshot_path = import_vicon_workspace(c3d_path, workspace)

    assert snapshot_path == current_project_snapshot_path(workspace)
    payload = read_vicon_json_payload(snapshot_path)
    assert payload["metadata"]["source"] == "vicon_import"
    assert payload["metadata"]["source_recording"] == c3d_path.resolve().as_posix()

    recording = load_workspace_vicon_recording(workspace)
    assert recording.path.is_file()
    assert recording.path.name == "trial.c3d"
    assert workspace in recording.path.parents
    assert recording.xcp_path is not None and recording.xcp_path.is_file()
    assert recording.vsk_path is not None and recording.vsk_path.is_file()
    assert recording.marker_names == ("center", "R_foot")
    assert [event.label for event in recording.events] == ["Foot Strike", "Start", "Foot Off"]
    assert [event.event_type for event in recording.gait_events] == [
        "foot_strike",
        "foot_off",
    ]
    assert recording.additional_points is not None
    assert recording.analog is not None

    artifact = pack_project(workspace, out=tmp_path / "Imported Vicon.expkg")
    validate_expkg(artifact)
    restored_root = unpack_project(artifact, tmp_path / "Restored Vicon Project")
    restored = load_workspace_vicon_recording(restored_root)

    assert restored.source_type == recording.source_type
    assert restored.frame_offset == recording.frame_offset
    assert restored.source_marker_labels == recording.source_marker_labels
    assert restored.events == recording.events
    assert restored.path.is_file()
    assert restored.path.name == "trial.c3d"
    assert restored_root in restored.path.parents
    assert restored.xcp_path is not None and restored.xcp_path.is_file()
    assert restored_root in restored.xcp_path.parents
    assert restored.vsk_path is not None and restored.vsk_path.is_file()
    assert restored_root in restored.vsk_path.parents
    np.testing.assert_allclose(restored.positions, recording.positions, equal_nan=True)
    np.testing.assert_array_equal(restored.marker_valid, recording.marker_valid)
    assert restored.analog is not None
    assert recording.analog is not None
    assert restored.analog.fps == recording.analog.fps
    assert restored.analog.samples_per_frame == recording.analog.samples_per_frame
    assert restored.analog.channel_names == recording.analog.channel_names
    assert restored.analog.channel_units == recording.analog.channel_units
    assert restored.analog.channel_descriptions == recording.analog.channel_descriptions
    assert restored.analog.candidate_emg_channel_names == ("Voltage.RTA",)
    np.testing.assert_allclose(
        restored.analog.values,
        recording.analog.values,
    )
    assert restored.additional_points is not None
    assert recording.additional_points is not None
    assert restored.additional_points.labels == recording.additional_points.labels
    np.testing.assert_allclose(
        restored.additional_points.values,
        recording.additional_points.values,
        equal_nan=True,
    )
    assert restored.model is not None
    assert recording.model is not None
    assert restored.model.marker_names == recording.model.marker_names
    assert restored.model.edges == recording.model.edges
    assert [camera.user_id for camera in restored.cameras] == [
        camera.user_id for camera in recording.cameras
    ]


def test_import_vicon_csv_workspace_loads_workspace_native_recording(tmp_path: Path) -> None:
    from xpkg.workspace import (
        current_project_snapshot_path,
        import_vicon_csv_workspace,
        load_workspace_vicon_recording,
    )

    csv_path = tmp_path / "trial.csv"
    write_sample_vicon_csv(csv_path)
    write_sample_vsk(csv_path.with_suffix(".vsk"))
    write_sample_xcp(csv_path.with_suffix(".xcp"))
    workspace = tmp_path / "Imported CSV Project"

    snapshot_path = import_vicon_csv_workspace(csv_path, workspace)

    assert snapshot_path == current_project_snapshot_path(workspace)
    recording = load_workspace_vicon_recording(workspace)
    assert recording.source_type == "csv"
    assert recording.path.is_file()
    assert workspace in recording.path.parents
    assert recording.vsk_path is not None and recording.vsk_path.is_file()
    assert recording.xcp_path is not None and recording.xcp_path.is_file()


def test_load_workspace_vicon_recording_rebuilds_tampered_snapshot_cache(tmp_path: Path) -> None:
    from xpkg.workspace import import_vicon_workspace, load_workspace_vicon_recording

    c3d_path = tmp_path / "trial.c3d"
    write_sample_vicon_c3d(c3d_path)
    write_sample_vsk(c3d_path.with_suffix(".vsk"))
    write_sample_xcp(c3d_path.with_suffix(".xcp"))
    workspace = tmp_path / "Tampered Vicon Project"

    snapshot_path = import_vicon_workspace(c3d_path, workspace)
    document = json.loads(snapshot_path.read_text(encoding="utf-8"))
    payload = document["payload"]
    payload["path"] = "broken/trial.c3d"
    snapshot_path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    recording = load_workspace_vicon_recording(workspace)

    assert recording.path.is_file()
    assert recording.path.name == "trial.c3d"
    assert [event.label for event in recording.events] == ["Foot Strike", "Start", "Foot Off"]
    rebuilt_document = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert rebuilt_document["payload"]["path"] != "broken/trial.c3d"
    assert [event["label"] for event in rebuilt_document["payload"]["events"]] == [
        "Foot Strike",
        "Start",
        "Foot Off",
    ]


def test_validate_workspace_rebuilds_missing_vicon_snapshot_cache(tmp_path: Path) -> None:
    from xpkg.workspace import import_vicon_workspace, validate_workspace

    c3d_path = tmp_path / "trial.c3d"
    write_sample_vicon_c3d(c3d_path)
    workspace = tmp_path / "Missing Snapshot Project"

    snapshot_path = import_vicon_workspace(c3d_path, workspace)
    snapshot_path.unlink()

    validate_workspace(workspace)

    assert snapshot_path.is_file()


def test_validate_workspace_rejects_missing_vicon_bundle_files(tmp_path: Path) -> None:
    from xpkg.workspace import import_vicon_workspace, validate_workspace

    c3d_path = tmp_path / "trial.c3d"
    write_sample_vicon_c3d(c3d_path)
    workspace = tmp_path / "Broken Vicon Project"

    import_vicon_workspace(c3d_path, workspace)
    bundled_recording = next((workspace / ".xpkg" / "imports").rglob("trial.c3d"))
    bundled_recording.unlink()

    with pytest.raises(FileNotFoundError, match="Vicon recording file missing"):
        validate_workspace(workspace)
