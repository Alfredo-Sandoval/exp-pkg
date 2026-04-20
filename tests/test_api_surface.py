"""Smoke tests for the xpkg public API facade."""

from __future__ import annotations

import xpkg.api as api


def test_xpkg_api_exposes_io_contract() -> None:
    expected = {
        "Labels",
        "ProjectDescriptor",
        "ConversionResult",
        "PoseTrack",
        "WorkspaceLayout",
        "WorkspaceService",
        "convert_dlc_csv",
        "convert_sleap_h5",
        "import_sleap_h5_workspace",
        "labels_from_json_payload",
        "labels_numpy",
        "labels_to_dataframe",
        "labels_to_json_payload",
        "pack_project",
        "read_pose_node_names",
        "read_pose_track",
        "resolve_pose_node_indices",
        "validate_workspace",
    }

    assert expected.issubset(set(api.__all__))
    assert "SleapTrack" not in api.__all__
    assert "read_sleap_node_names" not in api.__all__
    assert "read_sleap_track" not in api.__all__
    assert "resolve_sleap_node_indices" not in api.__all__
    assert api.Labels.__name__ == "Labels"
    assert api.ProjectDescriptor.__name__ == "ProjectDescriptor"
    assert api.PoseTrack.__name__ == "PoseTrack"
    assert api.WorkspaceService.__name__ == "WorkspaceService"
    assert callable(api.labels_from_json_payload)
    assert callable(api.labels_numpy)
    assert callable(api.labels_to_dataframe)
    assert callable(api.labels_to_json_payload)
    assert callable(api.read_pose_node_names)
    assert callable(api.read_pose_track)
    assert callable(api.resolve_pose_node_indices)
