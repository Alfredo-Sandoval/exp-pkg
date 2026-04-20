"""Smoke tests for the xpkg public API facade."""

from __future__ import annotations

import xpkg.api as api


def test_xpkg_api_exposes_workspace_first_contract() -> None:
    expected = {
        "Labels",
        "ProjectDescriptor",
        "PoseTrack",
        "WorkspaceImports",
        "WorkspaceLayout",
        "WorkspaceService",
        "current_project_snapshot_path",
        "current_project_state_path",
        "default_expkg_path",
        "import_detectron2_coco_workspace",
        "import_dlc_csv_workspace",
        "import_dlc_h5_workspace",
        "import_dlc_project_workspace",
        "import_mediapipe_pose_landmarks_json_workspace",
        "import_mmpose_topdown_json_workspace",
        "import_openpose_json_workspace",
        "import_sleap_h5_workspace",
        "import_sleap_package_workspace",
        "labels_from_json_payload",
        "labels_numpy",
        "labels_to_dataframe",
        "labels_to_json_payload",
        "migrate_legacy_archive",
        "pack_project",
        "read_pose_node_names",
        "read_pose_track",
        "resolve_pose_node_indices",
        "save_workspace_labels",
        "unpack_project",
        "validate_workspace",
    }

    assert expected.issubset(set(api.__all__))
    assert "ConversionResult" not in api.__all__
    assert "convert_dlc_csv" not in api.__all__
    assert "current_project_archive_path" not in api.__all__
    assert "export_project_archive" not in api.__all__
    assert "import_legacy_archive" not in api.__all__
    assert "SleapTrack" not in api.__all__
    assert "read_sleap_node_names" not in api.__all__
    assert "read_sleap_track" not in api.__all__
    assert "resolve_sleap_node_indices" not in api.__all__
    assert api.Labels.__name__ == "Labels"
    assert api.ProjectDescriptor.__name__ == "ProjectDescriptor"
    assert api.PoseTrack.__name__ == "PoseTrack"
    assert api.WorkspaceImports.__name__ == "WorkspaceImports"
    assert api.WorkspaceService.__name__ == "WorkspaceService"
    assert callable(api.labels_from_json_payload)
    assert callable(api.labels_numpy)
    assert callable(api.labels_to_dataframe)
    assert callable(api.labels_to_json_payload)
    assert callable(api.read_pose_node_names)
    assert callable(api.read_pose_track)
    assert callable(api.resolve_pose_node_indices)
    assert callable(api.migrate_legacy_archive)


def test_xpkg_api_lists_service_entrypoints_before_free_function_helpers() -> None:
    exports = api.__all__

    assert exports.index("WorkspaceService") < exports.index("WorkspaceImports")
    assert exports.index("WorkspaceImports") < exports.index("WorkspaceLayout")
    assert exports.index("WorkspaceService") < exports.index("import_dlc_csv_workspace")
    assert exports.index("WorkspaceService") < exports.index("migrate_legacy_archive")
    assert exports.index("import_dlc_csv_workspace") < exports.index("migrate_legacy_archive")
