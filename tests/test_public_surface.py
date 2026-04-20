from __future__ import annotations

import importlib

import pytest

import xpkg
from xpkg.codecs import (
    labels_from_json_payload,
    labels_numpy,
    labels_to_dataframe,
    labels_to_json_payload,
)
from xpkg.formats import (
    EXPKG_SUFFIX,
    PROJECT_DESCRIPTOR_FILENAME,
    ProjectDescriptor,
    current_project_snapshot_path,
    current_project_state_path,
    default_expkg_path,
    import_detectron2_coco_workspace,
    import_dlc_csv_workspace,
    import_dlc_h5_workspace,
    import_dlc_project_workspace,
    import_mediapipe_pose_landmarks_json_workspace,
    import_mmpose_topdown_json_workspace,
    import_openpose_json_workspace,
    import_sleap_h5_workspace,
    import_sleap_package_workspace,
    init_project,
    is_workspace_root,
    load_project_descriptor,
    migrate_legacy_archive,
    pack_project,
    project_descriptor_path,
    read_labels_json_payload,
    resolve_workspace_root,
    save_workspace_labels,
    unpack_project,
    validate_artifact,
    validate_expkg,
    validate_workspace,
    write_labels_json,
    write_project_descriptor,
)
from xpkg.model import (
    Instance,
    Keypoint,
    KPFlag,
    LabeledFrame,
    Labels,
    Point,
    PointArray,
    PredictedInstance,
    PredictedPoint,
    PredictedPointArray,
    Skeleton,
    SuggestionFrame,
    Track,
    Video,
    build_keypoint_skeleton,
    is_predicted_instance,
    load_skeleton,
    load_skeleton_dlc,
    load_skeleton_sleap,
    load_skeleton_ultralytics,
    load_skeleton_xpkg_json,
)
from xpkg.services import WorkspaceImports, WorkspaceLayout, WorkspaceService


def test_root_namespace_is_curated_to_workspace_first_modules() -> None:
    reloaded = importlib.reload(xpkg)
    reloaded.__dict__.pop("compat", None)
    reloaded.__dict__.pop("adapters", None)

    assert reloaded.__version__
    assert reloaded.__all__ == ["__version__", "api", "codecs", "formats", "model", "services"]
    assert reloaded.codecs is not None
    assert reloaded.formats is not None
    assert reloaded.model is not None
    assert reloaded.services is not None

    with pytest.raises(AttributeError):
        reloaded.__getattribute__("compat")

    with pytest.raises(AttributeError):
        reloaded.__getattribute__("adapters")


def test_public_exports_are_callable() -> None:
    assert EXPKG_SUFFIX == ".expkg"
    assert PROJECT_DESCRIPTOR_FILENAME == "PROJECT.json"
    assert ProjectDescriptor is not None
    assert callable(current_project_snapshot_path)
    assert callable(current_project_state_path)
    assert callable(default_expkg_path)
    assert callable(import_detectron2_coco_workspace)
    assert callable(import_dlc_csv_workspace)
    assert callable(import_dlc_h5_workspace)
    assert callable(import_dlc_project_workspace)
    assert callable(import_mediapipe_pose_landmarks_json_workspace)
    assert callable(import_mmpose_topdown_json_workspace)
    assert callable(import_openpose_json_workspace)
    assert callable(import_sleap_h5_workspace)
    assert callable(import_sleap_package_workspace)
    assert callable(init_project)
    assert callable(is_workspace_root)
    assert callable(load_project_descriptor)
    assert callable(migrate_legacy_archive)
    assert callable(pack_project)
    assert callable(project_descriptor_path)
    assert callable(read_labels_json_payload)
    assert callable(resolve_workspace_root)
    assert callable(save_workspace_labels)
    assert callable(unpack_project)
    assert callable(validate_artifact)
    assert callable(validate_expkg)
    assert callable(validate_workspace)
    assert callable(write_labels_json)
    assert callable(write_project_descriptor)
    assert callable(labels_from_json_payload)
    assert callable(labels_numpy)
    assert callable(labels_to_dataframe)
    assert callable(labels_to_json_payload)
    assert WorkspaceImports is not None
    assert WorkspaceLayout is not None
    assert WorkspaceService is not None


def test_services_surface_lists_workspace_service_first() -> None:
    assert xpkg.services.__all__ == ["WorkspaceService", "WorkspaceImports", "WorkspaceLayout"]


def test_workspace_imports_surface_covers_supported_workspace_importers() -> None:
    expected = {
        "detectron2_coco",
        "dlc_csv",
        "dlc_h5",
        "dlc_project",
        "mediapipe_pose_landmarks_json",
        "mmpose_topdown_json",
        "openpose_json",
        "sleap_h5",
        "sleap_package",
    }

    assert expected.issubset(set(dir(WorkspaceImports)))
    assert "legacy_archive" not in dir(WorkspaceImports)


def test_model_exports_are_available() -> None:
    assert Labels is not None
    assert SuggestionFrame is not None
    assert Skeleton is not None
    assert Keypoint is not None
    assert Track is not None
    assert LabeledFrame is not None
    assert Instance is not None
    assert PredictedInstance is not None
    assert Point is not None
    assert PredictedPoint is not None
    assert PointArray is not None
    assert PredictedPointArray is not None
    assert Video is not None
    assert KPFlag is not None
    assert callable(build_keypoint_skeleton)
    assert callable(is_predicted_instance)
    assert callable(load_skeleton)
    assert callable(load_skeleton_dlc)
    assert callable(load_skeleton_xpkg_json)
    assert callable(load_skeleton_sleap)
    assert callable(load_skeleton_ultralytics)
    assert "load_skeleton_archive_json" not in xpkg.model.__all__


def test_formats_surface_is_workspace_first_only() -> None:
    assert "read_archive" not in xpkg.formats.__all__
    assert "write_archive" not in xpkg.formats.__all__
    assert "read_xpkg" not in xpkg.formats.__all__
    assert "write_xpkg" not in xpkg.formats.__all__
    assert "export_project_archive" not in xpkg.formats.__all__
    assert "current_project_archive_path" not in xpkg.formats.__all__
    assert "import_legacy_archive" not in xpkg.formats.__all__
    assert "pack_project" in xpkg.formats.__all__
    assert "import_dlc_project_workspace" in xpkg.formats.__all__
    assert "migrate_legacy_archive" in xpkg.formats.__all__

    with pytest.raises(AttributeError):
        xpkg.formats.__getattribute__("read_archive")

    with pytest.raises(AttributeError):
        xpkg.formats.__getattribute__("create_store_from_archive")


def test_direct_compat_module_keeps_only_canonical_xpkg_names() -> None:
    compat = importlib.import_module("xpkg.compat")

    assert callable(compat.read_xpkg)
    assert callable(compat.write_xpkg)
    assert callable(compat.update_labels_xpkg)
    assert callable(compat.append_predictions_xpkg)
    assert callable(compat.merge_predictions_xpkg)
    assert callable(compat.summarize_xpkg)
    assert callable(compat.validate_xpkg)
    assert callable(compat.create_store_from_xpkg)
    assert callable(compat.read_metrics_table)
    assert callable(compat.write_metrics_table)
    assert "read_archive" not in compat.__all__
    assert "write_archive" not in compat.__all__
    assert "create_archive_store" not in compat.__all__

    with pytest.raises(AttributeError):
        compat.__getattribute__("read_archive")

    with pytest.raises(AttributeError):
        compat.__getattribute__("write_archive")


def test_codecs_surface_is_curated() -> None:
    assert sorted(xpkg.codecs.__all__) == [
        "labels_from_json_payload",
        "labels_numpy",
        "labels_to_dataframe",
        "labels_to_json_payload",
    ]
