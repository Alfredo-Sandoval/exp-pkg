from __future__ import annotations

import importlib

import pytest

import xpkg
from xpkg.codecs import (
    labels_from_json_payload,
    labels_numpy,
    labels_to_dataframe,
    labels_to_json_payload,
    read_vicon_json_payload,
    vicon_recording_from_json_payload,
    vicon_recording_to_json_payload,
)
from xpkg.formats import (
    ARTIFACT_INDEX_FILENAME,
    ARTIFACT_MANIFEST_FILENAME,
    ARTIFACT_SCHEMA_VERSION,
    ARTIFACTS_DIRNAME,
    EXPKG_SUFFIX,
    FIGURE_ARTIFACT_SCHEMA_VERSION,
    FIGURE_ARTIFACT_TYPE,
    FIGURE_MANIFEST_FILENAME,
    FIGURES_DIRNAME,
    PROJECT_DESCRIPTOR_FILENAME,
    ArtifactFile,
    ArtifactIndexEntry,
    ArtifactManifest,
    ArtifactOutputSpec,
    FigureArtifact,
    ProjectDescriptor,
    SegmentationFrame,
    WorkspaceInspection,
    artifact_kind_dir,
    clear_workspace_segmentation_masks,
    current_project_snapshot_path,
    current_project_state_path,
    default_expkg_path,
    import_dlc_csv_workspace,
    import_dlc_h5_workspace,
    import_dlc_project_workspace,
    import_lightning_pose_csv_workspace,
    import_mediapipe_pose_landmarks_json_workspace,
    import_mmpose_topdown_json_workspace,
    import_sleap_h5_workspace,
    import_sleap_package_workspace,
    import_vicon_c3d_workspace,
    import_vicon_csv_workspace,
    import_vicon_workspace,
    init_project,
    inspect_workspace,
    is_workspace_root,
    list_workspace_artifact_index,
    list_workspace_artifacts,
    list_workspace_figures,
    load_project_descriptor,
    load_workspace_artifact,
    load_workspace_figure,
    load_workspace_metadata,
    load_workspace_metadata_field,
    load_workspace_payload,
    load_workspace_segmentation_frames,
    load_workspace_segmentation_masks,
    load_workspace_vicon_recording,
    migrate_legacy_archive,
    pack_project,
    project_descriptor_path,
    read_labels_json_payload,
    rebuild_workspace_artifact_index,
    resolve_workspace_root,
    save_workspace_artifact,
    save_workspace_figure,
    save_workspace_labels,
    save_workspace_metadata,
    save_workspace_metadata_field,
    save_workspace_segmentation_masks,
    unpack_project,
    validate_artifact,
    validate_expkg,
    validate_workspace,
    validate_workspace_artifact,
    validate_workspace_artifacts,
    validate_workspace_figure,
    validate_workspace_figures,
    workspace_artifact_index_path,
    workspace_artifact_root,
    workspace_artifact_type_root,
    workspace_artifacts_root,
    workspace_figure_root,
    workspace_figures_root,
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
    ViconEvent,
    ViconRecording,
    Video,
    VideoStub,
    build_keypoint_skeleton,
    build_prediction_stub,
    is_predicted_instance,
    load_skeleton,
    load_skeleton_dlc,
    load_skeleton_sleap,
    load_skeleton_ultralytics,
    load_skeleton_xpkg_json,
)
from xpkg.services import (
    WorkspaceArtifacts,
    WorkspaceFigures,
    WorkspaceImports,
    WorkspaceLayout,
    WorkspaceSegmentation,
    WorkspaceService,
)
from xpkg.services import (
    WorkspaceInspection as ServiceWorkspaceInspection,
)


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
    assert ARTIFACTS_DIRNAME == "artifacts"
    assert ARTIFACT_INDEX_FILENAME == "index.json"
    assert ARTIFACT_MANIFEST_FILENAME == "manifest.json"
    assert ARTIFACT_SCHEMA_VERSION == "1.0.0"
    assert EXPKG_SUFFIX == ".expkg"
    assert FIGURE_ARTIFACT_SCHEMA_VERSION == "1.0.0"
    assert FIGURE_ARTIFACT_TYPE == "figure"
    assert FIGURE_MANIFEST_FILENAME == "manifest.json"
    assert FIGURES_DIRNAME == "figures"
    assert PROJECT_DESCRIPTOR_FILENAME == "PROJECT.json"
    assert ArtifactFile is not None
    assert ArtifactIndexEntry is not None
    assert ArtifactManifest is not None
    assert ArtifactOutputSpec is not None
    assert FigureArtifact is not None
    assert ProjectDescriptor is not None
    assert SegmentationFrame is not None
    assert WorkspaceInspection is not None
    assert callable(clear_workspace_segmentation_masks)
    assert callable(current_project_snapshot_path)
    assert callable(current_project_state_path)
    assert callable(default_expkg_path)
    assert callable(import_vicon_c3d_workspace)
    assert callable(import_vicon_csv_workspace)
    assert callable(import_vicon_workspace)
    assert callable(import_dlc_csv_workspace)
    assert callable(import_dlc_h5_workspace)
    assert callable(import_dlc_project_workspace)
    assert callable(import_lightning_pose_csv_workspace)
    assert callable(import_mediapipe_pose_landmarks_json_workspace)
    assert callable(import_mmpose_topdown_json_workspace)
    assert callable(import_sleap_h5_workspace)
    assert callable(import_sleap_package_workspace)
    assert callable(init_project)
    assert callable(inspect_workspace)
    assert callable(is_workspace_root)
    assert callable(artifact_kind_dir)
    assert callable(list_workspace_artifact_index)
    assert callable(list_workspace_artifacts)
    assert callable(list_workspace_figures)
    assert callable(load_workspace_artifact)
    assert callable(load_workspace_figure)
    assert callable(load_project_descriptor)
    assert callable(load_workspace_metadata)
    assert callable(load_workspace_metadata_field)
    assert callable(load_workspace_payload)
    assert callable(load_workspace_segmentation_frames)
    assert callable(load_workspace_segmentation_masks)
    assert callable(load_workspace_vicon_recording)
    assert callable(migrate_legacy_archive)
    assert callable(pack_project)
    assert callable(project_descriptor_path)
    assert callable(read_labels_json_payload)
    assert callable(resolve_workspace_root)
    assert callable(rebuild_workspace_artifact_index)
    assert callable(save_workspace_artifact)
    assert callable(save_workspace_figure)
    assert callable(save_workspace_metadata)
    assert callable(save_workspace_labels)
    assert callable(save_workspace_metadata_field)
    assert callable(save_workspace_segmentation_masks)
    assert callable(unpack_project)
    assert callable(validate_artifact)
    assert callable(validate_expkg)
    assert callable(validate_workspace_artifact)
    assert callable(validate_workspace_artifacts)
    assert callable(validate_workspace_figure)
    assert callable(validate_workspace_figures)
    assert callable(validate_workspace)
    assert callable(workspace_artifacts_root)
    assert callable(workspace_artifact_index_path)
    assert callable(workspace_artifact_root)
    assert callable(workspace_artifact_type_root)
    assert callable(workspace_figure_root)
    assert callable(workspace_figures_root)
    assert callable(write_labels_json)
    assert callable(write_project_descriptor)
    assert callable(labels_from_json_payload)
    assert callable(labels_numpy)
    assert callable(labels_to_dataframe)
    assert callable(labels_to_json_payload)
    assert callable(read_vicon_json_payload)
    assert callable(vicon_recording_from_json_payload)
    assert callable(vicon_recording_to_json_payload)
    assert WorkspaceImports is not None
    assert WorkspaceArtifacts is not None
    assert WorkspaceFigures is not None
    assert ServiceWorkspaceInspection is not None
    assert WorkspaceLayout is not None
    assert WorkspaceSegmentation is not None
    assert WorkspaceService is not None


def test_services_surface_lists_workspace_service_first() -> None:
    assert xpkg.services.__all__ == [
        "WorkspaceService",
        "WorkspaceImports",
        "WorkspaceLayout",
        "WorkspaceInspection",
        "WorkspaceArtifacts",
        "WorkspaceFigures",
        "WorkspaceSegmentation",
    ]


def test_workspace_imports_surface_covers_supported_workspace_importers() -> None:
    expected = {
        "dlc_csv",
        "dlc_h5",
        "dlc_project",
        "lightning_pose_csv",
        "mediapipe_pose_landmarks_json",
        "mmpose_topdown_json",
        "sleap_h5",
        "sleap_package",
        "vicon",
        "vicon_c3d",
        "vicon_csv",
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
    assert VideoStub is not None
    assert ViconRecording is not None
    assert ViconEvent is not None
    assert KPFlag is not None
    assert callable(build_prediction_stub)
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
    assert "export_workspace_archive" not in xpkg.formats.__all__
    assert "import_dlc_project_workspace" in xpkg.formats.__all__
    assert "import_lightning_pose_csv_workspace" in xpkg.formats.__all__
    assert "inspect_workspace" in xpkg.formats.__all__
    assert "load_workspace_payload" in xpkg.formats.__all__
    assert "list_workspace_figures" in xpkg.formats.__all__
    assert "save_workspace_figure" in xpkg.formats.__all__
    assert "workspace_artifacts_root" in xpkg.formats.__all__
    assert "load_workspace_metadata" in xpkg.formats.__all__
    assert "load_workspace_metadata_field" in xpkg.formats.__all__
    assert "import_vicon_workspace" in xpkg.formats.__all__
    assert "migrate_legacy_archive" in xpkg.formats.__all__
    assert "save_workspace_metadata" in xpkg.formats.__all__
    assert "save_workspace_metadata_field" in xpkg.formats.__all__
    assert "save_workspace_segmentation_masks" in xpkg.formats.__all__
    assert "load_workspace_segmentation_masks" in xpkg.formats.__all__
    assert "import_detectron2_coco_workspace" not in xpkg.formats.__all__
    assert "import_openpose_json_workspace" not in xpkg.formats.__all__

    with pytest.raises(AttributeError):
        xpkg.formats.__getattribute__("read_archive")

    with pytest.raises(AttributeError):
        xpkg.formats.__getattribute__("create_store_from_archive")


def test_direct_compat_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("xpkg.compat")


def test_codecs_surface_is_curated() -> None:
    assert sorted(xpkg.codecs.__all__) == [
        "labels_from_json_payload",
        "labels_numpy",
        "labels_to_dataframe",
        "labels_to_json_payload",
        "read_vicon_json_payload",
        "vicon_recording_from_json_payload",
        "vicon_recording_to_json_payload",
    ]
