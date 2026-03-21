from __future__ import annotations

import posetta
from posetta.adapters import (
    ConversionResult,
    convert_dlc_csv,
    convert_dlc_h5,
    convert_dlc_h5_project,
    convert_dlc_project,
    convert_sleap_package,
)
from posetta.formats import (
    POSEPROJ_SUFFIX,
    PROJECT_DESCRIPTOR_FILENAME,
    LazyDatasetHandle,
    MaxInstancesExceededError,
    PredictionAppendItem,
    ProjectDescriptor,
    SerializerPredictedInstance,
    SiestaStore,
    append_predictions_siesta,
    create_store_from_archive,
    create_store_from_sta,
    current_project_archive_path,
    default_poseproj_path,
    import_dlc_csv_workspace,
    import_dlc_h5_workspace,
    import_legacy_archive,
    import_sleap_package_workspace,
    init_project,
    is_workspace_root,
    load_project_descriptor,
    merge_predictions_siesta,
    migrate_legacy_archive,
    open_store,
    pack_project,
    project_descriptor_path,
    read_labels_json_payload,
    read_metrics_table,
    read_siesta,
    resolve_workspace_root,
    save_workspace_labels,
    summarize_project,
    unpack_project,
    update_labels_siesta,
    validate_artifact,
    validate_poseproj,
    validate_project,
    validate_workspace,
    write_labels_json,
    write_metrics_table,
    write_project_descriptor,
    write_siesta,
)
from posetta.model import (
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
    load_skeleton_siesta_json,
    load_skeleton_sleap,
    load_skeleton_ultralytics,
)
from posetta.services import WorkspaceLayout, WorkspaceService


def test_public_exports_are_callable() -> None:
    assert posetta.__version__
    assert posetta.adapters is not None
    assert posetta.formats is not None
    assert posetta.model is not None
    assert posetta.services is not None
    assert ConversionResult is not None
    assert LazyDatasetHandle is not None
    assert PredictionAppendItem is not None
    assert SerializerPredictedInstance is not None
    assert MaxInstancesExceededError is not None
    assert POSEPROJ_SUFFIX == ".poseproj"
    assert SiestaStore is not None
    assert PROJECT_DESCRIPTOR_FILENAME == "PROJECT.json"
    assert ProjectDescriptor is not None
    assert callable(append_predictions_siesta)
    assert callable(current_project_archive_path)
    assert callable(create_store_from_archive)
    assert callable(create_store_from_sta)
    assert callable(default_poseproj_path)
    assert callable(import_dlc_csv_workspace)
    assert callable(import_dlc_h5_workspace)
    assert callable(import_legacy_archive)
    assert callable(import_sleap_package_workspace)
    assert callable(init_project)
    assert callable(is_workspace_root)
    assert callable(load_project_descriptor)
    assert callable(merge_predictions_siesta)
    assert callable(migrate_legacy_archive)
    assert callable(open_store)
    assert callable(pack_project)
    assert callable(project_descriptor_path)
    assert callable(read_labels_json_payload)
    assert callable(read_metrics_table)
    assert callable(read_siesta)
    assert callable(resolve_workspace_root)
    assert callable(save_workspace_labels)
    assert callable(summarize_project)
    assert callable(unpack_project)
    assert callable(update_labels_siesta)
    assert callable(validate_artifact)
    assert callable(validate_poseproj)
    assert callable(validate_project)
    assert callable(validate_workspace)
    assert callable(write_labels_json)
    assert callable(write_project_descriptor)
    assert callable(write_metrics_table)
    assert callable(write_siesta)
    assert callable(convert_dlc_csv)
    assert callable(convert_dlc_h5)
    assert callable(convert_dlc_h5_project)
    assert callable(convert_dlc_project)
    assert callable(convert_sleap_package)
    assert WorkspaceLayout is not None
    assert WorkspaceService is not None


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
    assert callable(load_skeleton_siesta_json)
    assert callable(load_skeleton_sleap)
    assert callable(load_skeleton_ultralytics)
