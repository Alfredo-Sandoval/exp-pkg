from __future__ import annotations

import xpkg
from xpkg.adapters import (
    ConversionResult,
    convert_dlc_csv,
    convert_dlc_h5,
    convert_dlc_h5_project,
    convert_dlc_project,
    convert_sleap_package,
)
from xpkg.compat import (
    ArchiveStore,
    LazyDatasetHandle,
    MaxInstancesExceededError,
    PredictionAppendItem,
    SerializerPredictedInstance,
    SiestaStore,
    append_predictions_siesta,
    append_predictions_sta,
    append_predictions_xpkg,
    create_archive_store,
    create_store_from_archive,
    create_store_from_sta,
    create_store_from_xpkg,
    merge_predictions_siesta,
    merge_predictions_sta,
    merge_predictions_xpkg,
    open_archive_store,
    open_store,
    read_metrics_table,
    read_siesta,
    read_sta,
    read_xpkg,
    summarize_project,
    summarize_sta,
    summarize_xpkg,
    update_labels_siesta,
    update_labels_sta,
    update_labels_xpkg,
    validate_project,
    validate_sta,
    validate_xpkg,
    write_metrics_table,
    write_siesta,
    write_sta,
    write_xpkg,
)
from xpkg.formats import (
    EXPKG_SUFFIX,
    PROJECT_DESCRIPTOR_FILENAME,
    ProjectDescriptor,
    current_project_archive_path,
    current_project_snapshot_path,
    current_project_state_path,
    default_expkg_path,
    import_dlc_csv_workspace,
    import_dlc_h5_workspace,
    import_legacy_archive,
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
    load_skeleton_siesta_json,
    load_skeleton_sleap,
    load_skeleton_sta_json,
    load_skeleton_xpkg_json,
    load_skeleton_ultralytics,
)
from xpkg.services import WorkspaceLayout, WorkspaceService


def test_public_exports_are_callable() -> None:
    assert xpkg.__version__
    assert xpkg.adapters is not None
    assert xpkg.compat is not None
    assert xpkg.formats is not None
    assert xpkg.model is not None
    assert xpkg.services is not None
    assert ArchiveStore is not None
    assert ConversionResult is not None
    assert LazyDatasetHandle is not None
    assert PredictionAppendItem is not None
    assert SerializerPredictedInstance is not None
    assert MaxInstancesExceededError is not None
    assert EXPKG_SUFFIX == ".expkg"
    assert SiestaStore is not None
    assert PROJECT_DESCRIPTOR_FILENAME == "PROJECT.json"
    assert ProjectDescriptor is not None
    assert callable(append_predictions_siesta)
    assert callable(append_predictions_sta)
    assert callable(append_predictions_xpkg)
    assert callable(create_archive_store)
    assert callable(create_store_from_archive)
    assert callable(create_store_from_sta)
    assert callable(create_store_from_xpkg)
    assert callable(current_project_archive_path)
    assert callable(current_project_snapshot_path)
    assert callable(current_project_state_path)
    assert callable(default_expkg_path)
    assert callable(import_dlc_csv_workspace)
    assert callable(import_dlc_h5_workspace)
    assert callable(import_legacy_archive)
    assert callable(import_sleap_package_workspace)
    assert callable(init_project)
    assert callable(is_workspace_root)
    assert callable(load_project_descriptor)
    assert callable(merge_predictions_siesta)
    assert callable(merge_predictions_sta)
    assert callable(merge_predictions_xpkg)
    assert callable(migrate_legacy_archive)
    assert callable(open_archive_store)
    assert callable(open_store)
    assert callable(pack_project)
    assert callable(project_descriptor_path)
    assert callable(read_labels_json_payload)
    assert callable(read_metrics_table)
    assert callable(read_siesta)
    assert callable(read_sta)
    assert callable(read_xpkg)
    assert callable(resolve_workspace_root)
    assert callable(save_workspace_labels)
    assert callable(summarize_project)
    assert callable(summarize_sta)
    assert callable(summarize_xpkg)
    assert callable(unpack_project)
    assert callable(update_labels_siesta)
    assert callable(update_labels_sta)
    assert callable(update_labels_xpkg)
    assert callable(validate_artifact)
    assert callable(validate_expkg)
    assert callable(validate_project)
    assert callable(validate_sta)
    assert callable(validate_xpkg)
    assert callable(validate_workspace)
    assert callable(write_labels_json)
    assert callable(write_metrics_table)
    assert callable(write_project_descriptor)
    assert callable(write_siesta)
    assert callable(write_sta)
    assert callable(write_xpkg)
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
    assert callable(load_skeleton_xpkg_json)
    assert callable(load_skeleton_sta_json)
    assert callable(load_skeleton_siesta_json)
    assert callable(load_skeleton_sleap)
    assert callable(load_skeleton_ultralytics)


def test_formats_core_surface_excludes_compat_symbols() -> None:
    assert "read_siesta" not in xpkg.formats.__all__
    assert "write_siesta" not in xpkg.formats.__all__
    assert "create_store_from_sta" not in xpkg.formats.__all__
    assert "read_xpkg" not in xpkg.formats.__all__
    assert "write_xpkg" not in xpkg.formats.__all__
    assert "pack_project" in xpkg.formats.__all__
