from __future__ import annotations

import importlib
import warnings

import xpkg
from xpkg.adapters import (
    ConversionResult,
    convert_dlc_csv,
    convert_dlc_h5,
    convert_dlc_h5_project,
    convert_dlc_project,
    convert_sleap_h5,
    convert_sleap_package,
)
from xpkg.codecs import (
    labels_from_json_payload,
    labels_numpy,
    labels_to_dataframe,
    labels_to_json_payload,
)
from xpkg.compat import (
    append_predictions_xpkg,
    create_store_from_xpkg,
    merge_predictions_xpkg,
    read_metrics_table,
    read_xpkg,
    summarize_xpkg,
    update_labels_xpkg,
    validate_xpkg,
    write_metrics_table,
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
    import_dlc_project_workspace,
    import_legacy_archive,
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
from xpkg.services import WorkspaceLayout, WorkspaceService


def test_public_exports_are_callable() -> None:
    assert xpkg.__version__
    assert xpkg.adapters is not None
    assert xpkg.codecs is not None
    assert xpkg.compat is not None
    assert xpkg.formats is not None
    assert xpkg.model is not None
    assert xpkg.services is not None
    assert ConversionResult is not None
    assert EXPKG_SUFFIX == ".expkg"
    assert PROJECT_DESCRIPTOR_FILENAME == "PROJECT.json"
    assert ProjectDescriptor is not None
    assert callable(append_predictions_xpkg)
    assert callable(create_store_from_xpkg)
    assert callable(current_project_archive_path)
    assert callable(current_project_snapshot_path)
    assert callable(current_project_state_path)
    assert callable(default_expkg_path)
    assert callable(import_dlc_csv_workspace)
    assert callable(import_dlc_h5_workspace)
    assert callable(import_dlc_project_workspace)
    assert callable(import_legacy_archive)
    assert callable(import_sleap_h5_workspace)
    assert callable(import_sleap_package_workspace)
    assert callable(init_project)
    assert callable(is_workspace_root)
    assert callable(load_project_descriptor)
    assert callable(merge_predictions_xpkg)
    assert callable(migrate_legacy_archive)
    assert callable(pack_project)
    assert callable(project_descriptor_path)
    assert callable(read_labels_json_payload)
    assert callable(read_metrics_table)
    assert callable(read_xpkg)
    assert callable(resolve_workspace_root)
    assert callable(save_workspace_labels)
    assert callable(summarize_xpkg)
    assert callable(unpack_project)
    assert callable(update_labels_xpkg)
    assert callable(validate_artifact)
    assert callable(validate_expkg)
    assert callable(validate_xpkg)
    assert callable(validate_workspace)
    assert callable(write_labels_json)
    assert callable(write_metrics_table)
    assert callable(write_project_descriptor)
    assert callable(write_xpkg)
    assert callable(convert_dlc_csv)
    assert callable(convert_dlc_h5)
    assert callable(convert_dlc_h5_project)
    assert callable(convert_dlc_project)
    assert callable(convert_sleap_h5)
    assert callable(convert_sleap_package)
    assert callable(labels_from_json_payload)
    assert callable(labels_numpy)
    assert callable(labels_to_dataframe)
    assert callable(labels_to_json_payload)
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
    assert callable(load_skeleton_sleap)
    assert callable(load_skeleton_ultralytics)
    assert "load_skeleton_archive_json" not in xpkg.model.__all__


def test_formats_core_surface_excludes_compat_symbols() -> None:
    assert "read_archive" not in xpkg.formats.__all__
    assert "write_archive" not in xpkg.formats.__all__
    assert "read_xpkg" not in xpkg.formats.__all__
    assert "write_xpkg" not in xpkg.formats.__all__
    assert "pack_project" in xpkg.formats.__all__
    assert "import_dlc_project_workspace" in xpkg.formats.__all__
    assert "read_archive" not in dir(xpkg.formats)
    assert "write_archive" not in dir(xpkg.formats)
    assert "ArchiveStore" not in dir(xpkg.formats)


def test_compat_surface_prefers_canonical_xpkg_names() -> None:
    assert callable(read_xpkg)
    assert callable(write_xpkg)
    assert callable(update_labels_xpkg)
    assert callable(append_predictions_xpkg)
    assert callable(merge_predictions_xpkg)
    assert callable(summarize_xpkg)
    assert callable(validate_xpkg)
    assert callable(create_store_from_xpkg)
    assert callable(read_metrics_table)
    assert callable(write_metrics_table)


def test_compat_surface_hides_legacy_archive_aliases() -> None:
    compat = importlib.reload(xpkg.compat)
    assert "read_archive" not in compat.__all__
    assert "write_archive" not in compat.__all__
    assert "create_archive_store" not in compat.__all__

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert callable(compat.read_archive)
        assert callable(compat.write_archive)
        assert any("legacy alias" in str(item.message) for item in caught)


def test_codecs_surface_is_curated() -> None:
    assert sorted(xpkg.codecs.__all__) == [
        "labels_from_json_payload",
        "labels_numpy",
        "labels_to_dataframe",
        "labels_to_json_payload",
    ]
