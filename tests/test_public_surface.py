from __future__ import annotations

import importlib
import importlib.util

import pytest

import xpkg
from xpkg.adapters import (
    labels_from_json_payload,
    labels_numpy,
    labels_to_dataframe,
    labels_to_json_payload,
    read_vicon_json_payload,
    vicon_recording_from_json_payload,
    vicon_recording_to_json_payload,
)
from xpkg.model import (
    AcquisitionMetadata,
    CameraMetadata,
    DatasetShareMetadata,
    EMGSignalData,
    Event,
    EventTable,
    ForcePlateData,
    Instance,
    Keypoint,
    KPFlag,
    LabeledFrame,
    Labels,
    PhotometryChannel,
    PhotometryRecording,
    Point,
    PointArray,
    PredictedInstance,
    PredictedPoint,
    PredictedPointArray,
    RecordingSession,
    SignalChannel,
    Skeleton,
    SuggestionFrame,
    SyncEvent,
    Timebase,
    Timeline,
    TimeRange,
    TimeSeries,
    Track,
    ViconEvent,
    ViconForcePlatformMetadata,
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
from xpkg.project import (
    ACQUISITION_METADATA_FILENAME,
    ARTIFACT_INDEX_FILENAME,
    ARTIFACT_MANIFEST_FILENAME,
    ARTIFACT_SCHEMA_VERSION,
    ARTIFACTS_DIRNAME,
    DATASET_SHARE_METADATA_FILENAME,
    EXPKG_MANIFEST_FILENAME,
    EXPKG_SUFFIX,
    FIGURE_ARTIFACT_SCHEMA_VERSION,
    FIGURE_ARTIFACT_TYPE,
    FIGURE_MANIFEST_FILENAME,
    FIGURES_DIRNAME,
    PROJECT_DESCRIPTOR_FILENAME,
    PROJECT_METADATA_DIRNAME,
    ArtifactFile,
    ArtifactIndexEntry,
    ArtifactManifest,
    ArtifactOutputSpec,
    FigureArtifact,
    ProjectDescriptor,
    ProjectInspection,
    SegmentationFrame,
    artifact_kind_dir,
    clear_project_segmentation_masks,
    current_project_state_path,
    default_expkg_path,
    init_project,
    inspect_project,
    is_project_root,
    list_project_artifact_index,
    list_project_artifacts,
    list_project_figures,
    load_project_acquisition_metadata,
    load_project_artifact,
    load_project_dataset_share_metadata,
    load_project_datasheet,
    load_project_descriptor,
    load_project_figure,
    load_project_metadata,
    load_project_metadata_field,
    load_project_model_card,
    load_project_payload,
    load_project_pose_provenance,
    load_project_segmentation_frames,
    load_project_segmentation_masks,
    load_project_vicon_recording,
    pack_project,
    project_acquisition_metadata_path,
    project_artifact_index_path,
    project_artifact_root,
    project_artifact_type_root,
    project_artifacts_root,
    project_dataset_share_metadata_path,
    project_descriptor_path,
    project_figure_root,
    project_figures_root,
    project_metadata_root,
    read_labels_json_payload,
    rebuild_project_artifact_index,
    resolve_project_root,
    save_project_acquisition_metadata,
    save_project_artifact,
    save_project_dataset_share_metadata,
    save_project_datasheet,
    save_project_figure,
    save_project_labels,
    save_project_metadata,
    save_project_metadata_field,
    save_project_model_card,
    save_project_pose_provenance,
    save_project_segmentation_masks,
    unpack_project,
    validate_artifact,
    validate_expkg,
    validate_project,
    validate_project_artifact,
    validate_project_artifacts,
    validate_project_figure,
    validate_project_figures,
    write_labels_json,
    write_project_descriptor,
)
from xpkg.readers import (
    read_abf,
    read_doric_photometry,
    read_ephys_csv,
    read_events_csv,
    read_neurophotometrics_csv,
    read_photometry_csv,
    read_pmat_events_csv,
    read_pmat_photometry_csv,
    read_pyphotometry_csv,
    read_pyphotometry_ppd,
    read_rwd_ofrs_session,
    read_tdt_photometry_block,
    read_teleopto_h5,
)
from xpkg.services import (
    ProjectArtifacts,
    ProjectCalibrations,
    ProjectFigures,
    ProjectLayout,
    ProjectMetadata,
    ProjectSegmentation,
    ProjectService,
)
from xpkg.services import (
    ProjectInspection as ServiceProjectInspection,
)

if importlib.util.find_spec("primitives") is not None:
    from xpkg.adapters import (  # noqa: E402
        labels_to_primitives_session,
        project_to_primitives_session,
    )
else:
    labels_to_primitives_session = None
    project_to_primitives_session = None


def test_root_namespace_is_curated_to_project_first_modules() -> None:
    reloaded = importlib.reload(xpkg)
    reloaded.__dict__.pop("compat", None)
    reloaded.__dict__.pop("adapters", None)
    reloaded.__dict__.pop("json_utils", None)
    reloaded.__dict__.pop("media", None)
    reloaded.__dict__.pop("payloads", None)
    reloaded.__dict__.pop("pose", None)
    reloaded.__dict__.pop("project", None)
    reloaded.__dict__.pop("readers", None)
    reloaded.__dict__.pop("segmentation", None)

    assert reloaded.__version__
    assert reloaded.__all__ == [
        "__version__",
        "adapters",
        "json_utils",
        "media",
        "model",
        "payloads",
        "pose",
        "project",
        "readers",
        "segmentation",
        "services",
    ]
    assert reloaded.adapters is not None
    assert reloaded.json_utils is not None
    assert reloaded.media is not None
    assert reloaded.payloads is not None
    assert reloaded.project is not None
    assert reloaded.model is not None
    assert reloaded.pose is not None
    assert reloaded.readers is not None
    assert reloaded.segmentation is not None
    assert reloaded.services is not None

    assert callable(reloaded.readers.read_abf)
    assert callable(reloaded.readers.read_doric_photometry)
    assert callable(reloaded.readers.read_ephys_csv)
    assert callable(reloaded.readers.read_events_csv)
    assert callable(reloaded.readers.read_neurophotometrics_csv)
    assert callable(reloaded.readers.read_photometry_csv)
    assert callable(reloaded.readers.read_pmat_events_csv)
    assert callable(reloaded.readers.read_pmat_photometry_csv)
    assert callable(reloaded.readers.read_pyphotometry_csv)
    assert callable(reloaded.readers.read_pyphotometry_ppd)
    assert callable(reloaded.readers.read_rwd_ofrs_session)
    assert callable(reloaded.readers.read_tdt_photometry_block)
    assert callable(reloaded.readers.read_teleopto_h5)

    with pytest.raises(AttributeError):
        reloaded.__getattribute__("compat")

    with pytest.raises(AttributeError):
        reloaded.__getattribute__("api")

    with pytest.raises(AttributeError):
        reloaded.__getattribute__("read_doric_photometry")

    with pytest.raises(AttributeError):
        reloaded.__getattribute__("exchange")

    with pytest.raises(AttributeError):
        reloaded.__getattribute__("formats")

def test_public_exports_are_callable() -> None:
    assert ARTIFACTS_DIRNAME == "artifacts"
    assert ARTIFACT_INDEX_FILENAME == "index.json"
    assert ARTIFACT_MANIFEST_FILENAME == "manifest.json"
    assert ARTIFACT_SCHEMA_VERSION == "1.0.0"
    assert EXPKG_MANIFEST_FILENAME == "EXPKG.json"
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
    assert ProjectInspection is not None
    assert ACQUISITION_METADATA_FILENAME == "acquisition.json"
    assert DATASET_SHARE_METADATA_FILENAME == "dataset_share.json"
    assert PROJECT_METADATA_DIRNAME == "metadata"
    assert callable(clear_project_segmentation_masks)
    assert callable(current_project_state_path)
    assert callable(default_expkg_path)
    assert callable(init_project)
    assert callable(inspect_project)
    assert callable(is_project_root)
    assert callable(artifact_kind_dir)
    assert callable(list_project_artifact_index)
    assert callable(list_project_artifacts)
    assert callable(list_project_figures)
    assert callable(load_project_artifact)
    assert callable(load_project_acquisition_metadata)
    assert callable(load_project_figure)
    assert callable(load_project_descriptor)
    assert callable(load_project_dataset_share_metadata)
    assert callable(load_project_datasheet)
    assert callable(load_project_metadata)
    assert callable(load_project_metadata_field)
    assert callable(load_project_model_card)
    assert callable(load_project_payload)
    assert callable(load_project_pose_provenance)
    assert callable(load_project_segmentation_frames)
    assert callable(load_project_segmentation_masks)
    assert callable(load_project_vicon_recording)
    assert callable(pack_project)
    assert callable(project_acquisition_metadata_path)
    assert callable(project_descriptor_path)
    assert callable(read_labels_json_payload)
    assert callable(resolve_project_root)
    assert callable(rebuild_project_artifact_index)
    assert callable(save_project_artifact)
    assert callable(save_project_acquisition_metadata)
    assert callable(save_project_dataset_share_metadata)
    assert callable(save_project_datasheet)
    assert callable(save_project_figure)
    assert callable(save_project_metadata)
    assert callable(save_project_labels)
    assert callable(save_project_metadata_field)
    assert callable(save_project_model_card)
    assert callable(save_project_pose_provenance)
    assert callable(save_project_segmentation_masks)
    assert callable(unpack_project)
    assert callable(validate_artifact)
    assert callable(validate_expkg)
    assert callable(validate_project_artifact)
    assert callable(validate_project_artifacts)
    assert callable(validate_project_figure)
    assert callable(validate_project_figures)
    assert callable(validate_project)
    assert callable(project_artifacts_root)
    assert callable(project_artifact_index_path)
    assert callable(project_artifact_root)
    assert callable(project_artifact_type_root)
    assert callable(project_dataset_share_metadata_path)
    assert callable(project_figure_root)
    assert callable(project_figures_root)
    assert callable(project_metadata_root)
    assert callable(write_labels_json)
    assert callable(write_project_descriptor)
    assert callable(labels_from_json_payload)
    assert callable(labels_numpy)
    assert callable(labels_to_dataframe)
    assert callable(labels_to_json_payload)
    if labels_to_primitives_session is not None:
        assert callable(labels_to_primitives_session)
    if project_to_primitives_session is not None:
        assert callable(project_to_primitives_session)
    assert callable(read_vicon_json_payload)
    assert callable(vicon_recording_from_json_payload)
    assert callable(vicon_recording_to_json_payload)
    assert callable(read_abf)
    assert callable(read_doric_photometry)
    assert callable(read_ephys_csv)
    assert callable(read_events_csv)
    assert callable(read_neurophotometrics_csv)
    assert callable(read_photometry_csv)
    assert callable(read_pmat_events_csv)
    assert callable(read_pmat_photometry_csv)
    assert callable(read_pyphotometry_csv)
    assert callable(read_pyphotometry_ppd)
    assert callable(read_rwd_ofrs_session)
    assert callable(read_tdt_photometry_block)
    assert callable(read_teleopto_h5)
    assert ProjectArtifacts is not None
    assert ProjectCalibrations is not None
    assert ProjectFigures is not None
    assert ProjectMetadata is not None
    assert ServiceProjectInspection is not None
    assert ProjectLayout is not None
    assert ProjectSegmentation is not None
    assert ProjectService is not None


def test_services_surface_lists_project_service_first() -> None:
    assert xpkg.services.__all__ == [
        "ProjectService",
        "ProjectLayout",
        "ProjectInspection",
        "ProjectArtifacts",
        "ProjectCalibrations",
        "ProjectFigures",
        "ProjectMetadata",
        "ProjectSegmentation",
        "PoseFormat",
        "CalibrationFormat",
        "MotionFormat",
    ]


def test_project_service_dispatches_supported_pose_calibration_and_motion_formats() -> None:
    pose_formats: set[str] = {
        "dlc-csv",
        "dlc-h5",
        "dlc-project",
        "lightning-pose-csv",
        "mediapipe-pose-landmarks-json",
        "mmpose-topdown-json",
        "sleap-h5",
        "sleap-package",
    }
    calibration_formats: set[str] = {"anipose"}
    motion_formats: set[str] = {"vicon", "vicon-csv", "vicon-c3d"}

    from typing import get_args

    from xpkg.services import CalibrationFormat, MotionFormat, PoseFormat

    assert pose_formats == set(get_args(PoseFormat))
    assert calibration_formats == set(get_args(CalibrationFormat))
    assert motion_formats == set(get_args(MotionFormat))


def test_model_exports_are_available() -> None:
    assert AcquisitionMetadata is not None
    assert CameraMetadata is not None
    assert DatasetShareMetadata is not None
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
    assert EMGSignalData is not None
    assert Event is not None
    assert EventTable is not None
    assert ForcePlateData is not None
    assert PhotometryChannel is not None
    assert PhotometryRecording is not None
    assert RecordingSession is not None
    assert SignalChannel is not None
    assert SyncEvent is not None
    assert Timeline is not None
    assert TimeRange is not None
    assert TimeSeries is not None
    assert Timebase is not None
    assert ViconRecording is not None
    assert ViconEvent is not None
    assert ViconForcePlatformMetadata is not None
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


def test_media_surface_is_public() -> None:
    assert "media" in xpkg.__all__
    assert xpkg.media.Video is Video
    assert "VideoWithFrames" in xpkg.media.__all__
    assert "video_total_frames" in xpkg.media.__all__
    assert callable(xpkg.media.available_hardware_accelerators)
    assert callable(xpkg.media.hardware_acceleration_status)
    assert callable(xpkg.media.read_bgr)
    assert callable(xpkg.media.read_rgb)
    assert callable(xpkg.media.read_rgb_bytes)
    assert callable(xpkg.media.require_hardware_acceleration)
    assert callable(xpkg.media.video_total_frames)
    assert callable(xpkg.media.write_video)


def test_json_utils_surface_is_public() -> None:
    assert "json_utils" in xpkg.__all__
    assert xpkg.json_utils.__all__ == [
        "dump_json",
        "parse_json",
        "parse_json_dict",
        "load_json",
        "load_json_dict",
        "write_json",
    ]
    assert xpkg.json_utils.parse_json(xpkg.json_utils.dump_json({"ok": True})) == {"ok": True}
    assert callable(xpkg.json_utils.parse_json_dict)
    assert callable(xpkg.json_utils.load_json)
    assert callable(xpkg.json_utils.load_json_dict)
    assert callable(xpkg.json_utils.write_json)


def test_project_surface_is_project_first_only() -> None:
    # Removed compat / archive surface (never were public)
    assert "read_archive" not in xpkg.project.__all__
    assert "write_archive" not in xpkg.project.__all__
    assert "read_xpkg" not in xpkg.project.__all__
    assert "write_xpkg" not in xpkg.project.__all__
    assert "export_project_archive" not in xpkg.project.__all__
    assert "current_project_archive_path" not in xpkg.project.__all__
    assert "import_detectron2_coco_project" not in xpkg.project.__all__
    assert "import_openpose_json_project" not in xpkg.project.__all__

    # Curated stable public surface
    assert "pack_project" in xpkg.project.__all__
    assert "unpack_project" in xpkg.project.__all__
    assert "init_project" in xpkg.project.__all__
    assert "validate_project" in xpkg.project.__all__
    assert "validate_expkg" in xpkg.project.__all__
    assert "inspect_project" in xpkg.project.__all__
    assert "load_project_payload" in xpkg.project.__all__
    assert "save_project_labels" in xpkg.project.__all__
    assert "list_project_figures" in xpkg.project.__all__
    assert "save_project_figure" in xpkg.project.__all__
    assert "project_artifacts_root" in xpkg.project.__all__
    assert "save_project_segmentation_masks" in xpkg.project.__all__
    assert "load_project_segmentation_masks" in xpkg.project.__all__

    # Package-level format importers are intentionally not exposed; use
    # ProjectService.import_pose/import_calibration/import_motion instead.
    not_exported = {
        "import_anipose_calibration_project",
        "import_dlc_csv_project",
        "import_dlc_h5_project",
        "import_dlc_project_directory",
        "import_lightning_pose_csv_project",
        "import_mediapipe_pose_landmarks_json_project",
        "import_mmpose_topdown_json_project",
        "import_sleap_h5_project",
        "import_sleap_package_project",
        "import_vicon_c3d_project",
        "import_vicon_csv_project",
        "import_vicon_project",
    }
    assert not_exported.isdisjoint(set(xpkg.project.__all__))
    for name in not_exported:
        assert not hasattr(xpkg.project, name), f"{name} should not be a project export"

    metadata_surface = {
        "load_project_acquisition_metadata",
        "load_project_dataset_share_metadata",
        "load_project_datasheet",
        "load_project_metadata",
        "load_project_metadata_field",
        "load_project_model_card",
        "load_project_pose_provenance",
        "save_project_acquisition_metadata",
        "save_project_dataset_share_metadata",
        "save_project_datasheet",
        "save_project_metadata",
        "save_project_metadata_field",
        "save_project_model_card",
        "save_project_pose_provenance",
    }
    assert metadata_surface.issubset(set(xpkg.project.__all__))

    with pytest.raises(AttributeError):
        xpkg.project.__getattribute__("read_archive")

    with pytest.raises(AttributeError):
        xpkg.project.__getattribute__("create_store_from_archive")


def test_direct_compat_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("xpkg.compat")


def test_adapters_surface_is_curated() -> None:
    assert sorted(xpkg.adapters.__all__) == [
        "labels_from_json_payload",
        "labels_numpy",
        "labels_to_dataframe",
        "labels_to_json_payload",
        "labels_to_primitives_session",
        "project_to_primitives_session",
        "read_vicon_json_payload",
        "vicon_recording_from_json_payload",
        "vicon_recording_to_json_payload",
    ]
