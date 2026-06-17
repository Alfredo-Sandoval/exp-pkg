from __future__ import annotations

import importlib

import pytest

import xpkg
from xpkg.adapters import (
    labels_from_json_payload,
    labels_numpy,
    labels_to_dataframe,
    labels_to_json_payload,
)
from xpkg.model import (
    BEHAVIOR_LABELS_SCHEMA_VERSION,
    AcquisitionMetadata,
    BehaviorEmbedding,
    BehaviorFrameLabel,
    BehaviorInterval,
    BehaviorLabels,
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
    ArtifactFile,
    ArtifactIndexEntry,
    ArtifactManifest,
    ArtifactOutputSpec,
    FigureArtifact,
    ProjectDescriptor,
    ProjectInspection,
    ProjectSummaryIndex,
    SegmentationFrame,
    clear_project_segmentation_masks,
    current_project_state_path,
    default_expkg_path,
    init_project,
    inspect_project,
    is_project_root,
    labels_state_summary,
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
    load_project_summary,
    pack_project,
    read_labels_json_payload,
    rebuild_project_artifact_index,
    refresh_project_summary,
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
    KNOWN_BEHAVIOR_SOURCE_TYPES,
    is_doric_photometry_file,
    is_neurophotometrics_csv,
    is_pyphotometry_csv,
    is_pyphotometry_ppd_file,
    is_rwd_ofrs_session,
    is_tdt_block,
    is_teleopto_h5,
    parse_teleopto_h5_arrays,
    read_behavior_events_csv,
    read_behavior_events_json,
    read_boris_csv,
    read_bsoid_csv,
    read_doric_photometry,
    read_events_csv,
    read_keypoint_moseq_syllables_csv,
    read_neurophotometrics_csv,
    read_nwb_photometry,
    read_opencv_stereo_calibration,
    read_photometry_csv,
    read_pmat_events_csv,
    read_pmat_photometry_csv,
    read_pyphotometry_csv,
    read_pyphotometry_ppd,
    read_rwd_ofrs_session,
    read_simba_csv,
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
    # Each access goes through the lazy __getattr__ loader (the submodules were
    # popped above) and must resolve to the matching xpkg subpackage.
    import types

    for sub in (
        "adapters",
        "json_utils",
        "media",
        "payloads",
        "project",
        "model",
        "pose",
        "readers",
        "segmentation",
        "services",
    ):
        module = getattr(reloaded, sub)
        assert isinstance(module, types.ModuleType)
        assert module.__name__ == f"xpkg.{sub}"

    assert callable(reloaded.readers.read_boris_csv)
    assert callable(reloaded.readers.read_bsoid_csv)
    assert callable(reloaded.readers.read_behavior_events_csv)
    assert callable(reloaded.readers.read_behavior_events_json)
    assert callable(reloaded.readers.read_keypoint_moseq_syllables_csv)
    assert callable(reloaded.readers.read_simba_csv)
    assert callable(reloaded.readers.is_doric_photometry_file)
    assert callable(reloaded.readers.is_neurophotometrics_csv)
    assert callable(reloaded.readers.is_pyphotometry_csv)
    assert callable(reloaded.readers.is_pyphotometry_ppd_file)
    assert callable(reloaded.readers.is_rwd_ofrs_session)
    assert callable(reloaded.readers.is_tdt_block)
    assert callable(reloaded.readers.is_teleopto_h5)
    assert callable(reloaded.readers.parse_teleopto_h5_arrays)
    assert callable(reloaded.readers.read_doric_photometry)
    assert callable(reloaded.readers.read_events_csv)
    assert callable(reloaded.readers.read_neurophotometrics_csv)
    assert callable(reloaded.readers.read_nwb_photometry)
    assert callable(reloaded.readers.read_photometry_csv)
    assert callable(reloaded.readers.read_pmat_events_csv)
    assert callable(reloaded.readers.read_pmat_photometry_csv)
    assert callable(reloaded.readers.read_pyphotometry_csv)
    assert callable(reloaded.readers.read_pyphotometry_ppd)
    assert callable(reloaded.readers.read_rwd_ofrs_session)
    assert callable(reloaded.readers.read_tdt_photometry_block)
    assert callable(reloaded.readers.read_teleopto_h5)

    for removed_name in ("compat", "api", "read_doric_photometry", "exchange", "formats"):
        with pytest.raises(AttributeError, match=removed_name):
            reloaded.__getattribute__(removed_name)


def test_public_exports_are_callable() -> None:
    import types

    assert callable(ArtifactFile)
    assert callable(ArtifactIndexEntry)
    assert callable(ArtifactManifest)
    # ArtifactOutputSpec is a union type alias, not a constructor.
    assert isinstance(ArtifactOutputSpec, types.UnionType)
    assert callable(FigureArtifact)
    assert callable(ProjectDescriptor)
    assert callable(SegmentationFrame)
    assert callable(ProjectInspection)
    assert callable(ProjectSummaryIndex)
    assert callable(clear_project_segmentation_masks)
    assert callable(current_project_state_path)
    assert callable(default_expkg_path)
    assert callable(init_project)
    assert callable(inspect_project)
    assert callable(is_project_root)
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
    assert callable(load_project_summary)
    assert callable(pack_project)
    assert callable(read_labels_json_payload)
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
    assert callable(write_labels_json)
    assert callable(write_project_descriptor)
    assert callable(refresh_project_summary)
    assert callable(labels_state_summary)
    assert callable(labels_from_json_payload)
    assert callable(labels_numpy)
    assert callable(labels_to_dataframe)
    assert callable(labels_to_json_payload)
    assert "keypoint_moseq" in KNOWN_BEHAVIOR_SOURCE_TYPES
    assert callable(parse_teleopto_h5_arrays)
    assert callable(read_boris_csv)
    assert callable(read_bsoid_csv)
    assert callable(read_behavior_events_csv)
    assert callable(read_behavior_events_json)
    assert callable(read_keypoint_moseq_syllables_csv)
    assert callable(read_simba_csv)
    assert callable(is_doric_photometry_file)
    assert callable(is_neurophotometrics_csv)
    assert callable(is_pyphotometry_csv)
    assert callable(is_pyphotometry_ppd_file)
    assert callable(is_rwd_ofrs_session)
    assert callable(is_tdt_block)
    assert callable(is_teleopto_h5)
    assert callable(read_doric_photometry)
    assert callable(read_events_csv)
    assert callable(read_neurophotometrics_csv)
    assert callable(read_nwb_photometry)
    assert callable(read_opencv_stereo_calibration)
    assert callable(read_photometry_csv)
    assert callable(read_pmat_events_csv)
    assert callable(read_pmat_photometry_csv)
    assert callable(read_pyphotometry_csv)
    assert callable(read_pyphotometry_ppd)
    assert callable(read_rwd_ofrs_session)
    assert callable(read_tdt_photometry_block)
    assert callable(read_teleopto_h5)
    assert callable(ProjectArtifacts)
    assert callable(ProjectCalibrations)
    assert callable(ProjectFigures)
    assert callable(ProjectMetadata)
    assert callable(ServiceProjectInspection)
    assert callable(ProjectLayout)
    assert callable(ProjectSegmentation)
    assert callable(ProjectService)


def test_services_surface_lists_project_service_first() -> None:
    assert not hasattr(xpkg.services, "WorkspaceService")
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
    ]


def test_project_service_dispatches_supported_pose_and_calibration_formats() -> None:
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
    calibration_formats: set[str] = {"anipose", "opencv-stereo-yaml"}

    from typing import get_args

    from xpkg.services import CalibrationFormat, PoseFormat

    assert pose_formats == set(get_args(PoseFormat))
    assert calibration_formats == set(get_args(CalibrationFormat))


def test_model_exports_are_available() -> None:
    assert callable(AcquisitionMetadata)
    assert BEHAVIOR_LABELS_SCHEMA_VERSION == "xpkg.behavior_labels.v1"
    assert callable(BehaviorEmbedding)
    assert callable(BehaviorFrameLabel)
    assert callable(BehaviorInterval)
    assert callable(BehaviorLabels)
    assert callable(CameraMetadata)
    assert callable(DatasetShareMetadata)
    assert callable(Labels)
    assert callable(SuggestionFrame)
    assert callable(Skeleton)
    assert callable(Keypoint)
    assert callable(Track)
    assert callable(LabeledFrame)
    assert callable(Instance)
    assert callable(PredictedInstance)
    assert callable(Point)
    assert callable(PredictedPoint)
    assert callable(PointArray)
    assert callable(PredictedPointArray)
    assert callable(Video)
    assert callable(VideoStub)
    assert callable(EMGSignalData)
    assert callable(Event)
    assert callable(EventTable)
    assert callable(ForcePlateData)
    assert callable(PhotometryChannel)
    assert callable(PhotometryRecording)
    assert callable(RecordingSession)
    assert callable(SignalChannel)
    assert callable(SyncEvent)
    assert callable(Timeline)
    assert callable(TimeRange)
    assert callable(TimeSeries)
    assert callable(Timebase)
    assert callable(KPFlag)
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
    assert "load_project_summary" in xpkg.project.__all__
    assert "refresh_project_summary" in xpkg.project.__all__
    assert "save_project_labels" in xpkg.project.__all__
    assert "list_project_figures" in xpkg.project.__all__
    assert "save_project_figure" in xpkg.project.__all__
    assert "current_project_state_path" in xpkg.project.__all__
    assert "save_project_segmentation_masks" in xpkg.project.__all__
    assert "load_project_segmentation_masks" in xpkg.project.__all__

    # Package-level format importers are intentionally not exposed; use
    # ProjectService.import_pose/import_calibration instead.
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
    }
    assert not_exported.isdisjoint(set(xpkg.project.__all__))
    for name in not_exported:
        assert not hasattr(xpkg.project, name), f"{name} should not be a project export"

    # Private-store layout details -- the .xpkg/ directory and filename
    # constants and the project_* path helpers -- are intentionally not part of
    # the public surface; downstream code should locate project files through
    # ProjectService rather than hard-coding them. They remain importable from
    # their submodules (xpkg.project.layout, .metadata, .calibration, .artifact).
    layout_internals = {
        "ACQUISITION_METADATA_FILENAME",
        "ARTIFACTS_DIRNAME",
        "ARTIFACT_INDEX_FILENAME",
        "ARTIFACT_MANIFEST_FILENAME",
        "ARTIFACT_SCHEMA_VERSION",
        "CALIBRATION_FILENAME",
        "CALIBRATION_SOURCE_DIRNAME",
        "CALIBRATIONS_DIRNAME",
        "DATASET_SHARE_METADATA_FILENAME",
        "DATASHEET_FILENAME",
        "EXPKG_MANIFEST_FILENAME",
        "EXPKG_SUFFIX",
        "FIGURE_ARTIFACT_SCHEMA_VERSION",
        "FIGURE_ARTIFACT_TYPE",
        "FIGURE_MANIFEST_FILENAME",
        "FIGURES_DIRNAME",
        "INDEXES_DIRNAME",
        "MODEL_CARD_FILENAME",
        "POSE_PROVENANCE_FILENAME",
        "PROJECT_DESCRIPTOR_FILENAME",
        "PROJECT_METADATA_DIRNAME",
        "PROJECT_SUMMARY_FILENAME",
        "PROJECT_SUMMARY_SCHEMA_VERSION",
        "artifact_kind_dir",
        "project_acquisition_metadata_path",
        "project_artifact_index_path",
        "project_artifact_root",
        "project_artifact_type_root",
        "project_artifacts_root",
        "project_calibration_path",
        "project_calibration_root",
        "project_calibration_source_root",
        "project_calibrations_root",
        "project_dataset_share_metadata_path",
        "project_datasheet_path",
        "project_descriptor_path",
        "project_exports_root",
        "project_figure_root",
        "project_figures_root",
        "project_indexes_root",
        "project_media_root",
        "project_metadata_root",
        "project_model_card_path",
        "project_pose_provenance_path",
        "project_state_root",
        "project_store_root",
        "project_summary_path",
        "resolve_project_root",
    }
    assert layout_internals.isdisjoint(set(xpkg.project.__all__))
    for name in layout_internals:
        assert not hasattr(xpkg.project, name), f"{name} should not be a project export"
    # current_project_state_path is the one retained state-file locator.
    assert hasattr(xpkg.project, "current_project_state_path")

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
    ]
