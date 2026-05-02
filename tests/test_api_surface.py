"""Smoke tests for the xpkg public API facade."""

from __future__ import annotations

import xpkg.api as api


def test_xpkg_api_exposes_project_first_contract() -> None:
    expected = {
        "Labels",
        "ArtifactFile",
        "ArtifactIndexEntry",
        "ArtifactManifest",
        "EMGSignalData",
        "Event",
        "EventTable",
        "ForcePlateData",
        "FigureArtifact",
        "PhotometryChannel",
        "PhotometryRecording",
        "ProjectDescriptor",
        "PoseTrack",
        "RecordingSession",
        "SegmentationFrame",
        "SignalChannel",
        "SyncEvent",
        "Timeline",
        "TimeRange",
        "TimeSeries",
        "Timebase",
        "VideoStub",
        "ViconRecording",
        "ViconEvent",
        "ViconForcePlatformMetadata",
        "ProjectImports",
        "ProjectArtifacts",
        "ProjectFigures",
        "ProjectInspection",
        "ProjectLayout",
        "ProjectSegmentation",
        "ProjectService",
        "build_force_plate_data_from_vicon_recording",
        "build_prediction_stub",
        "candidate_vicon_emg_channels",
        "current_project_snapshot_path",
        "current_project_state_path",
        "default_expkg_path",
        "extract_vicon_emg",
        "import_vicon_c3d_project",
        "import_vicon_csv_project",
        "import_vicon_project",
        "import_dlc_csv_project",
        "import_dlc_h5_project",
        "import_dlc_project_directory",
        "import_lightning_pose_csv_project",
        "import_mediapipe_pose_landmarks_json_project",
        "import_mmpose_topdown_json_project",
        "import_sleap_h5_project",
        "import_sleap_package_project",
        "inspect_project",
        "list_project_artifact_index",
        "list_project_artifacts",
        "list_project_figures",
        "labels_from_json_payload",
        "labels_numpy",
        "labels_to_dataframe",
        "labels_to_json_payload",
        "load_project_figure",
        "load_project_artifact",
        "load_project_metadata",
        "load_project_metadata_field",
        "load_project_payload",
        "load_project_segmentation_frames",
        "load_project_segmentation_masks",
        "load_project_vicon_recording",
        "pack_project",
        "read_doric_photometry",
        "read_events_csv",
        "read_neurophotometrics_csv",
        "read_photometry_csv",
        "read_pmat_events_csv",
        "read_pmat_photometry_csv",
        "read_pose_node_names",
        "read_pose_track",
        "read_pyphotometry_csv",
        "read_pyphotometry_ppd",
        "read_rwd_ofrs_session",
        "read_tdt_photometry_block",
        "read_teleopto_h5",
        "read_vicon_c3d",
        "read_vicon_csv",
        "read_vicon_recording",
        "read_vicon_json_payload",
        "resolve_pose_node_indices",
        "rebuild_project_artifact_index",
        "save_project_artifact",
        "save_project_figure",
        "save_project_metadata",
        "save_project_labels",
        "save_project_metadata_field",
        "save_project_segmentation_masks",
        "clear_project_segmentation_masks",
        "unpack_project",
        "validate_project_artifact",
        "validate_project_artifacts",
        "validate_project_figure",
        "validate_project_figures",
        "validate_project",
        "vicon_recording_from_json_payload",
        "vicon_recording_to_json_payload",
    }

    assert expected.issubset(set(api.__all__))
    assert "ConversionResult" not in api.__all__
    assert "convert_dlc_csv" not in api.__all__
    assert "current_project_archive_path" not in api.__all__
    assert "export_project_archive" not in api.__all__
    assert "export_project_archive" not in api.__all__
    assert "import_detectron2_coco_project" not in api.__all__
    assert "import_openpose_json_project" not in api.__all__
    assert "SleapTrack" not in api.__all__
    assert "read_sleap_node_names" not in api.__all__
    assert "read_sleap_track" not in api.__all__
    assert "resolve_sleap_node_indices" not in api.__all__
    assert api.Labels.__name__ == "Labels"
    assert api.ArtifactFile.__name__ == "ArtifactFile"
    assert api.ArtifactIndexEntry.__name__ == "ArtifactIndexEntry"
    assert api.ArtifactManifest.__name__ == "ArtifactManifest"
    assert api.EMGSignalData.__name__ == "EMGSignalData"
    assert api.Event.__name__ == "Event"
    assert api.EventTable.__name__ == "EventTable"
    assert api.ForcePlateData.__name__ == "ForcePlateData"
    assert api.FigureArtifact.__name__ == "FigureArtifact"
    assert api.PhotometryChannel.__name__ == "PhotometryChannel"
    assert api.PhotometryRecording.__name__ == "PhotometryRecording"
    assert api.ProjectDescriptor.__name__ == "ProjectDescriptor"
    assert api.PoseTrack.__name__ == "PoseTrack"
    assert api.RecordingSession.__name__ == "RecordingSession"
    assert api.SignalChannel.__name__ == "SignalChannel"
    assert api.SyncEvent.__name__ == "SyncEvent"
    assert api.Timeline.__name__ == "Timeline"
    assert api.TimeRange.__name__ == "TimeRange"
    assert api.TimeSeries.__name__ == "TimeSeries"
    assert api.Timebase.__name__ == "Timebase"
    assert api.VideoStub.__name__ == "VideoStub"
    assert api.ViconEvent.__name__ == "ViconEvent"
    assert api.ViconForcePlatformMetadata.__name__ == "ViconForcePlatformMetadata"
    assert api.ViconRecording.__name__ == "ViconRecording"
    assert api.ProjectImports.__name__ == "ProjectImports"
    assert api.ProjectArtifacts.__name__ == "ProjectArtifacts"
    assert api.ProjectFigures.__name__ == "ProjectFigures"
    assert api.ProjectInspection.__name__ == "ProjectInspection"
    assert api.ProjectSegmentation.__name__ == "ProjectSegmentation"
    assert api.ProjectService.__name__ == "ProjectService"
    assert api.SegmentationFrame.__name__ == "SegmentationFrame"
    assert callable(api.build_prediction_stub)
    assert callable(api.build_force_plate_data_from_vicon_recording)
    assert callable(api.candidate_vicon_emg_channels)
    assert callable(api.inspect_project)
    assert callable(api.extract_vicon_emg)
    assert callable(api.list_project_artifact_index)
    assert callable(api.list_project_artifacts)
    assert callable(api.list_project_figures)
    assert callable(api.labels_from_json_payload)
    assert callable(api.labels_numpy)
    assert callable(api.labels_to_dataframe)
    assert callable(api.labels_to_json_payload)
    assert callable(api.load_project_metadata)
    assert callable(api.load_project_metadata_field)
    assert callable(api.load_project_payload)
    assert callable(api.load_project_artifact)
    assert callable(api.load_project_figure)
    assert callable(api.load_project_segmentation_frames)
    assert callable(api.load_project_segmentation_masks)
    assert callable(api.load_project_vicon_recording)
    assert callable(api.read_doric_photometry)
    assert callable(api.read_events_csv)
    assert callable(api.read_neurophotometrics_csv)
    assert callable(api.read_photometry_csv)
    assert callable(api.read_pmat_events_csv)
    assert callable(api.read_pmat_photometry_csv)
    assert callable(api.read_pose_node_names)
    assert callable(api.read_pose_track)
    assert callable(api.read_pyphotometry_csv)
    assert callable(api.read_pyphotometry_ppd)
    assert callable(api.read_rwd_ofrs_session)
    assert callable(api.read_tdt_photometry_block)
    assert callable(api.read_teleopto_h5)
    assert callable(api.read_vicon_c3d)
    assert callable(api.read_vicon_csv)
    assert callable(api.read_vicon_recording)
    assert callable(api.read_vicon_json_payload)
    assert callable(api.resolve_pose_node_indices)
    assert callable(api.rebuild_project_artifact_index)
    assert callable(api.save_project_artifact)
    assert callable(api.save_project_figure)
    assert callable(api.save_project_metadata)
    assert callable(api.save_project_metadata_field)
    assert callable(api.save_project_segmentation_masks)
    assert callable(api.clear_project_segmentation_masks)
    assert callable(api.validate_project_artifact)
    assert callable(api.validate_project_artifacts)
    assert callable(api.validate_project_figure)
    assert callable(api.validate_project_figures)
    assert callable(api.vicon_recording_from_json_payload)
    assert callable(api.vicon_recording_to_json_payload)


def test_xpkg_api_lists_service_entrypoints_before_free_function_helpers() -> None:
    exports = api.__all__

    assert exports.index("ProjectService") < exports.index("ProjectImports")
    assert exports.index("ProjectImports") < exports.index("ProjectLayout")
    assert exports.index("ProjectService") < exports.index("import_dlc_csv_project")
