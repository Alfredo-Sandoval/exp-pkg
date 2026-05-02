"""Smoke tests for the xpkg public API facade."""

from __future__ import annotations

import xpkg.api as api


def test_xpkg_api_exposes_workspace_first_contract() -> None:
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
        "WorkspaceImports",
        "WorkspaceArtifacts",
        "WorkspaceFigures",
        "WorkspaceInspection",
        "WorkspaceLayout",
        "WorkspaceSegmentation",
        "WorkspaceService",
        "build_force_plate_data_from_vicon_recording",
        "build_prediction_stub",
        "candidate_vicon_emg_channels",
        "current_project_snapshot_path",
        "current_project_state_path",
        "decode_hdf5_string",
        "default_expkg_path",
        "extract_vicon_emg",
        "import_vicon_c3d_workspace",
        "import_vicon_csv_workspace",
        "import_vicon_workspace",
        "import_dlc_csv_workspace",
        "import_dlc_h5_workspace",
        "import_dlc_project_workspace",
        "import_lightning_pose_csv_workspace",
        "import_mediapipe_pose_landmarks_json_workspace",
        "import_mmpose_topdown_json_workspace",
        "import_sleap_h5_workspace",
        "import_sleap_package_workspace",
        "inspect_workspace",
        "list_workspace_artifact_index",
        "list_workspace_artifacts",
        "list_workspace_figures",
        "labels_from_json_payload",
        "labels_numpy",
        "labels_to_dataframe",
        "labels_to_json_payload",
        "load_workspace_figure",
        "load_workspace_artifact",
        "load_workspace_metadata",
        "load_workspace_metadata_field",
        "load_workspace_payload",
        "load_workspace_segmentation_frames",
        "load_workspace_segmentation_masks",
        "load_workspace_vicon_recording",
        "pack_project",
        "read_events_csv",
        "read_hdf5_table",
        "read_hdf5_table_group",
        "read_photometry_csv",
        "read_pose_node_names",
        "read_pose_track",
        "read_pyphotometry_ppd",
        "read_vicon_c3d",
        "read_vicon_csv",
        "read_vicon_recording",
        "read_vicon_json_payload",
        "resolve_pose_node_indices",
        "rebuild_workspace_artifact_index",
        "save_workspace_artifact",
        "save_workspace_figure",
        "save_workspace_metadata",
        "save_workspace_labels",
        "save_workspace_metadata_field",
        "save_workspace_segmentation_masks",
        "clear_workspace_segmentation_masks",
        "unpack_project",
        "validate_workspace_artifact",
        "validate_workspace_artifacts",
        "validate_workspace_figure",
        "validate_workspace_figures",
        "validate_workspace",
        "vicon_recording_from_json_payload",
        "vicon_recording_to_json_payload",
        "write_hdf5_table",
        "write_hdf5_table_group",
    }

    assert expected.issubset(set(api.__all__))
    assert "ConversionResult" not in api.__all__
    assert "convert_dlc_csv" not in api.__all__
    assert "current_project_archive_path" not in api.__all__
    assert "export_workspace_archive" not in api.__all__
    assert "export_project_archive" not in api.__all__
    assert "import_detectron2_coco_workspace" not in api.__all__
    assert "import_openpose_json_workspace" not in api.__all__
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
    assert api.WorkspaceImports.__name__ == "WorkspaceImports"
    assert api.WorkspaceArtifacts.__name__ == "WorkspaceArtifacts"
    assert api.WorkspaceFigures.__name__ == "WorkspaceFigures"
    assert api.WorkspaceInspection.__name__ == "WorkspaceInspection"
    assert api.WorkspaceSegmentation.__name__ == "WorkspaceSegmentation"
    assert api.WorkspaceService.__name__ == "WorkspaceService"
    assert api.SegmentationFrame.__name__ == "SegmentationFrame"
    assert callable(api.build_prediction_stub)
    assert callable(api.build_force_plate_data_from_vicon_recording)
    assert callable(api.candidate_vicon_emg_channels)
    assert callable(api.decode_hdf5_string)
    assert callable(api.inspect_workspace)
    assert callable(api.extract_vicon_emg)
    assert callable(api.list_workspace_artifact_index)
    assert callable(api.list_workspace_artifacts)
    assert callable(api.list_workspace_figures)
    assert callable(api.labels_from_json_payload)
    assert callable(api.labels_numpy)
    assert callable(api.labels_to_dataframe)
    assert callable(api.labels_to_json_payload)
    assert callable(api.load_workspace_metadata)
    assert callable(api.load_workspace_metadata_field)
    assert callable(api.load_workspace_payload)
    assert callable(api.load_workspace_artifact)
    assert callable(api.load_workspace_figure)
    assert callable(api.load_workspace_segmentation_frames)
    assert callable(api.load_workspace_segmentation_masks)
    assert callable(api.load_workspace_vicon_recording)
    assert callable(api.read_events_csv)
    assert callable(api.read_hdf5_table)
    assert callable(api.read_hdf5_table_group)
    assert callable(api.read_photometry_csv)
    assert callable(api.read_pose_node_names)
    assert callable(api.read_pose_track)
    assert callable(api.read_pyphotometry_ppd)
    assert callable(api.read_vicon_c3d)
    assert callable(api.read_vicon_csv)
    assert callable(api.read_vicon_recording)
    assert callable(api.read_vicon_json_payload)
    assert callable(api.resolve_pose_node_indices)
    assert callable(api.rebuild_workspace_artifact_index)
    assert callable(api.save_workspace_artifact)
    assert callable(api.save_workspace_figure)
    assert callable(api.save_workspace_metadata)
    assert callable(api.save_workspace_metadata_field)
    assert callable(api.save_workspace_segmentation_masks)
    assert callable(api.clear_workspace_segmentation_masks)
    assert callable(api.validate_workspace_artifact)
    assert callable(api.validate_workspace_artifacts)
    assert callable(api.validate_workspace_figure)
    assert callable(api.validate_workspace_figures)
    assert callable(api.vicon_recording_from_json_payload)
    assert callable(api.vicon_recording_to_json_payload)
    assert callable(api.write_hdf5_table)
    assert callable(api.write_hdf5_table_group)


def test_xpkg_api_lists_service_entrypoints_before_free_function_helpers() -> None:
    exports = api.__all__

    assert exports.index("WorkspaceService") < exports.index("WorkspaceImports")
    assert exports.index("WorkspaceImports") < exports.index("WorkspaceLayout")
    assert exports.index("WorkspaceService") < exports.index("import_dlc_csv_workspace")
