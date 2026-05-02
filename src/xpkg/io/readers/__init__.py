"""Reader utilities for external tracking formats."""

from __future__ import annotations

from xpkg.io.readers._common import PoseTrack
from xpkg.io.readers.photometry import read_events_csv, read_photometry_csv
from xpkg.io.readers.pose import (
    read_pose_node_names,
    read_pose_track,
    resolve_pose_node_indices,
)
from xpkg.io.readers.pyphotometry import read_pyphotometry_ppd
from xpkg.io.readers.vicon import (
    ViconAdditionalPointData,
    ViconAnalogData,
    ViconCamera,
    ViconEvent,
    ViconForcePlatformMetadata,
    ViconMarkerModel,
    ViconRecording,
    read_vicon_c3d,
    read_vicon_csv,
    read_vicon_recording,
)
from xpkg.io.readers.vicon_emg import (
    candidate_vicon_emg_channels,
    extract_vicon_emg,
)
from xpkg.io.readers.vicon_force import build_force_plate_data_from_vicon_recording

__all__ = [
    "PoseTrack",
    "ViconAdditionalPointData",
    "ViconAnalogData",
    "ViconCamera",
    "ViconEvent",
    "ViconForcePlatformMetadata",
    "ViconMarkerModel",
    "ViconRecording",
    "build_force_plate_data_from_vicon_recording",
    "candidate_vicon_emg_channels",
    "extract_vicon_emg",
    "read_events_csv",
    "read_photometry_csv",
    "read_pose_node_names",
    "read_pose_track",
    "read_pyphotometry_ppd",
    "read_vicon_c3d",
    "read_vicon_csv",
    "read_vicon_recording",
    "resolve_pose_node_indices",
]
