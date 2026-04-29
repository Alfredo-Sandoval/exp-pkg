"""Reader utilities for external tracking formats."""

from __future__ import annotations

from xpkg.io.readers._common import PoseTrack
from xpkg.io.readers.pose import (
    read_pose_node_names,
    read_pose_track,
    resolve_pose_node_indices,
)
from xpkg.io.readers.vicon import (
    ViconAdditionalPointData,
    ViconAnalogData,
    ViconCamera,
    ViconEvent,
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

__all__ = [
    "PoseTrack",
    "ViconAdditionalPointData",
    "ViconAnalogData",
    "ViconCamera",
    "ViconEvent",
    "ViconMarkerModel",
    "ViconRecording",
    "candidate_vicon_emg_channels",
    "extract_vicon_emg",
    "read_pose_node_names",
    "read_pose_track",
    "read_vicon_c3d",
    "read_vicon_csv",
    "read_vicon_recording",
    "resolve_pose_node_indices",
]
