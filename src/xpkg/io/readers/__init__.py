"""Reader utilities for external tracking formats."""

from __future__ import annotations

from xpkg.io.readers.anipose import read_anipose_calibration, write_anipose_calibration
from xpkg.io.readers.ephys import read_abf, read_ephys_csv
from xpkg.io.readers.photometry import read_events_csv, read_photometry_csv
from xpkg.io.readers.photometry.fiber import (
    read_doric_photometry,
    read_neurophotometrics_csv,
    read_pmat_events_csv,
    read_pmat_photometry_csv,
    read_rwd_ofrs_session,
    read_tdt_photometry_block,
    read_teleopto_h5,
)
from xpkg.io.readers.photometry.pyphotometry import (
    read_pyphotometry_csv,
    read_pyphotometry_ppd,
)
from xpkg.io.readers.pose import (
    PoseTrack,
    read_pose_node_names,
    read_pose_track,
    resolve_pose_node_indices,
)
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
from xpkg.io.readers.vicon.emg import (
    candidate_vicon_emg_channels,
    extract_vicon_emg,
)
from xpkg.io.readers.vicon.force import build_force_plate_data_from_vicon_recording

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
    "read_abf",
    "read_anipose_calibration",
    "read_doric_photometry",
    "read_ephys_csv",
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
    "resolve_pose_node_indices",
    "write_anipose_calibration",
]
