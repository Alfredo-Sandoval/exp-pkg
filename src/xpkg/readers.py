"""Single public namespace for xpkg format readers.

This is the canonical import location for reader functions and reader-related
data classes. Everything here is re-exported from :mod:`xpkg.io.readers` (the
internal implementation home).

Example:
    >>> from xpkg.readers import read_dlc_csv  # doctest: +SKIP
    >>> from xpkg import readers              # doctest: +SKIP
    >>> readers.read_pose_track(...)          # doctest: +SKIP
"""

from __future__ import annotations

from xpkg.io.readers import (
    KNOWN_BEHAVIOR_SOURCE_TYPES,
    PoseTrack,
    ViconAdditionalPointData,
    ViconAnalogData,
    ViconCamera,
    ViconEvent,
    ViconForcePlatformMetadata,
    ViconMarkerModel,
    ViconRecording,
    build_force_plate_data_from_vicon_recording,
    candidate_vicon_emg_channels,
    extract_vicon_emg,
    read_abf,
    read_anipose_calibration,
    read_behavior_events_csv,
    read_behavior_events_json,
    read_boris_csv,
    read_doric_photometry,
    read_ephys_csv,
    read_events_csv,
    read_neurophotometrics_csv,
    read_photometry_csv,
    read_pmat_events_csv,
    read_pmat_photometry_csv,
    read_pose_node_names,
    read_pose_track,
    read_pyphotometry_csv,
    read_pyphotometry_ppd,
    read_rwd_ofrs_session,
    read_simba_csv,
    read_tdt_photometry_block,
    read_teleopto_h5,
    read_vicon_c3d,
    read_vicon_csv,
    read_vicon_recording,
    resolve_pose_node_indices,
    write_anipose_calibration,
)

__all__ = [
    "PoseTrack",
    "ViconAdditionalPointData",
    "ViconAnalogData",
    "ViconCamera",
    "ViconEvent",
    "ViconForcePlatformMetadata",
    "ViconMarkerModel",
    "ViconRecording",
    "KNOWN_BEHAVIOR_SOURCE_TYPES",
    "build_force_plate_data_from_vicon_recording",
    "candidate_vicon_emg_channels",
    "extract_vicon_emg",
    "read_abf",
    "read_anipose_calibration",
    "read_boris_csv",
    "read_behavior_events_csv",
    "read_behavior_events_json",
    "read_simba_csv",
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
