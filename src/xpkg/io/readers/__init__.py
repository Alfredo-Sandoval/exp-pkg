"""Reader utilities for external tracking formats."""

from __future__ import annotations

from xpkg.io.readers.anipose import read_anipose_calibration, write_anipose_calibration
from xpkg.io.readers.behavior import (
    KNOWN_BEHAVIOR_SOURCE_TYPES,
    read_behavior_events_csv,
    read_behavior_events_json,
    read_boris_csv,
    read_bsoid_csv,
    read_keypoint_moseq_syllables_csv,
    read_simba_csv,
)
from xpkg.io.readers.opencv_stereo import read_opencv_stereo_calibration
from xpkg.io.readers.photometry import read_events_csv, read_photometry_csv
from xpkg.io.readers.photometry.fiber import (
    is_doric_photometry_file,
    is_neurophotometrics_csv,
    is_rwd_ofrs_session,
    is_tdt_block,
    is_teleopto_h5,
    parse_teleopto_h5_arrays,
    read_doric_photometry,
    read_neurophotometrics_csv,
    read_pmat_events_csv,
    read_pmat_photometry_csv,
    read_rwd_ofrs_session,
    read_tdt_photometry_block,
    read_teleopto_h5,
)
from xpkg.io.readers.photometry.nwb import (
    is_nwb_photometry_file,
    read_nwb_photometry,
)
from xpkg.io.readers.photometry.pyphotometry import (
    is_pyphotometry_csv,
    is_pyphotometry_ppd_file,
    read_pyphotometry_csv,
    read_pyphotometry_ppd,
)
from xpkg.io.readers.pose import (
    PoseTrack,
    read_pose_node_names,
    read_pose_track,
    resolve_pose_node_indices,
)

__all__ = [
    "PoseTrack",
    "KNOWN_BEHAVIOR_SOURCE_TYPES",
    "is_doric_photometry_file",
    "is_neurophotometrics_csv",
    "is_rwd_ofrs_session",
    "is_tdt_block",
    "is_teleopto_h5",
    "parse_teleopto_h5_arrays",
    "read_anipose_calibration",
    "read_boris_csv",
    "read_bsoid_csv",
    "read_behavior_events_csv",
    "read_behavior_events_json",
    "read_keypoint_moseq_syllables_csv",
    "read_simba_csv",
    "read_doric_photometry",
    "read_events_csv",
    "read_neurophotometrics_csv",
    "is_nwb_photometry_file",
    "read_nwb_photometry",
    "read_opencv_stereo_calibration",
    "read_photometry_csv",
    "read_pmat_events_csv",
    "read_pmat_photometry_csv",
    "read_pose_node_names",
    "read_pose_track",
    "is_pyphotometry_csv",
    "is_pyphotometry_ppd_file",
    "read_pyphotometry_csv",
    "read_pyphotometry_ppd",
    "read_rwd_ofrs_session",
    "read_tdt_photometry_block",
    "read_teleopto_h5",
    "resolve_pose_node_indices",
    "write_anipose_calibration",
]
