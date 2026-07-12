"""Opt-in contract tests against genuine pose/behavior tool exports.

Mirrors ``test_fiber_photometry_external_fixtures`` for the pose and behavior
readers: point ``XPKG_POSE_FIXTURE_ROOT`` / ``XPKG_BEHAVIOR_FIXTURE_ROOT`` at a
directory of real exports and the readers are exercised against the actual
bytes the upstream tools write. The explicit vendor test target fails when a
configured corpus is incomplete.

The committed byte-faithful fixtures in ``tests/fixtures`` pin the same formats
deterministically in CI; these tests add coverage against genuine captures (and
the formats committed fixtures cannot fully reproduce, e.g. real binary H5).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from xpkg.io.readers import (
    read_boris_csv,
    read_bsoid_csv,
    read_keypoint_moseq_syllables_csv,
    read_pose_track,
    read_simba_csv,
)
from xpkg.io.readers.pose._common import PoseTrack
from xpkg.model import BehaviorLabels

pytestmark = pytest.mark.vendorfixtures


def _pose_root() -> Path:
    return _required_root("XPKG_POSE_FIXTURE_ROOT")


def _behavior_root() -> Path:
    return _required_root("XPKG_BEHAVIOR_FIXTURE_ROOT")


def _required_root(environment_variable: str) -> Path:
    raw_root = os.environ.get(environment_variable)
    if not raw_root:
        raise RuntimeError(
            f"{environment_variable} is required for vendor-fixture tests. "
            "Run them through `make test-vendor`."
        )
    return Path(raw_root)


def _fixture(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(f"Required vendor fixture does not exist: {path}")
    return path


def _assert_pose(track: PoseTrack) -> None:
    assert isinstance(track, PoseTrack)
    assert track.coords.ndim == 3
    assert track.coords.shape[2] == 2
    assert track.coords.shape[0] > 0
    assert len(track.node_names) == track.coords.shape[1] > 0


def test_external_dlc_csv_fixture_loads() -> None:
    path = _fixture(_pose_root(), "dlc/tracking.csv")
    _assert_pose(read_pose_track(path, software="DLC", file_type="csv"))


def test_external_dlc_h5_fixture_loads() -> None:
    path = _fixture(_pose_root(), "dlc/tracking.h5")
    _assert_pose(read_pose_track(path, software="DLC", file_type="h5"))


def test_external_sleap_analysis_h5_fixture_loads() -> None:
    path = _fixture(_pose_root(), "sleap/labels.analysis.h5")
    _assert_pose(read_pose_track(path, software="SLEAP", file_type="h5"))


def test_external_mmpose_demo_json_fixture_loads() -> None:
    path = _fixture(_pose_root(), "mmpose/results.json")
    _assert_pose(read_pose_track(path, software="MMPOSE", file_type="json"))


def test_external_mediapipe_json_fixture_loads() -> None:
    path = _fixture(_pose_root(), "mediapipe/landmarks.json")
    _assert_pose(read_pose_track(path, software="MEDIAPIPE", file_type="json"))


def test_external_boris_tabular_fixture_loads() -> None:
    path = _fixture(_behavior_root(), "boris/tabular.csv")
    labels = read_boris_csv(path)
    assert isinstance(labels, BehaviorLabels)
    assert labels.metadata["source"]["format"] == "tabular_events_csv"
    assert len(labels.intervals) > 0


def test_external_boris_aggregated_fixture_loads() -> None:
    path = _fixture(_behavior_root(), "boris/aggregated.csv")
    labels = read_boris_csv(path)
    assert labels.metadata["source"]["format"] == "aggregated_events_csv"
    assert len(labels.intervals) > 0


def test_external_bsoid_fixture_loads() -> None:
    path = _fixture(_behavior_root(), "bsoid/labels.csv")
    labels = read_bsoid_csv(path)
    assert isinstance(labels, BehaviorLabels)
    assert labels.frame_labels or labels.intervals


def test_external_simba_fixture_loads() -> None:
    path = _fixture(_behavior_root(), "simba/machine_results.csv")
    labels = read_simba_csv(path)
    assert len(labels.frame_labels) > 0


def test_external_keypoint_moseq_fixture_loads() -> None:
    path = _fixture(_behavior_root(), "keypoint_moseq/syllables.csv")
    labels = read_keypoint_moseq_syllables_csv(path)
    assert labels.frame_labels or labels.intervals
