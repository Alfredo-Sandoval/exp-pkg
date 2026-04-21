from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.vicon_helpers import (
    write_sample_vicon_c3d,
    write_sample_vicon_csv,
    write_sample_vsk,
    write_sample_xcp,
)


def test_read_vicon_csv_preserves_marker_labels_gaps_and_sidecars(tmp_path: Path) -> None:
    from xpkg.io.readers import read_vicon_csv

    csv_path = tmp_path / "trial.csv"
    write_sample_vicon_csv(csv_path)
    write_sample_vsk(csv_path.with_suffix(".vsk"))
    write_sample_xcp(csv_path.with_suffix(".xcp"))

    recording = read_vicon_csv(csv_path)

    assert recording.source_type == "csv"
    assert recording.fps == 100
    assert recording.frame_offset == 101
    assert recording.marker_names == ("center", "R_foot", "L_foot")
    assert recording.source_marker_labels == ("Mouse:center", "Mouse:R_foot", "Mouse:L_foot")
    assert recording.positions.shape == (2, 3, 3)
    assert recording.marker_valid.tolist() == [[True, True, True], [True, True, False]]
    assert np.isnan(recording.positions[1, 2]).all()
    assert recording.model is not None
    assert recording.model.source == "vsk"
    assert recording.vsk_path == csv_path.with_suffix(".vsk")
    assert recording.has_cameras
    assert [camera.user_id for camera in recording.cameras] == [1, 2]
    assert recording.xcp_path == csv_path.with_suffix(".xcp")


def test_read_vicon_c3d_preserves_analog_and_additional_points(tmp_path: Path) -> None:
    from xpkg.io.readers import read_vicon_c3d

    c3d_path = tmp_path / "trial.c3d"
    write_sample_vicon_c3d(c3d_path)
    write_sample_vsk(c3d_path.with_suffix(".vsk"))
    write_sample_xcp(c3d_path.with_suffix(".xcp"))

    recording = read_vicon_c3d(c3d_path)

    assert recording.source_type == "c3d"
    assert recording.fps == 100
    assert recording.frame_offset == 11
    assert recording.marker_names == ("center", "R_foot")
    assert recording.source_marker_labels == ("Mouse:center", "Mouse:R_foot")
    assert recording.positions.shape == (2, 2, 3)
    assert recording.marker_valid.tolist() == [[True, True], [True, False]]
    assert np.isnan(recording.positions[1, 1]).all()
    assert recording.has_analog
    assert recording.analog is not None
    assert recording.analog.channel_names == ("Fx", "Fy", "Fz")
    assert recording.analog.samples_per_frame == 2
    assert recording.analog.values.shape == (4, 3)
    np.testing.assert_allclose(
        recording.analog.values,
        np.array(
            [
                [10.0, 30.0, 50.0],
                [20.0, 40.0, 60.0],
                [11.0, 31.0, 51.0],
                [21.0, 41.0, 61.0],
            ]
        ),
    )
    assert recording.has_additional_points
    assert recording.additional_points is not None
    assert recording.additional_points.labels == ("Model:HipMoment",)
    assert recording.additional_points.values.shape == (2, 1, 5)
    assert recording.additional_points.valid.tolist() == [[True], [True]]
    assert recording.model is not None
    assert recording.model.marker_names == ("center", "R_foot")
    assert recording.model.edges == (("center", "R_foot"),)
    assert recording.model.source == "vsk"
    assert [camera.user_id for camera in recording.cameras] == [1, 2]


def test_read_vicon_recording_dispatches_from_suffix(tmp_path: Path) -> None:
    from xpkg.io.readers import read_vicon_recording

    csv_path = tmp_path / "trial.csv"
    write_sample_vicon_csv(csv_path)

    recording = read_vicon_recording(csv_path)

    assert recording.source_type == "csv"


def test_select_marker_labels_preserves_multi_subject_marker_namespaces() -> None:
    from xpkg.io.readers.vicon import select_marker_labels

    marker_names, source_marker_labels, model = select_marker_labels(
        ("MouseA:LASI", "MouseB:LASI", "MouseA:RASI"),
    )

    assert marker_names == ("MouseA:LASI", "MouseB:LASI", "MouseA:RASI")
    assert source_marker_labels == ("MouseA:LASI", "MouseB:LASI", "MouseA:RASI")
    assert model.marker_names == marker_names


def test_vicon_lookup_requires_namespaced_query_when_suffix_is_ambiguous() -> None:
    from xpkg.model import ViconAnalogData, ViconRecording

    recording = ViconRecording(
        path=Path("trial.c3d"),
        source_type="c3d",
        fps=100,
        marker_names=("MouseA:LASI", "MouseB:LASI"),
        source_marker_labels=("MouseA:LASI", "MouseB:LASI"),
        positions=np.zeros((1, 2, 3), dtype=np.float64),
        marker_valid=np.ones((1, 2), dtype=bool),
        frame_offset=1,
        analog=ViconAnalogData(
            fps=1000,
            samples_per_frame=1,
            channel_names=("FP1:Fz", "FP2:Fz"),
            values=np.zeros((1, 2), dtype=np.float64),
        ),
    )

    assert recording.marker_index("MouseB:LASI") == 1
    assert recording.analog is not None
    assert recording.analog.channel_index("FP2:Fz") == 1
    with pytest.raises(KeyError, match="ambiguous"):
        recording.marker_index("LASI")
    with pytest.raises(KeyError, match="ambiguous"):
        recording.analog.channel_index("Fz")
