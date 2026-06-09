from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.factories import write_dummy_video, write_sleap_analysis_h5


def test_convert_sleap_h5_builds_multi_track_labels(tmp_path: Path) -> None:
    from xpkg.io.converters.sleap_import import convert_sleap_h5

    tracking_path = tmp_path / "analysis.h5"
    write_sleap_analysis_h5(tracking_path)
    video_path = tmp_path / "session.avi"
    write_dummy_video(video_path, frame_count=10)

    result = convert_sleap_h5(
        tracking_path,
        video_path,
        skeleton_name="subject",
    )

    assert result.metadata["source"] == "sleap_h5_import"
    assert result.metadata["source_h5"] == tracking_path.name
    assert result.metadata["source_video"] == video_path.name

    labels = result.labels
    assert len(labels.videos) == 1
    assert len(labels.labeled_frames) == 10

    first_frame = next(frame for frame in labels.labeled_frames if frame.frame_idx == 0)
    assert sorted(
        inst.track.id for inst in first_frame.instances if inst.track is not None
    ) == [0, 1]
    assert [record.track_id for record in labels.identity_provenance] == ["0", "1"]
    assert [record.track_name for record in labels.identity_provenance] == [
        "track_0",
        "track_1",
    ]
    assert {record.source_tool for record in labels.identity_provenance} == {"sleap"}
    assert {record.source_file for record in labels.identity_provenance} == {
        tracking_path.name
    }
    assert {record.identity_source for record in labels.identity_provenance} == {"unknown"}

    frame_five = next(frame for frame in labels.labeled_frames if frame.frame_idx == 5)
    track_one = next(inst for inst in frame_five.instances if inst.track and inst.track.id == 1)
    points = track_one.point_records(copy=False)
    assert int(np.count_nonzero(~np.isnan(points["x"]))) == 3
