from __future__ import annotations

from xpkg.media.video import Video
from xpkg.model import LabeledFrame
from xpkg.model.labels_query import (
    build_frame_index_map,
    fancy_frame_indices,
    find_frames,
    group_labeled_frames_by_video,
)
from xpkg.model.video_types import VideoProtocol


def _video(filename: str) -> Video:
    video = Video.__new__(Video)
    video.filename = filename
    video.id = None
    video.label = None
    video.sha256 = None
    video._image_filenames = []
    video.width = 64
    video.height = 64
    video.frames = 20
    video.fps = 0.0
    video.channels = 3
    video.backend = "test"
    video.last_frame_idx = 19
    return video


def test_group_labeled_frames_by_video_includes_unknown_frame_videos() -> None:
    video = _video("known.mp4")
    other_video = _video("other.mp4")
    frame0 = LabeledFrame(video=video, frame_idx=0)
    frame1 = LabeledFrame(video=other_video, frame_idx=1)

    result = group_labeled_frames_by_video([frame0, frame1], [video])

    assert result[video] == [frame0]
    assert result[other_video] == [frame1]


def test_build_frame_index_map_indexes_each_video_by_frame_number() -> None:
    video = _video("movie.mp4")
    frame0 = LabeledFrame(video=video, frame_idx=0)
    frame5 = LabeledFrame(video=video, frame_idx=5)

    result = build_frame_index_map({video: [frame0, frame5]})

    assert result == {video: {0: frame0, 5: frame5}}


def test_find_frames_supports_all_single_multiple_and_missing_queries() -> None:
    video = _video("movie.mp4")
    unknown_video = _video("unknown.mp4")
    frame0 = LabeledFrame(video=video, frame_idx=0)
    frame5 = LabeledFrame(video=video, frame_idx=5)
    frame_idx_map: dict[VideoProtocol, dict[int, LabeledFrame]] = {
        video: {0: frame0, 5: frame5}
    }

    assert find_frames(frame_idx_map, video) == [frame0, frame5]
    assert find_frames(frame_idx_map, video, frame_idx=5) == [frame5]
    assert find_frames(frame_idx_map, video, frame_idx=[0, 999]) == [frame0]
    assert find_frames(frame_idx_map, video, frame_idx=999) is None
    assert find_frames(frame_idx_map, unknown_video) is None


def test_fancy_frame_indices_orders_nearest_then_wraps() -> None:
    frame_map = {
        0: LabeledFrame(video=_video("movie.mp4"), frame_idx=0),
        5: LabeledFrame(video=_video("movie.mp4"), frame_idx=5),
        10: LabeledFrame(video=_video("movie.mp4"), frame_idx=10),
    }

    assert fancy_frame_indices(frame_map, from_frame_idx=5, reverse=False) == [10, 0, 5]
    assert fancy_frame_indices(frame_map, from_frame_idx=10, reverse=True) == [5, 10, 0]
    assert fancy_frame_indices(frame_map, from_frame_idx=6, reverse=False) == [10, 0, 5]
    assert fancy_frame_indices({}, from_frame_idx=0, reverse=False) == []
