"""Labels package for Posetta."""

from pathlib import Path

from posetta.io.labels.cache import LabelsDataCache
from posetta.io.labels.merge import (
    complex_merge_between,
    finish_complex_merge,
    merge_container_dicts,
    merge_matching_frames,
    unify_video_references,
)
from posetta.io.labels.model import Labels, SuggestionFrame
from posetta.io.labels.query import (
    LabelsQuery,
    build_frame_index_map,
    fancy_frame_indices,
    find_frames,
    group_labeled_frames_by_video,
)
from posetta.io.labels.tracks import (
    add_track,
    find_track_occupancy,
    get_track_occupancy,
    remove_all_tracks,
    remove_track,
    remove_unused_tracks,
    track_set_instance,
    track_swap,
)


def make_video_callback():
    """Validate video paths (manifest-only, no search or GUI)."""

    def video_callback(video_list: list[dict]) -> None:
        filenames = [str(item["backend"]["filename"]) for item in video_list]
        if not filenames:
            raise ValueError("No video filenames provided for resolution")

        for raw in filenames:
            candidate = Path(str(raw)).expanduser()
            if not candidate.is_absolute():
                raise FileNotFoundError(f"Video filename must be absolute: {candidate}")
            if not candidate.exists():
                raise FileNotFoundError(f"Video file not found: {candidate}")

        for idx, item in enumerate(video_list):
            item["backend"]["filename"] = str(Path(filenames[idx]).resolve())

    return video_callback


__all__ = [
    "Labels",
    "LabelsDataCache",
    "LabelsQuery",
    "SuggestionFrame",
    "add_track",
    "build_frame_index_map",
    "complex_merge_between",
    "fancy_frame_indices",
    "find_frames",
    "find_track_occupancy",
    "finish_complex_merge",
    "get_track_occupancy",
    "group_labeled_frames_by_video",
    "make_video_callback",
    "merge_container_dicts",
    "merge_matching_frames",
    "remove_all_tracks",
    "remove_track",
    "remove_unused_tracks",
    "track_set_instance",
    "track_swap",
    "unify_video_references",
]
