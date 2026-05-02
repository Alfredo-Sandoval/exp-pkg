"""Convert official MMPose top-down demo JSON exports into project-ready labels."""

from __future__ import annotations

from pathlib import Path

from xpkg.io.converters.converter_helpers import (
    ConversionResult,
    ProgressCallback,
    _emit,
)
from xpkg.io.converters.dlc_import import (
    _resolve_tracking_path,
    _resolve_video_path,
    _tracking_conversion_result,
    _validate_video_alignment,
)
from xpkg.io.converters.pose_track_import import labels_from_pose_tracks
from xpkg.io.readers import read_pose_track
from xpkg.io.readers.pose.mmpose import read_sequence_dataset_name, read_skeleton_links
from xpkg.media.video import Video

_MMPOSE_JSON_READ_MARKER = "MMPOSE_JSON_IMPORT STEP: read_json"
_MMPOSE_JSON_VALIDATE_VIDEO_MARKER = "MMPOSE_JSON_IMPORT STEP: validate_video"
_MMPOSE_JSON_BUILD_LABELS_MARKER = "MMPOSE_JSON_IMPORT STEP: build_labels"
_MMPOSE_JSON_PREPARE_RESULT_MARKER = "MMPOSE_JSON_IMPORT STEP: prepare_project_state"
_MMPOSE_JSON_DONE_MARKER = "MMPOSE_JSON_IMPORT DONE"

MMPOSE_TOPDOWN_JSON_PROGRESS_MARKERS: tuple[tuple[str, int], ...] = (
    (_MMPOSE_JSON_READ_MARKER, 10),
    (_MMPOSE_JSON_VALIDATE_VIDEO_MARKER, 35),
    (_MMPOSE_JSON_BUILD_LABELS_MARKER, 60),
    (_MMPOSE_JSON_PREPARE_RESULT_MARKER, 80),
    (_MMPOSE_JSON_DONE_MARKER, 100),
)


def convert_mmpose_topdown_json(
    json_path: Path | str,
    video_path: Path | str,
    *,
    skeleton_name: str = "imported",
    instance_index: int = 0,
    likelihood_threshold: float = 0.0,
    progress_callback: ProgressCallback | None = None,
) -> ConversionResult:
    """Convert official MMPose top-down JSON into project-ready labels."""

    resolved_json_path = _resolve_tracking_path(json_path)
    resolved_video_path = _resolve_video_path(video_path)

    _emit(progress_callback, _MMPOSE_JSON_READ_MARKER)
    track = read_pose_track(
        resolved_json_path,
        software="MMPose",
        file_type="json",
        track_index=int(instance_index),
    )
    dataset_name = read_sequence_dataset_name(resolved_json_path)
    skeleton_links = read_skeleton_links(resolved_json_path)
    frame_count = int(track.coords.shape[0])
    _emit(
        progress_callback,
        "IMPORT: Found "
        f"{len(track.node_names)} keypoints across {frame_count} frames "
        f"for MMPose dataset {dataset_name!r}",
    )

    _emit(progress_callback, _MMPOSE_JSON_VALIDATE_VIDEO_MARKER)
    video = Video.from_filename(resolved_video_path.as_posix())
    _validate_video_alignment([video], required_frames=frame_count)

    _emit(progress_callback, _MMPOSE_JSON_BUILD_LABELS_MARKER)
    labels = labels_from_pose_tracks(
        [track],
        skeleton_name=skeleton_name,
        video=video,
        likelihood_threshold=float(likelihood_threshold),
        skeleton_links=skeleton_links,
    )
    labels.validate()

    _emit(progress_callback, _MMPOSE_JSON_PREPARE_RESULT_MARKER)
    result = _tracking_conversion_result(
        labels,
        data_path=resolved_json_path,
        video_path=resolved_video_path,
        source_label="mmpose_topdown_json_import",
        source_metadata_key="source_json",
        progress_callback=progress_callback,
    )
    _emit(progress_callback, _MMPOSE_JSON_DONE_MARKER)
    return result


__all__ = [
    "MMPOSE_TOPDOWN_JSON_PROGRESS_MARKERS",
    "ConversionResult",
    "ProgressCallback",
    "convert_mmpose_topdown_json",
]
