"""Convert serialized MediaPipe pose-landmarks JSON into project-ready labels."""

from __future__ import annotations

from pathlib import Path

from xpkg.io.converters.dlc_import import (
    _resolve_tracking_path,
    _resolve_video_path,
    _tracking_conversion_result,
    _validate_video_alignment,
)
from xpkg.io.converters.pose_track_import import labels_from_pose_tracks
from xpkg.io.converters.progress import ProgressCallback, emit_progress
from xpkg.io.converters.result import ConversionResult
from xpkg.io.readers.pose.mediapipe_pose_landmarks import (
    MEDIAPIPE_POSE_CONNECTIONS,
    MEDIAPIPE_POSE_LANDMARK_NAMES,
    read_image_size,
    read_track,
)
from xpkg.media.video import Video

_MEDIAPIPE_READ_JSON_MARKER = "MEDIAPIPE_IMPORT STEP: read_json"
_MEDIAPIPE_VALIDATE_VIDEO_MARKER = "MEDIAPIPE_IMPORT STEP: validate_video"
_MEDIAPIPE_BUILD_LABELS_MARKER = "MEDIAPIPE_IMPORT STEP: build_labels"
_MEDIAPIPE_PREPARE_RESULT_MARKER = "MEDIAPIPE_IMPORT STEP: prepare_project_state"
_MEDIAPIPE_DONE_MARKER = "MEDIAPIPE_IMPORT DONE"

MEDIAPIPE_POSE_LANDMARKS_JSON_PROGRESS_MARKERS: tuple[tuple[str, int], ...] = (
    (_MEDIAPIPE_READ_JSON_MARKER, 10),
    (_MEDIAPIPE_VALIDATE_VIDEO_MARKER, 35),
    (_MEDIAPIPE_BUILD_LABELS_MARKER, 60),
    (_MEDIAPIPE_PREPARE_RESULT_MARKER, 80),
    (_MEDIAPIPE_DONE_MARKER, 100),
)


def _mediapipe_skeleton_links() -> list[tuple[int, int]]:
    index_by_name = {
        node_name: node_index for node_index, node_name in enumerate(MEDIAPIPE_POSE_LANDMARK_NAMES)
    }
    return [
        (index_by_name[start_name], index_by_name[end_name])
        for start_name, end_name in MEDIAPIPE_POSE_CONNECTIONS
    ]


def convert_mediapipe_pose_landmarks_json(
    json_path: Path | str,
    video_path: Path | str,
    *,
    skeleton_name: str = "mediapipe_pose",
    likelihood_threshold: float = 0.0,
    progress_callback: ProgressCallback | None = None,
) -> ConversionResult:
    """Convert serialized MediaPipe pose-landmarks JSON plus a video into project-ready labels."""

    resolved_json_path = _resolve_tracking_path(json_path)
    resolved_video_path = _resolve_video_path(video_path)

    emit_progress(progress_callback, _MEDIAPIPE_READ_JSON_MARKER)
    track = read_track(resolved_json_path, track_index=0)
    image_width, image_height = read_image_size(resolved_json_path)
    emit_progress(
        progress_callback,
        "IMPORT: Found "
        f"{len(track.node_names)} keypoints, {track.coords.shape[0]} frames, "
        f"image size {image_width}x{image_height}",
    )

    emit_progress(progress_callback, _MEDIAPIPE_VALIDATE_VIDEO_MARKER)
    video = Video.from_filename(resolved_video_path.as_posix())
    _validate_video_alignment([video], required_frames=int(track.coords.shape[0]))
    if int(video.width) != image_width or int(video.height) != image_height:
        raise ValueError(
            "MediaPipe pose-landmarks JSON image size "
            f"{image_width}x{image_height} does not match video size "
            f"{int(video.width)}x{int(video.height)} for {resolved_video_path}"
        )

    emit_progress(progress_callback, _MEDIAPIPE_BUILD_LABELS_MARKER)
    labels = labels_from_pose_tracks(
        [track],
        skeleton_name=skeleton_name,
        video=video,
        likelihood_threshold=float(likelihood_threshold),
        skeleton_links=_mediapipe_skeleton_links(),
    )
    labels.validate()

    emit_progress(progress_callback, _MEDIAPIPE_PREPARE_RESULT_MARKER)
    result = _tracking_conversion_result(
        labels,
        data_path=resolved_json_path,
        video_path=resolved_video_path,
        source_label="mediapipe_pose_landmarks_json_import",
        source_metadata_key="source_json",
        progress_callback=progress_callback,
    )
    emit_progress(progress_callback, _MEDIAPIPE_DONE_MARKER)
    return result


__all__ = [
    "MEDIAPIPE_POSE_LANDMARKS_JSON_PROGRESS_MARKERS",
    "convert_mediapipe_pose_landmarks_json",
]
