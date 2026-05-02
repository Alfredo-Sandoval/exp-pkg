"""Convert serialized MediaPipe pose-landmarks JSON into native archives."""

from __future__ import annotations

from pathlib import Path

from xpkg._core.path_registry import resolve_path
from xpkg.io.archive_format.shared import CANONICAL_ARCHIVE_SUFFIX
from xpkg.io.converters.converter_helpers import (
    ConversionResult,
    ProgressCallback,
    _emit,
)
from xpkg.io.converters.dlc_import import (
    _resolve_tracking_path,
    _resolve_video_path,
    _validate_video_alignment,
    _write_tracking_archive,
)
from xpkg.io.converters.pose_track_import import labels_from_pose_tracks
from xpkg.io.readers.mediapipe_pose_landmarks import (
    MEDIAPIPE_POSE_CONNECTIONS,
    MEDIAPIPE_POSE_LANDMARK_NAMES,
    read_image_size,
    read_track,
)
from xpkg.io.video import Video

_MEDIAPIPE_READ_JSON_MARKER = "MEDIAPIPE_IMPORT STEP: read_json"
_MEDIAPIPE_VALIDATE_VIDEO_MARKER = "MEDIAPIPE_IMPORT STEP: validate_video"
_MEDIAPIPE_BUILD_LABELS_MARKER = "MEDIAPIPE_IMPORT STEP: build_labels"
_MEDIAPIPE_WRITE_ARCHIVE_MARKER = "MEDIAPIPE_IMPORT STEP: write_archive"
_MEDIAPIPE_DONE_MARKER = "MEDIAPIPE_IMPORT DONE"

MEDIAPIPE_POSE_LANDMARKS_JSON_PROGRESS_MARKERS: tuple[tuple[str, int], ...] = (
    (_MEDIAPIPE_READ_JSON_MARKER, 10),
    (_MEDIAPIPE_VALIDATE_VIDEO_MARKER, 35),
    (_MEDIAPIPE_BUILD_LABELS_MARKER, 60),
    (_MEDIAPIPE_WRITE_ARCHIVE_MARKER, 80),
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
    out_path: Path | str,
    *,
    skeleton_name: str = "mediapipe_pose",
    likelihood_threshold: float = 0.0,
    archive_extension: str = CANONICAL_ARCHIVE_SUFFIX,
    progress_callback: ProgressCallback | None = None,
) -> ConversionResult:
    """Convert serialized MediaPipe pose-landmarks JSON plus a video into an archive."""

    resolved_json_path = _resolve_tracking_path(json_path)
    resolved_video_path = _resolve_video_path(video_path)
    resolved_out_path = resolve_path(out_path)
    if archive_extension != CANONICAL_ARCHIVE_SUFFIX:
        raise ValueError(
            "MediaPipe pose-landmarks JSON conversion writes a single archive file; "
            f"archive_extension must be {CANONICAL_ARCHIVE_SUFFIX!r}."
        )

    _emit(progress_callback, _MEDIAPIPE_READ_JSON_MARKER)
    track = read_track(resolved_json_path, track_index=0)
    image_width, image_height = read_image_size(resolved_json_path)
    _emit(
        progress_callback,
        "IMPORT: Found "
        f"{len(track.node_names)} keypoints, {track.coords.shape[0]} frames, "
        f"image size {image_width}x{image_height}",
    )

    _emit(progress_callback, _MEDIAPIPE_VALIDATE_VIDEO_MARKER)
    video = Video.from_filename(resolved_video_path.as_posix())
    _validate_video_alignment([video], required_frames=int(track.coords.shape[0]))
    if int(video.width) != image_width or int(video.height) != image_height:
        raise ValueError(
            "MediaPipe pose-landmarks JSON image size "
            f"{image_width}x{image_height} does not match video size "
            f"{int(video.width)}x{int(video.height)} for {resolved_video_path}"
        )

    _emit(progress_callback, _MEDIAPIPE_BUILD_LABELS_MARKER)
    labels = labels_from_pose_tracks(
        [track],
        skeleton_name=skeleton_name,
        video=video,
        likelihood_threshold=float(likelihood_threshold),
        skeleton_links=_mediapipe_skeleton_links(),
    )
    labels.validate()

    _emit(progress_callback, _MEDIAPIPE_WRITE_ARCHIVE_MARKER)
    result = _write_tracking_archive(
        labels,
        out_path=resolved_out_path,
        data_path=resolved_json_path,
        video_path=resolved_video_path,
        source_label="mediapipe_pose_landmarks_json_import",
        source_metadata_key="source_json",
        progress_callback=progress_callback,
    )
    _emit(progress_callback, _MEDIAPIPE_DONE_MARKER)
    return result


__all__ = [
    "MEDIAPIPE_POSE_LANDMARKS_JSON_PROGRESS_MARKERS",
    "convert_mediapipe_pose_landmarks_json",
]
