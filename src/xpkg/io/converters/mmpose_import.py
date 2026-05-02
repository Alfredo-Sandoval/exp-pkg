"""Convert official MMPose top-down demo JSON exports into native archives."""

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
from xpkg.io.readers import read_pose_track
from xpkg.io.readers.mmpose import read_sequence_dataset_name, read_skeleton_links
from xpkg.io.video import Video

_MMPOSE_JSON_READ_MARKER = "MMPOSE_JSON_IMPORT STEP: read_json"
_MMPOSE_JSON_VALIDATE_VIDEO_MARKER = "MMPOSE_JSON_IMPORT STEP: validate_video"
_MMPOSE_JSON_BUILD_LABELS_MARKER = "MMPOSE_JSON_IMPORT STEP: build_labels"
_MMPOSE_JSON_WRITE_ARCHIVE_MARKER = "MMPOSE_JSON_IMPORT STEP: write_archive"
_MMPOSE_JSON_DONE_MARKER = "MMPOSE_JSON_IMPORT DONE"

MMPOSE_TOPDOWN_JSON_PROGRESS_MARKERS: tuple[tuple[str, int], ...] = (
    (_MMPOSE_JSON_READ_MARKER, 10),
    (_MMPOSE_JSON_VALIDATE_VIDEO_MARKER, 35),
    (_MMPOSE_JSON_BUILD_LABELS_MARKER, 60),
    (_MMPOSE_JSON_WRITE_ARCHIVE_MARKER, 80),
    (_MMPOSE_JSON_DONE_MARKER, 100),
)


def convert_mmpose_topdown_json(
    json_path: Path | str,
    video_path: Path | str,
    out_path: Path | str,
    *,
    skeleton_name: str = "imported",
    instance_index: int = 0,
    likelihood_threshold: float = 0.0,
    archive_extension: str = CANONICAL_ARCHIVE_SUFFIX,
    progress_callback: ProgressCallback | None = None,
) -> ConversionResult:
    """Convert official `topdown_demo_with_mmdet.py --save-predictions` JSON into xpkg."""

    resolved_json_path = _resolve_tracking_path(json_path)
    resolved_video_path = _resolve_video_path(video_path)
    resolved_out_path = resolve_path(out_path)
    if not str(archive_extension).startswith("."):
        raise ValueError(
            f"archive_extension must start with '.', got {archive_extension!r}."
        )

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

    _emit(progress_callback, _MMPOSE_JSON_WRITE_ARCHIVE_MARKER)
    result = _write_tracking_archive(
        labels,
        out_path=resolved_out_path,
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
