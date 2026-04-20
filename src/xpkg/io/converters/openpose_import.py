"""Convert OpenPose BODY_25 ``--write_json`` directories into native archives."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from xpkg.core.path_registry import ensure_dir, resolve_path
from xpkg.io.archive_format import write_archive
from xpkg.io.converters.converter_helpers import (
    ConversionResult,
    ProgressCallback,
    _emit,
    points_from_coords_scores,
)
from xpkg.io.converters.dlc_import import _resolve_video_path, _validate_video_alignment
from xpkg.io.converters.pose_track_import import build_pose_track_skeleton
from xpkg.io.readers.openpose import (
    BODY_25_SKELETON_LINKS,
    OpenPosePerson,
    OpenPoseSequence,
    read_sequence,
)
from xpkg.io.video import Video

if TYPE_CHECKING:
    from xpkg.core.annotations import Point
    from xpkg.core.skeleton import Keypoint
    from xpkg.model import Labels as _Labels

_OPENPOSE_READ_JSON_MARKER = "OPENPOSE_IMPORT STEP: read_json"
_OPENPOSE_VALIDATE_VIDEO_MARKER = "OPENPOSE_IMPORT STEP: validate_video"
_OPENPOSE_BUILD_LABELS_MARKER = "OPENPOSE_IMPORT STEP: build_labels"
_OPENPOSE_WRITE_ARCHIVE_MARKER = "OPENPOSE_IMPORT STEP: write_archive"
_OPENPOSE_DONE_MARKER = "OPENPOSE_IMPORT DONE"

OPENPOSE_JSON_PROGRESS_MARKERS: tuple[tuple[str, int], ...] = (
    (_OPENPOSE_READ_JSON_MARKER, 10),
    (_OPENPOSE_VALIDATE_VIDEO_MARKER, 35),
    (_OPENPOSE_BUILD_LABELS_MARKER, 60),
    (_OPENPOSE_WRITE_ARCHIVE_MARKER, 80),
    (_OPENPOSE_DONE_MARKER, 100),
)


def _points_for_person(
    person: OpenPosePerson,
    node_names: tuple[str, ...],
    *,
    likelihood_threshold: float,
) -> dict[str | Keypoint, Point]:
    return points_from_coords_scores(
        node_names,
        person.coords,
        person.scores,
        likelihood_threshold=likelihood_threshold,
    )


def _labels_from_openpose_sequence(
    sequence: OpenPoseSequence,
    *,
    video_path: Path,
    skeleton_name: str,
    likelihood_threshold: float,
) -> _Labels:
    from xpkg.core.annotations import Instance, LabeledFrame
    from xpkg.model import Labels

    skeleton = build_pose_track_skeleton(
        sequence.node_names,
        skeleton_name=skeleton_name,
        skeleton_links=BODY_25_SKELETON_LINKS,
    )
    video = Video.from_filename(video_path.as_posix())
    labels = Labels(skeletons=[skeleton], videos=[video])

    for frame_idx, frame in enumerate(sequence.frames):
        instances: list[Instance] = []
        for person in frame.people:
            points = _points_for_person(
                person,
                sequence.node_names,
                likelihood_threshold=likelihood_threshold,
            )
            if not points:
                continue
            instances.append(Instance(skeleton=skeleton, init_points=points))

        if not instances:
            continue
        labels.append(LabeledFrame(video=video, frame_idx=frame_idx, instances=instances))

    labels.update_cache()
    return labels


def _write_openpose_archive(
    labels: _Labels,
    *,
    json_dir: Path,
    video_path: Path,
    out_path: Path,
    progress_callback: ProgressCallback | None,
) -> ConversionResult:
    # The shared DLC writer helper assumes a file-like tracking source, while OpenPose imports
    # are directory-shaped and should report that directory as the source root.
    ensure_dir(out_path.parent)
    metadata = {
        "source": "openpose_json_import",
        "source_json_dir": json_dir.as_posix(),
        "source_video": video_path.as_posix(),
        "pose_model": "BODY_25",
    }
    _emit(progress_callback, f"IMPORT: Writing {out_path.name}")
    write_archive(out_path, labels, metadata=metadata)
    _emit(progress_callback, "IMPORT: Done")
    return ConversionResult(
        source_dir=json_dir,
        project_root=out_path.parent,
        videos=[video_path],
        archive_path=out_path,
    )


def convert_openpose_json(
    json_dir: Path | str,
    video_path: Path | str,
    out_path: Path | str,
    *,
    skeleton_name: str = "imported",
    likelihood_threshold: float = 0.0,
    progress_callback: ProgressCallback | None = None,
) -> ConversionResult:
    """Convert an OpenPose BODY_25 ``--write_json`` directory plus video into an archive."""

    resolved_json_dir = resolve_path(json_dir)
    resolved_video_path = _resolve_video_path(video_path)
    resolved_out_path = resolve_path(out_path)

    _emit(progress_callback, _OPENPOSE_READ_JSON_MARKER)
    sequence = read_sequence(resolved_json_dir)
    _emit(
        progress_callback,
        "IMPORT: Found "
        f"{len(sequence.frames)} JSON frames, {len(sequence.node_names)} BODY_25 keypoints",
    )

    _emit(progress_callback, _OPENPOSE_VALIDATE_VIDEO_MARKER)
    video = Video.from_filename(resolved_video_path.as_posix())
    _validate_video_alignment([video], required_frames=len(sequence.frames))

    _emit(progress_callback, _OPENPOSE_BUILD_LABELS_MARKER)
    labels = _labels_from_openpose_sequence(
        sequence,
        video_path=resolved_video_path,
        skeleton_name=skeleton_name,
        likelihood_threshold=likelihood_threshold,
    )
    labels.validate()

    _emit(progress_callback, _OPENPOSE_WRITE_ARCHIVE_MARKER)
    result = _write_openpose_archive(
        labels,
        json_dir=resolved_json_dir,
        video_path=resolved_video_path,
        out_path=resolved_out_path,
        progress_callback=progress_callback,
    )
    _emit(progress_callback, _OPENPOSE_DONE_MARKER)
    return result


__all__ = [
    "OPENPOSE_JSON_PROGRESS_MARKERS",
    "convert_openpose_json",
]
