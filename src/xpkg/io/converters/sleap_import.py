"""Convert a SLEAP `.pkg.slp` directly into native project archives."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from xpkg._core.json_utils import parse_json_dict
from xpkg._core.path_registry import ensure_dir, resolve_path
from xpkg.io.archive_format import write_archive
from xpkg.io.archive_format.shared import CANONICAL_ARCHIVE_SUFFIX
from xpkg.io.converters.converter_helpers import (
    ConversionResult,
    ProgressCallback,
    _emit,
    project_archive_path,
    rebase_image_sequences,
    remap_labels_to_videos,
)
from xpkg.io.converters.converter_helpers import (
    encode_videos as _encode_videos,
)
from xpkg.io.converters.dlc_import import (
    _resolve_tracking_path,
    _resolve_video_path,
    _validate_video_alignment,
    _write_tracking_archive,
)
from xpkg.io.converters.pose_track_import import (
    labels_from_pose_tracks,
    validate_pose_tracks_consistency,
)
from xpkg.io.converters.sleap_helpers import extract_frames, extract_labels_step4
from xpkg.io.readers.sleap_analysis_h5 import (
    read_node_names as _read_sleap_node_names,
)
from xpkg.io.readers.sleap_analysis_h5 import (
    read_track as _read_sleap_track,
)
from xpkg.io.readers.sleap_analysis_h5 import (
    read_track_count as _read_sleap_track_count,
)
from xpkg.io.readers.sleap_analysis_h5 import (
    read_track_names as _read_sleap_track_names,
)
from xpkg.io.skeleton_loaders import build_sleap_skeleton
from xpkg.io.video import Video

if TYPE_CHECKING:
    from xpkg.model import Labels as _Labels
    from xpkg.pose.skeleton import Keypoint
    from xpkg.pose.skeleton import Skeleton as _Skeleton

_NAT_SORT_RE = re.compile(r"(\d+)")

_START_EXTRACTING_FRAMES_MARKER = "XPKG_IMPORT START: extracting_frames"
_OK_FRAMES_EXTRACTED_MARKER = "XPKG_IMPORT OK: frames_extracted"
_START_BUILD_LABEL_TABLE_MARKER = "XPKG_IMPORT START: build_label_table"
_OK_LABEL_TABLE_READY_MARKER = "XPKG_IMPORT OK: label_table_ready"
_ASSEMBLE_LABELS_MARKER = "XPKG_IMPORT STEP: assemble_labels"
_BUILD_VIDEO_MARKER = "XPKG_IMPORT STEP: build_video"
_COPY_FRAMES_MARKER = "XPKG_IMPORT STEP: copy_frames"
_WRITE_ARCHIVE_MARKER = "XPKG_IMPORT STEP: write_archive"
_OK_ARCHIVE_WRITTEN_MARKER = "XPKG_IMPORT OK: archive_written"
_CLEANUP_TEMP_MARKER = "XPKG_IMPORT STEP: cleanup_temp_folders"
_DONE_MARKER = "XPKG_IMPORT DONE"

SLEAP_PACKAGE_PROGRESS_MARKERS: tuple[tuple[str, int], ...] = (
    (_START_EXTRACTING_FRAMES_MARKER, 10),
    (_OK_FRAMES_EXTRACTED_MARKER, 30),
    (_START_BUILD_LABEL_TABLE_MARKER, 35),
    (_OK_LABEL_TABLE_READY_MARKER, 45),
    (_ASSEMBLE_LABELS_MARKER, 55),
    (_BUILD_VIDEO_MARKER, 70),
    (_COPY_FRAMES_MARKER, 72),
    (_WRITE_ARCHIVE_MARKER, 80),
    (_OK_ARCHIVE_WRITTEN_MARKER, 92),
    (_CLEANUP_TEMP_MARKER, 96),
    (_DONE_MARKER, 100),
)

_SLEAP_H5_READ_TRACKS_MARKER = "SLEAP_H5_IMPORT STEP: read_h5"
_SLEAP_H5_VALIDATE_VIDEO_MARKER = "SLEAP_H5_IMPORT STEP: validate_video"
_SLEAP_H5_BUILD_LABELS_MARKER = "SLEAP_H5_IMPORT STEP: build_labels"
_SLEAP_H5_WRITE_ARCHIVE_MARKER = "SLEAP_H5_IMPORT STEP: write_archive"
_SLEAP_H5_DONE_MARKER = "SLEAP_H5_IMPORT DONE"

SLEAP_H5_PROGRESS_MARKERS: tuple[tuple[str, int], ...] = (
    (_SLEAP_H5_READ_TRACKS_MARKER, 10),
    (_SLEAP_H5_VALIDATE_VIDEO_MARKER, 35),
    (_SLEAP_H5_BUILD_LABELS_MARKER, 60),
    (_SLEAP_H5_WRITE_ARCHIVE_MARKER, 80),
    (_SLEAP_H5_DONE_MARKER, 100),
)


def _frame_sort_key(path: Path) -> list[int | str]:
    """Sort key for natural sorting (for example img1, img2, img10)."""

    parts = _NAT_SORT_RE.split(path.stem)
    return [int(part) if part.isdigit() else part for part in parts]


def _labels_from_step4_table(
    table: pd.DataFrame,
    labeled_root: Path,
    skeleton: _Skeleton,
) -> _Labels:
    from xpkg.io.video import Video as _Video
    from xpkg.model import Labels as _Labels
    from xpkg.pose.annotations import Instance as _Instance
    from xpkg.pose.annotations import LabeledFrame as _LabeledFrame
    from xpkg.pose.annotations import Point as _Point

    labels = _Labels()

    if table.empty:
        labels.skeletons = [skeleton]
        return labels

    table = table.drop_duplicates()

    frames_by_dir: dict[Path, set[Path]] = {}
    rows_by_frame: dict[Path, list[pd.Series]] = {}
    for _, row in table.iterrows():
        rel_frame = Path(str(row["frame"]))
        abs_frame = rel_frame if rel_frame.is_absolute() else (labeled_root / rel_frame)
        abs_frame = abs_frame.resolve()
        frames_by_dir.setdefault(abs_frame.parent, set()).add(abs_frame)
        rows_by_frame.setdefault(abs_frame, []).append(row)

    videos_by_dir: dict[Path, Any] = {}
    frame_idx_map: dict[str, int] = {}
    for video_dir, frame_paths in frames_by_dir.items():
        ordered = sorted(frame_paths, key=_frame_sort_key)
        ordered_str = [path.as_posix() for path in ordered]
        if not ordered_str:
            continue
        video = _Video.from_image_filenames(ordered_str)
        videos_by_dir[video_dir] = video
        for idx, frame_path in enumerate(ordered_str):
            frame_idx_map[frame_path] = idx

    keypoints = [str(col[:-2]) for col in table.columns if str(col).endswith("_x")]

    for abs_frame, row_group in rows_by_frame.items():
        video = videos_by_dir.get(abs_frame.parent)
        if video is None:
            continue
        frame_idx = frame_idx_map.get(abs_frame.as_posix())
        if frame_idx is None:
            continue

        instances: list[_Instance] = []
        for row in row_group:
            points: dict[str | Keypoint, _Point] = {}
            for kp in keypoints:
                x_val = row.get(f"{kp}_x")
                y_val = row.get(f"{kp}_y")
                if pd.isna(x_val) or pd.isna(y_val) or x_val is None or y_val is None:
                    continue
                points[kp] = _Point(float(x_val), float(y_val), visible=True, complete=True)

            if not points:
                continue
            instances.append(_Instance(skeleton=skeleton, init_points=points))

        if not instances:
            continue

        labeled_frame = _LabeledFrame(video=video, frame_idx=int(frame_idx), instances=instances)
        labels.append(labeled_frame)

    if not labels.skeletons:
        labels.skeletons = [skeleton]
    labels.update_cache()
    return labels


def _sleap_h5_points_for_frame(
    coords: np.ndarray,
    scores: np.ndarray,
    node_names: list[str],
    *,
    likelihood_threshold: float,
) -> dict[str | Keypoint, Any]:
    from xpkg.pose.annotations import Point

    points: dict[str | Keypoint, Any] = {}
    for node_idx, node_name in enumerate(node_names):
        x_val = float(coords[node_idx, 0])
        y_val = float(coords[node_idx, 1])
        score_val = float(scores[node_idx])
        if np.isnan(x_val) or np.isnan(y_val) or np.isnan(score_val):
            continue
        if score_val < likelihood_threshold:
            continue
        points[node_name] = Point(x_val, y_val, visible=True, complete=True)
    return points


def _validate_sleap_tracks_consistency(
    tracks: list[Any],
    *,
    source_path: Path,
) -> int:
    frame_count, _node_names = validate_pose_tracks_consistency(
        tracks,
        source_label=f"SLEAP analysis H5 {source_path}",
    )
    return frame_count


def _labels_from_sleap_h5_tracks(
    tracks: list[Any],
    track_names: list[str],
    *,
    skeleton_name: str,
    video: Any,
    likelihood_threshold: float,
) -> _Labels:
    return labels_from_pose_tracks(
        tracks,
        track_names=track_names,
        skeleton_name=skeleton_name,
        video=video,
        likelihood_threshold=likelihood_threshold,
    )


def convert_sleap_h5(
    h5_path: Path | str,
    video_path: Path | str,
    out_path: Path | str,
    *,
    skeleton_name: str = "imported",
    likelihood_threshold: float = 0.0,
    archive_extension: str = CANONICAL_ARCHIVE_SUFFIX,
    progress_callback: ProgressCallback | None = None,
) -> ConversionResult:
    """Convert a SLEAP analysis H5 export plus its video into a native archive."""

    resolved_h5_path = _resolve_tracking_path(h5_path)
    resolved_video_path = _resolve_video_path(video_path)
    resolved_out_path = resolve_path(out_path)

    _emit(progress_callback, _SLEAP_H5_READ_TRACKS_MARKER)
    track_count = _read_sleap_track_count(resolved_h5_path)
    if track_count <= 0:
        raise ValueError(f"SLEAP analysis H5 contains no tracks: {resolved_h5_path}")
    node_names = _read_sleap_node_names(resolved_h5_path)
    track_names = _read_sleap_track_names(resolved_h5_path)
    tracks = [
        _read_sleap_track(resolved_h5_path, track_index=track_idx)
        for track_idx in range(track_count)
    ]
    frame_count = _validate_sleap_tracks_consistency(tracks, source_path=resolved_h5_path)
    _emit(
        progress_callback,
        f"IMPORT: Found {track_count} tracks, {len(node_names)} keypoints, {frame_count} frames",
    )

    _emit(progress_callback, _SLEAP_H5_VALIDATE_VIDEO_MARKER)
    video = Video.from_filename(resolved_video_path.as_posix())
    _validate_video_alignment([video], required_frames=frame_count)

    _emit(progress_callback, _SLEAP_H5_BUILD_LABELS_MARKER)
    labels = _labels_from_sleap_h5_tracks(
        tracks,
        track_names,
        skeleton_name=skeleton_name,
        video=video,
        likelihood_threshold=likelihood_threshold,
    )
    labels.validate()

    _emit(progress_callback, _SLEAP_H5_WRITE_ARCHIVE_MARKER)
    result = _write_tracking_archive(
        labels,
        out_path=resolved_out_path,
        data_path=resolved_h5_path,
        video_path=resolved_video_path,
        source_label="sleap_h5_import",
        source_metadata_key="source_h5",
        progress_callback=progress_callback,
    )
    _emit(progress_callback, _SLEAP_H5_DONE_MARKER)
    return result


def convert_sleap_package(
    slp: Path | str,
    out_dir: Path | str,
    *,
    fps: int = 30,
    encode_videos: bool | None = None,
    archive_extension: str = CANONICAL_ARCHIVE_SUFFIX,
    progress_callback: ProgressCallback | None = None,
) -> ConversionResult:
    """Convert a SLEAP `.pkg.slp` archive into a native project archive."""

    slp_path = resolve_path(slp)
    proj_root = resolve_path(out_dir)
    ensure_dir(proj_root)
    tmp_extract = proj_root / "_tmp_extract"
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract.as_posix())
    ensure_dir(tmp_extract)

    _emit(progress_callback, "XPKG_IMPORT: extracting frames + labels")
    _emit(progress_callback, _START_EXTRACTING_FRAMES_MARKER)
    extract_frames(slp_path.as_posix(), tmp_extract.as_posix())
    _emit(progress_callback, _OK_FRAMES_EXTRACTED_MARKER)

    _emit(progress_callback, _START_BUILD_LABEL_TABLE_MARKER)
    label_table = extract_labels_step4(slp_path.as_posix(), tmp_extract.as_posix())
    _emit(progress_callback, _OK_LABEL_TABLE_READY_MARKER)

    import h5py as _h5

    from xpkg.pose.skeleton import Skeleton as _Skeleton

    with _h5.File(slp_path.as_posix(), "r") as hdf_for_skeleton:
        metadata = parse_json_dict(hdf_for_skeleton["metadata"].attrs.get("json", "{}"))
    skeleton = _Skeleton.from_dict(build_sleap_skeleton(metadata), normalize_names=True)

    _emit(progress_callback, _ASSEMBLE_LABELS_MARKER)
    labels = _labels_from_step4_table(label_table, tmp_extract, skeleton)
    if not labels.skeletons:
        labels.skeletons = [skeleton]
    labels.validate()

    videos: list[Path] = []
    if encode_videos is None or encode_videos:
        videos = _encode_videos(tmp_extract, proj_root, fps=int(fps), progress=progress_callback)
    else:
        _emit(progress_callback, "XPKG_IMPORT: skipping mp4 encoding (no-videos)")
        labeled_src = tmp_extract / "labeled-data"
        if labeled_src.exists():
            labeled_dst = proj_root / "videos" / "labeled-data"
            ensure_dir(labeled_dst.parent)
            _emit(progress_callback, _COPY_FRAMES_MARKER)
            shutil.copytree(labeled_src.as_posix(), labeled_dst.as_posix(), dirs_exist_ok=True)
            rebase_image_sequences(labels, labeled_src, labeled_dst)

    if videos:
        remap_labels_to_videos(labels, videos, proj_root)

    archive_path = project_archive_path(proj_root, archive_extension=archive_extension)
    metadata = {
        "project_name": proj_root.name,
        "source": "sleap_pkg_import",
        "source_package": slp_path.as_posix(),
    }
    _emit(progress_callback, _WRITE_ARCHIVE_MARKER)
    write_archive(archive_path, labels, metadata=metadata)
    _emit(progress_callback, _OK_ARCHIVE_WRITTEN_MARKER)

    _emit(progress_callback, _CLEANUP_TEMP_MARKER)
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract.as_posix())

    result = ConversionResult(
        source_dir=slp_path,
        project_root=proj_root,
        videos=videos,
        archive_path=archive_path,
    )

    _emit(progress_callback, _DONE_MARKER)
    return result


__all__ = [
    "SLEAP_H5_PROGRESS_MARKERS",
    "SLEAP_PACKAGE_PROGRESS_MARKERS",
    "convert_sleap_h5",
    "convert_sleap_package",
]
