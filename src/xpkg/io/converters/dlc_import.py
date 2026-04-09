"""Convert DeepLabCut-style tracking data into native bundle archives."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from xpkg.core.path_registry import ensure_dir, resolve_path
from xpkg.core.skeleton import build_keypoint_skeleton
from xpkg.io.siesta_format.shared import CANONICAL_BUNDLE_SUFFIX
from xpkg.io.converters.converter_helpers import (
    ConversionResult,
    ProgressCallback,
    _emit,
    project_bundle_path,
)
from xpkg.io.readers.dlc import read_dlc_csv_table, read_dlc_h5_table
from xpkg.io.siesta_format import write_siesta
from xpkg.io.video import Video

if TYPE_CHECKING:
    from xpkg.core.skeleton import Keypoint
    from xpkg.core.skeleton import Skeleton as _Skeleton
    from xpkg.model import Labels as _Labels

DlcReader = Callable[[Path], tuple[pd.DataFrame, list[str]]]

_DLC_READ_H5_MARKER = "DLC_IMPORT STEP: read_h5"
_DLC_VALIDATE_VIDEOS_MARKER = "DLC_IMPORT STEP: validate_videos"
_DLC_BUILD_LABELS_MARKER = "DLC_IMPORT STEP: build_labels"
_DLC_WRITE_BUNDLE_MARKER = "DLC_IMPORT STEP: write_bundle"
_DLC_DONE_MARKER = "DLC_IMPORT DONE"

DLC_H5_PROJECT_PROGRESS_MARKERS: tuple[tuple[str, int], ...] = (
    (_DLC_READ_H5_MARKER, 10),
    (_DLC_VALIDATE_VIDEOS_MARKER, 30),
    (_DLC_BUILD_LABELS_MARKER, 55),
    (_DLC_WRITE_BUNDLE_MARKER, 80),
    (_DLC_DONE_MARKER, 100),
)


def _read_dlc_csv(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read a DLC-style multi-index CSV and return a flat table plus keypoint names."""
    return read_dlc_csv_table(csv_path)


def _read_dlc_h5(h5_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read a DLC-style H5 tracking file and return a flat table plus keypoint names."""
    return read_dlc_h5_table(h5_path)


def _required_frame_count(index: pd.Index) -> int:
    if len(index) == 0:
        return 0
    return max(int(frame_idx) for frame_idx in index) + 1


def _coerce_frame_idx(frame_idx: object) -> int:
    if isinstance(frame_idx, bool) or not isinstance(frame_idx, int | float | str):
        raise TypeError(f"DLC frame index must be int-like, got {type(frame_idx).__name__}")
    return int(frame_idx)


def _stored_project_path(path: Path, *, project_root: Path) -> str:
    resolved = resolve_path(path)
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve_tracking_path(data_path: Path | str) -> Path:
    resolved_data_path = resolve_path(data_path)
    if not resolved_data_path.exists():
        raise FileNotFoundError(f"Tracking file not found: {resolved_data_path}")
    return resolved_data_path


def _resolve_video_path(video_path: Path | str) -> Path:
    resolved_video_path = resolve_path(video_path)
    if not resolved_video_path.exists():
        raise FileNotFoundError(f"Video file not found: {resolved_video_path}")
    return resolved_video_path


def _resolve_video_paths(video_paths: Sequence[Path | str]) -> list[Path]:
    if not video_paths:
        raise ValueError("DLC import requires at least one explicit video path")

    resolved: list[Path] = []
    for raw_path in video_paths:
        path = resolve_path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")
        resolved.append(path)
    return resolved


def _load_project_videos(
    video_paths: Sequence[Path],
    *,
    project_root: Path,
) -> list[Video]:
    videos: list[Video] = []
    for path in video_paths:
        video = Video.from_filename(path.as_posix())
        video.filename = _stored_project_path(path, project_root=project_root)
        videos.append(video)
    return videos


def _validate_video_alignment(videos: Sequence[Video], *, required_frames: int) -> None:
    if not videos:
        raise ValueError("At least one DLC video is required")

    reference = videos[0]
    if reference.frames < required_frames:
        raise ValueError(
            f"DLC video has {reference.frames} frames but tracking requires {required_frames}"
        )

    for video in videos[1:]:
        if video.frames != reference.frames:
            raise ValueError("All DLC videos must share the same frame count")
        if video.width != reference.width or video.height != reference.height:
            raise ValueError("All DLC videos must share the same frame size")


def _row_points(
    row: pd.Series,
    keypoints: Sequence[str],
    *,
    likelihood_threshold: float,
) -> dict[str, tuple[float, float]]:
    points: dict[str, tuple[float, float]] = {}
    for keypoint in keypoints:
        x_col = f"{keypoint}_x"
        y_col = f"{keypoint}_y"
        lh_col = f"{keypoint}_likelihood"
        if x_col not in row.index or y_col not in row.index:
            continue
        x_val = row[x_col]
        y_val = row[y_col]
        if lh_col in row.index:
            likelihood = row[lh_col]
            if pd.isna(likelihood) or float(likelihood) < likelihood_threshold:
                continue
        if pd.isna(x_val) or pd.isna(y_val):
            continue
        points[keypoint] = (float(x_val), float(y_val))
    return points


def _instance_points(points: dict[str, tuple[float, float]]) -> dict[str | Keypoint, Any]:
    from xpkg.core.annotations import Point

    return {
        keypoint: Point(x, y, visible=True, complete=True)
        for keypoint, (x, y) in points.items()
    }


def _labels_from_tracking_df(
    df: pd.DataFrame,
    keypoints: list[str],
    skeleton: _Skeleton,
    video_path: Path,
    *,
    stored_video_filename: str | None = None,
    likelihood_threshold: float = 0.0,
) -> _Labels:
    """Convert a tracking table into the canonical `xpkg.model.Labels` object."""

    from xpkg.core.annotations import Instance, LabeledFrame, Point
    from xpkg.model import Labels

    video = Video.from_filename(str(video_path))
    if stored_video_filename is not None:
        video.filename = stored_video_filename
    labels = Labels(skeletons=[skeleton], videos=[video])

    for frame_idx in df.index:
        row = df.loc[frame_idx]

        points: dict[str | Keypoint, Point] = {}
        for kp in keypoints:
            x_col = f"{kp}_x"
            y_col = f"{kp}_y"
            lh_col = f"{kp}_likelihood"

            if x_col not in row or y_col not in row:
                continue

            x_val = row[x_col]
            y_val = row[y_col]

            if lh_col in row:
                likelihood = row[lh_col]
                if pd.isna(likelihood) or float(likelihood) < likelihood_threshold:
                    continue

            if pd.isna(x_val) or pd.isna(y_val):
                continue

            points[kp] = Point(float(x_val), float(y_val), visible=True, complete=True)

        if not points:
            continue

        instance = Instance(skeleton=skeleton, init_points=points)
        labeled_frame = LabeledFrame(video=video, frame_idx=int(frame_idx), instances=[instance])
        labels.append(labeled_frame)

    labels.update_cache()
    return labels


def _labels_from_tracking_df_project(
    df: pd.DataFrame,
    keypoints: Sequence[str],
    *,
    videos: Sequence[Video],
    skeleton_name: str,
    likelihood_threshold: float,
) -> _Labels:
    from xpkg.core.annotations import Instance, LabeledFrame
    from xpkg.model import Labels

    skeleton = build_keypoint_skeleton(list(keypoints), name=skeleton_name)
    labels = Labels(skeletons=[skeleton], videos=list(videos))

    for frame_idx, row in df.iterrows():
        points = _row_points(row, keypoints, likelihood_threshold=likelihood_threshold)
        if not points:
            continue
        resolved_frame_idx = _coerce_frame_idx(frame_idx)
        instance_points = _instance_points(points)
        for video in videos:
            labels.append(
                LabeledFrame(
                    video=video,
                    frame_idx=resolved_frame_idx,
                    instances=[Instance(skeleton=skeleton, init_points=instance_points)],
                )
            )

    labels.update_cache()
    return labels


def _read_tracking_inputs(
    data_path: Path,
    *,
    read_tracking: DlcReader,
    read_step_label: str,
    progress_callback: ProgressCallback | None,
) -> tuple[pd.DataFrame, list[str]]:
    _emit(progress_callback, f"IMPORT: Reading {read_step_label} {data_path.name}")
    df, keypoints = read_tracking(data_path)
    _emit(progress_callback, f"IMPORT: Found {len(keypoints)} keypoints, {len(df)} frames")
    return df, keypoints


def _build_tracking_labels(
    df: pd.DataFrame,
    keypoints: list[str],
    *,
    skeleton_name: str,
    video_path: Path,
    stored_video_filename: str | None = None,
    likelihood_threshold: float,
    progress_callback: ProgressCallback | None,
) -> _Labels:
    _emit(progress_callback, "IMPORT: Building skeleton")
    skeleton = build_keypoint_skeleton(keypoints, name=skeleton_name)

    _emit(progress_callback, "IMPORT: Converting to labels")
    labels = _labels_from_tracking_df(
        df,
        keypoints,
        skeleton,
        video_path,
        stored_video_filename=stored_video_filename,
        likelihood_threshold=likelihood_threshold,
    )
    labels.validate()
    return labels


def _write_tracking_bundle(
    labels: _Labels,
    *,
    out_path: Path,
    data_path: Path,
    video_path: Path,
    source_label: str,
    source_metadata_key: str,
    progress_callback: ProgressCallback | None,
) -> ConversionResult:
    ensure_dir(out_path.parent)
    metadata = {
        "source": source_label,
        source_metadata_key: data_path.as_posix(),
        "source_video": video_path.as_posix(),
    }

    _emit(progress_callback, f"IMPORT: Writing {out_path.name}")
    write_siesta(out_path, labels, metadata=metadata)
    _emit(progress_callback, "IMPORT: Done")

    return ConversionResult(
        source_dir=data_path.parent,
        project_root=out_path.parent,
        videos=[video_path],
        bundle_path=out_path,
    )


def _convert_dlc_tracking(
    data_path: Path | str,
    video_path: Path | str,
    out_path: Path | str,
    *,
    read_tracking: DlcReader,
    source_label: str,
    source_metadata_key: str,
    read_step_label: str,
    skeleton_name: str,
    likelihood_threshold: float,
    progress_callback: ProgressCallback | None,
) -> ConversionResult:
    """Run the single-file DLC data-file to archive conversion pipeline."""

    resolved_data_path = _resolve_tracking_path(data_path)
    resolved_video_path = _resolve_video_path(video_path)
    resolved_out_path = resolve_path(out_path)

    df, keypoints = _read_tracking_inputs(
        resolved_data_path,
        read_tracking=read_tracking,
        read_step_label=read_step_label,
        progress_callback=progress_callback,
    )
    labels = _build_tracking_labels(
        df,
        keypoints,
        skeleton_name=skeleton_name,
        video_path=resolved_video_path,
        likelihood_threshold=likelihood_threshold,
        progress_callback=progress_callback,
    )
    return _write_tracking_bundle(
        labels,
        out_path=resolved_out_path,
        data_path=resolved_data_path,
        video_path=resolved_video_path,
        source_label=source_label,
        source_metadata_key=source_metadata_key,
        progress_callback=progress_callback,
    )


def convert_dlc_csv(
    csv_path: Path | str,
    video_path: Path | str,
    out_path: Path | str,
    *,
    skeleton_name: str = "imported",
    likelihood_threshold: float = 0.0,
    progress_callback: ProgressCallback | None = None,
) -> ConversionResult:
    """Convert a DLC CSV file and matching video into a native archive."""

    return _convert_dlc_tracking(
        csv_path,
        video_path,
        out_path,
        read_tracking=_read_dlc_csv,
        source_label="dlc_csv_import",
        source_metadata_key="source_csv",
        read_step_label="CSV",
        skeleton_name=skeleton_name,
        likelihood_threshold=likelihood_threshold,
        progress_callback=progress_callback,
    )


def convert_dlc_h5(
    h5_path: Path | str,
    video_path: Path | str,
    out_path: Path | str,
    *,
    skeleton_name: str = "imported",
    likelihood_threshold: float = 0.0,
    progress_callback: ProgressCallback | None = None,
) -> ConversionResult:
    """Convert a DLC H5 tracking file and matching video into a native archive."""

    return _convert_dlc_tracking(
        h5_path,
        video_path,
        out_path,
        read_tracking=_read_dlc_h5,
        source_label="dlc_h5_import",
        source_metadata_key="source_h5",
        read_step_label="H5",
        skeleton_name=skeleton_name,
        likelihood_threshold=likelihood_threshold,
        progress_callback=progress_callback,
    )


def convert_dlc_h5_project(
    h5_path: Path | str,
    video_paths: Sequence[Path | str],
    project_root: Path | str,
    *,
    likelihood_threshold: float = 0.0,
    progress_callback: ProgressCallback | None = None,
    bundle_extension: str = CANONICAL_BUNDLE_SUFFIX,
) -> ConversionResult:
    """Convert one DLC H5 tracking file plus explicit videos into a project archive."""

    resolved_h5 = resolve_path(h5_path)
    resolved_project_root = ensure_dir(project_root)
    resolved_video_paths = _resolve_video_paths(video_paths)
    videos = _load_project_videos(resolved_video_paths, project_root=resolved_project_root)
    bundle_path = project_bundle_path(
        resolved_project_root,
        bundle_extension=bundle_extension,
    )
    try:
        _emit(progress_callback, f"{_DLC_READ_H5_MARKER} {resolved_h5.name}")
        df, keypoints = _read_dlc_h5(resolved_h5)
        _emit(progress_callback, _DLC_VALIDATE_VIDEOS_MARKER)
        _validate_video_alignment(videos, required_frames=_required_frame_count(df.index))
        _emit(progress_callback, _DLC_BUILD_LABELS_MARKER)
        labels = _labels_from_tracking_df_project(
            df,
            keypoints,
            videos=videos,
            skeleton_name=resolved_project_root.name or "dlc",
            likelihood_threshold=likelihood_threshold,
        )
        labels.validate()
        metadata = {
            "project_name": resolved_project_root.name,
            "source": "dlc_h5_import",
            "source_h5": _stored_project_path(resolved_h5, project_root=resolved_project_root),
            "source_videos": [
                _stored_project_path(video_path, project_root=resolved_project_root)
                for video_path in resolved_video_paths
            ],
        }
        _emit(progress_callback, f"{_DLC_WRITE_BUNDLE_MARKER} {bundle_path.name}")
        write_siesta(bundle_path, labels, metadata=metadata)
    finally:
        for video in videos:
            video.close()
    _emit(progress_callback, _DLC_DONE_MARKER)
    return ConversionResult(
        source_dir=resolved_h5.parent,
        project_root=resolved_project_root,
        videos=list(resolved_video_paths),
        bundle_path=bundle_path,
    )


def convert_dlc_project(
    project_dir: Path | str,
    out_dir: Path | str,
    *,
    likelihood_threshold: float = 0.0,
    progress_callback: ProgressCallback | None = None,
) -> list[ConversionResult]:
    """Convert an entire DLC project directory into native `.xpkg` bundles."""

    project_dir = resolve_path(project_dir)
    out_dir = resolve_path(out_dir)
    ensure_dir(out_dir)
    results: list[ConversionResult] = []

    labeled_data = project_dir / "labeled-data"
    videos_dir = project_dir / "videos"

    if not labeled_data.exists():
        raise FileNotFoundError(f"No labeled-data directory in {project_dir}")

    for subdir in sorted(labeled_data.iterdir()):
        if not subdir.is_dir():
            continue

        csv_files = list(subdir.glob("CollectedData*.csv"))
        h5_files = list(subdir.glob("CollectedData*.h5"))

        data_file = None
        is_h5 = False
        if csv_files:
            data_file = csv_files[0]
        elif h5_files:
            data_file = h5_files[0]
            is_h5 = True

        if data_file is None:
            _emit(progress_callback, f"IMPORT: Skipping {subdir.name} (no data file)")
            continue

        video_file = None
        for extension in (".mp4", ".avi", ".mov", ".mkv"):
            candidate = videos_dir / f"{subdir.name}{extension}"
            if candidate.exists():
                video_file = candidate
                break

        if video_file is None:
            _emit(progress_callback, f"IMPORT: Skipping {subdir.name} (no video found)")
            continue

        out_path = out_dir / f"{subdir.name}{CANONICAL_BUNDLE_SUFFIX}"
        _emit(progress_callback, f"IMPORT: Converting {subdir.name}")

        if is_h5:
            result = convert_dlc_h5(
                data_file,
                video_file,
                out_path,
                skeleton_name=project_dir.name,
                likelihood_threshold=likelihood_threshold,
                progress_callback=progress_callback,
            )
        else:
            result = convert_dlc_csv(
                data_file,
                video_file,
                out_path,
                skeleton_name=project_dir.name,
                likelihood_threshold=likelihood_threshold,
                progress_callback=progress_callback,
            )

        results.append(result)

    return results


__all__ = [
    "DLC_H5_PROJECT_PROGRESS_MARKERS",
    "convert_dlc_csv",
    "convert_dlc_h5",
    "convert_dlc_h5_project",
    "convert_dlc_project",
]
