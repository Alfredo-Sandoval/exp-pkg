"""Convert DeepLabCut-style tracking data into native bundles."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd

from posetta.core.path_registry import ensure_dir, resolve_path
from posetta.core.skeleton import build_keypoint_skeleton
from posetta.io.converters.converter_helpers import (
    ConversionResult,
    ProgressCallback,
    _emit,
    project_bundle_path,
)
from posetta.io.siesta_format import write_siesta
from posetta.io.video import Video

if TYPE_CHECKING:
    from posetta.core.skeleton import Keypoint
    from posetta.core.skeleton import Skeleton as _Skeleton
    from posetta.model import Labels as _Labels

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


def _extract_keypoints_from_columns(columns: pd.Index) -> list[str]:
    """Extract keypoint names from flattened DLC-style coordinate columns."""

    keypoints: list[str] = []
    seen: set[str] = set()
    for col in columns:
        col_name = str(col)
        if not col_name.endswith("_x"):
            continue
        keypoint = col_name[:-2]
        if keypoint in seen:
            continue
        keypoints.append(keypoint)
        seen.add(keypoint)
    return keypoints


def _read_dlc_csv(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read DLC-style multi-index CSV and return flat DataFrame + keypoint names."""

    header_df = pd.read_csv(csv_path, header=None, nrows=4)

    n_header = 3
    if header_df.iloc[0, 0] in ("scorer", "model"):
        n_header = 3
    elif header_df.iloc[0, 0] in ("bodyparts", "keypoint"):
        n_header = 2
    else:
        for i in range(min(4, len(header_df))):
            row_vals = header_df.iloc[i].astype(str).str.lower()
            if any("likelihood" in value or row_vals.tolist().count("x") > 1 for value in row_vals):
                n_header = i + 1
                break

    df = pd.read_csv(csv_path, header=list(range(n_header)), index_col=0)

    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            if len(col) == 3:
                new_cols.append(f"{col[1]}_{col[2]}")
            elif len(col) == 2:
                new_cols.append(f"{col[0]}_{col[1]}")
            else:
                new_cols.append("_".join(str(value) for value in col))
        df.columns = new_cols

    return df, _extract_keypoints_from_columns(df.columns)


def _read_dlc_h5(h5_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read DLC-style H5 tracking file and return flat DataFrame + keypoint names."""
    df = cast(pd.DataFrame, pd.read_hdf(h5_path))

    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            if len(col) >= 2:
                new_cols.append(f"{col[-2]}_{col[-1]}")
            else:
                new_cols.append(str(col))
        df.columns = new_cols

    return df, _extract_keypoints_from_columns(df.columns)


def _labels_from_tracking_df(
    df: pd.DataFrame,
    keypoints: list[str],
    skeleton: _Skeleton,
    video_path: Path,
    *,
    stored_video_filename: str | None = None,
    likelihood_threshold: float = 0.0,
) -> _Labels:
    """Convert tracking DataFrame to the canonical `posetta.model.Labels` object."""
    from posetta.core.annotations import Instance, LabeledFrame, Point
    from posetta.model import Labels

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
                if pd.isna(likelihood) or likelihood < likelihood_threshold:
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
        siesta_path=out_path,
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
    """Run the canonical DLC data-file -> native bundle conversion pipeline."""

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


def _select_explicit_video_path(
    video_paths: Path | str | Sequence[Path | str],
) -> Path | str:
    if isinstance(video_paths, (str, Path)):
        return video_paths
    normalized_video_paths = list(video_paths)
    if len(normalized_video_paths) != 1:
        raise ValueError(
            "convert_dlc_h5_project expects exactly one explicit video path for one H5 file"
    )
    return normalized_video_paths[0]


def _relative_video_filename(video_path: Path, project_root: Path) -> str | None:
    try:
        return video_path.relative_to(project_root).as_posix()
    except ValueError:
        return None


def convert_dlc_csv(
    csv_path: Path | str,
    video_path: Path | str,
    out_path: Path | str,
    *,
    skeleton_name: str = "imported",
    likelihood_threshold: float = 0.0,
    progress_callback: ProgressCallback | None = None,
) -> ConversionResult:
    """Convert a DLC CSV file + video into a native bundle."""
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
    """Convert a DLC-style H5 tracking file + video into a native bundle."""
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
    video_paths: Path | str | Sequence[Path | str],
    project_root: Path | str,
    *,
    likelihood_threshold: float = 0.0,
    progress_callback: ProgressCallback | None = None,
    bundle_extension: str = ".sta",
) -> ConversionResult:
    """Convert one DLC H5 tracking file plus one explicit video into a project bundle."""

    resolved_project_root = resolve_path(project_root)
    ensure_dir(resolved_project_root)
    out_path = project_bundle_path(resolved_project_root, bundle_extension=bundle_extension)

    resolved_data_path = _resolve_tracking_path(h5_path)
    explicit_video_path = _select_explicit_video_path(video_paths)
    _emit(progress_callback, _DLC_READ_H5_MARKER)
    df, keypoints = _read_tracking_inputs(
        resolved_data_path,
        read_tracking=_read_dlc_h5,
        read_step_label="H5",
        progress_callback=progress_callback,
    )

    _emit(progress_callback, _DLC_VALIDATE_VIDEOS_MARKER)
    resolved_video_path = _resolve_video_path(explicit_video_path)

    _emit(progress_callback, _DLC_BUILD_LABELS_MARKER)
    labels = _build_tracking_labels(
        df,
        keypoints,
        skeleton_name=resolved_project_root.name,
        video_path=resolved_video_path,
        stored_video_filename=_relative_video_filename(resolved_video_path, resolved_project_root),
        likelihood_threshold=likelihood_threshold,
        progress_callback=progress_callback,
    )

    _emit(progress_callback, _DLC_WRITE_BUNDLE_MARKER)
    result = _write_tracking_bundle(
        labels,
        out_path=out_path,
        data_path=resolved_data_path,
        video_path=resolved_video_path,
        source_label="dlc_h5_import",
        source_metadata_key="source_h5",
        progress_callback=progress_callback,
    )
    _emit(progress_callback, _DLC_DONE_MARKER)
    return result


def convert_dlc_project(
    project_dir: Path | str,
    out_dir: Path | str,
    *,
    likelihood_threshold: float = 0.0,
    progress_callback: ProgressCallback | None = None,
) -> list[ConversionResult]:
    """Convert an entire DLC project directory to native bundle format."""
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

        out_path = out_dir / f"{subdir.name}.sta"
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
