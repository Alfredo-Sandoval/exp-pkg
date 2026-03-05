"""Convert DeepLabCut-style tracking data into `.siesta` projects.

Supports:
- DLC CSV format (multi-index headers: scorer, bodyparts, coords)
- DLC-style H5 tracking files
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd

from posetta.core.logging_utils import get_logger
from posetta.core.path_registry import ensure_dir, resolve_path
from posetta.core.skeleton import build_keypoint_skeleton
from posetta.io.converters.converter_helpers import (
    ConversionResult,
    ProgressCallback,
    _emit,
)
from posetta.io.siesta_format import write_siesta
from posetta.io.video import Video

_LOGGER = get_logger(__name__)

if TYPE_CHECKING:
    from posetta.core.skeleton import Keypoint
    from posetta.core.skeleton import Skeleton as _Skeleton
    from posetta.io.labels import Labels as _Labels

DlcReader = Callable[[Path], tuple[pd.DataFrame, list[str]]]


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
    """Read DLC-style multi-index CSV and return flat DataFrame + keypoint names.

    DLC CSV structure:
        Row 0: scorer (ignored)
        Row 1: bodyparts (keypoint names, repeated for x/y/likelihood)
        Row 2: coords (x, y, likelihood)
        Row 3+: data

    Returns:
        df: DataFrame with columns like 'nose_x', 'nose_y', 'nose_likelihood', ...
        keypoints: List of unique keypoint names in order
    """

    header_df = pd.read_csv(csv_path, header=None, nrows=4)

    n_header = 3
    if header_df.iloc[0, 0] in ("scorer", "model"):
        n_header = 3
    elif header_df.iloc[0, 0] in ("bodyparts", "keypoint"):
        n_header = 2
    else:
        for i in range(min(4, len(header_df))):
            row_vals = header_df.iloc[i].astype(str).str.lower()
            if any("likelihood" in v or row_vals.tolist().count("x") > 1 for v in row_vals):
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
                new_cols.append("_".join(str(c) for c in col))
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
    likelihood_threshold: float = 0.0,
) -> _Labels:
    """Convert tracking DataFrame to Labels object.

    Args:
        df: DataFrame with columns like 'nose_x', 'nose_y', 'nose_likelihood', ...
             Index should be frame indices (0-based)
        keypoints: List of keypoint names
        skeleton: Skeleton object
        video_path: Path to the video file
        likelihood_threshold: Minimum likelihood to include a point (0 = include all)
    """
    from posetta.core.annotations import Instance, LabeledFrame, Point
    from posetta.io.labels import Labels

    video = Video.from_filename(str(video_path))
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
                lh = row[lh_col]
                if pd.isna(lh) or lh < likelihood_threshold:
                    continue

            if pd.isna(x_val) or pd.isna(y_val):
                continue

            points[kp] = Point(float(x_val), float(y_val), visible=True, complete=True)

        if not points:
            continue

        instance = Instance(skeleton=skeleton, init_points=points)
        lf = LabeledFrame(video=video, frame_idx=int(frame_idx), instances=[instance])
        labels.append(lf)

    labels.update_cache()
    return labels


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
    """Run the canonical DLC data-file -> .siesta conversion pipeline."""

    resolved_data_path = resolve_path(data_path)
    resolved_video_path = resolve_path(video_path)
    resolved_out_path = resolve_path(out_path)

    _emit(progress_callback, f"IMPORT: Reading {read_step_label} {resolved_data_path.name}")
    df, keypoints = read_tracking(resolved_data_path)
    _emit(progress_callback, f"IMPORT: Found {len(keypoints)} keypoints, {len(df)} frames")

    _emit(progress_callback, "IMPORT: Building skeleton")
    skeleton = build_keypoint_skeleton(keypoints, name=skeleton_name)

    _emit(progress_callback, "IMPORT: Converting to labels")
    labels = _labels_from_tracking_df(
        df,
        keypoints,
        skeleton,
        resolved_video_path,
        likelihood_threshold,
    )
    labels.validate()

    ensure_dir(resolved_out_path.parent)
    metadata = {
        "source": source_label,
        source_metadata_key: resolved_data_path.as_posix(),
        "source_video": resolved_video_path.as_posix(),
    }

    _emit(progress_callback, f"IMPORT: Writing {resolved_out_path.name}")
    write_siesta(resolved_out_path, labels, metadata=metadata)
    _emit(progress_callback, "IMPORT: Done")

    return ConversionResult(
        source_dir=resolved_data_path.parent,
        project_root=resolved_out_path.parent,
        videos=[resolved_video_path],
        siesta_path=resolved_out_path,
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
    """Convert a DLC CSV file + video into a .siesta project.

    Args:
        csv_path: Path to the DLC CSV file
        video_path: Path to the corresponding video file
        out_path: Output .siesta file path
        skeleton_name: Name for the skeleton
        likelihood_threshold: Minimum likelihood to include points (0-1)
        progress_callback: Optional progress callback

    Returns:
        ConversionResult with paths to created files
    """
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
    """Convert a DLC-style H5 tracking file + video into a .siesta project."""
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


def convert_dlc_project(
    project_dir: Path | str,
    out_dir: Path | str,
    *,
    likelihood_threshold: float = 0.0,
    progress_callback: ProgressCallback | None = None,
) -> list[ConversionResult]:
    """Convert an entire DLC project directory to .siesta format.

    Expects DLC project structure:
        project/
            videos/
                video1.mp4
                video2.avi
            labeled-data/
                video1/
                    CollectedData_*.csv or CollectedData_*.h5
                video2/
                    ...

    Returns list of ConversionResult for each video converted.
    """
    project_dir = resolve_path(project_dir)
    out_dir = resolve_path(out_dir)
    ensure_dir(out_dir)
    results: list[ConversionResult] = []

    labeled_data = project_dir / "labeled-data"
    videos_dir = project_dir / "videos"

    if not labeled_data.exists():
        raise FileNotFoundError(f"No labeled-data directory in {project_dir}")

    for sub in sorted(labeled_data.iterdir()):
        if not sub.is_dir():
            continue

        csv_files = list(sub.glob("CollectedData*.csv"))
        h5_files = list(sub.glob("CollectedData*.h5"))

        data_file = None
        is_h5 = False
        if csv_files:
            data_file = csv_files[0]
        elif h5_files:
            data_file = h5_files[0]
            is_h5 = True

        if data_file is None:
            _emit(progress_callback, f"IMPORT: Skipping {sub.name} (no data file)")
            continue

        video_file = None
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            candidate = videos_dir / f"{sub.name}{ext}"
            if candidate.exists():
                video_file = candidate
                break

        if video_file is None:
            _emit(progress_callback, f"IMPORT: Skipping {sub.name} (no video found)")
            continue

        out_path = out_dir / f"{sub.name}.siesta"
        _emit(progress_callback, f"IMPORT: Converting {sub.name}")

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
