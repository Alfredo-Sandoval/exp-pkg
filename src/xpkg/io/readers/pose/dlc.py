"""Low-level readers for single-animal DeepLabCut tracking and labeled data."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from xpkg.io.readers._normalization import normalize_file_type as _normalize_file_type
from xpkg.io.readers.pose._common import (
    PoseTrack,
    build_pose_track,
    resolve_node_indices_from_names,
)

_DLC_H5_FILE_TYPES = {"h5", "hdf5"}


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


def _flatten_dlc_columns(columns: pd.Index) -> list[str]:
    """Flatten DLC column MultiIndex entries into keypoint coordinate names."""

    flat_columns: list[str] = []
    for col in columns:
        if not isinstance(col, tuple):
            flat_columns.append(str(col))
            continue
        if len(col) == 3:
            flat_columns.append(f"{col[1]}_{col[2]}")
            continue
        if len(col) == 2:
            flat_columns.append(f"{col[0]}_{col[1]}")
            continue
        if len(col) >= 2:
            flat_columns.append(f"{col[-2]}_{col[-1]}")
            continue
        flat_columns.append(str(col))
    return flat_columns


def read_dlc_csv_table(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read a DLC-style multi-index CSV into a flat coordinate table."""

    header_df = pd.read_csv(csv_path, header=None, nrows=4)
    first_header = str(header_df.iloc[0, 0]).lower()

    n_header = 3
    if first_header in ("scorer", "model"):
        n_header = 3
    elif first_header in ("bodyparts", "keypoint"):
        n_header = 2
    else:
        for i in range(min(4, len(header_df))):
            row_values = [str(value).lower() for value in header_df.iloc[i].tolist()]
            if any("likelihood" in value for value in row_values) or row_values.count("x") > 1:
                n_header = i + 1
                break

    df = pd.read_csv(csv_path, header=list(range(n_header)), index_col=0)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = _flatten_dlc_columns(df.columns)
    return df, _extract_keypoints_from_columns(df.columns)


def _reject_pickled_hdf_nodes(h5_path: Path) -> None:
    """Reject PyTables object nodes before pandas can deserialize them."""

    import tables

    with tables.open_file(h5_path.as_posix(), mode="r") as handle:
        for node in handle.walk_nodes("/"):
            atom = getattr(node, "atom", None)
            atom_kind = str(getattr(atom, "kind", "")).lower()
            atom_type = str(getattr(atom, "type", "")).lower()
            attrs = node._v_attrs
            pseudoatom = str(getattr(attrs, "PSEUDOATOM", "")).lower()
            node_class = str(getattr(attrs, "CLASS", "")).lower()
            if (
                atom_kind == "object"
                or atom_type == "object"
                or pseudoatom == "object"
                or node_class == "vlarray"
            ):
                raise ValueError(
                    "DLC H5 file contains object/pickled PyTables data and is not safe "
                    f"to read as an untrusted input: {h5_path}"
                )


def read_dlc_h5_table(h5_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read a DLC-style H5 file into a flat coordinate table."""

    _reject_pickled_hdf_nodes(h5_path)
    df = cast(pd.DataFrame, pd.read_hdf(h5_path))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = _flatten_dlc_columns(df.columns)
    return df, _extract_keypoints_from_columns(df.columns)


def _read_dlc_table(path: Path, *, file_type: str) -> tuple[pd.DataFrame, list[str]]:
    normalized_type = _normalize_file_type(file_type)
    if normalized_type == "csv":
        return read_dlc_csv_table(path)
    if normalized_type in _DLC_H5_FILE_TYPES:
        return read_dlc_h5_table(path)
    raise ValueError(
        f"Unsupported DLC file_type {file_type!r}. Expected one of ['csv', 'h5', 'hdf5']."
    )


def _validate_dlc_columns(
    df: pd.DataFrame,
    keypoints: Sequence[str],
    *,
    path: Path,
) -> bool:
    if not keypoints:
        raise ValueError(f"No DLC keypoints found in {path}.")
    if not df.columns.is_unique:
        raise ValueError(
            "DLC low-level readers only support single-animal inputs with unique bodypart "
            f"columns; repeated flattened columns found in {path}."
        )

    missing_coordinates: list[str] = []
    for keypoint in keypoints:
        for suffix in ("x", "y"):
            column_name = f"{keypoint}_{suffix}"
            if column_name not in df.columns:
                missing_coordinates.append(column_name)
    if missing_coordinates:
        raise ValueError(
            f"DLC file {path} is missing required coordinate columns: {sorted(missing_coordinates)}"
        )

    likelihood_columns = [f"{keypoint}_likelihood" for keypoint in keypoints]
    present_likelihood = [column in df.columns for column in likelihood_columns]
    if all(present_likelihood):
        return True
    if not any(present_likelihood):
        return False
    missing_likelihood = [
        column
        for column, present in zip(likelihood_columns, present_likelihood, strict=True)
        if not present
    ]
    raise ValueError(
        f"DLC file {path} has partial likelihood data; missing columns: {missing_likelihood}"
    )


def read_node_names(path: Path, *, file_type: str) -> list[str]:
    """Return decoded node names from a DLC CSV or H5 file."""

    _df, keypoints = _read_dlc_table(Path(path), file_type=file_type)
    return keypoints


def read_track(path: Path, *, file_type: str, track_index: int) -> PoseTrack:
    """Read one DLC track or labeled-data table as a PoseTrack."""

    idx = int(track_index)
    if idx != 0:
        raise ValueError(
            "DLC low-level readers currently support only single-animal exports; "
            f"track_index must be 0, got {track_index!r}."
        )

    resolved_path = Path(path)
    normalized_type = _normalize_file_type(file_type)
    df, keypoints = _read_dlc_table(resolved_path, file_type=normalized_type)
    has_likelihood = _validate_dlc_columns(df, keypoints, path=resolved_path)

    coords = np.empty((len(df), len(keypoints), 2), dtype=np.float64)
    coords.fill(np.nan)
    scores = np.empty((len(df), len(keypoints)), dtype=np.float64)
    scores.fill(np.nan)

    for node_idx, keypoint in enumerate(keypoints):
        x_values = df[f"{keypoint}_x"].to_numpy(dtype=np.float64, copy=False)
        y_values = df[f"{keypoint}_y"].to_numpy(dtype=np.float64, copy=False)
        coords[:, node_idx, 0] = x_values
        coords[:, node_idx, 1] = y_values
        if has_likelihood:
            scores[:, node_idx] = df[f"{keypoint}_likelihood"].to_numpy(
                dtype=np.float64,
                copy=False,
            )
        else:
            scores[:, node_idx] = np.where(
                np.isfinite(x_values) & np.isfinite(y_values),
                1.0,
                np.nan,
            )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        instance_score = np.nanmean(scores, axis=1)

    return build_pose_track(
        coords=coords,
        scores=scores,
        node_names=keypoints,
        instance_score=instance_score,
        source_label=f"DLC file {resolved_path}",
        metadata={
            "source": {
                "type": "dlc_h5" if normalized_type in _DLC_H5_FILE_TYPES else "dlc_csv",
                "path": str(resolved_path),
            },
            "software": "DLC",
            "file_type": normalized_type,
            "track_index": idx,
            "confidence_source": "likelihood" if has_likelihood else "labeled_data",
        },
    )


def resolve_node_indices(
    path: Path,
    *,
    file_type: str,
    target_names: Sequence[str],
) -> list[int]:
    """Map target node names to their indices in a DLC CSV or H5 file."""

    return resolve_node_indices_from_names(
        read_node_names(path, file_type=file_type),
        target_names,
    )


__all__ = [
    "PoseTrack",
    "read_dlc_csv_table",
    "read_dlc_h5_table",
    "read_node_names",
    "read_track",
    "resolve_node_indices",
]
