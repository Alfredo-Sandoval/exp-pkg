"""Shared HDF5 reader core for `.xpkg` / legacy archive payloads.

This lives outside either product reader because both repos share the same
on-disk schema, but they finalize product-specific metadata differently.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from xpkg.core.json_utils import parse_json
from xpkg.core.path_registry import make_path_id
from xpkg.io.manifest import (
    AssetType,
    coerce_manifest,
    resolve_asset_path,
    resolve_project_path,
)
from xpkg.io.archive_format.shared import (
    _DEFAULT_PROVENANCE_MAX_BYTES,
    _PROVENANCE_SCHEMA_VERSION,
    LABEL_TRACK_ID_DATASET,
    LABEL_VISIBILITY_DATASET,
    ARCHIVE_SCHEMA_NAME,
    _coerce_int,
    _mapping_to_str_key_dict,
    _normalize_predictions_committed_length,
)

__all__ = [
    "LazyDatasetHandle",
    "LazyArchiveHandle",
    "ReaderCommonState",
    "_decode_optional_mapping_attr",
    "_decode_utf8_field",
    "_looks_like_iso_timestamp",
    "_normalize_attr_value",
    "_read_str_dataset",
    "build_common_reader_state",
    "read_archive_with_assembler",
]


_ISO_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$"
)


@dataclass(slots=True)
class ReaderCommonState:
    """Format-level reader result reused by xpkg and Siesta wrappers."""

    result: dict[str, Any]
    metadata: dict[str, Any]
    preferences_override: dict[str, Any]


class LazyDatasetHandle:
    """Lightweight wrapper around an h5py.Dataset for lazy materialization."""

    def __init__(self, dataset: h5py.Dataset, dtype: Any, length: int | None = None) -> None:
        self.dataset = dataset
        self.dtype = dtype
        self.length = length

    def materialize(self) -> np.ndarray:
        if not self.dataset.id.valid:
            raise RuntimeError(
                "Cannot materialize lazy dataset after the owning .sta handle is closed"
            )
        data = self.dataset[...]
        if self.length is not None:
            data = data[: self.length]
        return np.asarray(data, dtype=self.dtype)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        arr = self.materialize()
        if dtype is None:
            out = arr
        else:
            out = np.asarray(arr, dtype=dtype)
        if copy:
            return np.array(out, copy=True)
        return out

    @property
    def shape(self) -> tuple[int, ...]:
        if not self.dataset.id.valid:
            raise RuntimeError(
                "Cannot read lazy dataset shape after the owning .sta handle is closed"
            )
        base = tuple(self.dataset.shape)
        if not base or self.length is None:
            return base
        return (min(self.length, base[0]), *base[1:])


class LazyArchiveHandle:
    """Owns an open h5py.File returned by read_archive(lazy=True)."""

    def __init__(self, file_handle: h5py.File) -> None:
        self._file_handle = file_handle

    @property
    def closed(self) -> bool:
        return not self._file_handle.id.valid

    @property
    def file(self) -> h5py.File:
        if self.closed:
            raise RuntimeError("LazyArchiveHandle is closed")
        return self._file_handle

    def close(self) -> None:
        if self._file_handle.id.valid:
            self._file_handle.close()

    def __enter__(self) -> LazyArchiveHandle:
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        self.close()


def _decode_utf8_field(
    raw: bytes | bytearray | np.bytes_,
    *,
    field: str,
) -> str:
    """Decode a UTF-8 byte field with structured error context."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{field} is not valid UTF-8") from exc


def _decode_project_metadata_utf8(raw: bytes | bytearray | np.bytes_, *, field: str) -> str:
    """Decode UTF-8 metadata bytes with explicit error context."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Project metadata attribute {field} is not valid UTF-8") from exc


def _decode_optional_mapping_attr(
    raw_value: Any,
    *,
    field: str,
    type_message: str,
) -> dict[str, Any] | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, Mapping):
        return _mapping_to_str_key_dict(raw_value, name=field)
    if isinstance(raw_value, bytes | bytearray | np.bytes_):
        raw_value = _decode_project_metadata_utf8(raw_value, field=field)
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return None
        parsed = parse_json(stripped)
        if not isinstance(parsed, Mapping):
            raise TypeError(f"{field} must decode to a mapping")
        return _mapping_to_str_key_dict(parsed, name=field)
    raise TypeError(type_message)


def _read_str_dataset(ds):
    """Read a string dataset robustly (vlen UTF-8 or bytes)."""
    if isinstance(ds, h5py.Dataset):
        is_vlen_str = h5py.check_dtype(vlen=ds.dtype) is str
        is_fixed_str = np.dtype(ds.dtype).kind in ("S", "U")
        data = ds.asstr()[...] if (is_vlen_str or is_fixed_str) else ds[...]
    else:
        data = np.asarray(ds)

    field_name = ds.name if isinstance(ds, h5py.Dataset) else "byte dataset"

    if np.ndim(data) >= 2 and int(np.shape(data)[-1]) == 2:
        rows = []
        for row in data:
            first, second = row[0], row[1]
            first_value = (
                _decode_utf8_field(first, field=f"{field_name}[row,0]")
                if isinstance(first, bytes | bytearray | np.bytes_)
                else str(first)
            )
            second_value = (
                _decode_utf8_field(second, field=f"{field_name}[row,1]")
                if isinstance(second, bytes | bytearray | np.bytes_)
                else str(second)
            )
            rows.append([first_value, second_value])
        return rows

    flat = np.ravel(data)
    values: list[str] = []
    for idx, item in enumerate(flat):
        if isinstance(item, bytes | bytearray | np.bytes_):
            values.append(_decode_utf8_field(item, field=f"{field_name}[{idx}]"))
            continue
        values.append(str(item))
    return values


def _normalize_iso_timestamp(text: str) -> str:
    candidate = text.strip()
    if not candidate:
        return ""
    adjusted = candidate[:-1] + "+00:00" if candidate.endswith("Z") else candidate
    dt = datetime.fromisoformat(adjusted)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt.isoformat().replace("+00:00", "Z")


def _looks_like_iso_timestamp(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return bool(_ISO_TIMESTAMP_RE.match(stripped))


def _normalize_attr_value(value: Any) -> Any:
    current = value
    if isinstance(current, np.ndarray) and current.size == 1:
        current = current.item()
    if isinstance(current, bytes | bytearray | np.bytes_):
        try:
            decoded = current.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("project_metadata attribute value is not valid UTF-8") from exc
        stripped = decoded.strip()
        if stripped and _looks_like_iso_timestamp(stripped):
            return _normalize_iso_timestamp(stripped)
        return decoded
    if isinstance(current, np.ndarray):
        return current.tolist()
    if isinstance(current, np.bool_):
        return bool(current)
    if isinstance(current, bool):
        return current
    if isinstance(current, np.integer):
        return int(current)
    if isinstance(current, int):
        return current
    if isinstance(current, np.floating):
        return float(current)
    if isinstance(current, float):
        return current
    if isinstance(current, str):
        stripped = current.strip()
        if not stripped:
            return ""
        if _looks_like_iso_timestamp(stripped):
            return _normalize_iso_timestamp(stripped)
        return stripped
    return current


def _read_dataset_slice(
    dataset: h5py.Dataset,
    dtype: Any,
    *,
    length: int | None = None,
    lazy: bool = False,
) -> LazyDatasetHandle | np.ndarray:
    if lazy:
        return LazyDatasetHandle(dataset=dataset, dtype=dtype, length=length)
    data = dataset[...] if length is None else dataset[:length]
    return np.asarray(data, dtype=dtype)


def _empty_labels_payload() -> dict[str, Any]:
    return {
        "frames": {
            "video_index": np.zeros((0,), dtype=np.int32),
            "frame_index": np.zeros((0,), dtype=np.int32),
            "num_instances": np.zeros((0,), dtype=np.int32),
        },
        "data": {
            "keypoints": np.zeros((0, 0, 0, 3), dtype=np.float32),
            "flags": np.zeros((0, 0, 0), dtype=np.uint8),
            LABEL_TRACK_ID_DATASET: np.zeros((0, 0), dtype=np.int32),
            LABEL_VISIBILITY_DATASET: np.zeros((0, 0, 0), dtype=np.uint8),
        },
        "metadata": {
            "num_frames": 0,
            "max_instances": 0,
            "num_keypoints": 0,
        },
    }


def _empty_predictions_payload() -> dict[str, Any]:
    return {
        "frames": {
            "video_index": np.zeros((0,), dtype=np.int32),
            "frame_index": np.zeros((0,), dtype=np.int32),
            "num_instances": np.zeros((0,), dtype=np.int32),
        },
        "data": {
            "keypoints": np.zeros((0, 0, 0, 3), dtype=np.float32),
            "keypoint_score": np.zeros((0, 0, 0), dtype=np.float32),
            "instance_score": np.zeros((0, 0), dtype=np.float32),
            "track_id": np.zeros((0, 0), dtype=np.int32),
            "deleted": np.zeros((0, 0), dtype=np.uint8),
            "heatmaps": None,
        },
        "attrs": {"committed_length": 0},
        "metadata": {
            "num_frames": 0,
            "max_instances": 0,
            "num_keypoints": 0,
            "heatmap_height": 0,
            "heatmap_width": 0,
        },
    }


def _read_videos_group(root: h5py.File) -> dict[str, Any]:
    group = root.get("videos")
    if not isinstance(group, h5py.Group):
        return {
            "base_dir": "",
            "filenames": [],
            "backends": [],
            "sha256": [],
            "shapes": np.zeros((0, 4), dtype=np.int32),
        }

    base_dir_raw = group.attrs.get("base_dir", "")
    base_dir_val = _normalize_attr_value(base_dir_raw)
    if not isinstance(base_dir_val, str):
        base_dir_val = "" if base_dir_val is None else str(base_dir_val)

    filenames = _read_str_dataset(group["filenames"]) if "filenames" in group else []
    image_filenames_json = (
        _read_str_dataset(group["image_filenames_json"]) if "image_filenames_json" in group else []
    )
    backends = _read_str_dataset(group["backends"]) if "backends" in group else []
    sha256 = _read_str_dataset(group["sha256"]) if "sha256" in group else []

    shapes_ds = group.get("shapes")
    shapes = (
        np.asarray(shapes_ds[...], dtype=np.int32)
        if isinstance(shapes_ds, h5py.Dataset)
        else np.zeros((0, 4), dtype=np.int32)
    )

    image_filenames: list[list[str]] = []
    for raw_entry in image_filenames_json:
        entry = str(raw_entry).strip()
        if not entry:
            image_filenames.append([])
            continue
        parsed = parse_json(entry)
        if not isinstance(parsed, list):
            raise TypeError("videos.image_filenames_json entries must decode to path lists")
        image_filenames.append([str(path).strip() for path in parsed if str(path).strip()])

    return {
        "base_dir": base_dir_val,
        "filenames": filenames,
        "image_filenames": image_filenames,
        "backends": backends,
        "sha256": sha256,
        "shapes": shapes,
    }


def _read_skeleton_group(root: h5py.File) -> dict[str, Any]:
    group = root.get("skeleton")
    if not isinstance(group, h5py.Group):
        return {
            "names": [],
            "links": np.zeros((0, 2), dtype=np.int32),
            "symmetry": {},
        }

    names = _read_str_dataset(group["names"]) if "names" in group else []
    roles = _read_str_dataset(group["roles"]) if "roles" in group else []

    links_ds = group.get("links")
    if isinstance(links_ds, h5py.Dataset):
        links_arr = np.asarray(links_ds[...])
        if links_arr.size == 0:
            links = np.zeros((0, 2), dtype=np.int32)
        else:
            if not np.issubdtype(links_arr.dtype, np.integer):
                raise ValueError("skeleton links dataset must be integer typed")
            if links_arr.shape[-1] != 2:
                raise ValueError("skeleton links dataset must store point pairs")
            links = links_arr.reshape(-1, 2).astype(np.int32, copy=False)
    else:
        links = np.zeros((0, 2), dtype=np.int32)

    symmetry_raw = group.attrs.get("symmetry", "")
    if isinstance(symmetry_raw, bytes | bytearray | np.bytes_):
        symmetry_raw = _decode_project_metadata_utf8(symmetry_raw, field="skeleton.symmetry")
    symmetry: dict[str, str] = {}
    if isinstance(symmetry_raw, str) and symmetry_raw:
        parsed = parse_json(symmetry_raw)
        if not isinstance(parsed, Mapping):
            raise ValueError("skeleton symmetry metadata must be a mapping")
        symmetry = {str(key): str(value) for key, value in parsed.items()}

    return {
        "names": names,
        "roles": roles,
        "links": np.asarray(links, dtype=np.int32),
        "symmetry": symmetry,
    }


def _read_labels_group(root: h5py.File, *, lazy_read: bool) -> dict[str, Any]:
    group = root.get("labels")
    if not isinstance(group, h5py.Group):
        return _empty_labels_payload()

    frames_group = group.get("frames")
    data_group = group.get("data")

    video_idx_ds = frames_group.get("video_index") if isinstance(frames_group, h5py.Group) else None
    frame_idx_ds = frames_group.get("frame_index") if isinstance(frames_group, h5py.Group) else None
    num_inst_ds = (
        frames_group.get("num_instances") if isinstance(frames_group, h5py.Group) else None
    )

    keypoints_ds = data_group.get("keypoints") if isinstance(data_group, h5py.Group) else None
    flags_ds = data_group.get("flags") if isinstance(data_group, h5py.Group) else None
    track_ds = (
        data_group.get(LABEL_TRACK_ID_DATASET)
        if isinstance(data_group, h5py.Group) and LABEL_TRACK_ID_DATASET in data_group
        else data_group.get("track_ids")
        if isinstance(data_group, h5py.Group)
        else None
    )
    visibility_ds = (
        data_group.get(LABEL_VISIBILITY_DATASET) if isinstance(data_group, h5py.Group) else None
    )

    n_frames = 0
    if isinstance(keypoints_ds, h5py.Dataset):
        n_frames = int(keypoints_ds.shape[0])
    if n_frames == 0 and isinstance(video_idx_ds, h5py.Dataset):
        n_frames = int(video_idx_ds.shape[0])
    if n_frames == 0 and isinstance(frame_idx_ds, h5py.Dataset):
        n_frames = int(frame_idx_ds.shape[0])
    if n_frames == 0 and isinstance(num_inst_ds, h5py.Dataset):
        n_frames = int(num_inst_ds.shape[0])

    max_inst = (
        int(keypoints_ds.shape[1])
        if isinstance(keypoints_ds, h5py.Dataset) and keypoints_ds.ndim > 1
        else 0
    )
    num_kpts = (
        int(keypoints_ds.shape[2])
        if isinstance(keypoints_ds, h5py.Dataset) and keypoints_ds.ndim > 2
        else 0
    )

    if max_inst == 0 and isinstance(num_inst_ds, h5py.Dataset):
        num_inst_arr = np.asarray(num_inst_ds[...], dtype=np.int32)
        if num_inst_arr.size:
            max_inst = int(np.max(num_inst_arr))

    max_inst = max(max_inst, 0)
    num_kpts = max(num_kpts, 0)

    video_index = (
        _read_dataset_slice(video_idx_ds, np.int32, lazy=lazy_read)
        if isinstance(video_idx_ds, h5py.Dataset)
        else np.zeros((n_frames,), dtype=np.int32)
    )
    frame_index = (
        _read_dataset_slice(frame_idx_ds, np.int32, lazy=lazy_read)
        if isinstance(frame_idx_ds, h5py.Dataset)
        else np.arange(n_frames, dtype=np.int32)
    )
    num_instances = (
        _read_dataset_slice(num_inst_ds, np.int32, lazy=lazy_read)
        if isinstance(num_inst_ds, h5py.Dataset)
        else np.zeros((n_frames,), dtype=np.int32)
    )

    if isinstance(keypoints_ds, h5py.Dataset):
        keypoints = _read_dataset_slice(keypoints_ds, np.float32, lazy=lazy_read)
    else:
        keypoints = np.full((n_frames, max_inst, num_kpts, 3), np.nan, dtype=np.float32)

    flags = (
        _read_dataset_slice(flags_ds, np.uint8, lazy=lazy_read)
        if isinstance(flags_ds, h5py.Dataset)
        else np.zeros((n_frames, max_inst, num_kpts), dtype=np.uint8)
    )
    track_ids = (
        _read_dataset_slice(track_ds, np.int32, lazy=lazy_read)
        if isinstance(track_ds, h5py.Dataset)
        else np.full((n_frames, max_inst), -1, dtype=np.int32)
    )
    visibility = (
        _read_dataset_slice(visibility_ds, np.uint8, lazy=lazy_read)
        if isinstance(visibility_ds, h5py.Dataset)
        else np.zeros((n_frames, max_inst, num_kpts), dtype=np.uint8)
    )

    return {
        "frames": {
            "video_index": video_index,
            "frame_index": frame_index,
            "num_instances": num_instances,
        },
        "data": {
            "keypoints": keypoints,
            "flags": flags,
            LABEL_TRACK_ID_DATASET: track_ids,
            LABEL_VISIBILITY_DATASET: visibility,
        },
        "metadata": {
            "num_frames": int(n_frames),
            "max_instances": int(max_inst),
            "num_keypoints": int(num_kpts),
        },
    }


def _read_predictions_group(root: h5py.File, *, lazy_read: bool) -> dict[str, Any]:
    group = root.get("predictions")
    if not isinstance(group, h5py.Group):
        return _empty_predictions_payload()

    frames_group = group.get("frames")
    data_group = group.get("data")

    keypoints_ds = data_group.get("keypoints") if isinstance(data_group, h5py.Group) else None
    if not isinstance(keypoints_ds, h5py.Dataset):
        payload = _empty_predictions_payload()
        committed = _normalize_predictions_committed_length(
            group,
            total_rows=0,
            missing_default=0,
            enforce_upper_bound=False,
        )
        payload["attrs"]["committed_length"] = int(committed)
        payload["metadata"]["num_frames"] = int(committed)
        return payload

    total_frames = int(keypoints_ds.shape[0])
    max_inst = int(keypoints_ds.shape[1]) if keypoints_ds.ndim > 1 else 0
    num_kpts = int(keypoints_ds.shape[2]) if keypoints_ds.ndim > 2 else 0
    committed_length = _normalize_predictions_committed_length(
        group,
        total_rows=total_frames,
        missing_default=0,
        exceed_message="Committed length {committed} exceeds dataset length {total_rows}",
    )

    video_idx_ds = frames_group.get("video_index") if isinstance(frames_group, h5py.Group) else None
    frame_idx_ds = frames_group.get("frame_index") if isinstance(frames_group, h5py.Group) else None
    num_inst_ds = (
        frames_group.get("num_instances") if isinstance(frames_group, h5py.Group) else None
    )

    video_index = (
        _read_dataset_slice(video_idx_ds, np.int32, length=committed_length, lazy=lazy_read)
        if isinstance(video_idx_ds, h5py.Dataset)
        else np.zeros((committed_length,), dtype=np.int32)
    )
    frame_index = (
        _read_dataset_slice(frame_idx_ds, np.int32, length=committed_length, lazy=lazy_read)
        if isinstance(frame_idx_ds, h5py.Dataset)
        else np.arange(committed_length, dtype=np.int32)
    )
    num_instances = (
        _read_dataset_slice(num_inst_ds, np.int32, length=committed_length, lazy=lazy_read)
        if isinstance(num_inst_ds, h5py.Dataset)
        else np.zeros((committed_length,), dtype=np.int32)
    )
    keypoints = _read_dataset_slice(
        keypoints_ds,
        np.float32,
        length=committed_length,
        lazy=lazy_read,
    )

    keypoint_score_ds = (
        data_group.get("keypoint_score") if isinstance(data_group, h5py.Group) else None
    )
    if isinstance(keypoint_score_ds, h5py.Dataset):
        keypoint_score = _read_dataset_slice(
            keypoint_score_ds,
            np.float32,
            length=committed_length,
            lazy=lazy_read,
        )
    elif lazy_read:
        keypoint_score = np.zeros((committed_length, max_inst, num_kpts), dtype=np.float32)
    else:
        keypoints_arr = np.asarray(keypoints, dtype=np.float32)
        if keypoints_arr.shape[-1] >= 3:
            keypoint_score = keypoints_arr[..., 2].astype(np.float32, copy=False)
        else:
            keypoint_score = np.zeros((committed_length, max_inst, num_kpts), dtype=np.float32)

    instance_score_ds = (
        data_group.get("instance_score") if isinstance(data_group, h5py.Group) else None
    )
    instance_score = (
        _read_dataset_slice(
            instance_score_ds,
            np.float32,
            length=committed_length,
            lazy=lazy_read,
        )
        if isinstance(instance_score_ds, h5py.Dataset)
        else np.zeros((committed_length, max_inst), dtype=np.float32)
    )

    track_ds = data_group.get("track_id") if isinstance(data_group, h5py.Group) else None
    track_id = (
        _read_dataset_slice(track_ds, np.int32, length=committed_length, lazy=lazy_read)
        if isinstance(track_ds, h5py.Dataset)
        else np.full((committed_length, max_inst), -1, dtype=np.int32)
    )

    deleted_ds = data_group.get("deleted") if isinstance(data_group, h5py.Group) else None
    deleted = (
        _read_dataset_slice(deleted_ds, np.uint8, length=committed_length, lazy=lazy_read)
        if isinstance(deleted_ds, h5py.Dataset)
        else np.zeros((committed_length, max_inst), dtype=np.uint8)
    )

    heatmaps_ds = data_group.get("heatmaps") if isinstance(data_group, h5py.Group) else None
    if isinstance(heatmaps_ds, h5py.Dataset):
        heatmaps = _read_dataset_slice(
            heatmaps_ds,
            np.float16,
            length=committed_length,
            lazy=lazy_read,
        )
        heatmap_h = int(heatmaps_ds.shape[2]) if heatmaps_ds.ndim > 2 else 0
        heatmap_w = int(heatmaps_ds.shape[3]) if heatmaps_ds.ndim > 3 else 0
    else:
        heatmaps = None
        heatmap_h = 0
        heatmap_w = 0

    return {
        "frames": {
            "video_index": video_index,
            "frame_index": frame_index,
            "num_instances": num_instances,
        },
        "data": {
            "keypoints": keypoints,
            "keypoint_score": keypoint_score,
            "instance_score": instance_score,
            "track_id": track_id,
            "deleted": deleted,
            "heatmaps": heatmaps,
        },
        "attrs": {"committed_length": int(committed_length)},
        "metadata": {
            "num_frames": int(committed_length),
            "max_instances": int(max_inst),
            "num_keypoints": int(num_kpts),
            "heatmap_height": int(heatmap_h),
            "heatmap_width": int(heatmap_w),
        },
    }


def _read_suggestions_group(root: h5py.File, *, lazy_read: bool) -> dict[str, Any]:
    group = root.get("suggestions")
    if not isinstance(group, h5py.Group):
        return {
            "video_indices": np.zeros((0,), dtype=np.int32),
            "frame_indices": np.zeros((0,), dtype=np.int32),
            "scores": None,
        }

    video_idx_ds = group.get("video_indices")
    frame_idx_ds = group.get("frame_indices")
    scores_ds = group.get("scores")

    return {
        "video_indices": (
            _read_dataset_slice(video_idx_ds, np.int32, lazy=lazy_read)
            if isinstance(video_idx_ds, h5py.Dataset)
            else np.zeros((0,), dtype=np.int32)
        ),
        "frame_indices": (
            _read_dataset_slice(frame_idx_ds, np.int32, lazy=lazy_read)
            if isinstance(frame_idx_ds, h5py.Dataset)
            else np.zeros((0,), dtype=np.int32)
        ),
        "scores": (
            _read_dataset_slice(scores_ds, np.float32, lazy=lazy_read)
            if isinstance(scores_ds, h5py.Dataset)
            else None
        ),
    }


def _read_runs_group(root: h5py.File) -> dict[str, Any]:
    group = root.get("runs")
    default_table = {
        "run_id": np.zeros((0,), dtype=np.int32),
        "created_ns": np.zeros((0,), dtype=np.int64),
        "config_json": np.zeros((0,), dtype=object),
    }
    if not isinstance(group, h5py.Group):
        return {"table": default_table, "entries": []}

    table_group = group.get("table")
    if not isinstance(table_group, h5py.Group):
        return {"table": default_table, "entries": []}

    run_ids = default_table["run_id"]
    if "run_id" in table_group:
        run_id_ds = table_group.get("run_id")
        if not isinstance(run_id_ds, h5py.Dataset):
            raise TypeError("runs/table/run_id must be an h5py Dataset")
        run_ids = np.asarray(run_id_ds[...], dtype=np.int32)

    created_ns = default_table["created_ns"]
    if "created_ns" in table_group:
        created_ds = table_group.get("created_ns")
        if not isinstance(created_ds, h5py.Dataset):
            raise TypeError("runs/table/created_ns must be an h5py Dataset")
        created_ns = np.asarray(created_ds[...], dtype=np.int64)

    if "config_json" in table_group:
        config_ds = table_group.get("config_json")
        if not isinstance(config_ds, h5py.Dataset):
            raise TypeError("runs/table/config_json must be an h5py Dataset")
        raw_config = np.asarray(config_ds[...], dtype=object).ravel()
        config_values: list[str] = []
        for item in raw_config:
            if isinstance(item, bytes | bytearray | np.bytes_):
                config_values.append(
                    _decode_project_metadata_utf8(item, field="runs.table.config_json")
                )
                continue
            if item is None:
                raise ValueError("config_json entries cannot be null")
            config_values.append(str(item))
        config_json_arr = np.asarray(config_values, dtype=object)
    else:
        config_json_arr = default_table["config_json"]

    count = min(run_ids.shape[0], created_ns.shape[0], config_json_arr.shape[0])
    if count == 0:
        return {
            "table": default_table,
            "entries": [],
        }

    run_ids = run_ids[:count]
    created_ns = created_ns[:count]
    config_json_arr = config_json_arr[:count]
    entries = [
        {
            "run_id": int(run_ids[idx]),
            "created_ns": int(created_ns[idx]) if created_ns.size else 0,
            "config_json": str(config_json_arr[idx]) if config_json_arr.size else "",
        }
        for idx in range(count)
    ]

    return {
        "table": {
            "run_id": run_ids,
            "created_ns": created_ns,
            "config_json": config_json_arr,
        },
        "entries": entries,
    }


def _read_project_metadata(
    handle: h5py.File,
    *,
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_group = handle.get("project_metadata")
    metadata: dict[str, Any] = {"path": str(path)}
    if isinstance(metadata_group, h5py.Group):
        for key, value in metadata_group.attrs.items():
            if key == "provenance_json":
                continue
            metadata[key] = _normalize_attr_value(value)
        max_bytes_val = metadata.get("provenance_max_bytes", _DEFAULT_PROVENANCE_MAX_BYTES)
        max_bytes = _coerce_int(max_bytes_val, default=_DEFAULT_PROVENANCE_MAX_BYTES)
        metadata["provenance_max_bytes"] = int(
            _DEFAULT_PROVENANCE_MAX_BYTES if max_bytes is None else max_bytes
        )
        provenance = _read_provenance(metadata_group)
    else:
        metadata["provenance_max_bytes"] = _DEFAULT_PROVENANCE_MAX_BYTES
        provenance = {"events": []}
    return metadata, provenance


def _read_provenance(metadata_group: h5py.Group) -> dict[str, Any]:
    provenance = _decode_optional_mapping_attr(
        metadata_group.attrs.get("provenance_json"),
        field="project_metadata.provenance_json",
        type_message="project_metadata.provenance_json must be a mapping or JSON string mapping",
    )
    if provenance is None:
        return {"events": [], "schema_version": _PROVENANCE_SCHEMA_VERSION}
    events = provenance.get("events")
    if not isinstance(events, list):
        provenance["events"] = []
    provenance.setdefault("schema_version", _PROVENANCE_SCHEMA_VERSION)
    return provenance


def _decode_runtime_config(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    return _decode_optional_mapping_attr(
        metadata.get("runtime_config"),
        field="project_metadata.runtime_config",
        type_message="project_metadata.runtime_config must be a mapping or JSON string mapping",
    )


def _decode_preferences_override(metadata: Mapping[str, Any]) -> dict[str, Any]:
    preferences_raw = metadata.get("preferences")
    if isinstance(preferences_raw, Mapping):
        return _mapping_to_str_key_dict(preferences_raw, name="project_metadata.preferences")
    preferences = _decode_optional_mapping_attr(
        metadata.get("preferences_json"),
        field="project_metadata.preferences_json",
        type_message="project_metadata.preferences_json must be a mapping or JSON string mapping",
    )
    return {} if preferences is None else preferences


def _decode_session_state(metadata: Mapping[str, Any]) -> dict[str, Any]:
    session_state = _decode_optional_mapping_attr(
        metadata.get("session_json"),
        field="project_metadata.session_json",
        type_message="project_metadata.session_json must be a mapping or JSON string mapping",
    )
    return {} if session_state is None else session_state


def _decode_manifest(metadata: dict[str, Any]):
    manifest = _decode_optional_mapping_attr(
        metadata.get("manifest_json"),
        field="project_metadata.manifest_json",
        type_message="project_metadata.manifest_json must be a mapping or JSON string mapping",
    )
    if manifest is None:
        metadata["manifest"] = None
        return None
    metadata["manifest"] = manifest
    return coerce_manifest(manifest)


def _resolve_video_entries(
    filenames: list[str],
    *,
    manifest_obj: Any,
    bundle_root: Path,
) -> tuple[list[str], list[bool], list[str], list[str]]:
    resolved_paths: list[str] = [""] * len(filenames)
    resolved_exists: list[bool] = [False] * len(filenames)
    video_ids: list[str] = [""] * len(filenames)
    video_labels: list[str] = [""] * len(filenames)

    if manifest_obj is not None:
        for entry in manifest_obj.find_by_type(AssetType.VIDEO):
            raw_index = entry.metadata.get("index") if isinstance(entry.metadata, dict) else None
            if not isinstance(raw_index, int) or not (0 <= raw_index < len(filenames)):
                continue
            raw_path = entry.metadata.get("resolved_path", entry.path)
            _, resolved_path = resolve_project_path(raw_path, project_root=bundle_root)
            resolved_paths[raw_index] = str(resolved_path)
            resolved_exists[raw_index] = resolved_path.exists()
            video_ids[raw_index] = str(entry.id)
            video_labels[raw_index] = str(entry.label)

    for idx, raw_name in enumerate(filenames):
        if not resolved_paths[idx]:
            resolved_candidate = resolve_asset_path(
                raw_name,
                asset_type=AssetType.VIDEO,
                manifest=manifest_obj,
                project_root=bundle_root,
                strict=False,
            )
            resolved_paths[idx] = str(resolved_candidate)
            resolved_exists[idx] = resolved_candidate.exists()
        if not video_ids[idx]:
            video_path_id = make_path_id(Path(resolved_paths[idx]), prefix="video")
            video_ids[idx] = video_path_id.id
            video_labels[idx] = video_path_id.label

    return resolved_paths, resolved_exists, video_ids, video_labels


def _read_videos_info(
    handle: h5py.File,
    *,
    bundle_root: Path,
    manifest_obj: Any,
) -> dict[str, Any]:
    videos_info = _read_videos_group(handle)
    filenames = list(videos_info.get("filenames", []))
    resolved = _resolve_video_entries(
        filenames,
        manifest_obj=manifest_obj,
        bundle_root=bundle_root,
    )
    (
        videos_info["resolved_paths"],
        videos_info["resolved_exists"],
        videos_info["video_ids"],
        videos_info["video_labels"],
    ) = resolved
    return videos_info


def _attach_labels_payload(
    handle: h5py.File,
    *,
    lazy_read: bool,
    videos_info: dict[str, Any],
    skeleton_info: dict[str, Any],
) -> dict[str, Any]:
    labels_payload = _read_labels_group(handle, lazy_read=lazy_read)
    labels_payload["skeleton"] = skeleton_info
    labels_payload["videos"] = videos_info
    labels_payload.setdefault("provenance", {"events": []})
    labels_payload.setdefault("metadata", {})
    return labels_payload


def _attach_predictions_payload(
    handle: h5py.File,
    *,
    lazy_read: bool,
    runtime_config: dict[str, Any] | None,
) -> dict[str, Any]:
    predictions_payload = _read_predictions_group(handle, lazy_read=lazy_read)
    predictions_payload.setdefault("provenance", {"events": []})
    if runtime_config is not None:
        predictions_payload["metadata"]["runtime_config"] = runtime_config
    return predictions_payload


def _read_metrics_info(handle: h5py.File) -> tuple[int, int]:
    metrics_group = handle.get("metrics")
    if not isinstance(metrics_group, h5py.Group):
        return 0, 0
    table_count = len(list(metrics_group.keys()))
    schema_raw = metrics_group.attrs.get("schema_version", 0)
    schema_version = int(schema_raw) if isinstance(schema_raw, int | float) else 0
    return table_count, schema_version


def _populate_metadata_defaults(
    metadata: dict[str, Any],
    *,
    labels_payload: dict[str, Any],
    predictions_payload: dict[str, Any],
    metrics_table_count: int,
    metrics_schema_ver: int,
    videos_info: dict[str, Any],
    runs_payload: dict[str, Any],
) -> None:
    metadata.setdefault("n_labels", labels_payload["metadata"]["num_frames"])
    metadata.setdefault(
        "n_predictions_committed",
        predictions_payload["attrs"]["committed_length"],
    )
    metadata.setdefault("max_inst_preds", predictions_payload["metadata"]["max_instances"])
    metadata.setdefault("n_metric_tables", metrics_table_count)
    if metrics_schema_ver > 0:
        metadata.setdefault("metrics_schema_version", metrics_schema_ver)
    metadata["videos"] = videos_info
    metadata.setdefault("runs_count", len(runs_payload["entries"]))
    if runs_payload["entries"]:
        metadata["runs"] = runs_payload["entries"]


def build_common_reader_state(
    handle: h5py.File,
    *,
    path: Path,
    bundle_root: Path,
    lazy_read: bool,
) -> ReaderCommonState:
    metadata, provenance = _read_project_metadata(handle, path=path)
    if not metadata.get("schema_version") and metadata.get("version"):
        metadata["schema_version"] = metadata["version"]
    metadata.setdefault("schema_name", ARCHIVE_SCHEMA_NAME)
    runtime_config = _decode_runtime_config(metadata)
    if runtime_config is not None:
        metadata["runtime_config"] = runtime_config
    preferences_override = _decode_preferences_override(metadata)
    session_state = _decode_session_state(metadata)
    manifest_obj = _decode_manifest(metadata)
    videos_info = _read_videos_info(handle, bundle_root=bundle_root, manifest_obj=manifest_obj)
    skeleton_info = _read_skeleton_group(handle)
    labels_payload = _attach_labels_payload(
        handle,
        lazy_read=lazy_read,
        videos_info=videos_info,
        skeleton_info=skeleton_info,
    )
    predictions_payload = _attach_predictions_payload(
        handle,
        lazy_read=lazy_read,
        runtime_config=runtime_config,
    )
    metrics_table_count, metrics_schema_ver = _read_metrics_info(handle)
    suggestions_payload = _read_suggestions_group(handle, lazy_read=lazy_read)
    runs_payload = _read_runs_group(handle)
    _populate_metadata_defaults(
        metadata,
        labels_payload=labels_payload,
        predictions_payload=predictions_payload,
        metrics_table_count=metrics_table_count,
        metrics_schema_ver=metrics_schema_ver,
        videos_info=videos_info,
        runs_payload=runs_payload,
    )
    result = {
        "labels": labels_payload,
        "predictions": predictions_payload,
        "metrics": {
            "schema_version": metrics_schema_ver,
            "tables": {},
            "metadata": {},
        },
        "suggestions": suggestions_payload,
        "runs": runs_payload,
        "metadata": metadata,
        "provenance": provenance,
        "session": session_state,
    }
    return ReaderCommonState(
        result=result,
        metadata=metadata,
        preferences_override=preferences_override,
    )


def read_archive_with_assembler(
    path: Path,
    *,
    lazy: bool,
    assemble_result: Callable[[h5py.File, Path, Path, bool], dict[str, Any]],
) -> dict[str, Any]:
    """Open an archive and let a repo-specific wrapper finalize the payload."""
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    handle = h5py.File(str(path), mode="r")
    bundle_root = path.parent

    with contextlib.ExitStack() as stack:
        stack.callback(handle.close)
        result = assemble_result(handle, path, bundle_root, lazy)
        if lazy:
            stack.pop_all()
            result["h5_handle"] = LazyArchiveHandle(handle)
        return result
