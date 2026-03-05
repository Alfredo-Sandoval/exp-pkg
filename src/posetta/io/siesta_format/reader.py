"""Read-only helpers for `.siesta` archives."""

from __future__ import annotations

import contextlib
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from posetta.core.json_utils import parse_json
from posetta.core.path_registry import make_path_id
from posetta.io.manifest import (
    AssetType,
    coerce_manifest,
    resolve_asset_path,
    resolve_project_path,
)
from posetta.io.siesta_format.project_validation import (
    ProjectSummary,
    summarize_project,
    validate_project,
)
from posetta.io.siesta_format.shared import (
    _DEFAULT_PROVENANCE_MAX_BYTES,
    _PROVENANCE_SCHEMA_VERSION,
    _coerce_int,
    _looks_like_int,
    _mapping_to_str_key_dict,
    _normalize_predictions_committed_length,
)


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
        for r in data:
            a, b = r[0], r[1]
            sa = (
                _decode_utf8_field(a, field=f"{field_name}[row,0]")
                if isinstance(a, bytes | bytearray | np.bytes_)
                else str(a)
            )
            sb = (
                _decode_utf8_field(b, field=f"{field_name}[row,1]")
                if isinstance(b, bytes | bytearray | np.bytes_)
                else str(b)
            )
            rows.append([sa, sb])
        return rows
    flat = np.ravel(data)
    out: list[str] = []
    for idx, x in enumerate(flat):
        if isinstance(x, bytes | bytearray | np.bytes_):
            out.append(_decode_utf8_field(x, field=f"{field_name}[{idx}]"))
        else:
            out.append(str(x))
    return out


__all__ = [
    "LazyDatasetHandle",
    "LazySiestaHandle",
    "ProjectSummary",
    "_looks_like_int",
    "_looks_like_iso_timestamp",
    "read_siesta",
    "summarize_project",
    "validate_project",
]


def _decode_project_metadata_utf8(raw: bytes | bytearray | np.bytes_, *, field: str) -> str:
    """Decode UTF-8 metadata bytes with explicit error context."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Project metadata attribute {field} is not valid UTF-8") from exc


class LazyDatasetHandle:
    """Lightweight wrapper around an h5py.Dataset for lazy materialization."""

    def __init__(self, dataset: h5py.Dataset, dtype: Any, length: int | None = None) -> None:
        self.dataset = dataset
        self.dtype = dtype
        self.length = length

    def materialize(self) -> np.ndarray:
        if not self.dataset.id.valid:
            raise RuntimeError(
                "Cannot materialize lazy dataset after the owning .siesta handle is closed"
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
                "Cannot read lazy dataset shape after the owning .siesta handle is closed"
            )
        base = tuple(self.dataset.shape)
        if not base or self.length is None:
            return base
        return (min(self.length, base[0]), *base[1:])


class LazySiestaHandle:
    """Owns an open h5py.File returned by read_siesta(lazy=True)."""

    def __init__(self, file_handle: h5py.File) -> None:
        self._file_handle = file_handle

    @property
    def closed(self) -> bool:
        return not self._file_handle.id.valid

    @property
    def file(self) -> h5py.File:
        if self.closed:
            raise RuntimeError("LazySiestaHandle is closed")
        return self._file_handle

    def close(self) -> None:
        if self._file_handle.id.valid:
            self._file_handle.close()

    def __enter__(self) -> LazySiestaHandle:
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        self.close()


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
    backends = _read_str_dataset(group["backends"]) if "backends" in group else []
    sha256 = _read_str_dataset(group["sha256"]) if "sha256" in group else []

    shapes_ds = group.get("shapes")
    if isinstance(shapes_ds, h5py.Dataset):
        shapes = np.asarray(shapes_ds[...], dtype=np.int32)
    else:
        shapes = np.zeros((0, 4), dtype=np.int32)

    return {
        "base_dir": base_dir_val,
        "filenames": filenames,
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
        symmetry = {str(k): str(v) for k, v in parsed.items()}

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
    track_ds = data_group.get("track_ids") if isinstance(data_group, h5py.Group) else None

    n_frames = 0
    if isinstance(keypoints_ds, h5py.Dataset):
        n_frames = keypoints_ds.shape[0]

    if n_frames == 0 and isinstance(video_idx_ds, h5py.Dataset):
        n_frames = int(video_idx_ds.shape[0])
    if n_frames == 0 and isinstance(frame_idx_ds, h5py.Dataset):
        n_frames = int(frame_idx_ds.shape[0])
    if n_frames == 0 and isinstance(num_inst_ds, h5py.Dataset):
        n_frames = int(num_inst_ds.shape[0])

    max_inst = 0
    if isinstance(keypoints_ds, h5py.Dataset) and keypoints_ds.ndim > 1:
        max_inst = keypoints_ds.shape[1]

    num_kpts = 0
    if isinstance(keypoints_ds, h5py.Dataset) and keypoints_ds.ndim > 2:
        num_kpts = keypoints_ds.shape[2]

    if max_inst == 0 and isinstance(num_inst_ds, h5py.Dataset):
        ds_inst = num_inst_ds
        num_inst_arr = np.asarray(ds_inst[...], dtype=np.int32)
        if num_inst_arr.size:
            max_inst = int(np.max(num_inst_arr))

    if max_inst < 0:
        max_inst = 0
    if num_kpts < 0:
        num_kpts = 0

    if isinstance(video_idx_ds, h5py.Dataset):
        video_index = _read_dataset_slice(video_idx_ds, np.int32, lazy=lazy_read)
    else:
        video_index = np.zeros((n_frames,), dtype=np.int32)

    if isinstance(frame_idx_ds, h5py.Dataset):
        frame_index = _read_dataset_slice(frame_idx_ds, np.int32, lazy=lazy_read)
    else:
        frame_index = np.arange(n_frames, dtype=np.int32)

    if isinstance(num_inst_ds, h5py.Dataset):
        num_instances = _read_dataset_slice(num_inst_ds, np.int32, lazy=lazy_read)
    else:
        num_instances = np.zeros((n_frames,), dtype=np.int32)

    if isinstance(keypoints_ds, h5py.Dataset):
        keypoints = _read_dataset_slice(keypoints_ds, np.float32, lazy=lazy_read)
    else:
        # Preserve missing keypoints as NaN so the GUI can hide them instead of
        # drawing bogus (0,0) markers.
        keypoints = np.full((n_frames, max_inst, num_kpts, 3), np.nan, dtype=np.float32)

    if isinstance(flags_ds, h5py.Dataset):
        flags = _read_dataset_slice(flags_ds, np.uint8, lazy=lazy_read)
    else:
        flags = np.zeros((n_frames, max_inst, num_kpts), dtype=np.uint8)

    if isinstance(track_ds, h5py.Dataset):
        track_ids = _read_dataset_slice(track_ds, np.int32, lazy=lazy_read)
    else:
        track_ids = np.full((n_frames, max_inst), -1, dtype=np.int32)

    metadata = {
        "num_frames": int(n_frames),
        "max_instances": int(max_inst),
        "num_keypoints": int(num_kpts),
    }

    return {
        "frames": {
            "video_index": video_index,
            "frame_index": frame_index,
            "num_instances": num_instances,
        },
        "data": {
            "keypoints": keypoints,
            "flags": flags,
            "track_ids": track_ids,
        },
        "metadata": metadata,
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

    if isinstance(video_idx_ds, h5py.Dataset):
        video_index = _read_dataset_slice(
            video_idx_ds, np.int32, length=committed_length, lazy=lazy_read
        )
    else:
        video_index = np.zeros((committed_length,), dtype=np.int32)

    if isinstance(frame_idx_ds, h5py.Dataset):
        frame_index = _read_dataset_slice(
            frame_idx_ds, np.int32, length=committed_length, lazy=lazy_read
        )
    else:
        frame_index = np.arange(committed_length, dtype=np.int32)

    if isinstance(num_inst_ds, h5py.Dataset):
        num_instances = _read_dataset_slice(
            num_inst_ds,
            np.int32,
            length=committed_length,
            lazy=lazy_read,
        )
    else:
        num_instances = np.zeros((committed_length,), dtype=np.int32)

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
        kp_arr = np.asarray(keypoints, dtype=np.float32)
        if kp_arr.shape[-1] >= 3:
            keypoint_score = kp_arr[..., 2].astype(np.float32, copy=False)
        else:
            keypoint_score = np.zeros((committed_length, max_inst, num_kpts), dtype=np.float32)

    instance_score_ds = (
        data_group.get("instance_score") if isinstance(data_group, h5py.Group) else None
    )
    if isinstance(instance_score_ds, h5py.Dataset):
        instance_score = _read_dataset_slice(
            instance_score_ds,
            np.float32,
            length=committed_length,
            lazy=lazy_read,
        )
    else:
        instance_score = np.zeros((committed_length, max_inst), dtype=np.float32)

    track_ds = data_group.get("track_id") if isinstance(data_group, h5py.Group) else None
    if isinstance(track_ds, h5py.Dataset):
        track_id = _read_dataset_slice(
            track_ds,
            np.int32,
            length=committed_length,
            lazy=lazy_read,
        )
    else:
        track_id = np.full((committed_length, max_inst), -1, dtype=np.int32)

    deleted_ds = data_group.get("deleted") if isinstance(data_group, h5py.Group) else None
    if isinstance(deleted_ds, h5py.Dataset):
        deleted = _read_dataset_slice(
            deleted_ds,
            np.uint8,
            length=committed_length,
            lazy=lazy_read,
        )
    else:
        deleted = np.zeros((committed_length, max_inst), dtype=np.uint8)

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

    metadata = {
        "num_frames": int(committed_length),
        "max_instances": int(max_inst),
        "num_keypoints": int(num_kpts),
        "heatmap_height": int(heatmap_h),
        "heatmap_width": int(heatmap_w),
    }

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
        "metadata": metadata,
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

    if isinstance(video_idx_ds, h5py.Dataset):
        video_indices = _read_dataset_slice(video_idx_ds, np.int32, lazy=lazy_read)
    else:
        video_indices = np.zeros((0,), dtype=np.int32)

    if isinstance(frame_idx_ds, h5py.Dataset):
        frame_indices = _read_dataset_slice(frame_idx_ds, np.int32, lazy=lazy_read)
    else:
        frame_indices = np.zeros((0,), dtype=np.int32)

    if isinstance(scores_ds, h5py.Dataset):
        scores = _read_dataset_slice(scores_ds, np.float32, lazy=lazy_read)
    else:
        scores = None

    return {
        "video_indices": video_indices,
        "frame_indices": frame_indices,
        "scores": scores,
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
        raw_config = config_ds[...]
        flattened = np.asarray(raw_config, dtype=object).ravel()
        config_values: list[str] = []
        for item in flattened:
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
            "table": {
                "run_id": default_table["run_id"],
                "created_ns": default_table["created_ns"],
                "config_json": default_table["config_json"],
            },
            "entries": [],
        }

    run_ids = run_ids[:count]
    created_ns = created_ns[:count]
    config_json_arr = config_json_arr[:count]

    entries = [
        {
            "run_id": int(run_ids[idx]),
            "created_ns": int(created_ns[idx]) if created_ns.size else 0,
            "config_json": (str(config_json_arr[idx]) if config_json_arr.size else ""),
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


def _assemble_result(
    handle: h5py.File,
    *,
    path: Path,
    bundle_root: Path,
    lazy_read: bool,
) -> dict[str, Any]:
    metadata_group = handle.get("project_metadata")
    metadata: dict[str, Any] = {"path": str(path)}
    provenance: dict[str, Any] = {"events": []}

    if isinstance(metadata_group, h5py.Group):
        for key, value in metadata_group.attrs.items():
            if key == "provenance_json":
                continue
            metadata[key] = _normalize_attr_value(value)

        max_bytes_val = metadata.get("provenance_max_bytes", _DEFAULT_PROVENANCE_MAX_BYTES)
        max_bytes = _coerce_int(max_bytes_val, default=_DEFAULT_PROVENANCE_MAX_BYTES)
        if max_bytes is None:
            max_bytes = _DEFAULT_PROVENANCE_MAX_BYTES
        metadata["provenance_max_bytes"] = int(max_bytes)

        provenance_raw = metadata_group.attrs.get("provenance_json")
        if isinstance(provenance_raw, bytes | bytearray | np.bytes_):
            provenance_raw = _decode_project_metadata_utf8(
                provenance_raw, field="project_metadata.provenance_json"
            )
        if isinstance(provenance_raw, str) and provenance_raw:
            parsed_provenance = parse_json(provenance_raw)
        else:
            parsed_provenance = {}
        if isinstance(parsed_provenance, Mapping):
            provenance = _mapping_to_str_key_dict(
                parsed_provenance, name="project_metadata.provenance_json"
            )
        else:
            provenance = {}
        events = provenance.get("events")
        if not isinstance(events, list):
            provenance["events"] = []
        provenance.setdefault("schema_version", _PROVENANCE_SCHEMA_VERSION)
    else:
        metadata["provenance_max_bytes"] = _DEFAULT_PROVENANCE_MAX_BYTES

    runtime_config_raw = metadata.get("runtime_config")
    runtime_config: dict[str, Any] | None = None
    if runtime_config_raw is not None:
        if isinstance(runtime_config_raw, Mapping):
            runtime_config = _mapping_to_str_key_dict(
                runtime_config_raw,
                name="project_metadata.runtime_config",
            )
        elif isinstance(runtime_config_raw, str):
            runtime_config_str = runtime_config_raw.strip()
            if runtime_config_str:
                parsed_runtime_config = parse_json(runtime_config_str)
                if not isinstance(parsed_runtime_config, Mapping):
                    raise TypeError("project_metadata.runtime_config must decode to a mapping")
                runtime_config = _mapping_to_str_key_dict(
                    parsed_runtime_config,
                    name="project_metadata.runtime_config",
                )
        else:
            raise TypeError(
                "project_metadata.runtime_config must be a mapping or JSON string mapping"
            )
    if runtime_config is not None:
        metadata["runtime_config"] = runtime_config

    preferences_override: dict[str, Any] = {}
    if isinstance(metadata.get("preferences"), Mapping):
        preferences_override = dict(metadata["preferences"])
    else:
        pref_raw = metadata.get("preferences_json")
        if isinstance(pref_raw, str) and pref_raw:
            parsed = parse_json(pref_raw)
            if isinstance(parsed, Mapping):
                preferences_override = _mapping_to_str_key_dict(
                    parsed, name="project_metadata.preferences_json"
                )
            else:
                raise ValueError("preferences_json must decode to a mapping")

    session_state: dict[str, Any] = {}
    session_raw = metadata.get("session_json")
    if session_raw is None and isinstance(metadata_group, h5py.Group):
        session_raw = metadata_group.attrs.get("session_json")
    if isinstance(session_raw, bytes | bytearray | np.bytes_):
        session_raw = _decode_project_metadata_utf8(
            session_raw, field="project_metadata.session_json"
        )
    if isinstance(session_raw, str) and session_raw:
        parsed_session = parse_json(session_raw)
        if isinstance(parsed_session, Mapping):
            session_state = _mapping_to_str_key_dict(
                parsed_session, name="project_metadata.session_json"
            )

    metadata["preferences"] = preferences_override

    manifest_raw = metadata.get("manifest_json")
    if manifest_raw is None and isinstance(metadata_group, h5py.Group):
        manifest_raw = metadata_group.attrs.get("manifest_json")
    if isinstance(manifest_raw, bytes | bytearray | np.bytes_):
        manifest_raw = _decode_project_metadata_utf8(
            manifest_raw, field="project_metadata.manifest_json"
        )
    if not manifest_raw:
        raise ValueError("Manifest is required in project_metadata.manifest_json")
    parsed_manifest = parse_json(manifest_raw)
    if not isinstance(parsed_manifest, Mapping):
        raise TypeError("manifest_json must decode to a mapping")
    manifest = _mapping_to_str_key_dict(parsed_manifest, name="project_metadata.manifest_json")
    if not manifest.get("entries"):
        raise ValueError("Manifest entries are required to resolve assets")
    metadata["manifest"] = manifest
    manifest_obj = coerce_manifest(manifest)

    videos_info = _read_videos_group(handle)
    filenames = list(videos_info.get("filenames", []))
    resolved_paths: list[str] = [""] * len(filenames)
    resolved_exists: list[bool] = [False] * len(filenames)
    video_ids: list[str] = [""] * len(filenames)
    video_labels: list[str] = [""] * len(filenames)

    if manifest_obj is not None:
        video_entries = manifest_obj.find_by_type(AssetType.VIDEO)
        for entry in video_entries:
            raw_index = entry.metadata.get("index") if isinstance(entry.metadata, dict) else None
            if not isinstance(raw_index, int):
                continue
            if raw_index < 0 or raw_index >= len(filenames):
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
            )
            resolved_paths[idx] = str(resolved_candidate)
            resolved_exists[idx] = resolved_candidate.exists()
        if not video_ids[idx]:
            vid_id = make_path_id(Path(resolved_paths[idx]), prefix="video")
            video_ids[idx] = vid_id.id
            video_labels[idx] = vid_id.label
    videos_info["resolved_paths"] = resolved_paths
    videos_info["resolved_exists"] = resolved_exists
    videos_info["video_ids"] = video_ids
    videos_info["video_labels"] = video_labels

    skeleton_info = _read_skeleton_group(handle)

    labels_payload = _read_labels_group(handle, lazy_read=lazy_read)
    labels_payload["skeleton"] = skeleton_info
    labels_payload["videos"] = videos_info
    labels_payload.setdefault("provenance", {"events": []})

    if "metadata" not in labels_payload:
        labels_payload["metadata"] = {}
    labels_payload["metadata"]["preferences"] = metadata.get("preferences", {})

    predictions_payload = _read_predictions_group(handle, lazy_read=lazy_read)
    predictions_payload.setdefault("provenance", {"events": []})
    if runtime_config is not None:
        predictions_payload["metadata"]["runtime_config"] = runtime_config

    metrics_group = handle.get("metrics")
    metrics_table_count = 0
    metrics_schema_ver = 0
    if isinstance(metrics_group, h5py.Group):
        metrics_table_count = len(list(metrics_group.keys()))
        ver_val = metrics_group.attrs.get("schema_version", 0)
        metrics_schema_ver = int(ver_val) if isinstance(ver_val, int | float) else 0

    suggestions_payload = _read_suggestions_group(handle, lazy_read=lazy_read)
    runs_payload = _read_runs_group(handle)

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

    return result


def read_siesta(
    path: Path,
    *,
    lazy: bool = False,
) -> dict[str, Any]:
    """Load a `.siesta` project archive from disk.

    Args:
        path: Path to the `.siesta` file.
        lazy: If True, return lazy dataset handles instead of materializing arrays.
            The return payload includes ``h5_handle`` (LazySiestaHandle), and the
            caller must close it after materializing lazy datasets.

    Returns:
        dict: Project data including videos, labels, predictions, and metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
    """

    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    handle = h5py.File(str(path), mode="r")
    bundle_root = path.parent

    with contextlib.ExitStack() as stack:
        stack.callback(handle.close)
        result = _assemble_result(handle, path=path, bundle_root=bundle_root, lazy_read=lazy)
        if lazy:
            stack.pop_all()
            result["h5_handle"] = LazySiestaHandle(handle)
        return result


def _normalize_iso_timestamp(text: str) -> str:
    candidate = text.strip()
    if not candidate:
        return ""
    adjusted = candidate
    if adjusted.endswith("Z"):
        adjusted = adjusted[:-1] + "+00:00"
    dt = datetime.fromisoformat(adjusted)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt.isoformat().replace("+00:00", "Z")


_FLOAT_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_ISO_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$"
)


def _looks_like_float(text: str) -> bool:
    if not text:
        return False
    s = text.strip()
    lowered = s.lower()
    if lowered in {"nan", "inf", "+inf", "-inf"}:
        return True
    return bool(_FLOAT_PATTERN.match(s))


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
        stripped_decoded = decoded.strip()
        if stripped_decoded and _looks_like_iso_timestamp(stripped_decoded):
            return _normalize_iso_timestamp(stripped_decoded)
        return decoded
    if isinstance(current, np.ndarray):
        return current.tolist()
    if isinstance(current, np.bool_):
        return bool(current)
    if isinstance(current, bool):
        return bool(current)
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

    ds = dataset
    if length is None:
        data = ds[...]
    else:
        data = ds[:length]

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
            "track_ids": np.zeros((0, 0), dtype=np.int32),
        },
        "metadata": {"num_frames": 0, "max_instances": 0, "num_keypoints": 0},
        "skeleton": {
            "names": [],
            "links": np.zeros((0, 2), dtype=np.int32),
            "symmetry": {},
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
