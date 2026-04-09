"""Unified native archive serializer (core writer)."""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from xpkg.core.json_utils import parse_json
from xpkg.core.logging_utils import get_logger
from xpkg.core.path_registry import ensure_dir, make_path_id
from xpkg.core.skeleton import Keypoint
from xpkg.io.archive_format.manifest_policy import (
    load_manifest_from_metadata,
    register_archive,
    register_metadata_assets,
    register_videos,
)
from xpkg.io.archive_format.metrics_hdf5 import (
    write_table_to_handle as write_metrics_table_to_handle,
)
from xpkg.io.archive_format.prediction_coerce import (
    _coerce_prediction_items,
    _infer_prediction_keypoint_count,
    coerce_predictions_from_labels,
)
from xpkg.io.archive_format.predictions_datasets import (
    PredictionDatasetMap,
    _assert_prediction_dataset_alignment,
    _bootstrap_predictions_group,
    _fill_prediction_slice,
    _infer_batch_heatmap_hw,
    _normalize_append_batch,
    predicted_instance_types,
)
from xpkg.io.archive_format.segmentation_hdf5 import (
    write_segmentation_group,
)
from xpkg.io.archive_format.shared import (
    _COERCE_PRIMITIVE_SENTINEL,
    _DEFAULT_PROVENANCE_MAX_BYTES,
    ARCHIVE_SCHEMA_NAME,
    ARCHIVE_SCHEMA_VERSION,
    CANONICAL_ARCHIVE_SUFFIX,
    LABEL_TRACK_ID_DATASET,
    LABEL_VISIBILITY_DATASET,
    _coerce_int,
    _coerce_primitive,
    _default_provenance_entry,
    _mapping_to_str_key_dict,
    _normalize_runs_entries,
    _now_utc_iso,
    _require_project_metadata_group,
    _serialize_json,
)
from xpkg.io.archive_format.tracks_hdf5 import read_tracks_group, write_tracks_group
from xpkg.io.archive_format.transaction import (
    ArchiveFileLock,
    _append_provenance,
    _ensure_journal_attr,
    _JournalTransaction,
)
from xpkg.io.labels.model import Labels as LabelsModel
from xpkg.io.manifest import ProjectManifest, coerce_manifest, resolve_project_path
from xpkg.version import __version__ as package_version

logger = get_logger(__name__)


def _write_preferences_attr(meta_group: h5py.Group, preferences: Mapping[str, Any]) -> None:
    if not preferences:
        return
    payload = _mapping_to_str_key_dict(preferences, name="labels.preferences")
    meta_group.attrs["preferences_json"] = _serialize_json(payload)


def _write_segmentation_from_labels(h5file: h5py.File, labels) -> None:
    """Collect masks and ROIs from labeled frames and write to HDF5."""
    from xpkg.core.annotations.regions import ROI, SegmentationMask

    masks_by_frame: list[tuple[int, int, list[SegmentationMask]]] = []
    rois_by_frame: list[tuple[int, int, list[ROI]]] = []
    video_lookup = {video: idx for idx, video in enumerate(labels.videos)}

    for lf in labels.labeled_frames:
        vi = video_lookup.get(lf.video, 0)
        fi = int(lf.frame_idx)
        if hasattr(lf, "masks") and lf.masks:
            masks_by_frame.append((vi, fi, list(lf.masks)))
        if hasattr(lf, "rois") and lf.rois:
            rois_by_frame.append((vi, fi, list(lf.rois)))

    if masks_by_frame or rois_by_frame:
        write_segmentation_group(h5file, masks_by_frame, rois_by_frame)


def write_archive(
    path: Path,
    labels,
    predictions=None,
    suggestions=None,
    metadata=None,
    metrics=None,
    manifest=None,
) -> None:
    """Create or overwrite a canonical `.xpkg` archive atomically."""
    parent = path.parent
    project_root = path.parent
    if not parent.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {parent}")

    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping when provided")

    metadata_input = dict(metadata or {})
    runs_input = metadata_input.pop("runs", None)
    raw_provenance = metadata_input.pop("provenance_json", None)

    preferences = labels.preferences

    def _coerce_attr_value(value: Any) -> Any:
        if value is None:
            return ""
        primitive = _coerce_primitive(value)
        if primitive is not _COERCE_PRIMITIVE_SENTINEL:
            return primitive
        return _serialize_json(value)

    def _normalize_metrics_entries(value: Any) -> list[tuple[str, pd.DataFrame]]:
        if value is None:
            return []
        if isinstance(value, Mapping):
            candidates = list(value.items())
        elif isinstance(value, Sequence) and not isinstance(value, bytes | bytearray | str):
            candidates = list(value)
        else:
            raise TypeError(
                "metrics must be a mapping or a sequence of (name, pandas.DataFrame) pairs"
            )

        normalized: list[tuple[str, pd.DataFrame]] = []
        for item in candidates:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "metrics entries must be (name, pandas.DataFrame) pairs; "
                    f"received {type(item).__name__}"
                )
            table_name, dataframe = item
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError(
                    f"metrics entry '{table_name}' must map to a pandas.DataFrame, "
                    f"received {type(dataframe).__name__}"
                )
            normalized.append((str(table_name), dataframe))
        return normalized

    now_iso = _now_utc_iso()

    runs_entries = _normalize_runs_entries(runs_input)

    defaults: dict[str, Any] = {
        "schema_name": ARCHIVE_SCHEMA_NAME,
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "version": ARCHIVE_SCHEMA_VERSION,
        "created": now_iso,
        "modified": now_iso,
        "archive_version": str(package_version),
        "archive_suffix": CANONICAL_ARCHIVE_SUFFIX,
        "instance_layout": "dense_fixed_width",
        "base_dir": path.name,
        "provenance_max_bytes": _DEFAULT_PROVENANCE_MAX_BYTES,
        "journal": "{}",
    }

    merged_metadata: dict[str, Any] = {**defaults, **metadata_input}
    if not merged_metadata.get("created"):
        merged_metadata["created"] = now_iso
    if not merged_metadata.get("modified"):
        merged_metadata["modified"] = now_iso
    if not merged_metadata.get("archive_version"):
        merged_metadata["archive_version"] = str(package_version)
    if not merged_metadata.get("schema_name"):
        merged_metadata["schema_name"] = ARCHIVE_SCHEMA_NAME
    if not merged_metadata.get("schema_version"):
        merged_metadata["schema_version"] = merged_metadata.get("version") or ARCHIVE_SCHEMA_VERSION
    merged_metadata["version"] = str(
        merged_metadata.get("version") or merged_metadata["schema_version"]
    )
    merged_metadata.setdefault("runs_count", len(runs_entries))

    journal_val = merged_metadata.get("journal", "{}")
    if isinstance(journal_val, Mapping):
        journal_val = _serialize_json(journal_val)
    merged_metadata["journal"] = str(journal_val) if journal_val else "{}"

    merged_metadata["base_dir"] = str(merged_metadata.get("base_dir") or path.name)
    merged_metadata["provenance_max_bytes"] = int(
        merged_metadata.get("provenance_max_bytes", _DEFAULT_PROVENANCE_MAX_BYTES)
    )

    def _max_instances_from_labels(lbls: LabelsModel) -> int:
        max_inst = 0
        for frame in lbls.labeled_frames:
            instances = list(frame.instances)
            user_count = sum(
                1 for inst in instances if not isinstance(inst, predicted_instance_types())
            )
            if user_count > max_inst:
                max_inst = user_count
        return int(max_inst)

    max_inst_labels = _max_instances_from_labels(labels)
    merged_metadata["max_inst_labels"] = int(max_inst_labels)

    provenance_payload: dict[str, Any] = {}
    if raw_provenance is not None:
        if isinstance(raw_provenance, str):
            parsed = parse_json(raw_provenance)
            if isinstance(parsed, Mapping):
                provenance_payload = _mapping_to_str_key_dict(
                    parsed, name="project_metadata.provenance_json"
                )
        elif isinstance(raw_provenance, Mapping):
            provenance_payload = _mapping_to_str_key_dict(
                raw_provenance, name="project_metadata.provenance_json"
            )
    else:
        provenance_payload = _mapping_to_str_key_dict(labels.provenance, name="labels.provenance")

    videos = list(labels.videos or [])
    video_lookup, video_path_lookup = build_video_lookups(videos, project_root=project_root)

    manifest_obj = coerce_manifest(manifest)
    if manifest_obj is None:
        manifest_obj = ProjectManifest()
    register_videos(manifest_obj, videos=videos, project_root=project_root)
    register_archive(manifest_obj, path)
    register_metadata_assets(manifest_obj, archive_path=path, metadata_input=metadata_input)

    suggestions_data = suggestions
    if suggestions_data is None:
        suggestions_data = labels.suggestions
    if suggestions_data is not None and not isinstance(suggestions_data, Sequence):
        suggestions_data = list(suggestions_data)

    tmp_path: Path | None = None
    with ArchiveFileLock(path):
        tmp_handle = tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}_", suffix=".tmp", dir=str(parent), delete=False
        )
        tmp_path = Path(tmp_handle.name)
        tmp_handle.close()

        with contextlib.ExitStack() as cleanup_stack:
            cleanup_stack.callback(
                lambda temp_file=tmp_path: temp_file.exists() and temp_file.unlink()
            )
            with h5py.File(str(tmp_path), mode="w") as h5file:
                meta_group = h5file.require_group("project_metadata")

                if provenance_payload:
                    meta_group.attrs["provenance_json"] = _serialize_json(provenance_payload)

                if preferences:
                    _write_preferences_attr(meta_group, preferences)

                session_data = labels.session
                if session_data:
                    meta_group.attrs["session_json"] = _serialize_json(session_data)

                manifest_data = manifest_obj.to_dict()
                meta_group.attrs["manifest_json"] = _serialize_json(manifest_data)

                write_skeleton_group(h5file, labels)
                write_videos_group(h5file, videos, base_dir=merged_metadata["base_dir"])

                labels_max = max_inst_labels if max_inst_labels > 0 else None
                write_labels_group(h5file, labels, max_inst=labels_max)

                labels_frames = 0
                if "labels" in h5file:
                    lbl_grp = h5file["labels"]
                    if isinstance(lbl_grp, h5py.Group) and "frames" in lbl_grp:
                        fr_grp = lbl_grp["frames"]
                        if isinstance(fr_grp, h5py.Group) and "video_index" in fr_grp:
                            vid_idx = fr_grp["video_index"]
                            if isinstance(vid_idx, h5py.Dataset):
                                labels_frames = int(vid_idx.shape[0])

                track_prediction_items = (
                    list(_coerce_prediction_items(predictions, video_index_lookup=video_lookup))
                    if predictions is not None
                    else None
                )

                max_inst_preds = write_predictions_group(
                    h5file,
                    predictions,
                    video_index_lookup=video_lookup,
                )
                write_tracks_group(
                    h5file,
                    labels=labels,
                    prediction_items=track_prediction_items,
                )
                predictions_group = h5file["predictions"]
                cl_val = predictions_group.attrs.get("committed_length", 0)
                committed_length = _coerce_int(cl_val, default=0) or 0

                if suggestions_data is not None:
                    suggestion_items = list(suggestions_data)
                    sugg_group = h5file.require_group("suggestions")
                    write_suggestions_datasets(
                        sugg_group,
                        suggestion_items,
                        video_lookup,
                        video_path_lookup,
                        project_root=project_root,
                    )

                _write_segmentation_from_labels(h5file, labels)

                metrics_count = 0
                for table_name, dataframe in _normalize_metrics_entries(metrics):
                    write_metrics_table_to_handle(
                        h5file,
                        name=table_name,
                        dataframe=dataframe,
                        mode="append",
                    )
                    metrics_count += 1

                runs_group = h5file.require_group("runs")
                write_runs_table(runs_group, runs_entries)

                merged_metadata["max_inst_preds"] = int(max_inst_preds)
                merged_metadata["n_labels"] = int(labels_frames)
                merged_metadata["n_predictions_committed"] = int(committed_length)
                merged_metadata["n_metric_tables"] = int(metrics_count)

                for key, value in merged_metadata.items():
                    if key == "provenance_json":
                        continue
                    meta_group.attrs[key] = _coerce_attr_value(value)

                _ensure_journal_attr(meta_group)

                creation_entry = _default_provenance_entry(
                    "create",
                    archive_version=str(merged_metadata.get("archive_version")),
                    schema_version=str(merged_metadata.get("schema_version")),
                )
                _append_provenance(
                    meta_group,
                    creation_entry,
                    max_bytes=int(merged_metadata["provenance_max_bytes"]),
                )

                h5file.flush()

            os.replace(tmp_path, path)
            cleanup_stack.pop_all()

    return None


def update_labels_archive(
    path: Path,
    labels,
    *,
    journal: bool = True,
    regenerate_predictions: bool = False,
) -> None:
    """Overwrite label data while preserving other archive stores by default."""

    if not path.exists():
        raise FileNotFoundError(f"archive does not exist: {path}")

    tmp_path: Path | None = None

    with ArchiveFileLock(path):
        parent = path.parent
        ensure_dir(parent)
        tmp_handle = tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}_labels_",
            suffix=".tmp",
            dir=str(parent),
            delete=False,
        )
        tmp_path = Path(tmp_handle.name)
        tmp_handle.close()

        with contextlib.ExitStack() as cleanup_stack:
            cleanup_stack.callback(
                lambda temp_file=tmp_path: temp_file.exists() and temp_file.unlink()
            )
            with h5py.File(str(path), "r") as src:
                _require_project_metadata_group(
                    src,
                    missing_message="archive is missing the /project_metadata group",
                )

                existing_tracks = read_tracks_group(src)
                rewritten_groups = {"labels", "videos", "suggestions", "skeleton", "tracks"}
                if regenerate_predictions:
                    rewritten_groups.add("predictions")

                with h5py.File(str(tmp_path), mode="w") as dst:
                    for group_name, obj in src.items():
                        if group_name in rewritten_groups:
                            continue
                        src.copy(obj, dst, name=group_name)

                    prediction_items = None
                    if regenerate_predictions:
                        prediction_items = coerce_predictions_from_labels(labels)
                        write_predictions_group(
                            dst,
                            prediction_items,
                            committed_length=None,
                        )

                    preferences = labels.preferences
                    if preferences:
                        meta_group = dst.require_group("project_metadata")
                        _write_preferences_attr(meta_group, preferences)

                    session_data = labels.session
                    if session_data:
                        meta_group = dst.require_group("project_metadata")
                        meta_group.attrs["session_json"] = _serialize_json(session_data)

                    write_skeleton_group(dst, labels)

                    videos = list(labels.videos or [])
                    base_dir = ""
                    src_videos_group = src.get("videos")
                    if isinstance(src_videos_group, h5py.Group):
                        base_dir_raw = src_videos_group.attrs.get("base_dir", "")
                        if isinstance(base_dir_raw, bytes | bytearray | np.bytes_):
                            base_dir = base_dir_raw.decode("utf-8")
                        elif base_dir_raw is not None:
                            base_dir = str(base_dir_raw)

                    write_videos_group(dst, videos, base_dir=base_dir)
                    write_tracks_group(
                        dst,
                        existing=existing_tracks,
                        labels=labels,
                        prediction_items=prediction_items,
                    )

                    suggestions_data = labels.suggestions
                    suggestion_items = (
                        list(suggestions_data or []) if suggestions_data is not None else []
                    )
                    if suggestion_items:
                        video_lookup, video_path_lookup = build_video_lookups(
                            videos, project_root=None
                        )
                        sugg_group = dst.require_group("suggestions")
                        write_suggestions_datasets(
                            sugg_group,
                            suggestion_items,
                            video_lookup,
                            video_path_lookup,
                        )
                    elif suggestions_data is not None:
                        video_lookup, video_path_lookup = build_video_lookups(
                            videos, project_root=None
                        )
                        sugg_group = dst.require_group("suggestions")
                        write_suggestions_datasets(
                            sugg_group,
                            [],
                            video_lookup,
                            video_path_lookup,
                        )
                    elif "suggestions" in src:
                        src.copy("suggestions", dst, name="suggestions")

                    metadata_dst = dst.get("project_metadata")
                    if not isinstance(metadata_dst, h5py.Group):
                        if "project_metadata" in dst:
                            del dst["project_metadata"]
                        metadata_dst = dst.create_group("project_metadata")

                    manifest_obj = load_manifest_from_metadata(metadata_dst)
                    project_root = path.parent
                    register_videos(manifest_obj, videos=videos, project_root=project_root)
                    register_archive(manifest_obj, path)
                    metadata_dst.attrs["manifest_json"] = _serialize_json(manifest_obj.to_dict())

                    max_bytes_attr = metadata_dst.attrs.get(
                        "provenance_max_bytes",
                        _DEFAULT_PROVENANCE_MAX_BYTES,
                    )
                    if isinstance(max_bytes_attr, int | float):
                        provenance_max_bytes = int(max_bytes_attr)
                    else:
                        provenance_max_bytes = _DEFAULT_PROVENANCE_MAX_BYTES

                    txn_context: Any = contextlib.nullcontext()
                    if journal:
                        committed_len = 0
                        preds_grp = dst.get("predictions")
                        if isinstance(preds_grp, h5py.Group):
                            val = preds_grp.attrs.get("committed_length", 0)
                            committed_len = int(val)

                        txn_context = _JournalTransaction(
                            dst.require_group("predictions"),
                            old_len=committed_len,
                            new_len=committed_len,
                            operation="labels.save",
                            enabled=True,
                        )

                    with txn_context:
                        write_labels_group(dst, labels)

                        labels_group = dst.get("labels")
                        labels_count = 0
                        max_inst_labels: int | None = None

                        if isinstance(labels_group, h5py.Group):
                            frames_group = labels_group.get("frames")
                            if isinstance(frames_group, h5py.Group):
                                video_index_ds = frames_group.get("video_index")
                                if isinstance(video_index_ds, h5py.Dataset):
                                    labels_count = int(video_index_ds.shape[0])

                            data_group = labels_group.get("data")
                            if isinstance(data_group, h5py.Group):
                                keypoints_ds = data_group.get("keypoints")
                                if isinstance(keypoints_ds, h5py.Dataset):
                                    shape = keypoints_ds.shape
                                    if len(shape) >= 2:
                                        max_inst_labels = int(shape[1])

                        if max_inst_labels is None:
                            existing = metadata_dst.attrs.get("max_inst_labels")
                            max_inst_labels = int(existing)

                        now_iso = _now_utc_iso()
                        metadata_dst.attrs["modified"] = now_iso
                        metadata_dst.attrs["n_labels"] = int(labels_count)
                        if max_inst_labels is not None:
                            metadata_dst.attrs["max_inst_labels"] = int(max_inst_labels)

                        _ensure_journal_attr(metadata_dst)

                        provenance_entry_payload: dict[str, Any] = {"frames": labels_count}
                        if max_inst_labels is not None:
                            provenance_entry_payload["max_instances"] = int(max_inst_labels)

                        _append_provenance(
                            metadata_dst,
                            _default_provenance_entry("labels.save", **provenance_entry_payload),
                            max_bytes=provenance_max_bytes,
                        )

                        dst.flush()

            os.replace(tmp_path, path)
            cleanup_stack.pop_all()

def write_predictions_group(
    file: h5py.File,
    predictions,
    *,
    max_inst_preds: int | None = None,
    committed_length: int | None = None,
    video_index_lookup: Mapping[object, int] | Sequence[object] | None = None,
) -> int:
    """Write predictions data to the `/predictions` group in a native archive."""
    prediction_items = _coerce_prediction_items(
        predictions,
        video_index_lookup=video_index_lookup,
    )
    prediction_items = _normalize_append_batch(list(prediction_items))
    n_rows = len(prediction_items)

    detected_max_inst = max((len(item.instances or []) for item in prediction_items), default=0)
    final_max_inst = int(max_inst_preds) if max_inst_preds is not None else int(detected_max_inst)
    if detected_max_inst > final_max_inst:
        raise ValueError(
            f"Data requires {detected_max_inst} instances but max_inst_preds={final_max_inst}"
        )
    max_inst_val = max(final_max_inst, 1)

    keypoint_count = _infer_prediction_keypoint_count(file, prediction_items)

    expected_heatmap_hw = _infer_batch_heatmap_hw(
        prediction_items,
        keypoint_count=keypoint_count,
        require_all=False,
    )
    if expected_heatmap_hw is not None:
        missing_heatmaps = [item for item in prediction_items if item.heatmaps is None]
        if missing_heatmaps:
            raise ValueError("All prediction entries must include heatmaps when any are present")

    committed = n_rows if committed_length is None else int(committed_length)
    if committed < 0 or committed > n_rows:
        raise ValueError("committed_length must be within [0, N]")

    initial_rows = max(n_rows, 1)
    _preds_group, frames_group, data_group, heatmaps_ds = _bootstrap_predictions_group(
        file,
        max_instances=max_inst_val,
        keypoint_count=keypoint_count,
        initial_length=initial_rows,
        committed_length=committed,
        expected_heatmap_hw=expected_heatmap_hw,
    )

    video_ds = frames_group["video_index"]
    frame_ds = frames_group["frame_index"]
    num_inst_ds = frames_group["num_instances"]

    keypoints_ds = data_group["keypoints"]
    keypoint_score_ds = data_group["keypoint_score"]
    instance_score_ds = data_group["instance_score"]
    track_id_ds = data_group["track_id"]
    deleted_ds = data_group["deleted"]

    if expected_heatmap_hw is not None and isinstance(heatmaps_ds, h5py.Dataset):
        hm_h, hm_w = expected_heatmap_hw
        if n_rows == 0:
            heatmaps_ds.resize((0, keypoint_count, hm_h, hm_w))

    datasets: PredictionDatasetMap = {
        "video_index": video_ds,
        "frame_index": frame_ds,
        "num_instances": num_inst_ds,
        "keypoints": keypoints_ds,
        "keypoint_score": keypoint_score_ds,
        "instance_score": instance_score_ds,
        "track_id": track_id_ds,
        "deleted": deleted_ds,
        "heatmaps": heatmaps_ds,
    }

    if n_rows > 0:
        _fill_prediction_slice(
            datasets,
            prediction_items,
            start=0,
            max_inst=max_inst_val,
            keypoint_count=keypoint_count,
        )
    else:
        video_ds.resize((0,))
        frame_ds.resize((0,))
        num_inst_ds.resize((0,))
        keypoints_ds.resize((0, max_inst_val, keypoint_count, 3))
        keypoint_score_ds.resize((0, max_inst_val, keypoint_count))
        instance_score_ds.resize((0, max_inst_val))
        track_id_ds.resize((0, max_inst_val))
        deleted_ds.resize((0, max_inst_val))

    _assert_prediction_dataset_alignment(datasets, expected_length=n_rows)

    return max_inst_val


def append_run_entry(group: h5py.Group, entry: dict[str, Any]) -> int:
    """Append or upsert a single run entry under `/runs/table` datasets."""
    str_dtype = h5py.string_dtype("utf-8")

    run_ds = group.get("run_id")
    created_ds = group.get("created_ns")
    config_ds = group.get("config_json")

    if not isinstance(run_ds, h5py.Dataset) or not isinstance(created_ds, h5py.Dataset):
        for name in ("run_id", "created_ns", "config_json"):
            if name in group:
                del group[name]
        group.create_dataset(
            "run_id",
            data=np.array([entry["run_id"]], dtype=np.int32),
            dtype=np.int32,
            maxshape=(None,),
            chunks=True,
        )
        group.create_dataset(
            "created_ns",
            data=np.array([entry["created_ns"]], dtype=np.int64),
            dtype=np.int64,
            maxshape=(None,),
            chunks=True,
        )
        group.create_dataset(
            "config_json",
            data=np.array([entry["config_json"]], dtype=object),
            dtype=str_dtype,
            maxshape=(None,),
            chunks=True,
        )
        return 1

    run_ids = np.asarray(run_ds[...], dtype=np.int32)
    existing_idx = np.flatnonzero(run_ids == int(entry["run_id"])).tolist()
    if existing_idx:
        idx = int(existing_idx[0])
        run_ds[idx] = int(entry["run_id"])
        created_ds[idx] = int(entry["created_ns"])
        if isinstance(config_ds, h5py.Dataset):
            config_ds[idx] = str(entry["config_json"])
        return run_ids.shape[0]

    current_size = run_ids.shape[0]
    new_size = current_size + 1

    run_ds.resize((new_size,))
    run_ds[current_size] = int(entry["run_id"])

    created_ds.resize((new_size,))
    created_ds[current_size] = int(entry["created_ns"])

    if not isinstance(config_ds, h5py.Dataset):
        config_ds = group.create_dataset(
            "config_json",
            data=np.array([], dtype=object),
            dtype=str_dtype,
            maxshape=(None,),
            chunks=True,
        )
    config_ds.resize((new_size,))
    config_ds[current_size] = str(entry["config_json"])

    return new_size


def write_runs_table(runs_group: h5py.Group, entries: Sequence[dict[str, Any]]) -> None:
    """Write the full `/runs/table` datasets from normalized run entries."""
    table_group = runs_group.require_group("table")
    for name in list(table_group.keys()):
        del table_group[name]

    run_ids = (
        np.asarray([entry["run_id"] for entry in entries], dtype=np.int32)
        if entries
        else np.zeros((0,), dtype=np.int32)
    )
    created_ns = (
        np.asarray([entry["created_ns"] for entry in entries], dtype=np.int64)
        if entries
        else np.zeros((0,), dtype=np.int64)
    )
    config_values = [str(entry["config_json"]) for entry in entries] if entries else []
    str_dtype = h5py.string_dtype("utf-8")

    table_group.create_dataset(
        "run_id",
        data=run_ids,
        dtype=np.int32,
        maxshape=(None,),
        chunks=True,
    )
    table_group.create_dataset(
        "created_ns",
        data=created_ns,
        dtype=np.int64,
        maxshape=(None,),
        chunks=True,
    )
    if config_values:
        table_group.create_dataset(
            "config_json",
            data=np.array(config_values, dtype=object),
            dtype=str_dtype,
            maxshape=(None,),
            chunks=True,
        )
    else:
        table_group.create_dataset(
            "config_json",
            shape=(0,),
            dtype=str_dtype,
            maxshape=(None,),
            chunks=True,
        )


def build_video_lookups(
    videos: Sequence[Any],
    *,
    project_root: Path | None,
) -> tuple[dict[Any, int], dict[Path, int]]:
    video_lookup = {video: idx for idx, video in enumerate(videos)}
    video_path_lookup: dict[Path, int] = {}
    for video, idx in video_lookup.items():
        video_name = str(video.filename or "").strip()
        if not video_name:
            continue
        if project_root is not None:
            _, resolved_path = resolve_project_path(video_name, project_root=project_root)
            video_path_lookup[resolved_path] = idx
        else:
            video_path_lookup[Path(video_name).resolve()] = idx
    return video_lookup, video_path_lookup


def write_videos_group(
    h5file: h5py.File,
    videos: Sequence[Any],
    *,
    base_dir: str,
) -> None:
    videos_group = h5file.require_group("videos")
    videos_group.attrs["base_dir"] = base_dir

    str_dtype = h5py.string_dtype("utf-8")

    filenames = [str(v.filename or "") for v in videos]
    image_filenames_json = [
        _serialize_json([str(path) for path in (v.image_filenames or [])])
        if getattr(v, "image_filenames", None)
        else ""
        for v in videos
    ]
    backends = [str(v.backend or "opencv") for v in videos]
    sha256_hashes = [str(v.sha256 or "") for v in videos]

    video_ids: list[str] = []
    video_labels: list[str] = []
    for name in filenames:
        vid_id = make_path_id(str(name), prefix="video")
        video_ids.append(vid_id.id)
        video_labels.append(vid_id.label)

    shapes = np.zeros((len(videos), 4), dtype=np.int32)
    for idx, video in enumerate(videos):
        frames = video.frames
        height = video.height
        width = video.width
        channels = video.channels
        shapes[idx] = (frames, height, width, channels)

    videos_group.create_dataset(
        "filenames",
        data=np.array(filenames, dtype=object),
        dtype=str_dtype,
    )
    videos_group.create_dataset(
        "image_filenames_json",
        data=np.array(image_filenames_json, dtype=object),
        dtype=str_dtype,
    )
    videos_group.create_dataset(
        "backends",
        data=np.array(backends, dtype=object),
        dtype=str_dtype,
    )
    videos_group.create_dataset("shapes", data=shapes)
    videos_group.create_dataset(
        "sha256",
        data=np.array(sha256_hashes, dtype=object),
        dtype=str_dtype,
    )
    videos_group.create_dataset(
        "video_ids",
        data=np.array(video_ids, dtype=object),
        dtype=str_dtype,
    )
    videos_group.create_dataset(
        "video_labels",
        data=np.array(video_labels, dtype=object),
        dtype=str_dtype,
    )


def write_suggestions_datasets(
    sugg_group: h5py.Group,
    suggestion_items: list,
    video_lookup: dict,
    video_path_lookup: dict[Path, int],
    project_root: Path | None = None,
) -> None:
    """Write suggestion video_indices, frame_indices, and scores datasets."""
    count = len(suggestion_items)
    video_indices = np.zeros(count, dtype=np.int32)
    frame_indices = np.zeros(count, dtype=np.int32)
    scores = np.zeros(count, dtype=np.float32)
    keep_scores = False

    def _resolve_video_index(candidate: Any) -> int:
        if candidate in video_lookup:
            return int(video_lookup[candidate])
        raw_name = str(candidate.filename or "").strip()
        if not raw_name:
            raise ValueError("Suggestion references a video with no filename")
        if project_root is not None:
            _, key = resolve_project_path(raw_name, project_root=project_root)
        else:
            key = Path(raw_name).resolve()
        if key in video_path_lookup:
            return int(video_path_lookup[key])
        raise ValueError("Suggestion references a video not present in labels.videos")

    for idx, item in enumerate(suggestion_items):
        video_indices[idx] = _resolve_video_index(item.video)
        frame_indices[idx] = int(item.frame_idx)
        if item.score is not None:
            scores[idx] = float(item.score)
            keep_scores = True

    sugg_group.create_dataset("video_indices", data=video_indices)
    sugg_group.create_dataset("frame_indices", data=frame_indices)
    if keep_scores:
        sugg_group.create_dataset("scores", data=scores)


def write_skeleton_group(h5file: h5py.File, labels) -> None:
    """Persist skeleton metadata under the `/skeleton` group."""
    skeleton_group = h5file.require_group("skeleton")
    if "symmetry" in skeleton_group.attrs:
        del skeleton_group.attrs["symmetry"]

    for ds_name in ["names", "links", "roles"]:
        if ds_name in skeleton_group:
            del skeleton_group[ds_name]

    skeletons = list(labels.skeletons or [])
    if len(skeletons) > 1:
        raise ValueError(
            "The native archive schema currently supports exactly one skeleton per archive; "
            f"received {len(skeletons)} skeletons."
        )
    if skeletons:
        skeleton = skeletons[0]
        keypoint_names = [str(kp.name) for kp in skeleton.keypoints]
        keypoint_roles = [str(kp.role or "") for kp in skeleton.keypoints]
        id_to_index = {kp.id: idx for idx, kp in enumerate(skeleton.keypoints)}
        links_idx: list[tuple[int, int]] = []
        for edge in skeleton.links_ids or []:
            if not isinstance(edge, Sequence) or len(edge) != 2:
                continue
            a, b = edge
            ai = None
            bi = None
            if isinstance(a, int | np.integer):
                ai = id_to_index.get(int(a))
            elif isinstance(a, Keypoint):
                ai = id_to_index.get(int(a.id))
            elif isinstance(a, str):
                ai = next(
                    (idx for idx, kp in enumerate(skeleton.keypoints) if kp.name == a),
                    None,
                )
            if isinstance(b, int | np.integer):
                bi = id_to_index.get(int(b))
            elif isinstance(b, Keypoint):
                bi = id_to_index.get(int(b.id))
            elif isinstance(b, str):
                bi = next(
                    (idx for idx, kp in enumerate(skeleton.keypoints) if kp.name == b),
                    None,
                )
            if ai is None or bi is None:
                continue
            links_idx.append((int(ai), int(bi)))

        names_arr = np.array(keypoint_names, dtype=object)
        skeleton_group.create_dataset(
            "names",
            data=names_arr,
            dtype=h5py.string_dtype("utf-8"),
        )

        roles_arr = np.array(keypoint_roles, dtype=object)
        skeleton_group.create_dataset(
            "roles",
            data=roles_arr,
            dtype=h5py.string_dtype("utf-8"),
        )

        links_arr = np.asarray(links_idx, dtype=np.int32)
        if links_arr.size == 0:
            links_arr = np.zeros((0, 2), dtype=np.int32)
        skeleton_group.create_dataset("links", data=links_arr)

        symmetry_map = {
            str(kp.name): str(kp.mirror_partner) for kp in skeleton.keypoints if kp.mirror_partner
        }
        if symmetry_map:
            skeleton_group.attrs["symmetry"] = _serialize_json(symmetry_map)
    else:
        skeleton_group.create_dataset(
            "names",
            data=np.array([], dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        skeleton_group.create_dataset(
            "links",
            data=np.zeros((0, 2), dtype=np.int32),
        )


def write_labels_group(
    file: h5py.File,
    labels,
    *,
    max_inst: int | None = None,
) -> None:
    """Write labels data to the `/labels` group in a native archive."""
    from xpkg.core.annotations import KPFlag

    frames_data = []
    detected_max_inst = 0

    for vi, video in enumerate(labels.videos):
        lfs = labels.query.find(video) or []
        for lf in lfs:
            inst_list = [
                inst for inst in lf.instances if not isinstance(inst, predicted_instance_types())
            ]
            if inst_list:
                frames_data.append((vi, lf.frame_idx, inst_list))
                detected_max_inst = max(detected_max_inst, len(inst_list))

    if max_inst is None:
        max_inst = detected_max_inst
    elif detected_max_inst > max_inst:
        raise ValueError(f"Data requires {detected_max_inst} instances but max_inst={max_inst}")

    n_rows = len(frames_data)

    skeleton = None
    skeleton_keypoints: Sequence[Any] = ()
    if len(labels.skeletons) > 1:
        raise ValueError(
            "The native archive schema currently supports exactly one skeleton per archive; "
            f"received {len(labels.skeletons)} skeletons."
        )
    if labels.skeletons:
        skeleton = labels.skeletons[0]
        if skeleton is None:
            raise ValueError("labels.skeletons[0] is None")
        skeleton_keypoints = list(skeleton.keypoints)
    else:
        skeleton_keypoints = list(labels.keypoints or [])
        if n_rows > 0:
            raise ValueError("Labels contain user instances but no skeleton")

    keypoint_count = len(skeleton_keypoints)

    labels_group = file.require_group("labels")

    frames_group = labels_group.require_group("frames")

    if max_inst is None:
        max_inst = detected_max_inst
    max_inst = max(int(max_inst), 1)

    video_indices = np.zeros(n_rows, dtype=np.int32)
    frame_indices = np.zeros(n_rows, dtype=np.int32)
    num_instances = np.zeros(n_rows, dtype=np.int32)
    keypoints = np.full((n_rows, max_inst, keypoint_count, 3), np.nan, dtype=np.float32)
    visibility = np.zeros((n_rows, max_inst, keypoint_count), dtype=np.uint8)
    flags = np.zeros((n_rows, max_inst, keypoint_count), dtype=np.uint8)
    track_ids = np.full((n_rows, max_inst), -1, dtype=np.int32)

    for idx, (vi, fi, inst_list) in enumerate(frames_data):
        video_indices[idx] = vi
        frame_indices[idx] = fi
        num_instances[idx] = len(inst_list)

        for ii, inst in enumerate(inst_list):
            if inst.track is not None:
                track_ids[idx, ii] = int(inst.track.id)

            pts = inst.get_points_array(copy=False, full=True)

            keypoints[idx, ii, :, 0] = pts["x"]
            keypoints[idx, ii, :, 1] = pts["y"]

            visibility[idx, ii, :] = np.asarray(pts["visible"], dtype=np.uint8)

            flags[idx, ii, :] = pts["flags"] & KPFlag.NO_TRAIN

    frames_group.create_dataset("video_index", data=video_indices)
    frames_group.create_dataset("frame_index", data=frame_indices)
    frames_group.create_dataset("num_instances", data=num_instances)

    data_group = labels_group.require_group("data")

    use_chunking = (n_rows > 0) and (keypoint_count > 0)

    if use_chunking:
        chunk_rows = min(128, n_rows)
        chunk_k = keypoint_count
        maxshape_k = keypoint_count if skeleton is not None else None

        keypoints_ds = data_group.create_dataset(
            "keypoints",
            shape=(n_rows, max_inst, keypoint_count, 3),
            dtype=np.float32,
            chunks=(chunk_rows, max_inst, chunk_k, 3),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            maxshape=(None, max_inst, maxshape_k, 3),
        )
        flags_ds = data_group.create_dataset(
            "flags",
            shape=(n_rows, max_inst, keypoint_count),
            dtype=np.uint8,
            chunks=(chunk_rows, max_inst, chunk_k),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            maxshape=(None, max_inst, maxshape_k),
        )
        visibility_ds = data_group.create_dataset(
            LABEL_VISIBILITY_DATASET,
            shape=(n_rows, max_inst, keypoint_count),
            dtype=np.uint8,
            chunks=(chunk_rows, max_inst, chunk_k),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            maxshape=(None, max_inst, maxshape_k),
        )
        track_ids_ds = data_group.create_dataset(
            LABEL_TRACK_ID_DATASET,
            shape=(n_rows, max_inst),
            dtype=np.int32,
            chunks=(min(128, n_rows), max_inst),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            maxshape=(None, max_inst),
        )
    else:
        keypoints_ds = data_group.create_dataset(
            "keypoints",
            shape=(n_rows, max_inst, keypoint_count, 3),
            dtype=np.float32,
        )
        flags_ds = data_group.create_dataset(
            "flags",
            shape=(n_rows, max_inst, keypoint_count),
            dtype=np.uint8,
        )
        visibility_ds = data_group.create_dataset(
            LABEL_VISIBILITY_DATASET,
            shape=(n_rows, max_inst, keypoint_count),
            dtype=np.uint8,
        )
        track_ids_ds = data_group.create_dataset(
            LABEL_TRACK_ID_DATASET,
            shape=(n_rows, max_inst),
            dtype=np.int32,
        )

    if n_rows > 0:
        if keypoint_count > 0:
            keypoints_ds[...] = keypoints
            visibility_ds[...] = visibility
            flags_ds[...] = flags
        track_ids_ds[...] = track_ids


__all__ = [
    "append_run_entry",
    "build_video_lookups",
    "coerce_manifest",
    "update_labels_archive",
    "write_labels_group",
    "write_predictions_group",
    "write_runs_table",
    "write_archive",
    "write_skeleton_group",
    "write_suggestions_datasets",
    "write_videos_group",
]
