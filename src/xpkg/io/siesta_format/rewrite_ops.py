"""Prediction rewrite helpers for native archives.

The rewrite path was extracted from append_ops because it is a large, independently
testable operation family (>200 LOC) that rewrites archive layout/state rather than
performing in-place append/merge updates.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from xpkg.io.siesta_format.predictions_datasets import (
    MaxInstancesExceededError,
    PredictionAppendItem,
    PredictionDatasetMap,
    _assert_prediction_dataset_alignment,
    _bootstrap_predictions_group,
    _copy_dataset_rows,
    _create_heatmaps_dataset,
    _create_predictions_datasets,
    _existing_instances_for_index,
    _fill_prediction_slice,
    _infer_batch_heatmap_hw,
    _instance_keypoint_length,
    _normalize_heatmaps_frame,
)
from xpkg.io.siesta_format.shared import (
    _DEFAULT_PROVENANCE_MAX_BYTES,
    _coerce_int,
    _default_provenance_entry,
    _normalize_predictions_committed_length,
    _now_utc_iso,
    _require_project_metadata_group,
    _skeleton_keypoint_count,
)
from xpkg.io.siesta_format.tracks_hdf5 import read_tracks_group, write_tracks_group
from xpkg.io.siesta_format.transaction import (
    _append_provenance,
    _flush_file,
    _JournalTransaction,
)
from xpkg.io.siesta_format.writer_core import append_run_entry


def _prediction_batch_keypoint_count(batch: Sequence[PredictionAppendItem]) -> int:
    keypoint_count = 0
    for item in batch:
        for inst in item.instances or []:
            keypoint_count = max(keypoint_count, _instance_keypoint_length(inst))
        if item.heatmaps is not None:
            heatmaps = np.asarray(item.heatmaps)
            if heatmaps.ndim == 3:
                keypoint_count = max(keypoint_count, int(heatmaps.shape[0]))
    return keypoint_count


def _optional_prediction_datasets(
    data_group: h5py.Group,
) -> tuple[
    h5py.Dataset | None,
    h5py.Dataset | None,
    h5py.Dataset | None,
    h5py.Dataset | None,
]:
    keypoint_score_ds = data_group.get("keypoint_score")
    if keypoint_score_ds is not None and not isinstance(keypoint_score_ds, h5py.Dataset):
        raise TypeError("Predictions keypoint_score must be an h5py Dataset")
    instance_score_ds = data_group.get("instance_score")
    if instance_score_ds is not None and not isinstance(instance_score_ds, h5py.Dataset):
        raise TypeError("Predictions instance_score must be an h5py Dataset")
    track_id_ds = data_group.get("track_id")
    if track_id_ds is not None and not isinstance(track_id_ds, h5py.Dataset):
        raise TypeError("Predictions track_id must be an h5py Dataset")
    deleted_ds = data_group.get("deleted")
    if deleted_ds is not None and not isinstance(deleted_ds, h5py.Dataset):
        raise TypeError("Predictions deleted must be an h5py Dataset")
    return keypoint_score_ds, instance_score_ds, track_id_ds, deleted_ds


def _require_predictions_groups(
    src_file: h5py.File,
    *,
    predictions_type_error: str = "Predictions entry must be an h5py Group",
    frames_group_type_error: str = "Predictions frames/data must be h5py Groups",
) -> tuple[h5py.Group, h5py.Group, h5py.Group]:
    preds_group = src_file.get("predictions")
    if preds_group is None:
        raise ValueError("archive is missing the /predictions group")
    if not isinstance(preds_group, h5py.Group):
        raise TypeError(predictions_type_error)

    frames_group = preds_group.get("frames")
    data_group = preds_group.get("data")
    if frames_group is None or data_group is None:
        raise ValueError("archive is missing predictions frames/data groups")
    if not isinstance(frames_group, h5py.Group) or not isinstance(data_group, h5py.Group):
        raise TypeError(frames_group_type_error)
    return preds_group, frames_group, data_group


def _require_predictions_keypoints_dataset(
    data_group: h5py.Group,
    *,
    keypoints_type_error: str = "Predictions keypoints must be an h5py Dataset",
) -> h5py.Dataset:
    keypoints_ds = data_group.get("keypoints")
    if keypoints_ds is None:
        raise ValueError("archive is missing predictions/data/keypoints dataset")
    if not isinstance(keypoints_ds, h5py.Dataset):
        raise TypeError(keypoints_type_error)
    return keypoints_ds


def _rewrite_with_larger_max(
    path: Path | str,
    batch: Sequence[PredictionAppendItem],
    new_max_inst: int,
    *,
    journal: bool = True,
) -> int:
    """Slow-path rewrite when batch exceeds max instances; rewrites file with larger capacity."""

    if not isinstance(path, Path):
        path = Path(path)

    batch_list = list(batch)
    if not batch_list:
        return 0

    new_max_inst = max(int(new_max_inst), 1)

    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f"{path.name}.rewrite-",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(temp_fd)
    temp_path = Path(temp_name)

    with contextlib.ExitStack() as cleanup_stack:
        cleanup_stack.callback(
            lambda temp_file=temp_path: temp_file.exists() and temp_file.unlink()
        )
        with h5py.File(str(path), "r") as src_file:
            meta_group = _require_project_metadata_group(src_file)

            preds_group, frames_group, data_group = _require_predictions_groups(
                src_file,
                frames_group_type_error="Expected Groups for frames/data",
            )
            keypoints_src = _require_predictions_keypoints_dataset(
                data_group,
                keypoints_type_error="Expected Dataset for keypoints",
            )

            total_rows = int(keypoints_src.shape[0]) if keypoints_src.shape else 0
            old_max_inst = int(keypoints_src.shape[1]) if keypoints_src.ndim > 1 else 0
            keypoint_count = int(keypoints_src.shape[2]) if keypoints_src.ndim > 2 else 0
            value_channels = int(keypoints_src.shape[3]) if keypoints_src.ndim > 3 else 0
            if value_channels not in (0, 3):
                raise ValueError("Unexpected keypoints dataset layout; expected last dim=3")
            if value_channels <= 0:
                value_channels = 3

            batch_keypoint_count = _prediction_batch_keypoint_count(batch_list)

            if batch_keypoint_count > keypoint_count:
                skeleton_kp_count = _skeleton_keypoint_count(src_file, default=0)
                if batch_keypoint_count <= skeleton_kp_count:
                    keypoint_count = batch_keypoint_count

            if keypoint_count <= 0:
                keypoint_count = max(batch_keypoint_count, 1)

            committed_length = _normalize_predictions_committed_length(
                preds_group,
                total_rows=total_rows,
            )

            incoming_frames = len(batch_list)
            total_frames = committed_length + incoming_frames

            for item in batch_list:
                inst_count = len(item.instances or [])
                if inst_count > new_max_inst:
                    raise MaxInstancesExceededError(
                        f"Frame {item.frame_index} requires {inst_count} instances "
                        f"(> {new_max_inst})"
                    )

            heatmaps_src = data_group.get("heatmaps")
            heatmaps_hw: tuple[int, int] | None = None
            if isinstance(heatmaps_src, h5py.Dataset):
                if heatmaps_src.ndim != 4:
                    raise ValueError("heatmaps dataset must have rank 4 (N,K,H,W)")
                if int(heatmaps_src.shape[1]) == int(keypoint_count):
                    heatmaps_hw = (int(heatmaps_src.shape[2]), int(heatmaps_src.shape[3]))
            if heatmaps_hw is None:
                heatmaps_hw = _infer_batch_heatmap_hw(
                    batch_list,
                    keypoint_count=keypoint_count,
                    require_all=False,
                )

            provenance_max_bytes = _coerce_int(
                meta_group.attrs.get("provenance_max_bytes"), default=_DEFAULT_PROVENANCE_MAX_BYTES
            )

            with h5py.File(str(temp_path), "w") as dst_file:
                for name, obj in src_file.items():
                    if name == "predictions":
                        continue
                    src_file.copy(obj, dst_file, name=name)

                preds_dst = dst_file.create_group("predictions")
                frame_templates = {
                    name: dataset
                    for name in ("video_index", "frame_index", "num_instances")
                    if isinstance((dataset := frames_group.get(name)), h5py.Dataset)
                }
                data_templates = {
                    name: dataset
                    for name in (
                        "keypoints",
                        "keypoint_score",
                        "instance_score",
                        "track_id",
                        "deleted",
                    )
                    if isinstance((dataset := data_group.get(name)), h5py.Dataset)
                }
                frames_dst, data_dst = _create_predictions_datasets(
                    preds_dst,
                    new_max_inst,
                    keypoint_count,
                    total_frames,
                    frame_templates=frame_templates,
                    data_templates=data_templates,
                )

                for name in ("video_index", "frame_index", "num_instances"):
                    src_ds = frames_group.get(name)
                    if not isinstance(src_ds, h5py.Dataset):
                        raise TypeError(f"Expected Dataset for predictions/frames/{name}")
                    _copy_dataset_rows(src_ds, frames_dst[name], committed_length)

                _copy_dataset_rows(keypoints_src, data_dst["keypoints"], committed_length)
                for name in ("keypoint_score", "instance_score", "track_id", "deleted"):
                    src_ds = data_group.get(name)
                    if isinstance(src_ds, h5py.Dataset):
                        _copy_dataset_rows(src_ds, data_dst[name], committed_length)

                heatmaps_dst: h5py.Dataset | None = None
                if heatmaps_hw is not None:
                    hm_h, hm_w = heatmaps_hw
                    heatmaps_dst = _create_heatmaps_dataset(
                        data_dst,
                        initial_length=total_frames,
                        keypoint_count=keypoint_count,
                        height=hm_h,
                        width=hm_w,
                        template=heatmaps_src if isinstance(heatmaps_src, h5py.Dataset) else None,
                    )
                    if isinstance(heatmaps_src, h5py.Dataset):
                        _copy_dataset_rows(heatmaps_src, heatmaps_dst, committed_length)

                datasets: PredictionDatasetMap = {
                    "video_index": frames_dst["video_index"],
                    "frame_index": frames_dst["frame_index"],
                    "num_instances": frames_dst["num_instances"],
                    "keypoints": data_dst["keypoints"],
                    "keypoint_score": data_dst["keypoint_score"],
                    "instance_score": data_dst["instance_score"],
                    "track_id": data_dst["track_id"],
                    "deleted": data_dst["deleted"],
                    "heatmaps": heatmaps_dst,
                }

                with _JournalTransaction(
                    preds_dst,
                    old_len=committed_length,
                    new_len=total_frames,
                    operation="predictions.rewrite",
                    enabled=journal,
                ):
                    _fill_prediction_slice(
                        datasets,
                        batch_list,
                        start=committed_length,
                        max_inst=new_max_inst,
                        keypoint_count=keypoint_count,
                    )

                    _assert_prediction_dataset_alignment(datasets, expected_length=total_frames)
                    preds_dst.attrs["committed_length"] = int(total_frames)
                    meta_dst = _require_project_metadata_group(dst_file)
                    meta_dst.attrs["modified"] = _now_utc_iso()
                    meta_dst.attrs["n_predictions_committed"] = int(total_frames)
                    meta_dst.attrs["max_inst_preds"] = int(new_max_inst)
                    write_tracks_group(
                        dst_file,
                        existing=read_tracks_group(dst_file),
                        prediction_items=batch_list,
                    )
                    meta_dst.attrs["provenance_max_bytes"] = int(
                        provenance_max_bytes or _DEFAULT_PROVENANCE_MAX_BYTES
                    )
                    _append_provenance(
                        meta_dst,
                        _default_provenance_entry(
                            "predictions.rewrite",
                            frames=len(batch_list),
                            committed=total_frames,
                            old_max=old_max_inst,
                            new_max=new_max_inst,
                        ),
                        max_bytes=int(provenance_max_bytes or _DEFAULT_PROVENANCE_MAX_BYTES),
                    )

                    _flush_file(dst_file, fsync=True)

        os.replace(temp_path, path)
        cleanup_stack.pop_all()

    return len(batch_list)


def _create_empty_predictions_group(
    h5file: h5py.File,
    batch_list: list[PredictionAppendItem],
    max_instances: int,
) -> h5py.Group:
    """Create an empty predictions group structure for first-time append.

    This is called when appending to a file that doesn't have a predictions group yet
    (e.g., after the predictions were deleted or on a newly created file).
    """

    keypoint_count = 0
    for item in batch_list:
        for inst in item.instances or []:
            keypoint_count = max(keypoint_count, _instance_keypoint_length(inst))

    if keypoint_count == 0:
        skeleton_grp = h5file.get("skeleton")
        if skeleton_grp is not None and isinstance(skeleton_grp, h5py.Group):
            names_ds = skeleton_grp.get("names")
            if names_ds is not None and isinstance(names_ds, h5py.Dataset):
                keypoint_count = int(names_ds.shape[0]) if names_ds.shape else 31

    keypoint_count = max(keypoint_count, 1)
    max_inst = max(max_instances, 1)

    expected_hw = _infer_batch_heatmap_hw(
        batch_list,
        keypoint_count=keypoint_count,
        require_all=False,
    )
    preds_group, _frames_group, _data_group, _heatmaps_ds = _bootstrap_predictions_group(
        h5file,
        max_instances=max_inst,
        keypoint_count=keypoint_count,
        initial_length=0,
        committed_length=0,
        expected_heatmap_hw=expected_hw,
    )
    return preds_group


def _rewrite_predictions_with_updates(
    path: Path,
    updates: dict[tuple[int, int], PredictionAppendItem],
    *,
    keypoint_count: int,
    new_max_inst: int,
    committed_length: int,
    journal: bool,
    run_entry: dict[str, Any] | None,
    fsync: bool,
) -> int:
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f"{path.name}.rewrite-",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(temp_fd)
    temp_path = Path(temp_name)

    with contextlib.ExitStack() as cleanup_stack:
        cleanup_stack.callback(
            lambda temp_file=temp_path: temp_file.exists() and temp_file.unlink()
        )
        with h5py.File(str(path), "r") as src_file:
            meta_group = _require_project_metadata_group(src_file)

            _preds_group, frames_group, data_group = _require_predictions_groups(src_file)
            keypoints_ds = _require_predictions_keypoints_dataset(data_group)
            old_max_inst = int(keypoints_ds.shape[1]) if keypoints_ds.ndim > 1 else 0

            video_idx_ds = frames_group.get("video_index")
            frame_idx_ds = frames_group.get("frame_index")
            num_inst_ds = frames_group.get("num_instances")
            if video_idx_ds is None or frame_idx_ds is None:
                raise ValueError("Predictions frames datasets missing required entries")
            if not isinstance(video_idx_ds, h5py.Dataset) or not isinstance(
                frame_idx_ds, h5py.Dataset
            ):
                raise TypeError("Predictions frames datasets must be h5py Dataset")
            if num_inst_ds is None:
                raise ValueError("Predictions num_instances dataset missing")
            if not isinstance(num_inst_ds, h5py.Dataset):
                raise TypeError("Predictions num_instances dataset must be h5py Dataset")

            video_indices = np.asarray(video_idx_ds[:committed_length], dtype=np.int32)
            frame_indices = np.asarray(frame_idx_ds[:committed_length], dtype=np.int32)

            heatmaps_src = data_group.get("heatmaps")
            if heatmaps_src is not None and not isinstance(heatmaps_src, h5py.Dataset):
                raise TypeError("Predictions heatmaps must be an h5py Dataset")
            heatmaps_hw: tuple[int, int] | None = None
            if isinstance(heatmaps_src, h5py.Dataset):
                if heatmaps_src.ndim != 4:
                    raise ValueError("heatmaps dataset must have rank 4 (N,K,H,W)")
                if int(heatmaps_src.shape[1]) != int(keypoint_count):
                    raise ValueError("heatmaps dataset K does not match predictions keypoints K")
                heatmaps_hw = (int(heatmaps_src.shape[2]), int(heatmaps_src.shape[3]))
            else:
                expected_hw: tuple[int, int] | None = None
                for update_item in updates.values():
                    hm = update_item.heatmaps
                    if hm is None:
                        continue
                    _, hw = _normalize_heatmaps_frame(
                        hm,
                        keypoint_count=int(keypoint_count),
                        expected_hw=expected_hw,
                        frame_index=int(update_item.frame_index),
                    )
                    expected_hw = hw
                heatmaps_hw = expected_hw

            keypoint_score_src, instance_score_src, track_id_src, deleted_src = (
                _optional_prediction_datasets(data_group)
            )

            read_datasets: PredictionDatasetMap = {
                "video_index": video_idx_ds,
                "frame_index": frame_idx_ds,
                "num_instances": num_inst_ds,
                "keypoints": keypoints_ds,
                "keypoint_score": keypoint_score_src,
                "instance_score": instance_score_src,
                "track_id": track_id_src,
                "deleted": deleted_src,
                "heatmaps": heatmaps_src,
            }

            all_items: list[PredictionAppendItem] = []
            for idx in range(committed_length):
                key = (int(video_indices[idx]), int(frame_indices[idx]))
                base_instances = _existing_instances_for_index(idx, read_datasets, keypoint_count)
                base_heatmaps = None
                if isinstance(heatmaps_src, h5py.Dataset):
                    base_heatmaps = np.asarray(heatmaps_src[idx, :, :, :], dtype=np.float16)
                update_item = updates.get(key)
                merged_instances = (
                    base_instances
                    if update_item is None
                    else base_instances + list(update_item.instances or [])
                )
                merged_heatmaps = base_heatmaps
                if update_item is not None and update_item.heatmaps is not None:
                    merged_heatmaps, _ = _normalize_heatmaps_frame(
                        update_item.heatmaps,
                        keypoint_count=int(keypoint_count),
                        expected_hw=heatmaps_hw,
                        frame_index=int(update_item.frame_index),
                    )
                if heatmaps_hw is not None and merged_heatmaps is None:
                    hm_h, hm_w = heatmaps_hw
                    merged_heatmaps = np.zeros((keypoint_count, hm_h, hm_w), dtype=np.float16)
                all_items.append(
                    PredictionAppendItem(
                        video_index=int(video_indices[idx]),
                        frame_index=int(frame_indices[idx]),
                        instances=merged_instances,
                        detections=update_item.detections if update_item else None,
                        tracks=update_item.tracks if update_item else None,
                        heatmaps=merged_heatmaps,
                    )
                )

            rewrite_max_inst = max(
                new_max_inst,
                max((len(item.instances or []) for item in all_items), default=1),
            )
            provenance_max_bytes = _coerce_int(
                meta_group.attrs.get("provenance_max_bytes"), default=_DEFAULT_PROVENANCE_MAX_BYTES
            )

            with h5py.File(str(temp_path), "w") as dst_file:
                for name, obj in src_file.items():
                    if name == "predictions":
                        continue
                    src_file.copy(obj, dst_file, name=name)

                preds_dst = dst_file.create_group("predictions")
                frame_templates = {
                    name: dataset
                    for name in ("video_index", "frame_index", "num_instances")
                    if isinstance((dataset := frames_group.get(name)), h5py.Dataset)
                }
                data_templates = {
                    name: dataset
                    for name in (
                        "keypoints",
                        "keypoint_score",
                        "instance_score",
                        "track_id",
                        "deleted",
                    )
                    if isinstance((dataset := data_group.get(name)), h5py.Dataset)
                }
                frames_dst, data_dst = _create_predictions_datasets(
                    preds_dst,
                    rewrite_max_inst,
                    keypoint_count,
                    committed_length,
                    frame_templates=frame_templates,
                    data_templates=data_templates,
                )

                video_dst = frames_dst["video_index"]
                frame_dst = frames_dst["frame_index"]
                num_inst_dst = frames_dst["num_instances"]

                keypoints_dst = data_dst["keypoints"]
                keypoint_score_dst = data_dst["keypoint_score"]
                instance_score_dst = data_dst["instance_score"]
                track_id_dst = data_dst["track_id"]
                deleted_dst = data_dst["deleted"]
                heatmaps_dst = None
                if heatmaps_hw is not None:
                    hm_h, hm_w = heatmaps_hw
                    heatmaps_dst = _create_heatmaps_dataset(
                        data_dst,
                        initial_length=committed_length,
                        keypoint_count=keypoint_count,
                        height=hm_h,
                        width=hm_w,
                        template=heatmaps_src if isinstance(heatmaps_src, h5py.Dataset) else None,
                    )

                datasets: PredictionDatasetMap = {
                    "video_index": video_dst,
                    "frame_index": frame_dst,
                    "num_instances": num_inst_dst,
                    "keypoints": keypoints_dst,
                    "keypoint_score": keypoint_score_dst,
                    "instance_score": instance_score_dst,
                    "track_id": track_id_dst,
                    "deleted": deleted_dst,
                    "heatmaps": heatmaps_dst,
                }

                with _JournalTransaction(
                    preds_dst,
                    old_len=committed_length,
                    new_len=committed_length,
                    operation="predictions.rewrite",
                    commit_length=committed_length,
                    enabled=journal,
                ):
                    _fill_prediction_slice(
                        datasets,
                        all_items,
                        start=0,
                        max_inst=rewrite_max_inst,
                        keypoint_count=keypoint_count,
                    )

                    _assert_prediction_dataset_alignment(datasets, expected_length=committed_length)
                    preds_dst.attrs["committed_length"] = int(committed_length)
                    meta_dst = _require_project_metadata_group(dst_file)
                    meta_dst.attrs["modified"] = _now_utc_iso()
                    meta_dst.attrs["n_predictions_committed"] = int(committed_length)
                    meta_dst.attrs["max_inst_preds"] = int(rewrite_max_inst)
                    write_tracks_group(
                        dst_file,
                        existing=read_tracks_group(dst_file),
                        prediction_items=all_items,
                    )
                    meta_dst.attrs["provenance_max_bytes"] = int(
                        provenance_max_bytes or _DEFAULT_PROVENANCE_MAX_BYTES
                    )

                    runs_count: int | None = None
                    if run_entry is not None:
                        runs_group = dst_file.require_group("runs")
                        table_group = runs_group.require_group("table")
                        runs_count = append_run_entry(table_group, run_entry)
                        meta_dst.attrs["runs_count"] = int(runs_count)

                    _append_provenance(
                        meta_dst,
                        _default_provenance_entry(
                            "predictions.rewrite",
                            frames=len(updates),
                            committed=committed_length,
                            old_max=old_max_inst,
                            new_max=rewrite_max_inst,
                        ),
                        max_bytes=int(provenance_max_bytes or _DEFAULT_PROVENANCE_MAX_BYTES),
                    )

                    _flush_file(dst_file, fsync=fsync)

        os.replace(temp_path, path)
        cleanup_stack.pop_all()

    return len(updates)


__all__ = [
    "_create_empty_predictions_group",
    "_optional_prediction_datasets",
    "_prediction_batch_keypoint_count",
    "_require_predictions_groups",
    "_require_predictions_keypoints_dataset",
    "_rewrite_predictions_with_updates",
    "_rewrite_with_larger_max",
]
