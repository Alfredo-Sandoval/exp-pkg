"""Prediction append/merge helpers for native archives.

Rewrite operations were extracted to ``rewrite_ops.py`` because they are a large,
independently testable family with separate rewrite semantics from in-place
append/merge updates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from xpkg.io.archive_format.predictions_datasets import (
    MaxInstancesExceededError,
    PredictionAppendItem,
    PredictionDatasetMap,
    _assert_prediction_dataset_alignment,
    _create_heatmaps_dataset,
    _existing_instances_for_index,
    _fill_prediction_slice,
    _infer_batch_heatmap_hw,
    _normalize_append_batch,
    _resize_prediction_datasets,
)
from xpkg.io.archive_format.rewrite_ops import (
    _create_empty_predictions_group,
    _optional_prediction_datasets,
    _prediction_batch_keypoint_count,
    _require_predictions_groups,
    _require_predictions_keypoints_dataset,
    _rewrite_predictions_with_updates,
    _rewrite_with_larger_max,
)
from xpkg.io.archive_format.shared import (
    _DEFAULT_PROVENANCE_MAX_BYTES,
    _default_provenance_entry,
    _normalize_predictions_committed_length,
    _normalize_run_entry,
    _now_utc_iso,
    _require_project_metadata_group,
    _skeleton_keypoint_count,
)
from xpkg.io.archive_format.tracks_hdf5 import read_tracks_group, write_tracks_group
from xpkg.io.archive_format.transaction import (
    ArchiveFileLock,
    _append_provenance,
    _flush_file,
    _JournalTransaction,
)
from xpkg.io.archive_format.writer_core import append_run_entry


def append_predictions_archive(
    path: Path,
    batch: Sequence[PredictionAppendItem],
    *,
    allow_max_inst_growth: bool = False,
    journal: bool = True,
    fsync: bool = True,
    run_metadata: Mapping[str, Any] | None = None,
) -> int:
    """Append prediction batches to an existing native archive.

    Args:
        path: Path to the native archive file.
        batch: Sequence of PredictionAppendItem to append.
        allow_max_inst_growth: If True, allow the file to be rewritten with a larger
            max_instances if the batch exceeds the current limit.
        journal: If True, use a journal for atomic updates.
        fsync: If True, fsync the file after writing.
        run_metadata: Optional metadata for this specific run.

    Returns:
        int: Number of frames successfully appended.
    """

    batch_list = _normalize_append_batch(list(batch))
    if not batch_list:
        return 0

    run_entry: dict[str, Any] | None = None
    if run_metadata is not None:
        run_entry = _normalize_run_entry(run_metadata)

    batch_max_instances = max(len(item.instances or []) for item in batch_list)

    with ArchiveFileLock(path):
        with h5py.File(str(path), mode="r+") as h5file:
            meta_group = _require_project_metadata_group(h5file)

            preds_group = h5file.get("predictions")
            if preds_group is None:
                preds_group = _create_empty_predictions_group(
                    h5file, batch_list, batch_max_instances
                )
            if not isinstance(preds_group, h5py.Group):
                raise TypeError("Predictions entry must be an h5py Group")

            frames_group = preds_group.get("frames")
            data_group = preds_group.get("data")
            if frames_group is None or data_group is None:
                raise ValueError("archive is missing predictions frames/data groups")
            if not isinstance(frames_group, h5py.Group) or not isinstance(data_group, h5py.Group):
                raise TypeError("Predictions frames/data must be h5py Groups")

            keypoints_ds = data_group.get("keypoints")
            if keypoints_ds is None:
                raise ValueError("archive is missing predictions/data/keypoints dataset")
            if not isinstance(keypoints_ds, h5py.Dataset):
                raise TypeError("Predictions keypoints must be an h5py Dataset")

            max_inst = (
                int(keypoints_ds.shape[1])
                if keypoints_ds.shape and len(keypoints_ds.shape) > 1
                else 0
            )
            keypoint_count = (
                int(keypoints_ds.shape[2])
                if keypoints_ds.shape and len(keypoints_ds.shape) > 2
                else 0
            )

            batch_keypoint_count = _prediction_batch_keypoint_count(batch_list)

            if batch_keypoint_count > keypoint_count:
                skeleton_kp_count = _skeleton_keypoint_count(h5file, default=0)
                if skeleton_kp_count > 0 and batch_keypoint_count <= skeleton_kp_count:
                    return _rewrite_with_larger_max(
                        path,
                        batch_list,
                        max(batch_max_instances, max_inst),
                        journal=journal,
                    )
            total_length = int(keypoints_ds.shape[0]) if keypoints_ds.shape else 0

            committed_length = _normalize_predictions_committed_length(
                preds_group,
                total_rows=total_length,
            )

            if batch_max_instances > max_inst:
                if not allow_max_inst_growth:
                    raise MaxInstancesExceededError(
                        "Batch requires more instances than the file currently stores. "
                        "Re-run with allow_max_inst_growth=True to trigger rewrite."
                    )
                return _rewrite_with_larger_max(
                    path,
                    batch_list,
                    batch_max_instances,
                    journal=journal,
                )

            expected_heatmap_hw = _infer_batch_heatmap_hw(
                batch_list,
                keypoint_count=keypoint_count,
                require_all=False,
            )
            heatmaps_ds = data_group.get("heatmaps")
            if heatmaps_ds is not None and not isinstance(heatmaps_ds, h5py.Dataset):
                raise TypeError("Predictions heatmaps must be an h5py Dataset")
            if isinstance(heatmaps_ds, h5py.Dataset) and expected_heatmap_hw is None:
                raise ValueError(
                    "Prediction archive already contains heatmaps but append batch "
                    "did not supply heatmaps"
                )
            if expected_heatmap_hw is not None:
                hm_h, hm_w = expected_heatmap_hw
                if isinstance(heatmaps_ds, h5py.Dataset):
                    if heatmaps_ds.ndim != 4:
                        raise ValueError("heatmaps dataset must have rank 4 (N,K,H,W)")
                    if int(heatmaps_ds.shape[1]) != int(keypoint_count):
                        raise ValueError(
                            "heatmaps dataset K does not match predictions keypoints K"
                        )
                    if (int(heatmaps_ds.shape[2]), int(heatmaps_ds.shape[3])) != (hm_h, hm_w):
                        raise ValueError(
                            "heatmaps dataset spatial dims do not match incoming batch heatmaps"
                        )
                else:
                    initial_rows = max(total_length, 1)
                    heatmaps_ds = _create_heatmaps_dataset(
                        data_group,
                        initial_length=initial_rows,
                        keypoint_count=keypoint_count,
                        height=hm_h,
                        width=hm_w,
                    )
                    if total_length == 0:
                        heatmaps_ds.resize((0, keypoint_count, hm_h, hm_w))

            video_ds = frames_group.get("video_index")
            frame_ds = frames_group.get("frame_index")
            num_inst_ds = frames_group.get("num_instances")
            if video_ds is None or frame_ds is None or num_inst_ds is None:
                missing = [
                    name
                    for name, ds in (
                        ("video_index", video_ds),
                        ("frame_index", frame_ds),
                        ("num_instances", num_inst_ds),
                    )
                    if ds is None
                ]
                raise ValueError(
                    "Predictions frames datasets missing required entries: " + ", ".join(missing)
                )
            if (
                not isinstance(video_ds, h5py.Dataset)
                or not isinstance(frame_ds, h5py.Dataset)
                or not isinstance(num_inst_ds, h5py.Dataset)
            ):
                raise TypeError("Predictions frames datasets must be h5py Dataset")

            keypoint_score_ds, instance_score_ds, track_id_ds, deleted_ds = (
                _optional_prediction_datasets(data_group)
            )

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

            new_committed = committed_length + len(batch_list)

            def _is_length_capped(ds: h5py.Dataset | None) -> bool:
                if ds is None:
                    return False
                if ds.maxshape is None:
                    return new_committed > int(ds.shape[0] if ds.shape else 0)
                if len(ds.maxshape) > 0 and ds.maxshape[0] is not None:
                    return int(ds.maxshape[0]) < new_committed
                return False

            dataset_values: tuple[h5py.Dataset | None, ...] = (
                datasets["video_index"],
                datasets["frame_index"],
                datasets["num_instances"],
                datasets["keypoints"],
                datasets["keypoint_score"],
                datasets["instance_score"],
                datasets["track_id"],
                datasets["deleted"],
                datasets["heatmaps"],
            )
            length_capped = any(_is_length_capped(ds) for ds in dataset_values)
            if length_capped:
                return _rewrite_with_larger_max(
                    path,
                    batch_list,
                    max_inst,
                    journal=journal,
                )
            if new_committed > total_length:
                _resize_prediction_datasets(datasets, new_committed)

            with _JournalTransaction(
                preds_group,
                old_len=committed_length,
                new_len=new_committed,
                operation="predictions.append",
                commit_length=new_committed,
                enabled=journal,
            ):
                _fill_prediction_slice(
                    datasets,
                    batch_list,
                    start=committed_length,
                    max_inst=max_inst,
                    keypoint_count=keypoint_count,
                )

                _assert_prediction_dataset_alignment(datasets, expected_length=new_committed)
                preds_group.attrs["committed_length"] = int(new_committed)

                runs_count: int | None = None
                if run_entry is not None:
                    runs_group = h5file.require_group("runs")
                    table_group = runs_group.require_group("table")
                    runs_count = append_run_entry(table_group, run_entry)

                now_iso = _now_utc_iso()
                meta_group.attrs["modified"] = now_iso
                meta_group.attrs["n_predictions_committed"] = int(new_committed)
                meta_group.attrs["max_inst_preds"] = int(max_inst)
                write_tracks_group(
                    h5file,
                    existing=read_tracks_group(h5file),
                    prediction_items=batch_list,
                )
                if runs_count is not None:
                    meta_group.attrs["runs_count"] = int(runs_count)

                max_bytes = int(
                    meta_group.attrs.get("provenance_max_bytes", _DEFAULT_PROVENANCE_MAX_BYTES)
                )
                _append_provenance(
                    meta_group,
                    _default_provenance_entry(
                        "predictions.append",
                        frames=len(batch_list),
                        committed=new_committed,
                    ),
                    max_bytes=max_bytes,
                )

                _flush_file(h5file, fsync=fsync)

    return len(batch_list)

def merge_predictions_archive(
    path: Path,
    batch: Sequence[PredictionAppendItem],
    *,
    allow_max_inst_growth: bool = True,
    journal: bool = True,
    fsync: bool = True,
    run_metadata: Mapping[str, Any] | None = None,
) -> int:
    """Merge predictions for existing frames by appending instances to those frames.

    Args:
        path: Path to the native archive file.
        batch: Sequence of PredictionAppendItem to merge.
        allow_max_inst_growth: If True, allow the file to be rewritten with a larger
            max_instances if the merge exceeds the current limit.
        journal: If True, use a journal for atomic updates.
        fsync: If True, fsync the file after writing.
        run_metadata: Optional metadata for this specific run.

    Returns:
        int: Number of frames successfully merged.
    """

    batch_list = _normalize_append_batch(list(batch))
    if not batch_list:
        return 0

    run_entry: dict[str, Any] | None = None
    if run_metadata is not None:
        run_entry = _normalize_run_entry(run_metadata)

    with ArchiveFileLock(path):
        with h5py.File(str(path), "r") as probe_file:
            preds_group, frames_group, data_group = _require_predictions_groups(probe_file)
            keypoints_ds = _require_predictions_keypoints_dataset(data_group)

            keypoint_count = int(keypoints_ds.shape[2]) if keypoints_ds.ndim > 2 else 0
            max_inst = int(keypoints_ds.shape[1]) if keypoints_ds.ndim > 1 else 0
            total_rows = int(keypoints_ds.shape[0]) if keypoints_ds.shape else 0
            committed_length = _normalize_predictions_committed_length(
                preds_group,
                total_rows=total_rows,
            )

            expected_heatmap_hw = _infer_batch_heatmap_hw(
                batch_list,
                keypoint_count=keypoint_count,
                require_all=False,
            )
            heatmaps_probe = data_group.get("heatmaps")
            if heatmaps_probe is not None and not isinstance(heatmaps_probe, h5py.Dataset):
                raise TypeError("Predictions heatmaps must be an h5py Dataset")
            heatmaps_hw: tuple[int, int] | None = None
            if isinstance(heatmaps_probe, h5py.Dataset):
                if heatmaps_probe.ndim != 4:
                    raise ValueError("heatmaps dataset must have rank 4 (N,K,H,W)")
                if int(heatmaps_probe.shape[1]) != int(keypoint_count):
                    raise ValueError("heatmaps dataset K does not match predictions keypoints K")
                heatmaps_hw = (int(heatmaps_probe.shape[2]), int(heatmaps_probe.shape[3]))
                if expected_heatmap_hw is not None and expected_heatmap_hw != heatmaps_hw:
                    raise ValueError(
                        f"heatmaps batch spatial dims {expected_heatmap_hw} "
                        f"do not match existing archive heatmaps {heatmaps_hw}"
                    )

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
            num_instances = np.asarray(num_inst_ds[:committed_length], dtype=np.int32)

            index_map = {
                (int(video_indices[i]), int(frame_indices[i])): i for i in range(committed_length)
            }

            missing_keys = [
                (item.video_index, item.frame_index)
                for item in batch_list
                if (item.video_index, item.frame_index) not in index_map
            ]
            if missing_keys:
                raise ValueError(
                    "merge_predictions_archive only supports frames already present in the archive"
                )

            required_max = max_inst
            for item in batch_list:
                base_count = int(num_instances[index_map[(item.video_index, item.frame_index)]])
                required_max = max(required_max, base_count + len(item.instances or []))

        if required_max > max_inst and not allow_max_inst_growth:
            raise MaxInstancesExceededError(
                "Batch requires more instances than the file currently stores. "
                "Re-run with allow_max_inst_growth=True to trigger rewrite."
            )

        if required_max > max_inst:
            update_map = {(item.video_index, item.frame_index): item for item in batch_list}
            return _rewrite_predictions_with_updates(
                path,
                update_map,
                keypoint_count=keypoint_count,
                new_max_inst=required_max,
                committed_length=committed_length,
                journal=journal,
                run_entry=run_entry,
                fsync=fsync,
            )

        with h5py.File(str(path), "r+") as h5file:
            preds_group = h5file.get("predictions")
            if not isinstance(preds_group, h5py.Group):
                raise TypeError("Predictions entry must be an h5py Group")
            frames_group = preds_group.get("frames")
            data_group = preds_group.get("data")
            if not isinstance(frames_group, h5py.Group) or not isinstance(data_group, h5py.Group):
                raise TypeError("Predictions frames/data must be h5py Groups")

            heatmaps_ds = data_group.get("heatmaps")
            if expected_heatmap_hw is not None and not isinstance(heatmaps_ds, h5py.Dataset):
                hm_h, hm_w = expected_heatmap_hw
                keypoints_ds = data_group.get("keypoints")
                if not isinstance(keypoints_ds, h5py.Dataset):
                    raise TypeError("Predictions keypoints must be an h5py Dataset")
                total_rows = int(keypoints_ds.shape[0]) if keypoints_ds.shape else 0
                initial_rows = max(total_rows, 1)
                heatmaps_ds = _create_heatmaps_dataset(
                    data_group,
                    initial_length=initial_rows,
                    keypoint_count=keypoint_count,
                    height=hm_h,
                    width=hm_w,
                )
                if total_rows == 0:
                    heatmaps_ds.resize((0, keypoint_count, hm_h, hm_w))

            video_ds = frames_group.get("video_index")
            frame_ds = frames_group.get("frame_index")
            num_inst_ds = frames_group.get("num_instances")
            if (
                not isinstance(video_ds, h5py.Dataset)
                or not isinstance(frame_ds, h5py.Dataset)
                or not isinstance(num_inst_ds, h5py.Dataset)
            ):
                raise TypeError("Predictions frames datasets must be h5py Dataset")
            keypoints_ds = data_group.get("keypoints")
            if not isinstance(keypoints_ds, h5py.Dataset):
                raise TypeError("Predictions keypoints must be an h5py Dataset")

            keypoint_score_ds, instance_score_ds, track_id_ds, deleted_ds = (
                _optional_prediction_datasets(data_group)
            )

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

            meta_group = _require_project_metadata_group(h5file)
            runs_count: int | None = None
            with _JournalTransaction(
                preds_group,
                old_len=committed_length,
                new_len=committed_length,
                operation="predictions.merge",
                commit_length=committed_length,
                enabled=journal,
            ):
                for item in batch_list:
                    idx = index_map[(item.video_index, item.frame_index)]
                    base_instances = _existing_instances_for_index(idx, datasets, keypoint_count)
                    merged_instances = base_instances + list(item.instances or [])
                    merged_heatmaps = item.heatmaps
                    if isinstance(heatmaps_ds, h5py.Dataset) and merged_heatmaps is None:
                        merged_heatmaps = np.asarray(heatmaps_ds[idx, :, :, :], dtype=np.float16)
                    merged_item = PredictionAppendItem(
                        video_index=item.video_index,
                        frame_index=item.frame_index,
                        instances=merged_instances,
                        detections=item.detections,
                        tracks=item.tracks,
                        heatmaps=merged_heatmaps,
                    )
                    _fill_prediction_slice(
                        datasets,
                        [merged_item],
                        start=idx,
                        max_inst=max_inst,
                        keypoint_count=keypoint_count,
                    )

                _assert_prediction_dataset_alignment(datasets, expected_length=committed_length)
                if run_entry is not None:
                    runs_group = h5file.require_group("runs")
                    table_group = runs_group.require_group("table")
                    runs_count = append_run_entry(table_group, run_entry)

                meta_group.attrs["modified"] = _now_utc_iso()
                meta_group.attrs["n_predictions_committed"] = int(committed_length)
                meta_group.attrs["max_inst_preds"] = int(max_inst)
                write_tracks_group(
                    h5file,
                    existing=read_tracks_group(h5file),
                    prediction_items=batch_list,
                )
                if runs_count is not None:
                    meta_group.attrs["runs_count"] = int(runs_count)

                _flush_file(h5file, fsync=fsync)

    return len(batch_list)

__all__ = [
    "append_predictions_archive",
    "merge_predictions_archive",
]
