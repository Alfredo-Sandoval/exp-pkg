"""HDF5 dataset helpers for `.sta` predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

import h5py
import numpy as np

from xpkg.core.annotations import PredictedInstance, PredictedPoint

_PREDICTED_INSTANCE_TYPES: tuple[type, ...] = (PredictedInstance,)
_PREDICTED_POINT_TYPES: tuple[type, ...] = (PredictedPoint,)


def predicted_instance_types() -> tuple[type, ...]:
    return _PREDICTED_INSTANCE_TYPES


def predicted_point_types() -> tuple[type, ...]:
    return _PREDICTED_POINT_TYPES


class MaxInstancesExceededError(RuntimeError):
    """Raised when an append batch exceeds the configured max instances per frame."""


class SerializerPredictedInstance:
    """Lightweight predicted instance used by serializer tests."""

    __slots__ = ("deleted", "keypoint_scores", "keypoints", "score", "track_id")

    def __init__(
        self,
        *,
        keypoints: Sequence[Sequence[float]] | None = None,
        score: float | None = None,
        track_id: int | None = None,
        deleted: bool = False,
        keypoint_scores: Sequence[float] | None = None,
        **_kwargs: Any,
    ) -> None:
        if keypoints is None:
            self.keypoints = []
        else:
            self.keypoints = list(keypoints)
        self.score = float(score) if score is not None else None
        self.track_id = int(track_id) if track_id is not None else -1
        self.deleted = bool(deleted)
        self.keypoint_scores = (
            [float(val) for val in keypoint_scores] if keypoint_scores is not None else []
        )


class PredictionAppendItem:
    """Container for appending predictions to a `.sta` archive."""

    __slots__ = (
        "detections",
        "frame_index",
        "heatmaps",
        "instances",
        "tracks",
        "video_index",
    )

    def __init__(
        self,
        video_index: int,
        frame_index: int,
        instances,
        detections=None,
        tracks=None,
        heatmaps: Any | None = None,
    ):
        self.video_index = int(video_index)
        self.frame_index = int(frame_index)
        self.instances = instances
        self.tracks = tracks
        self.detections = detections
        self.heatmaps = heatmaps


class PredictionDatasetMap(TypedDict):
    video_index: h5py.Dataset
    frame_index: h5py.Dataset
    num_instances: h5py.Dataset
    keypoints: h5py.Dataset
    keypoint_score: h5py.Dataset | None
    instance_score: h5py.Dataset | None
    track_id: h5py.Dataset | None
    deleted: h5py.Dataset | None
    heatmaps: h5py.Dataset | None


_PREDICTION_COMPRESSION = "gzip"
_PREDICTION_COMPRESSION_LEVEL = 4
_PREDICTION_ROW_CHUNK_MAX = 128
_HEATMAP_ROW_CHUNK_MAX = 8
_NO_FILLVALUE = object()


def _normalize_heatmaps_frame(
    heatmaps: Any,
    *,
    keypoint_count: int,
    expected_hw: tuple[int, int] | None,
    frame_index: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    arr = np.asarray(heatmaps)
    if arr.ndim != 3:
        raise ValueError(f"heatmaps must be a (K,H,W) array for frame {frame_index}")
    k, h, w = (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))
    if k != int(keypoint_count):
        raise ValueError(
            f"heatmaps K={k} does not match keypoint_count={keypoint_count} for frame {frame_index}"
        )
    if h <= 0 or w <= 0:
        raise ValueError(f"heatmaps spatial dims must be positive for frame {frame_index}")
    if expected_hw is not None and (h, w) != expected_hw:
        raise ValueError(
            "heatmaps spatial dims "
            f"{(h, w)} do not match expected {expected_hw} for frame {frame_index}"
        )
    return arr.astype(np.float16, copy=False), (h, w)


def _infer_batch_heatmap_hw(
    batch: Sequence[PredictionAppendItem],
    *,
    keypoint_count: int,
    require_all: bool,
) -> tuple[int, int] | None:
    expected_hw: tuple[int, int] | None = None
    saw_any = False
    for item in batch:
        hm = item.heatmaps
        if hm is None:
            if require_all:
                raise ValueError(
                    "heatmaps are required when present in batch "
                    f"(missing for frame {item.frame_index})"
                )
            continue
        saw_any = True
        _, hw = _normalize_heatmaps_frame(
            hm,
            keypoint_count=keypoint_count,
            expected_hw=expected_hw,
            frame_index=int(item.frame_index),
        )
        expected_hw = hw
    return expected_hw if saw_any else None


def _format_location(
    frame_index: int, instance_index: int, keypoint_index: int | None = None
) -> str:
    base = f"frame={frame_index}, instance={instance_index}"
    if keypoint_index is not None:
        base += f", keypoint={keypoint_index}"
    return base


def _coerce_prediction_float(
    value: Any,
    *,
    field: str,
    frame_index: int,
    instance_index: int,
    keypoint_index: int | None = None,
) -> float:
    if isinstance(value, bool):
        raise TypeError(
            f"Invalid {field} at {_format_location(frame_index, instance_index, keypoint_index)}"
        )
    if not isinstance(value, int | float | np.integer | np.floating):
        raise TypeError(
            f"Invalid {field} at {_format_location(frame_index, instance_index, keypoint_index)}"
        )
    num = float(value)
    if not np.isfinite(num):
        raise ValueError(
            f"Non-finite {field} at {_format_location(frame_index, instance_index, keypoint_index)}"
        )
    return float(num)


def _bounded_sequence_length(
    seq: Sequence[Any] | None, *, max_len: int, kind: str, frame_index: int
) -> int:
    if seq is None:
        return 0
    length = len(seq)
    if length > max_len:
        raise ValueError(f"Too many {kind} ({length}) for frame {frame_index}; max {max_len}")
    return length


def _coerce_keypoint_triplet(
    value: Any, *, frame_index: int, instance_index: int, keypoint_index: int
) -> tuple[float, float, float]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, Sequence) or len(value) < 2:
        raise ValueError(
            "Keypoint value must be a sequence of at least two numbers at "
            f"{_format_location(frame_index, instance_index, keypoint_index)}"
        )
    x_val = _coerce_prediction_float(
        value[0],
        field="keypoint x",
        frame_index=frame_index,
        instance_index=instance_index,
        keypoint_index=keypoint_index,
    )
    y_val = _coerce_prediction_float(
        value[1],
        field="keypoint y",
        frame_index=frame_index,
        instance_index=instance_index,
        keypoint_index=keypoint_index,
    )
    score_val = 1.0
    if len(value) > 2:
        score_val = _coerce_prediction_float(
            value[2],
            field="keypoint score",
            frame_index=frame_index,
            instance_index=instance_index,
            keypoint_index=keypoint_index,
        )
    return x_val, y_val, score_val


def _normalize_keypoint_values(
    pt: Any,
    *,
    frame_index: int,
    instance_index: int,
    keypoint_index: int,
) -> tuple[float, float, float]:
    if isinstance(pt, predicted_point_types()):
        is_nan = bool(pt.isnan())
        x_val = (
            np.nan
            if is_nan
            else _coerce_prediction_float(
                pt.x,
                field="keypoint x",
                frame_index=frame_index,
                instance_index=instance_index,
                keypoint_index=keypoint_index,
            )
        )
        y_val = (
            np.nan
            if is_nan
            else _coerce_prediction_float(
                pt.y,
                field="keypoint y",
                frame_index=frame_index,
                instance_index=instance_index,
                keypoint_index=keypoint_index,
            )
        )
        score_value = None if pt.score is None else float(pt.score)
        confidence = 1.0 if pt.visible else 0.0
        if score_value is not None:
            confidence = _coerce_prediction_float(
                score_value,
                field="keypoint score",
                frame_index=frame_index,
                instance_index=instance_index,
                keypoint_index=keypoint_index,
            )
        return float(x_val), float(y_val), float(confidence)

    if isinstance(pt, Sequence) and not isinstance(pt, bytes | bytearray | str):
        x_val, y_val, score_val = _coerce_keypoint_triplet(
            pt,
            frame_index=frame_index,
            instance_index=instance_index,
            keypoint_index=keypoint_index,
        )
        return float(x_val), float(y_val), float(score_val)

    raise TypeError(
        f"Unsupported keypoint type at frame={frame_index}, instance={instance_index}, "
        f"keypoint={keypoint_index}"
    )


def _instance_keypoint_length(instance: Any) -> int:
    if isinstance(instance, SerializerPredictedInstance):
        return len(instance.keypoints)
    if isinstance(instance, predicted_instance_types()):
        return len(instance.skeleton.keypoints)
    return 0


def _normalize_append_batch(batch: list[PredictionAppendItem]) -> list[PredictionAppendItem]:
    normalized: list[PredictionAppendItem] = []
    for item in batch:
        if not isinstance(item, PredictionAppendItem):
            raise TypeError("append batch must contain PredictionAppendItem entries")
        normalized.append(item)
    normalized.sort(key=lambda x: (x.video_index, x.frame_index))
    for idx in range(1, len(normalized)):
        prev = normalized[idx - 1]
        curr = normalized[idx]
        if (prev.video_index, prev.frame_index) == (curr.video_index, curr.frame_index):
            raise ValueError(
                "append batch contains duplicate (video_index, frame_index) entries; "
                "merge instances per frame before writing"
            )
    return normalized


def _row_chunk_count(initial_length: int, *, max_rows: int) -> int:
    return min(max_rows, max(1, int(initial_length) if int(initial_length) > 0 else 1))


def _sanitize_chunks(chunks: tuple[int, ...], shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(chunks) != len(shape):
        raise ValueError("Template chunk rank does not match target dataset rank")

    normalized: list[int] = []
    for chunk_dim, shape_dim in zip(chunks, shape, strict=True):
        chunk_int = max(1, int(chunk_dim))
        shape_int = int(shape_dim)
        if shape_int > 0 and chunk_int > shape_int:
            chunk_int = shape_int
        normalized.append(chunk_int)
    return tuple(normalized)


def _dataset_create_kwargs(
    *,
    shape: tuple[int, ...],
    template: h5py.Dataset | None,
    default_chunks: tuple[int, ...] | bool,
    default_compression: str | None,
    default_compression_opts: int | None,
    default_shuffle: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}

    if template is not None and template.chunks is not None:
        kwargs["chunks"] = _sanitize_chunks(tuple(int(x) for x in template.chunks), shape)
    else:
        kwargs["chunks"] = default_chunks

    template_compression = (
        str(template.compression)
        if template is not None and template.compression is not None
        else None
    )
    if template_compression is not None:
        kwargs["compression"] = template_compression
        if template is not None and template.compression_opts is not None:
            kwargs["compression_opts"] = template.compression_opts
        if template is not None and bool(template.shuffle):
            kwargs["shuffle"] = True
    elif default_compression is not None:
        kwargs["compression"] = default_compression
        if default_compression_opts is not None:
            kwargs["compression_opts"] = default_compression_opts
    if template is not None and bool(template.fletcher32):
        kwargs["fletcher32"] = True
    if "shuffle" not in kwargs and default_shuffle:
        kwargs["shuffle"] = True
    return kwargs


def _create_prediction_dataset(
    group: h5py.Group,
    name: str,
    *,
    shape: tuple[int, ...],
    maxshape: tuple[int | None, ...],
    dtype: Any,
    template: h5py.Dataset | None,
    default_chunks: tuple[int, ...] | bool,
    default_compression: str | None,
    default_compression_opts: int | None,
    default_shuffle: bool,
    fillvalue: Any = _NO_FILLVALUE,
) -> h5py.Dataset:
    create_kwargs = _dataset_create_kwargs(
        shape=shape,
        template=template,
        default_chunks=default_chunks,
        default_compression=default_compression,
        default_compression_opts=default_compression_opts,
        default_shuffle=default_shuffle,
    )
    if fillvalue is _NO_FILLVALUE:
        return group.create_dataset(
            name,
            shape=shape,
            maxshape=maxshape,
            dtype=dtype,
            **create_kwargs,
        )
    return group.create_dataset(
        name,
        shape=shape,
        maxshape=maxshape,
        dtype=dtype,
        fillvalue=fillvalue,
        **create_kwargs,
    )


def _create_heatmaps_dataset(
    data_group: h5py.Group,
    *,
    initial_length: int,
    keypoint_count: int,
    height: int,
    width: int,
    template: h5py.Dataset | None = None,
) -> h5py.Dataset:
    shape = (int(initial_length), int(keypoint_count), int(height), int(width))
    maxshape = (None, int(keypoint_count), int(height), int(width))
    chunk_rows = _row_chunk_count(initial_length, max_rows=_HEATMAP_ROW_CHUNK_MAX)
    return _create_prediction_dataset(
        data_group,
        "heatmaps",
        shape=shape,
        maxshape=maxshape,
        dtype=np.float16,
        template=template,
        default_chunks=(chunk_rows, 1, int(height), int(width)),
        default_compression=_PREDICTION_COMPRESSION,
        default_compression_opts=_PREDICTION_COMPRESSION_LEVEL,
        default_shuffle=True,
    )


def _assert_prediction_dataset_alignment(
    datasets: PredictionDatasetMap,
    *,
    expected_length: int | None = None,
) -> None:
    lengths: dict[str, int] = {}
    for name, dataset in datasets.items():
        if not isinstance(dataset, h5py.Dataset):
            continue
        if len(dataset.shape) == 0:
            continue
        lengths[name] = int(dataset.shape[0])

    if not lengths:
        return

    observed_lengths = sorted(set(lengths.values()))
    if len(observed_lengths) != 1:
        parts = ", ".join(f"{key}={value}" for key, value in sorted(lengths.items()))
        raise ValueError(f"Prediction dataset lengths are misaligned: {parts}")

    if expected_length is not None and observed_lengths[0] != int(expected_length):
        parts = ", ".join(f"{key}={value}" for key, value in sorted(lengths.items()))
        raise ValueError(
            f"Prediction dataset length mismatch (expected {int(expected_length)}): {parts}"
        )


def _existing_instances_for_index(
    idx: int, datasets: PredictionDatasetMap, keypoint_count: int
) -> list[SerializerPredictedInstance]:
    num_inst_ds = datasets["num_instances"]
    keypoints_ds = datasets["keypoints"]
    kp_score_ds = datasets["keypoint_score"]
    inst_score_ds = datasets["instance_score"]
    track_id_ds = datasets["track_id"]
    deleted_ds = datasets["deleted"]

    inst_count = int(num_inst_ds[idx])
    keypoints = keypoints_ds[idx, :inst_count, :keypoint_count, :]
    kp_scores = None
    if isinstance(kp_score_ds, h5py.Dataset):
        kp_scores = kp_score_ds[idx, :inst_count, :keypoint_count]
    inst_scores = None
    if isinstance(inst_score_ds, h5py.Dataset):
        inst_scores = inst_score_ds[idx, :inst_count]
    track_ids = None
    if isinstance(track_id_ds, h5py.Dataset):
        track_ids = track_id_ds[idx, :inst_count]
    deleted = None
    if isinstance(deleted_ds, h5py.Dataset):
        deleted = deleted_ds[idx, :inst_count]

    keypoint_rows = np.asarray(keypoints[:, :, :3], dtype=np.float32)
    kp_score_rows = (
        np.asarray(kp_scores[:, :keypoint_count], dtype=np.float32)
        if kp_scores is not None
        else None
    )
    inst_score_arr = np.asarray(inst_scores, dtype=np.float32) if inst_scores is not None else None
    track_id_arr = np.asarray(track_ids, dtype=np.int32) if track_ids is not None else None
    deleted_arr = np.asarray(deleted, dtype=np.uint8) if deleted is not None else None

    instances: list[SerializerPredictedInstance] = []
    for inst_idx in range(inst_count):
        keypoints_list = [tuple(row) for row in keypoint_rows[inst_idx].tolist()]
        score_val = float(inst_score_arr[inst_idx]) if inst_score_arr is not None else 0.0
        track_val = None
        if track_id_arr is not None:
            raw = int(track_id_arr[inst_idx])
            track_val = raw if raw >= 0 else None
        deleted_val = bool(deleted_arr[inst_idx]) if deleted_arr is not None else False
        kp_score_seq = None
        if kp_score_rows is not None:
            kp_score_seq = kp_score_rows[inst_idx].tolist()
        instances.append(
            SerializerPredictedInstance(
                keypoints=keypoints_list,
                score=score_val,
                track_id=track_val,
                deleted=deleted_val,
                keypoint_scores=kp_score_seq,
            )
        )
    return instances


def _resize_prediction_datasets(datasets: PredictionDatasetMap, new_length: int) -> None:
    for name, dataset in datasets.items():
        if not isinstance(dataset, h5py.Dataset):
            continue
        ds: h5py.Dataset = dataset
        if ds.shape and int(ds.shape[0]) >= new_length:
            continue
        if ds.maxshape is None:
            raise ValueError(f"Dataset {name} cannot be resized (maxshape=None)")
        maxshape = ds.maxshape
        if len(maxshape) == 0:
            raise ValueError(f"Dataset {name} has zero rank and cannot be resized")
        max_dim = maxshape[0]
        if max_dim is None or max_dim >= new_length:
            new_shape = list(ds.shape)
            new_shape[0] = new_length
            ds.resize(tuple(new_shape))
            continue
        raise ValueError(
            f"Dataset {name} cannot grow beyond {max_dim} rows (requested {new_length})"
        )


def _pack_instances_into_row(
    instances: Sequence[Any],
    *,
    frame_index: int,
    keypoint_slots: int,
    row_index: int,
    keypoints_arr: np.ndarray,
    keypoint_score_arr: np.ndarray | None,
    instance_score_arr: np.ndarray | None,
    track_id_arr: np.ndarray | None,
    deleted_arr: np.ndarray | None,
) -> int:
    inst_count = len(instances)
    for inst_idx, inst in enumerate(instances):
        if inst_idx >= keypoints_arr.shape[1]:
            break

        if isinstance(inst, SerializerPredictedInstance):
            score_raw = 0.0 if inst.score is None else inst.score
            track_id_raw: int | None = inst.track_id
            keypoints_src = list(inst.keypoints)
            keypoint_scores = list(inst.keypoint_scores)
            deleted = bool(inst.deleted)
        elif isinstance(inst, predicted_instance_types()):
            score_raw = inst.score
            track_id_raw = int(inst.track.spawned_on) if inst.track is not None else None
            keypoints_src = list(
                inst.get_points_array(full=True, copy=False, invisible_as_nan=False)
            )
            keypoint_scores = []
            deleted = False
        else:
            raise TypeError(
                "Prediction instances must be SerializerPredictedInstance "
                "or PredictedInstance entries"
            )

        score_val = _coerce_prediction_float(
            score_raw,
            field="instance score",
            frame_index=frame_index,
            instance_index=inst_idx,
        )

        if instance_score_arr is not None:
            instance_score_arr[row_index, inst_idx] = float(score_val)

        track_id = None
        if track_id_raw is not None and int(track_id_raw) >= 0:
            track_id = int(track_id_raw)
        if track_id_arr is not None and track_id is not None:
            track_id_arr[row_index, inst_idx] = int(track_id)

        if deleted_arr is not None:
            deleted_arr[row_index, inst_idx] = 1 if deleted else 0

        for kp_idx, raw_pt in enumerate(keypoints_src):
            if kp_idx >= keypoint_slots:
                break

            pt_value = raw_pt
            if keypoint_scores and kp_idx < len(keypoint_scores):
                if isinstance(raw_pt, Sequence) and not isinstance(raw_pt, bytes | bytearray | str):
                    if len(raw_pt) == 2:
                        pt_value = (raw_pt[0], raw_pt[1], keypoint_scores[kp_idx])

            x_val, y_val, conf_val = _normalize_keypoint_values(
                pt_value,
                frame_index=frame_index,
                instance_index=inst_idx,
                keypoint_index=kp_idx,
            )
            keypoints_arr[row_index, inst_idx, kp_idx, 0] = x_val
            keypoints_arr[row_index, inst_idx, kp_idx, 1] = y_val
            keypoints_arr[row_index, inst_idx, kp_idx, 2] = conf_val
            if keypoint_score_arr is not None:
                keypoint_score_arr[row_index, inst_idx, kp_idx] = conf_val

    return inst_count


def _fill_prediction_slice(
    datasets: PredictionDatasetMap,
    batch,
    *,
    start: int,
    max_inst: int,
    keypoint_count: int,
) -> None:
    video_ds = datasets["video_index"]
    frame_ds = datasets["frame_index"]
    num_inst_ds = datasets["num_instances"]
    keypoints_ds = datasets["keypoints"]

    keypoint_score_ds = datasets["keypoint_score"]
    instance_score_ds = datasets["instance_score"]
    track_id_ds = datasets["track_id"]
    deleted_ds = datasets["deleted"]
    heatmaps_ds = datasets["heatmaps"]

    end = start + len(batch)

    kp_shape = tuple(int(x) for x in keypoints_ds.shape)
    kps_per_instance = kp_shape[2] if len(kp_shape) > 2 else keypoint_count

    video_arr = np.zeros((len(batch),), dtype=np.int32)
    frame_arr = np.zeros((len(batch),), dtype=np.int32)
    num_inst_arr = np.zeros((len(batch),), dtype=np.int32)
    keypoints_arr = np.zeros((len(batch), max_inst, kps_per_instance, 3), dtype=np.float32)

    keypoint_score_arr = (
        np.zeros((len(batch), max_inst, kps_per_instance), dtype=np.float32)
        if keypoint_score_ds is not None
        else None
    )
    instance_score_arr = (
        np.zeros((len(batch), max_inst), dtype=np.float32)
        if instance_score_ds is not None
        else None
    )
    track_id_arr = (
        np.full((len(batch), max_inst), -1, dtype=np.int32) if track_id_ds is not None else None
    )
    deleted_arr = (
        np.zeros((len(batch), max_inst), dtype=np.uint8) if deleted_ds is not None else None
    )

    heatmaps_arr = None
    heatmaps_hw: tuple[int, int] | None = None
    if heatmaps_ds is not None:
        hm_shape = tuple(int(x) for x in heatmaps_ds.shape)
        if len(hm_shape) < 4:
            raise ValueError("heatmaps dataset must have shape (N,K,H,W)")
        hm_k, hm_h, hm_w = int(hm_shape[1]), int(hm_shape[2]), int(hm_shape[3])
        if hm_k != int(kps_per_instance):
            raise ValueError("heatmaps dataset K does not match keypoints dataset K")
        if hm_h <= 0 or hm_w <= 0:
            raise ValueError("heatmaps dataset spatial dims must be positive")
        heatmaps_hw = (hm_h, hm_w)
        heatmaps_arr = np.zeros((len(batch), hm_k, hm_h, hm_w), dtype=np.float16)

    for row, item in enumerate(batch):
        video_arr[row] = int(item.video_index)
        frame_arr[row] = int(item.frame_index)

        instances = list(item.instances or [])
        inst_count = _bounded_sequence_length(
            instances, max_len=max_inst, kind="instances", frame_index=item.frame_index
        )
        instances = instances[:inst_count]
        num_inst_arr[row] = _pack_instances_into_row(
            instances,
            frame_index=int(item.frame_index),
            keypoint_slots=int(kps_per_instance),
            row_index=row,
            keypoints_arr=keypoints_arr,
            keypoint_score_arr=keypoint_score_arr,
            instance_score_arr=instance_score_arr,
            track_id_arr=track_id_arr,
            deleted_arr=deleted_arr,
        )

        if heatmaps_arr is not None:
            hm = item.heatmaps
            if hm is None:
                raise ValueError(
                    "heatmaps dataset exists but PredictionAppendItem is missing heatmaps "
                    f"for frame {item.frame_index}"
                )
            hm_arr, _ = _normalize_heatmaps_frame(
                hm,
                keypoint_count=int(kps_per_instance),
                expected_hw=heatmaps_hw,
                frame_index=int(item.frame_index),
            )
            heatmaps_arr[row, :, :, :] = hm_arr

    slice_obj = slice(start, end)
    video_ds[slice_obj] = video_arr
    frame_ds[slice_obj] = frame_arr
    num_inst_ds[slice_obj] = num_inst_arr
    keypoints_ds[slice_obj, :, :, :] = keypoints_arr
    if keypoint_score_arr is not None and isinstance(keypoint_score_ds, h5py.Dataset):
        keypoint_score_ds[slice_obj, :, :] = keypoint_score_arr
    if instance_score_arr is not None and isinstance(instance_score_ds, h5py.Dataset):
        instance_score_ds[slice_obj, :] = instance_score_arr
    if track_id_arr is not None and isinstance(track_id_ds, h5py.Dataset):
        track_id_ds[slice_obj, :] = track_id_arr
    if deleted_arr is not None and isinstance(deleted_ds, h5py.Dataset):
        deleted_ds[slice_obj, :] = deleted_arr
    if heatmaps_arr is not None and isinstance(heatmaps_ds, h5py.Dataset):
        heatmaps_ds[slice_obj, :, :, :] = heatmaps_arr


def _create_predictions_datasets(
    preds_group: h5py.Group,
    max_instances: int,
    keypoint_count: int,
    initial_length: int = 0,
    *,
    frame_templates: Mapping[str, h5py.Dataset] | None = None,
    data_templates: Mapping[str, h5py.Dataset] | None = None,
) -> tuple[h5py.Group, h5py.Group]:
    """Create the standard predictions frames/data dataset structure.

    Returns the (frames_group, data_group) pair so callers can reference
    the individual datasets for copy/fill operations.
    """
    frames_group = preds_group.require_group("frames")
    data_group = preds_group.require_group("data")

    chunk_rows = _row_chunk_count(initial_length, max_rows=_PREDICTION_ROW_CHUNK_MAX)

    frame_specs = {
        "video_index": np.int32,
        "frame_index": np.int32,
        "num_instances": np.int32,
    }

    for name, dtype in frame_specs.items():
        frame_template = None if frame_templates is None else frame_templates.get(name)
        _create_prediction_dataset(
            frames_group,
            name,
            shape=(int(initial_length),),
            maxshape=(None,),
            dtype=dtype,
            template=frame_template,
            default_chunks=True,
            default_compression=None,
            default_compression_opts=None,
            default_shuffle=False,
        )

    data_specs: tuple[
        tuple[str, tuple[int, ...], Any, Any, tuple[int, ...]],
        ...,
    ] = (
        (
            "keypoints",
            (int(max_instances), int(keypoint_count), 3),
            np.float32,
            _NO_FILLVALUE,
            (chunk_rows, int(max_instances), int(keypoint_count), 3),
        ),
        (
            "keypoint_score",
            (int(max_instances), int(keypoint_count)),
            np.float32,
            _NO_FILLVALUE,
            (chunk_rows, int(max_instances), int(keypoint_count)),
        ),
        (
            "instance_score",
            (int(max_instances),),
            np.float32,
            _NO_FILLVALUE,
            (chunk_rows, int(max_instances)),
        ),
        (
            "track_id",
            (int(max_instances),),
            np.int32,
            -1,
            (chunk_rows, int(max_instances)),
        ),
        (
            "deleted",
            (int(max_instances),),
            np.uint8,
            _NO_FILLVALUE,
            (chunk_rows, int(max_instances)),
        ),
    )

    for name, trailing_shape, dtype, fillvalue, chunk_shape in data_specs:
        data_template = None if data_templates is None else data_templates.get(name)
        _create_prediction_dataset(
            data_group,
            name,
            shape=(int(initial_length), *trailing_shape),
            maxshape=(None, *trailing_shape),
            dtype=dtype,
            template=data_template,
            default_chunks=chunk_shape,
            default_compression=_PREDICTION_COMPRESSION,
            default_compression_opts=_PREDICTION_COMPRESSION_LEVEL,
            default_shuffle=True,
            fillvalue=fillvalue,
        )

    return frames_group, data_group


def _bootstrap_predictions_group(
    container: h5py.File | h5py.Group,
    *,
    max_instances: int,
    keypoint_count: int,
    initial_length: int,
    committed_length: int,
    expected_heatmap_hw: tuple[int, int] | None,
    frame_templates: Mapping[str, h5py.Dataset] | None = None,
    data_templates: Mapping[str, h5py.Dataset] | None = None,
    heatmaps_template: h5py.Dataset | None = None,
) -> tuple[h5py.Group, h5py.Group, h5py.Group, h5py.Dataset | None]:
    """Create `/predictions` datasets and optional heatmaps in one canonical step."""
    preds_group = container.require_group("predictions")
    frames_group, data_group = _create_predictions_datasets(
        preds_group,
        max_instances=int(max_instances),
        keypoint_count=int(keypoint_count),
        initial_length=int(initial_length),
        frame_templates=frame_templates,
        data_templates=data_templates,
    )

    heatmaps_ds: h5py.Dataset | None = None
    if expected_heatmap_hw is not None:
        hm_h, hm_w = expected_heatmap_hw
        heatmaps_ds = _create_heatmaps_dataset(
            data_group,
            initial_length=int(initial_length),
            keypoint_count=int(keypoint_count),
            height=int(hm_h),
            width=int(hm_w),
            template=heatmaps_template,
        )

    preds_group.attrs["committed_length"] = int(committed_length)
    return preds_group, frames_group, data_group, heatmaps_ds


def _copy_dataset_rows(src: h5py.Dataset, dst: h5py.Dataset, limit: int) -> None:
    if limit <= 0:
        return
    step = 1024
    if src.chunks and len(src.chunks) > 0 and src.chunks[0]:
        step = int(src.chunks[0])
    step = max(1, min(step, limit))
    offset = 0
    while offset < limit:
        end = min(limit, offset + step)
        dst[offset:end, ...] = src[offset:end, ...]
        offset = end


__all__ = [
    "MaxInstancesExceededError",
    "PredictionAppendItem",
    "PredictionDatasetMap",
    "SerializerPredictedInstance",
    "_assert_prediction_dataset_alignment",
    "_bootstrap_predictions_group",
    "_bounded_sequence_length",
    "_coerce_keypoint_triplet",
    "_coerce_prediction_float",
    "_copy_dataset_rows",
    "_create_heatmaps_dataset",
    "_create_predictions_datasets",
    "_existing_instances_for_index",
    "_fill_prediction_slice",
    "_format_location",
    "_infer_batch_heatmap_hw",
    "_instance_keypoint_length",
    "_normalize_append_batch",
    "_normalize_heatmaps_frame",
    "_normalize_keypoint_values",
    "_pack_instances_into_row",
    "_resize_prediction_datasets",
    "predicted_instance_types",
    "predicted_point_types",
]
