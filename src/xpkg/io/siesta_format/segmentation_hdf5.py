"""HDF5 read/write helpers for segmentation masks and ROIs in .sta archives."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np

from xpkg.core.annotations.regions import (
    ROI,
    MaskType,
    PromptType,
    SegmentationMask,
    SegmentationPrompt,
)

SEGMENTATION_SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_segmentation_group(
    file: h5py.File,
    masks_by_frame: list[tuple[int, int, list[SegmentationMask]]],
    rois_by_frame: list[tuple[int, int, list[ROI]]],
) -> None:
    """Write segmentation data to /segmentation in the HDF5 file.

    Args:
        file: Open HDF5 file handle.
        masks_by_frame: List of (video_index, frame_index, masks) tuples.
        rois_by_frame: List of (video_index, frame_index, rois) tuples.
    """
    seg_group = file.require_group("segmentation")
    seg_group.attrs["schema_version"] = SEGMENTATION_SCHEMA_VERSION

    _write_masks_subgroup(seg_group, masks_by_frame)
    _write_rois_subgroup(seg_group, rois_by_frame)


def _write_masks_subgroup(
    seg_group: h5py.Group,
    masks_by_frame: list[tuple[int, int, list[SegmentationMask]]],
) -> None:
    """Write mask data to /segmentation/masks."""
    masks_group = seg_group.require_group("masks")
    str_dtype = h5py.string_dtype("utf-8")

    all_masks: list[tuple[int, int, SegmentationMask]] = []
    for vi, fi, frame_masks in masks_by_frame:
        for mask in frame_masks:
            all_masks.append((vi, fi, mask))

    n = len(all_masks)
    if n == 0:
        masks_group.create_dataset("video_index", data=np.zeros(0, dtype=np.int32))
        masks_group.create_dataset("frame_index", data=np.zeros(0, dtype=np.int32))
        masks_group.create_dataset("mask_type", data=np.zeros(0, dtype=np.uint8))
        masks_group.attrs["count"] = 0
        return

    video_index = np.zeros(n, dtype=np.int32)
    frame_index = np.zeros(n, dtype=np.int32)
    mask_type = np.zeros(n, dtype=np.uint8)
    class_names: list[str] = []
    instance_ref = np.full(n, -1, dtype=np.int32)
    track_id = np.full(n, -1, dtype=np.int32)
    confidence = np.full(n, np.nan, dtype=np.float32)
    is_predicted = np.zeros(n, dtype=np.uint8)

    polygon_offsets = [0]
    polygon_ring_offsets = [0]
    all_polygon_vertices: list[np.ndarray] = []

    rle_offsets = [0]
    all_rle_counts: list[np.ndarray] = []
    rle_start = np.zeros(n, dtype=np.uint8)
    rle_height = np.zeros(n, dtype=np.int32)
    rle_width = np.zeros(n, dtype=np.int32)

    prompt_type = np.zeros(n, dtype=np.uint8)
    prompt_box = np.full((n, 4), np.nan, dtype=np.float32)
    prompt_text: list[str] = []
    prompt_model_id: list[str] = []
    prompt_backend: list[str] = []

    total_vertices = 0
    total_rle = 0

    for i, (vi, fi, mask) in enumerate(all_masks):
        video_index[i] = vi
        frame_index[i] = fi
        mask_type[i] = int(mask.mask_type)
        class_names.append(mask.class_name)
        instance_ref[i] = mask.instance_ref
        if mask.track is not None:
            track_id[i] = mask.track.id
        confidence[i] = mask.confidence
        is_predicted[i] = 1 if mask.is_predicted else 0

        if mask.mask_type == MaskType.POLYGON and mask.polygon_vertices is not None:
            for ring in mask.polygon_vertices:
                ring_arr = np.asarray(ring, dtype=np.float32).reshape(-1, 2)
                all_polygon_vertices.append(ring_arr)
                total_vertices += ring_arr.shape[0]
                polygon_ring_offsets.append(total_vertices)
        polygon_offsets.append(len(polygon_ring_offsets) - 1)

        if mask.mask_type == MaskType.RLE and mask.rle_counts is not None:
            counts_arr = np.asarray(mask.rle_counts, dtype=np.uint32)
            all_rle_counts.append(counts_arr)
            total_rle += counts_arr.shape[0]
            rle_start[i] = mask.rle_start
            rle_height[i] = mask.rle_height
            rle_width[i] = mask.rle_width
        rle_offsets.append(total_rle)

        if mask.prompt is not None and mask.prompt.prompt_type != PromptType.NONE:
            prompt_type[i] = int(mask.prompt.prompt_type)
            if mask.prompt.box is not None:
                prompt_box[i] = mask.prompt.box
            prompt_text.append(mask.prompt.text)
            prompt_model_id.append(mask.prompt.model_id)
            prompt_backend.append(mask.prompt.backend)
        else:
            prompt_text.append("")
            prompt_model_id.append("")
            prompt_backend.append("")

    masks_group.attrs["count"] = n
    masks_group.create_dataset("video_index", data=video_index)
    masks_group.create_dataset("frame_index", data=frame_index)
    masks_group.create_dataset("mask_type", data=mask_type)
    masks_group.create_dataset("class_name", data=class_names, dtype=str_dtype)
    masks_group.create_dataset("instance_ref", data=instance_ref)
    masks_group.create_dataset("track_id", data=track_id)
    masks_group.create_dataset("confidence", data=confidence)
    masks_group.create_dataset("is_predicted", data=is_predicted)

    masks_group.create_dataset(
        "polygon_offsets", data=np.array(polygon_offsets, dtype=np.int64)
    )
    masks_group.create_dataset(
        "polygon_ring_offsets", data=np.array(polygon_ring_offsets, dtype=np.int64)
    )
    if all_polygon_vertices:
        verts = np.concatenate(all_polygon_vertices, axis=0)
        masks_group.create_dataset(
            "polygon_vertices",
            data=verts,
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )
    else:
        masks_group.create_dataset(
            "polygon_vertices", data=np.zeros((0, 2), dtype=np.float32)
        )

    masks_group.create_dataset(
        "rle_offsets", data=np.array(rle_offsets, dtype=np.int64)
    )
    if all_rle_counts:
        rle_flat = np.concatenate(all_rle_counts)
        masks_group.create_dataset(
            "rle_counts",
            data=rle_flat,
            dtype=np.uint32,
            compression="gzip",
            compression_opts=4,
        )
    else:
        masks_group.create_dataset(
            "rle_counts", data=np.zeros(0, dtype=np.uint32)
        )
    masks_group.create_dataset("rle_start", data=rle_start)
    masks_group.create_dataset("rle_height", data=rle_height)
    masks_group.create_dataset("rle_width", data=rle_width)

    masks_group.create_dataset("prompt_type", data=prompt_type)
    masks_group.create_dataset(
        "prompt_box", data=prompt_box, dtype=np.float32
    )
    masks_group.create_dataset("prompt_text", data=prompt_text, dtype=str_dtype)
    masks_group.create_dataset("prompt_model_id", data=prompt_model_id, dtype=str_dtype)
    masks_group.create_dataset("prompt_backend", data=prompt_backend, dtype=str_dtype)


def _write_rois_subgroup(
    seg_group: h5py.Group,
    rois_by_frame: list[tuple[int, int, list[ROI]]],
) -> None:
    """Write ROI data to /segmentation/rois."""
    rois_group = seg_group.require_group("rois")
    str_dtype = h5py.string_dtype("utf-8")

    all_rois: list[tuple[int, int, ROI]] = []
    for vi, fi, frame_rois in rois_by_frame:
        for roi in frame_rois:
            all_rois.append((vi, fi, roi))

    n = len(all_rois)
    if n == 0:
        rois_group.create_dataset("video_index", data=np.zeros(0, dtype=np.int32))
        rois_group.create_dataset("frame_index", data=np.zeros(0, dtype=np.int32))
        rois_group.create_dataset("box", data=np.zeros((0, 4), dtype=np.float32))
        rois_group.attrs["count"] = 0
        return

    video_index = np.zeros(n, dtype=np.int32)
    frame_index = np.zeros(n, dtype=np.int32)
    box = np.zeros((n, 4), dtype=np.float32)
    class_names: list[str] = []
    instance_ref = np.full(n, -1, dtype=np.int32)
    track_id = np.full(n, -1, dtype=np.int32)
    confidence = np.full(n, np.nan, dtype=np.float32)
    is_predicted = np.zeros(n, dtype=np.uint8)

    for i, (vi, fi, roi) in enumerate(all_rois):
        video_index[i] = vi
        frame_index[i] = fi
        box[i] = [roi.x1, roi.y1, roi.x2, roi.y2]
        class_names.append(roi.class_name)
        instance_ref[i] = roi.instance_ref
        if roi.track is not None:
            track_id[i] = roi.track.id
        confidence[i] = roi.confidence
        is_predicted[i] = 1 if roi.is_predicted else 0

    rois_group.attrs["count"] = n
    rois_group.create_dataset("video_index", data=video_index)
    rois_group.create_dataset("frame_index", data=frame_index)
    rois_group.create_dataset("box", data=box, dtype=np.float32)
    rois_group.create_dataset("class_name", data=class_names, dtype=str_dtype)
    rois_group.create_dataset("instance_ref", data=instance_ref)
    rois_group.create_dataset("track_id", data=track_id)
    rois_group.create_dataset("confidence", data=confidence)
    rois_group.create_dataset("is_predicted", data=is_predicted)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def read_segmentation_group(
    file: h5py.File,
    tracks_by_id: dict[int, Any] | None = None,
) -> dict[str, Any]:
    """Read segmentation data from /segmentation in the HDF5 file.

    Returns a dict with:
        "masks": list of (video_index, frame_index, SegmentationMask)
        "rois": list of (video_index, frame_index, ROI)
        "schema_version": str
    """
    seg_group = file.get("segmentation")
    if not isinstance(seg_group, h5py.Group):
        return {"masks": [], "rois": [], "schema_version": ""}

    schema_version = str(seg_group.attrs.get("schema_version", ""))

    masks = _read_masks_subgroup(seg_group, tracks_by_id or {})
    rois = _read_rois_subgroup(seg_group, tracks_by_id or {})

    return {
        "masks": masks,
        "rois": rois,
        "schema_version": schema_version,
    }


def _read_str_dataset_list(ds: h5py.Dataset) -> list[str]:
    """Read a string dataset as a list of Python strings."""
    is_vlen = h5py.check_dtype(vlen=ds.dtype) is str
    is_fixed = np.dtype(ds.dtype).kind in ("S", "U")
    if is_vlen or is_fixed:
        data = ds.asstr()[...]
    else:
        data = ds[...]
    flat = np.ravel(data)
    result: list[str] = []
    for item in flat:
        if isinstance(item, bytes | bytearray | np.bytes_):
            result.append(item.decode("utf-8"))
        else:
            result.append(str(item))
    return result


def _read_masks_subgroup(
    seg_group: h5py.Group,
    tracks_by_id: dict[int, Any],
) -> list[tuple[int, int, SegmentationMask]]:
    """Read mask data from /segmentation/masks."""
    masks_group = seg_group.get("masks")
    if not isinstance(masks_group, h5py.Group):
        return []

    n = int(masks_group.attrs.get("count", 0))
    if n == 0:
        return []

    video_index = np.asarray(masks_group["video_index"][...], dtype=np.int32)
    frame_index = np.asarray(masks_group["frame_index"][...], dtype=np.int32)
    mask_type_arr = np.asarray(masks_group["mask_type"][...], dtype=np.uint8)

    class_names = (
        _read_str_dataset_list(masks_group["class_name"])
        if "class_name" in masks_group
        else [""] * n
    )
    instance_ref = (
        np.asarray(masks_group["instance_ref"][...], dtype=np.int32)
        if "instance_ref" in masks_group
        else np.full(n, -1, dtype=np.int32)
    )
    track_id_arr = (
        np.asarray(masks_group["track_id"][...], dtype=np.int32)
        if "track_id" in masks_group
        else np.full(n, -1, dtype=np.int32)
    )
    confidence_arr = (
        np.asarray(masks_group["confidence"][...], dtype=np.float32)
        if "confidence" in masks_group
        else np.full(n, np.nan, dtype=np.float32)
    )
    is_predicted_arr = (
        np.asarray(masks_group["is_predicted"][...], dtype=np.uint8)
        if "is_predicted" in masks_group
        else np.zeros(n, dtype=np.uint8)
    )

    polygon_offsets = (
        np.asarray(masks_group["polygon_offsets"][...], dtype=np.int64)
        if "polygon_offsets" in masks_group
        else np.zeros(n + 1, dtype=np.int64)
    )
    polygon_ring_offsets = (
        np.asarray(masks_group["polygon_ring_offsets"][...], dtype=np.int64)
        if "polygon_ring_offsets" in masks_group
        else np.zeros(1, dtype=np.int64)
    )
    polygon_vertices = (
        np.asarray(masks_group["polygon_vertices"][...], dtype=np.float32)
        if "polygon_vertices" in masks_group
        else np.zeros((0, 2), dtype=np.float32)
    )

    rle_offsets = (
        np.asarray(masks_group["rle_offsets"][...], dtype=np.int64)
        if "rle_offsets" in masks_group
        else np.zeros(n + 1, dtype=np.int64)
    )
    rle_counts_flat = (
        np.asarray(masks_group["rle_counts"][...], dtype=np.uint32)
        if "rle_counts" in masks_group
        else np.zeros(0, dtype=np.uint32)
    )
    rle_start_arr = (
        np.asarray(masks_group["rle_start"][...], dtype=np.uint8)
        if "rle_start" in masks_group
        else np.zeros(n, dtype=np.uint8)
    )
    rle_height_arr = (
        np.asarray(masks_group["rle_height"][...], dtype=np.int32)
        if "rle_height" in masks_group
        else np.zeros(n, dtype=np.int32)
    )
    rle_width_arr = (
        np.asarray(masks_group["rle_width"][...], dtype=np.int32)
        if "rle_width" in masks_group
        else np.zeros(n, dtype=np.int32)
    )

    prompt_type_arr = (
        np.asarray(masks_group["prompt_type"][...], dtype=np.uint8)
        if "prompt_type" in masks_group
        else np.zeros(n, dtype=np.uint8)
    )
    prompt_box_arr = (
        np.asarray(masks_group["prompt_box"][...], dtype=np.float32)
        if "prompt_box" in masks_group
        else np.full((n, 4), np.nan, dtype=np.float32)
    )
    prompt_text_list = (
        _read_str_dataset_list(masks_group["prompt_text"])
        if "prompt_text" in masks_group
        else [""] * n
    )
    prompt_model_id_list = (
        _read_str_dataset_list(masks_group["prompt_model_id"])
        if "prompt_model_id" in masks_group
        else [""] * n
    )
    prompt_backend_list = (
        _read_str_dataset_list(masks_group["prompt_backend"])
        if "prompt_backend" in masks_group
        else [""] * n
    )

    result: list[tuple[int, int, SegmentationMask]] = []
    for i in range(n):
        mt = MaskType(int(mask_type_arr[i]))

        polygon_verts = None
        if mt == MaskType.POLYGON:
            ring_start = int(polygon_offsets[i])
            ring_end = int(polygon_offsets[i + 1])
            rings = []
            for r in range(ring_start, ring_end):
                v_start = int(polygon_ring_offsets[r])
                v_end = (
                    int(polygon_ring_offsets[r + 1])
                    if (r + 1) < len(polygon_ring_offsets)
                    else v_start
                )
                if v_end > v_start:
                    rings.append(polygon_vertices[v_start:v_end].copy())
            polygon_verts = rings if rings else None

        rle_c = None
        if mt == MaskType.RLE:
            rle_start_idx = int(rle_offsets[i])
            rle_end_idx = int(rle_offsets[i + 1])
            if rle_end_idx > rle_start_idx:
                rle_c = rle_counts_flat[rle_start_idx:rle_end_idx].copy()

        track = tracks_by_id.get(int(track_id_arr[i]))

        prompt = None
        pt = PromptType(int(prompt_type_arr[i]))
        if pt != PromptType.NONE:
            box_vals = prompt_box_arr[i]
            prompt_box = None
            if not np.any(np.isnan(box_vals)):
                prompt_box = (
                    float(box_vals[0]),
                    float(box_vals[1]),
                    float(box_vals[2]),
                    float(box_vals[3]),
                )
            prompt = SegmentationPrompt(
                prompt_type=pt,
                box=prompt_box,
                text=prompt_text_list[i],
                model_id=prompt_model_id_list[i],
                backend=prompt_backend_list[i],
            )

        mask = SegmentationMask(
            mask_type=mt,
            polygon_vertices=polygon_verts,
            rle_counts=rle_c,
            rle_start=int(rle_start_arr[i]),
            rle_height=int(rle_height_arr[i]),
            rle_width=int(rle_width_arr[i]),
            class_name=class_names[i],
            confidence=float(confidence_arr[i]),
            track=track,
            instance_ref=int(instance_ref[i]),
            is_predicted=bool(is_predicted_arr[i]),
            prompt=prompt,
        )
        result.append((int(video_index[i]), int(frame_index[i]), mask))

    return result


def _read_rois_subgroup(
    seg_group: h5py.Group,
    tracks_by_id: dict[int, Any],
) -> list[tuple[int, int, ROI]]:
    """Read ROI data from /segmentation/rois."""
    rois_group = seg_group.get("rois")
    if not isinstance(rois_group, h5py.Group):
        return []

    n = int(rois_group.attrs.get("count", 0))
    if n == 0:
        return []

    video_index = np.asarray(rois_group["video_index"][...], dtype=np.int32)
    frame_index = np.asarray(rois_group["frame_index"][...], dtype=np.int32)
    box_arr = np.asarray(rois_group["box"][...], dtype=np.float32)

    class_names = (
        _read_str_dataset_list(rois_group["class_name"])
        if "class_name" in rois_group
        else [""] * n
    )
    instance_ref = (
        np.asarray(rois_group["instance_ref"][...], dtype=np.int32)
        if "instance_ref" in rois_group
        else np.full(n, -1, dtype=np.int32)
    )
    track_id_arr = (
        np.asarray(rois_group["track_id"][...], dtype=np.int32)
        if "track_id" in rois_group
        else np.full(n, -1, dtype=np.int32)
    )
    confidence_arr = (
        np.asarray(rois_group["confidence"][...], dtype=np.float32)
        if "confidence" in rois_group
        else np.full(n, np.nan, dtype=np.float32)
    )
    is_predicted_arr = (
        np.asarray(rois_group["is_predicted"][...], dtype=np.uint8)
        if "is_predicted" in rois_group
        else np.zeros(n, dtype=np.uint8)
    )

    result: list[tuple[int, int, ROI]] = []
    for i in range(n):
        track = tracks_by_id.get(int(track_id_arr[i]))
        roi = ROI(
            x1=float(box_arr[i, 0]),
            y1=float(box_arr[i, 1]),
            x2=float(box_arr[i, 2]),
            y2=float(box_arr[i, 3]),
            class_name=class_names[i],
            confidence=float(confidence_arr[i]),
            track=track,
            instance_ref=int(instance_ref[i]),
            is_predicted=bool(is_predicted_arr[i]),
        )
        result.append((int(video_index[i]), int(frame_index[i]), roi))

    return result


__all__ = [
    "SEGMENTATION_SCHEMA_VERSION",
    "read_segmentation_group",
    "write_segmentation_group",
]
