"""COCO instance-segmentation adapters.

The internal xpkg RLE contract is row-major and explicit about start value.
COCO RLE is column-major and always alternates from background counts. These
helpers keep that conversion boundary visible.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from xpkg._core.json_utils import load_json_dict
from xpkg.segmentation.model import SegmentationMask


def mask_to_coco_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask as uncompressed COCO RLE."""

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape {binary.shape}")
    height, width = binary.shape
    flat = binary.ravel(order="F")
    counts: list[int] = []
    current = 0
    run_length = 0
    for pixel in flat:
        value = int(pixel)
        if value == current:
            run_length += 1
            continue
        counts.append(run_length)
        current = value
        run_length = 1
    counts.append(run_length)
    return {"size": [int(height), int(width)], "counts": counts}


def coco_rle_to_mask(rle: Mapping[str, Any]) -> np.ndarray:
    """Decode uncompressed or pycocotools-compressed COCO RLE."""

    size = rle.get("size")
    if not isinstance(size, Sequence) or isinstance(size, str | bytes) or len(size) != 2:
        raise ValueError("COCO RLE field 'size' must be [height, width].")
    height = int(size[0])
    width = int(size[1])
    counts = rle.get("counts")
    if isinstance(counts, str | bytes):
        return _decode_compressed_coco_rle(rle).astype(np.uint8)
    if not isinstance(counts, Sequence):
        raise ValueError("COCO RLE field 'counts' must be a sequence or compressed string.")
    values = np.empty(len(counts), dtype=np.uint8)
    values[0::2] = 0
    values[1::2] = 1
    flat = np.repeat(values, np.asarray(counts, dtype=np.int64))
    expected = height * width
    if flat.size != expected:
        raise ValueError(f"COCO RLE decode size mismatch: expected {expected}, got {flat.size}.")
    return flat.reshape((height, width), order="F").astype(np.uint8)


def segmentation_to_masks(
    segmentation: Any,
    *,
    image_size: tuple[int, int] | None = None,
    class_name: str = "",
    confidence: float = float("nan"),
    is_predicted: bool = False,
) -> list[SegmentationMask]:
    """Convert a COCO ``segmentation`` field into xpkg masks."""

    if isinstance(segmentation, Mapping):
        mask = coco_rle_to_mask(segmentation)
        return [
            SegmentationMask.from_binary_mask(
                mask,
                class_name=class_name,
                confidence=confidence,
                is_predicted=is_predicted,
            )
        ]
    if isinstance(segmentation, Sequence) and not isinstance(segmentation, str | bytes):
        masks: list[SegmentationMask] = []
        for item in segmentation:
            coords = np.asarray(item, dtype=np.float32)
            if coords.size % 2 != 0:
                raise ValueError("COCO polygon segmentations must contain x/y pairs.")
            vertices = coords.reshape((-1, 2))
            masks.append(
                SegmentationMask.from_polygon(
                    vertices,
                    class_name=class_name,
                    confidence=confidence,
                    is_predicted=is_predicted,
                )
            )
        return masks
    if image_size is None:
        raise ValueError("Unsupported COCO segmentation payload.")
    return []


def annotation_to_masks(
    annotation: Mapping[str, Any],
    *,
    category_names: Mapping[int, str] | None = None,
    is_predicted: bool = False,
) -> list[SegmentationMask]:
    """Convert one COCO annotation into one or more xpkg masks."""

    category_id = int(annotation.get("category_id", -1))
    class_name = (
        str(category_names[category_id])
        if category_names is not None and category_id in category_names
        else str(category_id) if category_id >= 0 else ""
    )
    masks = segmentation_to_masks(
        annotation.get("segmentation"),
        class_name=class_name,
        confidence=float(annotation.get("score", float("nan"))),
        is_predicted=is_predicted,
    )
    for mask in masks:
        mask.instance_ref = int(annotation.get("id", -1))
    return masks


def annotations_to_masks(
    annotations: Sequence[Mapping[str, Any]],
    *,
    categories: Sequence[Mapping[str, Any]] | None = None,
    is_predicted: bool = False,
) -> list[SegmentationMask]:
    """Convert COCO annotations into xpkg segmentation masks."""

    category_names = _category_names(categories)
    masks: list[SegmentationMask] = []
    for annotation in annotations:
        masks.extend(
            annotation_to_masks(
                annotation,
                category_names=category_names,
                is_predicted=is_predicted,
            )
        )
    return masks


def mask_to_coco_annotation(
    mask: SegmentationMask,
    *,
    image_id: int | str,
    category_id: int,
    annotation_id: int | str | None = None,
    use_rle: bool = True,
) -> dict[str, Any]:
    """Convert one xpkg mask into a COCO annotation dictionary."""

    bbox = [float(value) for value in mask.bounding_box.tolist()]
    payload: dict[str, Any] = {
        "image_id": image_id,
        "category_id": int(category_id),
        "bbox": bbox,
        "iscrowd": 1 if use_rle else 0,
    }
    if annotation_id is not None:
        payload["id"] = annotation_id
    if not np.isnan(mask.confidence):
        payload["score"] = float(mask.confidence)

    if use_rle:
        binary = mask.to_binary_mask()
        payload["segmentation"] = mask_to_coco_rle(binary)
        payload["area"] = int(np.count_nonzero(binary))
        return payload

    if mask.polygon_vertices is None:
        binary = mask.to_binary_mask()
        payload["segmentation"] = mask_to_coco_polygons(binary)
        payload["area"] = int(np.count_nonzero(binary))
        return payload

    payload["segmentation"] = [
        [float(value) for value in vertices.reshape(-1).tolist()]
        for vertices in mask.polygon_vertices
    ]
    payload["area"] = float(_polygon_area(mask.polygon_vertices))
    return payload


def mask_to_coco_polygons(mask: np.ndarray) -> list[list[float]]:
    """Approximate a binary mask as COCO polygon contours."""

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in contours:
        contour = contour.reshape((-1, 2))
        if contour.shape[0] < 3:
            continue
        polygons.append([float(value) for value in contour.reshape(-1).tolist()])
    return polygons


def read_coco_annotations(
    path: str | Path,
    *,
    is_predicted: bool = False,
) -> list[SegmentationMask]:
    """Read a COCO JSON file and return segmentation masks."""

    payload = load_json_dict(Path(path))
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])
    if not isinstance(annotations, list):
        raise ValueError("COCO payload field 'annotations' must be a list.")
    if not isinstance(categories, list):
        raise ValueError("COCO payload field 'categories' must be a list.")
    return annotations_to_masks(
        annotations,
        categories=categories,
        is_predicted=is_predicted,
    )


def _category_names(categories: Sequence[Mapping[str, Any]] | None) -> dict[int, str]:
    if categories is None:
        return {}
    names: dict[int, str] = {}
    for category in categories:
        names[int(category["id"])] = str(category.get("name", category["id"]))
    return names


def _decode_compressed_coco_rle(rle: Mapping[str, Any]) -> np.ndarray:
    try:
        coco_mask = importlib.import_module("pycocotools.mask")
    except ImportError as exc:
        raise ImportError(
            "Compressed COCO RLE requires pycocotools. Install it in an optional "
            "segmentation environment to decode compressed counts."
        ) from exc
    return np.asarray(coco_mask.decode(dict(rle)), dtype=np.uint8)


def _polygon_area(vertices: Sequence[np.ndarray]) -> float:
    return float(sum(abs(cv2.contourArea(np.asarray(ring, dtype=np.float32))) for ring in vertices))


__all__ = [
    "annotation_to_masks",
    "annotations_to_masks",
    "coco_rle_to_mask",
    "mask_to_coco_annotation",
    "mask_to_coco_polygons",
    "mask_to_coco_rle",
    "read_coco_annotations",
    "segmentation_to_masks",
]
