"""Segmentation mask models, encodings, image IO, and interop adapters."""

from __future__ import annotations

from xpkg.segmentation.coco import (
    annotation_to_masks,
    annotations_to_masks,
    coco_rle_to_mask,
    mask_to_coco_annotation,
    mask_to_coco_polygons,
    mask_to_coco_rle,
    read_coco_annotations,
    segmentation_to_masks,
)
from xpkg.segmentation.images import (
    masks_from_label_image,
    read_binary_mask,
    read_label_image,
    write_binary_mask,
    write_label_image,
)
from xpkg.segmentation.model import (
    ROI,
    MaskType,
    PromptType,
    SegmentationFrame,
    SegmentationMask,
    SegmentationPrompt,
    rasterize_polygon,
)
from xpkg.segmentation.rle import (
    XPKG_RLE_ENCODING,
    XPKG_RLE_ORDER,
    decode_mask_rle,
    encode_mask_rle,
    encode_masks_rle,
    rle_decode,
    rle_encode,
)
from xpkg.segmentation.sam import (
    SamSegmentationResult,
    mask_from_sam_array,
    masks_from_fiesta_result_json,
    masks_from_fiesta_summary,
    masks_from_sam_arrays,
)

__all__ = [
    "XPKG_RLE_ENCODING",
    "XPKG_RLE_ORDER",
    "MaskType",
    "PromptType",
    "ROI",
    "SamSegmentationResult",
    "SegmentationFrame",
    "SegmentationMask",
    "SegmentationPrompt",
    "annotation_to_masks",
    "annotations_to_masks",
    "coco_rle_to_mask",
    "decode_mask_rle",
    "encode_mask_rle",
    "encode_masks_rle",
    "mask_from_sam_array",
    "mask_to_coco_annotation",
    "mask_to_coco_polygons",
    "mask_to_coco_rle",
    "masks_from_fiesta_result_json",
    "masks_from_fiesta_summary",
    "masks_from_label_image",
    "masks_from_sam_arrays",
    "rasterize_polygon",
    "read_binary_mask",
    "read_coco_annotations",
    "read_label_image",
    "rle_decode",
    "rle_encode",
    "segmentation_to_masks",
    "write_binary_mask",
    "write_label_image",
]
