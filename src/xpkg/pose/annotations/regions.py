"""Compatibility re-exports for segmentation annotation types.

The concrete implementation lives in :mod:`xpkg.segmentation`. This module
remains so older pose-facing imports keep working.
"""

from __future__ import annotations

from xpkg.segmentation import (
    ROI,
    MaskType,
    PromptType,
    SegmentationMask,
    SegmentationPrompt,
    rasterize_polygon,
    rle_decode,
    rle_encode,
)

__all__ = [
    "MaskType",
    "PromptType",
    "ROI",
    "SegmentationMask",
    "SegmentationPrompt",
    "rasterize_polygon",
    "rle_decode",
    "rle_encode",
]
