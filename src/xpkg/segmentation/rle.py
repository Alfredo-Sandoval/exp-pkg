"""Exact binary-mask RLE helpers for segmentation storage.

The xpkg RLE contract is intentionally simple and JSON-safe:

- row-major/C-order flattening
- an explicit starting pixel value
- integer run lengths
- explicit ``[height, width]`` size metadata

COCO RLE is handled separately in :mod:`xpkg.segmentation.coco` because COCO
uses its own ordering and compressed-count conventions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

XPKG_RLE_ENCODING = "xpkg.rle.v1"
XPKG_RLE_ORDER = "C"


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape {arr.shape}")
    return (arr > 0).astype(np.uint8, copy=False)


def _as_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"RLE field {field!r} must be an integer.")
    return int(value)


def rle_encode(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Encode a 2D binary mask as row-major run lengths (low-level numpy API).

    This is the low-level counterpart to :func:`encode_mask_rle`. Use this when
    you want the raw arrays without the JSON-payload wrapper; use
    :func:`encode_mask_rle` when you want the canonical xpkg RLE dict that
    embeds the encoding tag, ordering, and size metadata for serialization.

    Returns:
        ``(counts, start)`` where counts is a ``uint32`` array and start is
        the first pixel value, either ``0`` or ``1``.
    """

    binary = _binary_mask(mask)
    flat = binary.ravel(order=XPKG_RLE_ORDER)
    if flat.size == 0:
        return np.zeros(0, dtype=np.uint32), 0
    start = int(flat[0])
    transitions = np.flatnonzero(np.diff(flat) != 0) + 1
    boundaries = np.concatenate(([0], transitions, [flat.size]))
    counts = np.diff(boundaries).astype(np.uint32)
    return counts, start


def rle_decode(
    counts: np.ndarray | Sequence[int],
    start: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Decode row-major RLE counts into a ``uint8`` binary mask (low-level API).

    Low-level counterpart to :func:`decode_mask_rle`: takes the raw
    ``(counts, start, height, width)`` tuple form rather than the canonical
    xpkg RLE payload dict.
    """

    if start not in (0, 1):
        raise ValueError("RLE start must be 0 or 1.")
    if height < 0 or width < 0:
        raise ValueError("RLE height and width must be non-negative.")
    counts_arr = np.asarray(counts, dtype=np.int64)
    if np.any(counts_arr < 0):
        raise ValueError("RLE counts must be non-negative.")

    total_pixels = int(height) * int(width)
    if counts_arr.size == 0:
        if total_pixels == 0:
            return np.zeros((height, width), dtype=np.uint8)
        raise ValueError(
            f"RLE counts sum to 0 but expected {total_pixels} ({height}x{width})"
        )
    actual_total = int(counts_arr.sum())
    if actual_total != total_pixels:
        raise ValueError(
            f"RLE counts sum to {actual_total} but expected {total_pixels} "
            f"({height}x{width})"
        )

    values = np.empty(counts_arr.size, dtype=np.uint8)
    values[0::2] = int(start)
    values[1::2] = 1 - int(start)
    flat = np.repeat(values, counts_arr).astype(np.uint8, copy=False)
    return flat.reshape((height, width), order=XPKG_RLE_ORDER)


def encode_mask_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask into the canonical xpkg RLE payload (JSON dict API).

    This is the JSON-serializable counterpart to :func:`rle_encode`. The result
    is the dict form embedded in :class:`SegmentationMask` payloads and on disk
    (encoding tag, ``size``, ``order``, ``start``, ``counts``).
    """

    binary = _binary_mask(mask)
    counts, start = rle_encode(binary)
    height, width = binary.shape
    return {
        "encoding": XPKG_RLE_ENCODING,
        "size": [int(height), int(width)],
        "order": XPKG_RLE_ORDER,
        "start": int(start),
        "counts": [int(value) for value in counts.tolist()],
    }


def encode_masks_rle(masks: Sequence[np.ndarray]) -> list[dict[str, Any]]:
    """Encode a sequence of masks into xpkg RLE payloads."""

    return [encode_mask_rle(np.asarray(mask)) for mask in masks]


def decode_mask_rle(mask_rle: Mapping[str, Any]) -> np.ndarray:
    """Decode an xpkg RLE payload dict into a ``uint8`` binary mask (JSON dict API).

    JSON-payload counterpart to :func:`rle_decode`: takes the canonical xpkg RLE
    dict produced by :func:`encode_mask_rle` rather than the low-level
    ``(counts, start, height, width)`` tuple form.
    """

    size = mask_rle.get("size")
    if size is None and "height" in mask_rle and "width" in mask_rle:
        size = [mask_rle["height"], mask_rle["width"]]
    if not isinstance(size, Sequence) or isinstance(size, str | bytes) or len(size) != 2:
        raise ValueError("RLE field 'size' must be [height, width].")
    height = _as_int(size[0], field="size[0]")
    width = _as_int(size[1], field="size[1]")
    order = str(mask_rle.get("order", XPKG_RLE_ORDER))
    if order != XPKG_RLE_ORDER:
        raise ValueError(f"Unsupported xpkg RLE order {order!r}; expected 'C'.")
    start = _as_int(mask_rle.get("start", 0), field="start")
    counts_raw = mask_rle.get("counts")
    if not isinstance(counts_raw, Sequence) or isinstance(counts_raw, str | bytes):
        raise ValueError("RLE field 'counts' must be a sequence of integers.")
    counts = [_as_int(value, field="counts[]") for value in counts_raw]
    return rle_decode(counts, start, height, width)


__all__ = [
    "XPKG_RLE_ENCODING",
    "XPKG_RLE_ORDER",
    "decode_mask_rle",
    "encode_mask_rle",
    "encode_masks_rle",
    "rle_decode",
    "rle_encode",
]
