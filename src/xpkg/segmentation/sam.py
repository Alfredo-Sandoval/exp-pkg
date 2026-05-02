"""Adapters for SAM/SAM2/SAM3-style segmentation outputs.

This module deliberately does not run SAM models. It only converts model
outputs from callers such as Fiesta into xpkg segmentation masks and ROIs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from xpkg.segmentation.images import read_binary_mask
from xpkg.segmentation.model import (
    ROI,
    PromptType,
    SegmentationMask,
    SegmentationPrompt,
)
from xpkg.segmentation.rle import decode_mask_rle


@dataclass(frozen=True, slots=True)
class SamSegmentationResult:
    """Neutral mask/ROI output converted from a SAM-family result."""

    masks: tuple[SegmentationMask, ...]
    rois: tuple[ROI, ...] = ()


def mask_from_sam_array(
    mask: np.ndarray,
    *,
    class_name: str = "",
    confidence: float = float("nan"),
    prompt: SegmentationPrompt | None = None,
    backend: str = "",
    model_id: str = "",
    instance_ref: int = -1,
    is_predicted: bool = True,
) -> SegmentationMask:
    """Convert one SAM-family binary mask array into an xpkg RLE mask."""

    return SegmentationMask.from_binary_mask(
        np.asarray(mask),
        class_name=class_name,
        confidence=confidence,
        instance_ref=instance_ref,
        is_predicted=is_predicted,
        prompt=prompt
        if prompt is not None
        else _prompt(prompt_text="", backend=backend, model_id=model_id),
    )


def masks_from_sam_arrays(
    masks: Sequence[np.ndarray],
    *,
    boxes: Sequence[Sequence[float]] | None = None,
    scores: Sequence[float | None] | None = None,
    prompt_text: str = "",
    backend: str = "",
    model_id: str = "",
    class_names: Sequence[str] | None = None,
    is_predicted: bool = True,
) -> SamSegmentationResult:
    """Convert SAM masks, boxes, and scores into xpkg masks plus ROIs."""

    converted_masks: list[SegmentationMask] = []
    converted_rois: list[ROI] = []
    prompt = _prompt(prompt_text=prompt_text, backend=backend, model_id=model_id)
    for index, mask in enumerate(masks):
        class_name = _class_name(class_names, index)
        confidence = _score(scores, index)
        converted_masks.append(
            SegmentationMask.from_binary_mask(
                np.asarray(mask),
                class_name=class_name,
                confidence=confidence,
                instance_ref=index,
                is_predicted=is_predicted,
                prompt=prompt,
            )
        )
        box = _box(boxes, index)
        if box is not None:
            converted_rois.append(
                ROI(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    class_name=class_name,
                    confidence=confidence,
                    instance_ref=index,
                    is_predicted=is_predicted,
                )
            )
    return SamSegmentationResult(tuple(converted_masks), tuple(converted_rois))


def masks_from_fiesta_summary(
    summary: Mapping[str, Any],
    *,
    result_root: str | Path | None = None,
    load_mask_paths: bool = True,
    is_predicted: bool = True,
) -> SamSegmentationResult:
    """Convert a Fiesta ``result.json`` payload into xpkg masks and ROIs."""

    root = Path(result_root) if result_root is not None else Path(".")
    if "results" in summary and isinstance(summary["results"], list):
        all_masks: list[SegmentationMask] = []
        all_rois: list[ROI] = []
        for frame_payload in summary["results"]:
            if not isinstance(frame_payload, Mapping):
                continue
            result = _masks_from_fiesta_entry(
                frame_payload,
                root=root,
                inherited=summary,
                load_mask_paths=load_mask_paths,
                is_predicted=is_predicted,
            )
            all_masks.extend(result.masks)
            all_rois.extend(result.rois)
        return SamSegmentationResult(tuple(all_masks), tuple(all_rois))
    return _masks_from_fiesta_entry(
        summary,
        root=root,
        inherited=summary,
        load_mask_paths=load_mask_paths,
        is_predicted=is_predicted,
    )


def masks_from_fiesta_result_json(
    path: str | Path,
    *,
    load_mask_paths: bool = True,
    is_predicted: bool = True,
) -> SamSegmentationResult:
    """Read a Fiesta ``result.json`` and convert masks/ROIs to xpkg objects."""

    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    return masks_from_fiesta_summary(
        payload,
        result_root=target.parent,
        load_mask_paths=load_mask_paths,
        is_predicted=is_predicted,
    )


def _masks_from_fiesta_entry(
    entry: Mapping[str, Any],
    *,
    root: Path,
    inherited: Mapping[str, Any],
    load_mask_paths: bool,
    is_predicted: bool,
) -> SamSegmentationResult:
    prompt_text = str(entry.get("prompt", inherited.get("prompt", "")))
    backend = str(entry.get("backend", inherited.get("backend", "")))
    model_id = str(entry.get("model_id", inherited.get("model_id", "")))
    prompt = _prompt(prompt_text=prompt_text, backend=backend, model_id=model_id)
    scores = _sequence(entry.get("scores"))
    boxes = _sequence(entry.get("boxes"))
    class_names = _class_names(entry, inherited)

    masks: list[SegmentationMask] = []
    rois: list[ROI] = []
    for index, rle_payload in enumerate(_sequence(entry.get("mask_rles"))):
        if not isinstance(rle_payload, Mapping):
            raise ValueError("Fiesta mask_rles entries must be objects.")
        masks.append(
            SegmentationMask.from_binary_mask(
                decode_mask_rle(rle_payload),
                class_name=_class_name(class_names, index),
                confidence=_score(scores, index),
                instance_ref=index,
                is_predicted=is_predicted,
                prompt=prompt,
            )
        )

    if load_mask_paths:
        for index, raw_path in enumerate(_sequence(entry.get("mask_paths"))):
            if not isinstance(raw_path, str):
                raise ValueError("Fiesta mask_paths entries must be strings.")
            path = _resolve_relative_path(root, raw_path)
            masks.append(
                SegmentationMask.from_binary_mask(
                    read_binary_mask(path),
                    class_name=_class_name(class_names, index),
                    confidence=_score(scores, index),
                    instance_ref=index,
                    is_predicted=is_predicted,
                    prompt=prompt,
                    mask_path=path.as_posix(),
                )
            )

    for index, raw_box in enumerate(boxes):
        if not isinstance(raw_box, Sequence) or len(raw_box) != 4:
            continue
        rois.append(
            ROI(
                x1=float(raw_box[0]),
                y1=float(raw_box[1]),
                x2=float(raw_box[2]),
                y2=float(raw_box[3]),
                class_name=_class_name(class_names, index),
                confidence=_score(scores, index),
                instance_ref=index,
                is_predicted=is_predicted,
            )
        )
    return SamSegmentationResult(tuple(masks), tuple(rois))


def _prompt(*, prompt_text: str, backend: str, model_id: str) -> SegmentationPrompt:
    prompt_type = PromptType.TEXT if prompt_text.strip() else PromptType.NONE
    return SegmentationPrompt(
        prompt_type=prompt_type,
        text=prompt_text,
        backend=backend,
        model_id=model_id,
    )


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return list(value)
    return []


def _class_names(entry: Mapping[str, Any], inherited: Mapping[str, Any]) -> list[str] | None:
    for key in ("class_names", "track_labels", "labels"):
        values = _sequence(entry.get(key, inherited.get(key)))
        if values:
            return [str(value) for value in values]
    return None


def _class_name(class_names: Sequence[str] | None, index: int) -> str:
    if class_names is None or index >= len(class_names):
        return f"mask_{index:03d}"
    return str(class_names[index])


def _score(scores: Sequence[Any] | None, index: int) -> float:
    if scores is None or index >= len(scores) or scores[index] is None:
        return float("nan")
    score = float(scores[index])
    if not np.isfinite(score):
        return float("nan")
    return score


def _box(boxes: Sequence[Sequence[float]] | None, index: int) -> Sequence[float] | None:
    if boxes is None or index >= len(boxes):
        return None
    box = boxes[index]
    if len(box) != 4:
        return None
    return box


def _resolve_relative_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


__all__ = [
    "SamSegmentationResult",
    "mask_from_sam_array",
    "masks_from_fiesta_result_json",
    "masks_from_fiesta_summary",
    "masks_from_sam_arrays",
]
