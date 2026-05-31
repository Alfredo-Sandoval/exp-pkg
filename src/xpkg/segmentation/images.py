"""PNG/TIFF mask image helpers for segmentation masks."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Literal

import cv2
import numpy as np

ImageColorOrder = Literal["rgb", "bgr"]
type BoxXYXY = tuple[float, float, float, float]


class SupervisionOverlayError(RuntimeError):
    """Raised when Roboflow supervision cannot render a segmentation overlay."""


def _read_unchanged(path: str | Path) -> np.ndarray:
    target = Path(path)
    image = cv2.imread(target.as_posix(), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read mask image: {target}")
    if image.ndim == 3:
        image = image[:, :, 0]
    return np.asarray(image)


def _png_write_params(path: Path, png_compression: int | None) -> list[int]:
    if png_compression is None:
        return []
    compression = int(png_compression)
    if compression < 0 or compression > 9:
        raise ValueError("png_compression must be between 0 and 9.")
    if path.suffix.lower() != ".png":
        return []
    return [int(cv2.IMWRITE_PNG_COMPRESSION), compression]


def _as_rgb_uint8_image(image: np.ndarray, *, image_color: ImageColorOrder) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        rgb = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] in {3, 4}:
        channels = arr[:, :, :3]
        if image_color == "rgb":
            rgb = channels
        elif image_color == "bgr":
            rgb = channels[:, :, ::-1]
        else:
            raise ValueError("image_color must be 'rgb' or 'bgr'.")
    else:
        raise ValueError(f"Expected HxW, HxWx3, or HxWx4 image, got shape {arr.shape}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"Overlay image must use uint8 pixels, got {rgb.dtype}")
    return np.asarray(rgb).copy()


def _as_binary_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape {arr.shape}")
    return arr > 0


def _rgb_triplet(values: Sequence[int], *, field: str) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError(f"{field} must contain three RGB channel values.")
    red, green, blue = (int(value) for value in values)
    channels = (red, green, blue)
    if any(value < 0 or value > 255 for value in channels):
        raise ValueError(f"{field} channel values must be between 0 and 255.")
    return channels


def _box_xyxy(box: Sequence[float]) -> tuple[int, int, int, int]:
    if len(box) != 4:
        raise ValueError("box must contain four xyxy values.")
    x0, y0, x1, y1 = (int(round(float(value))) for value in box)
    return x0, y0, x1, y1


def _load_supervision() -> ModuleType:
    try:
        return import_module("supervision")
    except ModuleNotFoundError as exc:
        if exc.name != "supervision":
            raise
        raise SupervisionOverlayError(
            "Roboflow supervision is required to render segmentation overlays."
        ) from exc


def read_binary_mask(path: str | Path, *, threshold: int = 0) -> np.ndarray:
    """Read a PNG/TIFF mask image as a ``uint8`` binary array."""

    image = _read_unchanged(path)
    return (image > int(threshold)).astype(np.uint8)


def write_binary_mask(
    path: str | Path,
    mask: np.ndarray,
    *,
    true_value: int = 255,
    png_compression: int | None = None,
) -> Path:
    """Write a 2D binary mask as PNG/TIFF using ``0`` and ``true_value``."""

    target = Path(path)
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape {arr.shape}")
    target.parent.mkdir(parents=True, exist_ok=True)
    image = (arr > 0).astype(np.uint8) * int(true_value)
    params = _png_write_params(target, png_compression)
    if not cv2.imwrite(target.as_posix(), image, params):
        raise OSError(f"Could not write binary mask image: {target}")
    return target


def write_binary_masks(
    output_dir: str | Path,
    masks: Sequence[np.ndarray],
    *,
    file_prefix: str = "mask",
    true_value: int = 255,
    png_compression: int | None = None,
) -> list[Path]:
    """Write one binary PNG mask per array and return output paths in order."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, mask in enumerate(masks):
        path = root / f"{file_prefix}_{index:03d}.png"
        write_binary_mask(
            path,
            mask,
            true_value=true_value,
            png_compression=png_compression,
        )
        paths.append(path)
    return paths


def write_mask_overlay(
    path: str | Path,
    image: np.ndarray,
    mask: np.ndarray,
    *,
    tint_rgb: Sequence[int],
    opacity: float,
    box: Sequence[float] | None = None,
    box_outline_rgb: Sequence[int] = (255, 255, 0),
    resize_mask_to_image: bool = False,
    image_color: ImageColorOrder = "rgb",
    png_compression: int | None = None,
) -> Path:
    """Write an RGB/BGR image with a binary mask tint and optional xyxy box."""

    target = Path(path)
    rgb = _as_rgb_uint8_image(image, image_color=image_color)
    binary_mask = _as_binary_mask(mask)
    if binary_mask.shape != rgb.shape[:2]:
        if not resize_mask_to_image:
            raise ValueError(
                f"Mask shape {binary_mask.shape} does not match image shape {rgb.shape[:2]}"
            )
        width = int(rgb.shape[1])
        height = int(rgb.shape[0])
        binary_mask = cv2.resize(
            binary_mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    alpha = float(opacity)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("opacity must be between 0 and 1.")
    tint = np.asarray(_rgb_triplet(tint_rgb, field="tint_rgb"), dtype=np.float32)
    output = np.where(
        binary_mask[:, :, None],
        rgb.astype(np.float32) * (1.0 - alpha) + tint * alpha,
        rgb.astype(np.float32),
    )
    overlay = np.rint(output).clip(0, 255).astype(np.uint8)
    if box is not None:
        x0, y0, x1, y1 = _box_xyxy(box)
        outline = _rgb_triplet(box_outline_rgb, field="box_outline_rgb")
        cv2.rectangle(overlay, (x0, y0), (x1, y1), outline, thickness=3)

    target.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    params = _png_write_params(target, png_compression)
    if not cv2.imwrite(target.as_posix(), bgr, params):
        raise OSError(f"Could not write mask overlay image: {target}")
    return target


def render_supervision_overlay(
    image: np.ndarray,
    *,
    masks: Sequence[np.ndarray] | None = None,
    boxes_xyxy: Sequence[Sequence[float]] | None = None,
    labels: Sequence[str | None] | None = None,
    confidences: Sequence[float | None] | None = None,
    class_ids: Sequence[int | None] | None = None,
    class_names: Sequence[str | None] | None = None,
    tracker_ids: Sequence[int | None] | None = None,
    opacity: float = 0.35,
    resize_masks_to_image: bool = False,
    image_color: ImageColorOrder = "rgb",
    color_by_tracker_id: bool | None = None,
    palette_rgb: Sequence[Sequence[int]] | None = None,
    box_thickness: int = 2,
    text_scale: float = 0.45,
    text_thickness: int = 1,
    text_padding: int = 4,
    confidence_digits: int = 2,
    fallback_label: str = "object",
) -> np.ndarray:
    """Render masks, boxes, labels, and tracker IDs through Roboflow supervision."""

    supervision = _load_supervision()
    rgb = _as_rgb_uint8_image(image, image_color=image_color)
    mask_array = _overlay_masks(
        masks,
        image_shape=rgb.shape[:2],
        resize_masks_to_image=resize_masks_to_image,
    )
    boxes = _overlay_boxes(boxes_xyxy, mask_array=mask_array)
    count = int(boxes.shape[0])
    if count == 0:
        raise ValueError("supervision overlay requires at least one mask or box")

    label_values = format_supervision_overlay_labels(
        labels=labels,
        confidences=confidences,
        tracker_ids=tracker_ids,
        count=count,
        confidence_digits=confidence_digits,
        fallback_label=fallback_label,
    )
    tracker_array = _overlay_tracker_ids(tracker_ids, count=count)
    color_lookup = _overlay_color_lookup(
        supervision,
        tracker_array=tracker_array,
        color_by_tracker_id=color_by_tracker_id,
    )
    detections = supervision.Detections(
        xyxy=boxes,
        mask=mask_array,
        confidence=_overlay_confidences(confidences, count=count),
        class_id=_overlay_class_ids(class_ids, count=count),
        tracker_id=tracker_array,
        data=_overlay_data(
            labels=label_values,
            class_names=class_names,
            count=count,
        ),
    )
    scene = np.ascontiguousarray(rgb.copy())
    palette = _supervision_palette(supervision, palette_rgb)
    if mask_array is not None:
        mask_kwargs = _annotator_kwargs(
            color_lookup=color_lookup,
            color=palette,
            opacity=float(opacity),
        )
        scene = np.asarray(
            supervision.MaskAnnotator(**mask_kwargs).annotate(
                scene=scene,
                detections=detections,
            )
        )
    box_kwargs = _annotator_kwargs(
        color_lookup=color_lookup,
        color=palette,
        thickness=int(box_thickness),
    )
    scene = np.asarray(
        supervision.BoxAnnotator(**box_kwargs).annotate(
            scene=scene,
            detections=detections,
        )
    )
    label_kwargs = _annotator_kwargs(
        color_lookup=color_lookup,
        color=palette,
        text_color=supervision.Color.WHITE,
        text_scale=float(text_scale),
        text_thickness=int(text_thickness),
        text_padding=int(text_padding),
    )
    scene = np.asarray(
        supervision.LabelAnnotator(**label_kwargs).annotate(
            scene=scene,
            detections=detections,
            labels=label_values,
        )
    )
    return np.asarray(scene, dtype=np.uint8)


def write_supervision_overlay(
    path: str | Path,
    image: np.ndarray,
    *,
    masks: Sequence[np.ndarray] | None = None,
    boxes_xyxy: Sequence[Sequence[float]] | None = None,
    labels: Sequence[str | None] | None = None,
    confidences: Sequence[float | None] | None = None,
    class_ids: Sequence[int | None] | None = None,
    class_names: Sequence[str | None] | None = None,
    tracker_ids: Sequence[int | None] | None = None,
    opacity: float = 0.35,
    resize_masks_to_image: bool = False,
    image_color: ImageColorOrder = "rgb",
    color_by_tracker_id: bool | None = None,
    palette_rgb: Sequence[Sequence[int]] | None = None,
    box_thickness: int = 2,
    text_scale: float = 0.45,
    text_thickness: int = 1,
    text_padding: int = 4,
    confidence_digits: int = 2,
    fallback_label: str = "object",
    png_compression: int | None = None,
) -> Path:
    """Write a Roboflow supervision-rendered overlay image."""

    target = Path(path)
    overlay = render_supervision_overlay(
        image,
        masks=masks,
        boxes_xyxy=boxes_xyxy,
        labels=labels,
        confidences=confidences,
        class_ids=class_ids,
        class_names=class_names,
        tracker_ids=tracker_ids,
        opacity=opacity,
        resize_masks_to_image=resize_masks_to_image,
        image_color=image_color,
        color_by_tracker_id=color_by_tracker_id,
        palette_rgb=palette_rgb,
        box_thickness=box_thickness,
        text_scale=text_scale,
        text_thickness=text_thickness,
        text_padding=text_padding,
        confidence_digits=confidence_digits,
        fallback_label=fallback_label,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    params = _png_write_params(target, png_compression)
    if not cv2.imwrite(target.as_posix(), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), params):
        raise OSError(f"Could not write supervision overlay image: {target}")
    return target


def format_supervision_overlay_labels(
    *,
    labels: Sequence[str | None] | None,
    confidences: Sequence[float | None] | None = None,
    tracker_ids: Sequence[int | None] | None = None,
    count: int | None = None,
    confidence_digits: int = 2,
    fallback_label: str = "object",
) -> list[str]:
    """Format compact Roboflow labels with DETR-style tracker prefixes."""

    label_count = _resolve_label_count(labels, confidences, tracker_ids, count=count)
    label_values = _optional_rows(labels, count=label_count, label="labels")
    confidence_values = _optional_rows(confidences, count=label_count, label="confidences")
    tracker_values = _optional_rows(tracker_ids, count=label_count, label="tracker_ids")
    include_tracker_id = tracker_values is not None
    formatted: list[str] = []
    for index in range(label_count):
        label = fallback_label
        if label_values is not None and label_values[index] is not None:
            candidate = str(label_values[index]).strip()
            if candidate:
                label = candidate
        if include_tracker_id:
            if tracker_values is None:
                raise ValueError("tracker values unexpectedly missing")
            tracker_label = _tracker_label(tracker_values[index])
            label = f"#{tracker_label} {label}"
        if confidence_values is not None:
            confidence = _optional_confidence(confidence_values[index])
            if confidence is not None:
                label = f"{label} {confidence:.{confidence_digits}f}"
        formatted.append(label)
    return formatted


def _resolve_label_count(
    labels: Sequence[str | None] | None,
    confidences: Sequence[float | None] | None,
    tracker_ids: Sequence[int | None] | None,
    *,
    count: int | None,
) -> int:
    if count is not None:
        return int(count)
    for values in (labels, confidences, tracker_ids):
        if values is not None:
            return len(values)
    raise ValueError("label formatting requires count when no values are provided")


def _optional_rows(
    values: Sequence[object] | None,
    *,
    count: int,
    label: str,
) -> list[object] | None:
    if values is None:
        return None
    rows = list(values)
    if len(rows) != count:
        raise ValueError(f"supervision overlay {label} must contain {count} values")
    return rows


def _optional_confidence(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise TypeError("supervision overlay confidences must be numeric when provided")
    normalized = float(value)
    if np.isnan(normalized):
        return None
    if not np.isfinite(normalized):
        raise ValueError("supervision overlay confidences must be finite or NaN")
    return normalized


def _tracker_label(value: object) -> str:
    if value is None:
        return "pending"
    tracker_id = _tracker_id(value)
    return str(tracker_id)


def _tracker_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError("supervision overlay tracker_ids must be integers when provided")
    tracker_id = int(value)
    if tracker_id < 0:
        raise ValueError("supervision overlay tracker_ids must be non-negative when provided")
    return tracker_id


def _overlay_masks(
    masks: Sequence[np.ndarray] | None,
    *,
    image_shape: tuple[int, int],
    resize_masks_to_image: bool,
) -> np.ndarray | None:
    if masks is None:
        return None
    normalized = [
        _overlay_mask(
            mask,
            image_shape=image_shape,
            resize_masks_to_image=resize_masks_to_image,
        )
        for mask in masks
    ]
    if not normalized:
        return None
    return np.stack(normalized, axis=0).astype(bool, copy=False)


def _overlay_mask(
    mask: np.ndarray,
    *,
    image_shape: tuple[int, int],
    resize_masks_to_image: bool,
) -> np.ndarray:
    binary = _as_binary_mask(mask)
    if binary.shape == image_shape:
        return binary.astype(bool, copy=False)
    if not resize_masks_to_image:
        raise ValueError(f"Mask shape {binary.shape} does not match image shape {image_shape}")
    return cv2.resize(
        binary.astype(np.uint8),
        (int(image_shape[1]), int(image_shape[0])),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def _overlay_boxes(
    boxes_xyxy: Sequence[Sequence[float]] | None,
    *,
    mask_array: np.ndarray | None,
) -> np.ndarray:
    if boxes_xyxy is None:
        if mask_array is None:
            return np.zeros((0, 4), dtype=np.float32)
        return np.asarray([_box_from_mask(mask) for mask in mask_array], dtype=np.float32)
    boxes = [_validated_box_xyxy(box) for box in boxes_xyxy]
    if mask_array is not None and len(boxes) != int(mask_array.shape[0]):
        raise ValueError("supervision overlay boxes_xyxy length must match masks length")
    return np.asarray(boxes, dtype=np.float32)


def _validated_box_xyxy(box: Sequence[float]) -> BoxXYXY:
    if len(box) != 4:
        raise ValueError("supervision overlay box must contain four xyxy values")
    values = [float(value) for value in box]
    if not all(np.isfinite(value) for value in values):
        raise ValueError("supervision overlay box values must be finite")
    x0, y0, x1, y1 = values
    if x1 <= x0 or y1 <= y0:
        raise ValueError("supervision overlay boxes must have positive area")
    return (x0, y0, x1, y1)


def _box_from_mask(mask: np.ndarray) -> BoxXYXY:
    y_rows, x_cols = np.nonzero(mask)
    if x_cols.size == 0 or y_rows.size == 0:
        raise ValueError("supervision overlay masks must contain at least one positive pixel")
    return (
        float(np.min(x_cols)),
        float(np.min(y_rows)),
        float(np.max(x_cols) + 1),
        float(np.max(y_rows) + 1),
    )


def _overlay_confidences(
    confidences: Sequence[float | None] | None,
    *,
    count: int,
) -> np.ndarray | None:
    values = _optional_rows(confidences, count=count, label="confidences")
    if values is None or all(value is None for value in values):
        return None
    return np.asarray(
        [np.nan if value is None else _required_confidence(value) for value in values],
        dtype=np.float32,
    )


def _required_confidence(value: object) -> float:
    confidence = _optional_confidence(value)
    if confidence is None:
        return float("nan")
    return confidence


def _overlay_class_ids(
    class_ids: Sequence[int | None] | None,
    *,
    count: int,
) -> np.ndarray | None:
    values = _optional_rows(class_ids, count=count, label="class_ids")
    if values is None or any(value is None for value in values):
        return None
    return np.asarray([_class_id(value) for value in values], dtype=np.int32)


def _class_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError("supervision overlay class_ids must be integers when provided")
    return int(value)


def _overlay_tracker_ids(
    tracker_ids: Sequence[int | None] | None,
    *,
    count: int,
) -> np.ndarray | None:
    values = _optional_rows(tracker_ids, count=count, label="tracker_ids")
    if values is None or all(value is None for value in values):
        return None
    return np.asarray(
        [-1 if value is None else _tracker_id(value) for value in values],
        dtype=np.int32,
    )


def _overlay_data(
    *,
    labels: list[str],
    class_names: Sequence[str | None] | None,
    count: int,
) -> dict[str, list[str]]:
    data = {"label": labels}
    values = _optional_rows(class_names, count=count, label="class_names")
    if values is not None:
        data["class_name"] = ["" if value is None else str(value) for value in values]
    return data


def _overlay_color_lookup(
    supervision: ModuleType,
    *,
    tracker_array: np.ndarray | None,
    color_by_tracker_id: bool | None,
) -> object:
    if color_by_tracker_id is None:
        color_by_tracker_id = tracker_array is not None
    if color_by_tracker_id and tracker_array is not None:
        return supervision.ColorLookup.TRACK
    return supervision.ColorLookup.INDEX


def _supervision_palette(
    supervision: ModuleType,
    palette_rgb: Sequence[Sequence[int]] | None,
) -> object | None:
    if palette_rgb is None:
        return None
    colors = [
        supervision.Color(r=red, g=green, b=blue)
        for red, green, blue in (_rgb_triplet(color, field="palette_rgb") for color in palette_rgb)
    ]
    return supervision.ColorPalette(colors=colors)


def _annotator_kwargs(**values: object) -> dict[str, object]:
    return {key: value for key, value in values.items() if value is not None}


def best_mask_index(scores: Sequence[float]) -> int:
    """Return the highest-scoring mask index with finite-score validation."""

    score_arr = np.asarray(scores, dtype=float).reshape(-1)
    if score_arr.size == 0:
        return 0
    if not np.all(np.isfinite(score_arr)):
        raise ValueError("scores must contain only finite numeric values")
    return int(np.argmax(score_arr))


def select_masks_for_save(
    masks: Sequence[np.ndarray],
    *,
    save_masks: str,
) -> list[np.ndarray]:
    """Resolve saved mask payloads for a simple mask artifact policy."""

    if save_masks == "none":
        return []
    if save_masks == "top1":
        return [np.asarray(masks[0])] if masks else []
    return [np.asarray(mask) for mask in masks]


def read_label_image(path: str | Path) -> np.ndarray:
    """Read an instance/class label image, preserving integer labels."""

    image = _read_unchanged(path)
    if not np.issubdtype(image.dtype, np.integer):
        raise ValueError(f"Label image must use an integer dtype, got {image.dtype}")
    return image


def write_label_image(path: str | Path, labels: np.ndarray) -> Path:
    """Write an integer label image as PNG/TIFF."""

    target = Path(path)
    arr = np.asarray(labels)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D label image, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"Label image must use an integer dtype, got {arr.dtype}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(target.as_posix(), arr):
        raise OSError(f"Could not write label image: {target}")
    return target


def masks_from_label_image(labels: np.ndarray, *, background: int = 0) -> dict[int, np.ndarray]:
    """Split a label image into binary masks keyed by label value."""

    arr = np.asarray(labels)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D label image, got shape {arr.shape}")
    masks: dict[int, np.ndarray] = {}
    for value in np.unique(arr):
        label = int(value)
        if label == int(background):
            continue
        masks[label] = (arr == value).astype(np.uint8)
    return masks


__all__ = [
    "SupervisionOverlayError",
    "best_mask_index",
    "format_supervision_overlay_labels",
    "masks_from_label_image",
    "read_binary_mask",
    "read_label_image",
    "render_supervision_overlay",
    "select_masks_for_save",
    "write_binary_mask",
    "write_binary_masks",
    "write_label_image",
    "write_mask_overlay",
    "write_supervision_overlay",
]
