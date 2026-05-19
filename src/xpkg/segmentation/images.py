"""PNG/TIFF mask image helpers for segmentation masks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

ImageColorOrder = Literal["rgb", "bgr"]


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
    channels = tuple(int(value) for value in values)
    if any(value < 0 or value > 255 for value in channels):
        raise ValueError(f"{field} channel values must be between 0 and 255.")
    return channels


def _box_xyxy(box: Sequence[float]) -> tuple[int, int, int, int]:
    if len(box) != 4:
        raise ValueError("box must contain four xyxy values.")
    x0, y0, x1, y1 = (int(round(float(value))) for value in box)
    return x0, y0, x1, y1


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
    "best_mask_index",
    "masks_from_label_image",
    "read_binary_mask",
    "read_label_image",
    "select_masks_for_save",
    "write_binary_mask",
    "write_binary_masks",
    "write_label_image",
    "write_mask_overlay",
]
