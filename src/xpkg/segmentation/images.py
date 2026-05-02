"""PNG/TIFF mask image helpers for segmentation masks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _read_unchanged(path: str | Path) -> np.ndarray:
    target = Path(path)
    image = cv2.imread(target.as_posix(), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read mask image: {target}")
    if image.ndim == 3:
        image = image[:, :, 0]
    return np.asarray(image)


def read_binary_mask(path: str | Path, *, threshold: int = 0) -> np.ndarray:
    """Read a PNG/TIFF mask image as a ``uint8`` binary array."""

    image = _read_unchanged(path)
    return (image > int(threshold)).astype(np.uint8)


def write_binary_mask(
    path: str | Path,
    mask: np.ndarray,
    *,
    true_value: int = 255,
) -> Path:
    """Write a 2D binary mask as PNG/TIFF using ``0`` and ``true_value``."""

    target = Path(path)
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape {arr.shape}")
    target.parent.mkdir(parents=True, exist_ok=True)
    image = (arr > 0).astype(np.uint8) * int(true_value)
    if not cv2.imwrite(target.as_posix(), image):
        raise OSError(f"Could not write binary mask image: {target}")
    return target


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
    "masks_from_label_image",
    "read_binary_mask",
    "read_label_image",
    "write_binary_mask",
    "write_label_image",
]
