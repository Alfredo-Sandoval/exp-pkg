"""Minimal color conversion helpers used by the extracted IO stack."""

from __future__ import annotations

import cv2
import numpy as np


def ensure_three_channels(image: np.ndarray) -> np.ndarray:
    """Return a 3-channel uint8-like image."""
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Expected a 2D image or 3-channel image, got shape {image.shape}")


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return ensure_three_channels(image)[..., ::-1]


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return ensure_three_channels(image)[..., ::-1]


def bgr_to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(ensure_three_channels(image), cv2.COLOR_BGR2GRAY)


def ensure_bgr(image: np.ndarray, *, input_is_bgr: bool = False) -> np.ndarray:
    normalized = ensure_three_channels(image)
    return normalized if input_is_bgr else rgb_to_bgr(normalized)


__all__ = ["bgr_to_gray", "bgr_to_rgb", "ensure_bgr", "ensure_three_channels", "rgb_to_bgr"]
