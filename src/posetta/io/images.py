"""Minimal image IO helpers (centralized color policy boundary).

- read_bgr(path): returns BGR uint8 image or None
- read_rgb(path): returns RGB uint8 image or None
- read_rgb_bytes(buf): decode bytes into RGB uint8 image or raises

These functions keep dependencies light and centralize BGR<->RGB policy so
callers don't sprinkle cv2 and conversions throughout the codebase.
"""

from __future__ import annotations

import cv2
import numpy as np

from posetta.core.colors import bgr_to_rgb


def _cv2_imread_bgr(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img


def read_bgr(path: str) -> np.ndarray | None:
    """Read an image from disk as BGR uint8 or None when unavailable (cv2 only).

    Args:
        path: Path to the image file.

    Returns:
        np.ndarray | None: The BGR image array or None if reading failed.
    """
    return _cv2_imread_bgr(path)


def read_rgb(path: str) -> np.ndarray | None:
    """Read an image as RGB uint8 or None when unavailable (cv2 only).

    Args:
        path: Path to the image file.

    Returns:
        np.ndarray | None: The RGB image array or None if reading failed.
    """
    bgr = _cv2_imread_bgr(path)
    return None if bgr is None else bgr_to_rgb(bgr)


def read_rgb_bytes(buf: bytes | bytearray | memoryview) -> np.ndarray:
    """Decode an image from bytes into an RGB uint8 array.

    Args:
        buf: The image data buffer.

    Returns:
        np.ndarray: The decoded RGB image array.

    Raises:
        RuntimeError: If the buffer cannot be decoded.
    """
    data = np.frombuffer(buf, dtype=np.uint8)
    arr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError("Could not decode image bytes (cv2.imdecode returned None)")
    return bgr_to_rgb(arr)
