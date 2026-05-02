"""Frame transform helpers for canonical numpy media arrays."""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["augment_background", "resize_image", "resize_images"]


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize one image by a scale factor."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    if scale == 1.0:
        return image
    height, width = image.shape[:2]
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    if image.ndim == 3 and image.shape[2] == 1:
        resized = cv2.resize(image.squeeze(-1), (new_width, new_height))
        return resized[..., np.newaxis]
    return cv2.resize(image, (new_width, new_height))


def resize_images(images: np.ndarray, scale: float) -> np.ndarray:
    """Resize each image in `images` by `scale`."""
    if scale == 1.0:
        return images
    return np.stack([resize_image(image, scale) for image in images], axis=0)


def augment_background(images: np.ndarray, background: str | None) -> np.ndarray:
    """Fill with a solid color or keep original pixels."""
    if background is None or background == "original":
        return images

    fill_values = {"black": 0, "grey": 127, "white": 255}
    if background not in fill_values:
        valid = ", ".join(fill_values)
        raise ValueError(f"Invalid background color: {background}. Options include: {valid}")
    return np.full_like(images, fill_values[background])
