from __future__ import annotations

import numpy as np
import pytest

from xpkg.media.video import augment_background, resize_image, resize_images


@pytest.mark.parametrize(
    ("shape", "scale", "expected_shape"),
    [
        ((100, 100, 3), 0.5, (50, 50, 3)),
        ((50, 50, 3), 2.0, (100, 100, 3)),
        ((80, 120, 3), 0.5, (40, 60, 3)),
        ((100, 100, 1), 0.5, (50, 50, 1)),
    ],
)
def test_resize_image_preserves_dtype_and_expected_shape(
    shape: tuple[int, ...],
    scale: float,
    expected_shape: tuple[int, ...],
) -> None:
    image = np.full(shape, 128, dtype=np.uint8)

    resized = resize_image(image, scale)

    assert resized.shape == expected_shape
    assert resized.dtype == np.uint8
    assert np.mean(resized) == pytest.approx(128, abs=5)


def test_resize_image_scale_one_returns_original_array() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    assert resize_image(image, 1.0) is image


def test_resize_images_resizes_batch_and_preserves_passthrough() -> None:
    images = np.zeros((5, 20, 10, 3), dtype=np.uint8)

    resized = resize_images(images, 0.5)

    assert resized.shape == (5, 10, 5, 3)
    assert resized.dtype == np.uint8
    assert resize_images(images, 1.0) is images


@pytest.mark.parametrize(
    ("background", "expected"),
    [
        ("black", 0),
        ("grey", 127),
        ("white", 255),
    ],
)
def test_augment_background_fills_solid_colors(background: str, expected: int) -> None:
    images = np.ones((2, 8, 8, 3), dtype=np.uint8) * 23

    result = augment_background(images, background)

    assert result.shape == images.shape
    assert result.dtype == np.uint8
    assert np.all(result == expected)


@pytest.mark.parametrize("background", [None, "original"])
def test_augment_background_passthrough_modes(background: str | None) -> None:
    images = np.arange(2 * 4 * 4 * 1, dtype=np.uint8).reshape((2, 4, 4, 1))

    result = augment_background(images, background)

    np.testing.assert_array_equal(result, images)


def test_augment_background_rejects_unknown_color() -> None:
    images = np.zeros((1, 4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Invalid background color"):
        augment_background(images, "blue")
