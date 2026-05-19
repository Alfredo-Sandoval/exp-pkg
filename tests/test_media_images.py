from __future__ import annotations

import cv2
import numpy as np
import pytest


def test_read_bgr_returns_bgr_uint8_image(tmp_path) -> None:
    from xpkg.media.images import read_bgr

    image_path = tmp_path / "test.png"
    bgr_image = np.zeros((16, 16, 3), dtype=np.uint8)
    bgr_image[..., 2] = 255
    assert cv2.imwrite(image_path.as_posix(), bgr_image)

    result = read_bgr(image_path.as_posix())

    assert result is not None
    assert result.shape == (16, 16, 3)
    assert result.dtype == np.uint8
    assert result[..., 2].mean() == pytest.approx(255, abs=1)


@pytest.mark.parametrize("payload_type", [bytes, bytearray, memoryview])
def test_read_rgb_bytes_decodes_buffer_types(tmp_path, payload_type) -> None:
    from xpkg.media.images import read_rgb_bytes

    image_path = tmp_path / "test.png"
    bgr_image = np.zeros((8, 8, 3), dtype=np.uint8)
    bgr_image[..., 0] = 200
    assert cv2.imwrite(image_path.as_posix(), bgr_image)
    payload = image_path.read_bytes()
    if payload_type is bytearray:
        payload = bytearray(payload)
    elif payload_type is memoryview:
        payload = memoryview(payload)

    result = read_rgb_bytes(payload)

    assert result.shape == (8, 8, 3)
    assert result.dtype == np.uint8
    assert result[..., 2].mean() == pytest.approx(200, abs=1)


def test_read_rgb_bytes_rejects_invalid_payload() -> None:
    from xpkg.media.images import read_rgb_bytes

    with pytest.raises(RuntimeError, match="Could not decode"):
        read_rgb_bytes(b"not an image")


def test_collect_image_paths_returns_sorted_supported_files(tmp_path) -> None:
    from xpkg.media import collect_image_paths

    (tmp_path / "b.png").write_bytes(b"fake")
    (tmp_path / "a.JPG").write_bytes(b"fake")
    (tmp_path / "notes.txt").write_text("skip", encoding="utf-8")

    paths = collect_image_paths(tmp_path)

    assert [path.name for path in paths] == ["a.JPG", "b.png"]
