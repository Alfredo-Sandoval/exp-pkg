from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xpkg.io.masked_brightness import extract_masked_brightness_trace


@dataclass
class _StubVideo:
    frames_array: np.ndarray
    fps: float = 4.0

    @property
    def height(self) -> int:
        return int(self.frames_array.shape[1])

    @property
    def width(self) -> int:
        return int(self.frames_array.shape[2])

    @property
    def frames(self) -> int:
        return int(self.frames_array.shape[0])

    def get_frame(self, idx: int, *, approximate: bool = False) -> np.ndarray:
        del approximate
        return self.frames_array[idx]


def test_extract_masked_brightness_trace_captures_dark_to_bright_transition() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1

    frames = np.full((8, 8, 8, 3), 20, dtype=np.uint8)
    frames[:4, 2:6, 2:6, :] = 10
    frames[4:, 2:6, 2:6, :] = 220

    trace = extract_masked_brightness_trace(
        _StubVideo(frames),
        mask=mask,
        bbox_xyxy=(1.0, 1.0, 7.0, 7.0),
        sample_rate_hz=4.0,
        max_seconds=2.0,
        ring_px=1,
    )

    assert trace.frame_indices.tolist() == list(range(8))
    assert trace.times_seconds.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    assert float(trace.mean_brightness[0]) < 15.0
    assert float(trace.mean_brightness[-1]) > 200.0
    assert float(trace.contrast_peak[0]) < 0.0
    assert float(trace.contrast_peak[-1]) > 150.0


def test_extract_masked_brightness_trace_respects_bbox_crop() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:7, 4:7] = 1

    frames = np.full((4, 10, 10, 3), 30, dtype=np.uint8)
    frames[:, 4:7, 4:7, :] = 180

    trace = extract_masked_brightness_trace(
        _StubVideo(frames),
        mask=mask,
        bbox_xyxy=(4.0, 4.0, 7.0, 7.0),
        sample_rate_hz=2.0,
        max_seconds=2.0,
        ring_px=0,
    )

    assert trace.frame_indices.tolist() == [0, 2]
    assert all(value > 170.0 for value in trace.mean_brightness.tolist())
