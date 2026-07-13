"""Metadata-only model objects for prediction payloads and tests."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

from xpkg.model.labels import Labels
from xpkg.model.video_types import VideoProtocol
from xpkg.pose.skeleton import Keypoint, Skeleton


@dataclass(frozen=True, slots=True)
class VideoStub:
    """Video metadata object used when frame decoding is intentionally unavailable."""

    filename: str | None
    frames: int
    height: int
    width: int
    channels: int = 3
    backend: str = "opencv"
    sha256: str | None = None
    id: str | None = None
    label: str | None = None
    fps: float = 0.0
    _image_filenames: tuple[str, ...] = field(default_factory=tuple)
    last_frame_idx: int = field(init=False)

    def __post_init__(self) -> None:
        filename = str(self.filename).strip() if self.filename is not None else ""
        sha256 = str(self.sha256).strip() if self.sha256 is not None else ""
        image_filenames = tuple(
            str(path).strip() for path in self._image_filenames if str(path).strip()
        )
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "frames", int(self.frames))
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "channels", int(self.channels))
        object.__setattr__(self, "backend", str(self.backend))
        object.__setattr__(self, "sha256", sha256)
        object.__setattr__(self, "fps", float(self.fps))
        object.__setattr__(self, "_image_filenames", image_filenames)
        object.__setattr__(self, "last_frame_idx", max(0, int(self.frames) - 1))

    @property
    def image_filenames(self) -> list[str]:
        return list(self._image_filenames)

    def get_frame(self, idx: int) -> np.ndarray:
        raise RuntimeError(
            "VideoStub is metadata-only and cannot decode frames. "
            "Load the source video file to read frame data."
        )

    def iter_frames(self) -> Iterator[np.ndarray]:
        raise RuntimeError(
            "VideoStub is metadata-only and cannot decode frames. "
            "Load the source video file to iterate frame data."
        )

    def close(self) -> None:
        return None

    @property
    def uses_pyav(self) -> bool:
        return False


def build_prediction_stub(
    keypoint_names: Sequence[str],
    video_stub: VideoProtocol,
    *,
    skeleton_name: str = "xpkg-predict",
) -> Labels:
    """Build an empty labels container anchored to prediction video metadata."""
    keypoints = [Keypoint(id=idx, name=str(name)) for idx, name in enumerate(keypoint_names)]
    skeleton = Skeleton(name=str(skeleton_name), keypoints=keypoints, links_ids=[])
    return Labels(
        labeled_frames=[],
        videos=[video_stub],
        skeletons=[skeleton],
        keypoints=keypoints,
    )


__all__ = ["VideoStub", "build_prediction_stub"]
