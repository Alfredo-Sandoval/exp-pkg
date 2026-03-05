"""Factory and store helpers around `Labels`."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from posetta.core.skeleton import Keypoint, Skeleton
from posetta.io.labels.video_types import VideoProtocol

if TYPE_CHECKING:
    from posetta.io.labels.model import Labels


@dataclass(frozen=True)
class VideoStub:
    """A lightweight video metadata object used for prediction bundles."""

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
    last_frame_idx: int = field(init=False)

    def __post_init__(self) -> None:
        filename = str(self.filename).strip() if self.filename is not None else ""
        sha256 = str(self.sha256).strip() if self.sha256 is not None else ""
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "frames", int(self.frames))
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "channels", int(self.channels))
        object.__setattr__(self, "backend", str(self.backend))
        object.__setattr__(self, "sha256", sha256)
        object.__setattr__(self, "fps", float(self.fps))
        object.__setattr__(self, "last_frame_idx", max(0, int(self.frames) - 1))

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

class LabelsFactory:
    """Factory for creating Labels objects."""

    @staticmethod
    def build_prediction_stub(keypoint_names: Sequence[str], video_stub: VideoProtocol) -> Labels:
        """Build a Labels object stub for prediction bundles."""
        from posetta.io.labels.model import Labels

        keypoints = [Keypoint(id=idx, name=name) for idx, name in enumerate(keypoint_names)]
        skeleton = Skeleton(name="siesta-predict", keypoints=keypoints, links_ids=[])
        return Labels(
            labeled_frames=[],
            videos=[video_stub],
            skeletons=[skeleton],
            keypoints=keypoints,
        )


@dataclass
class FileLabelStore:
    """LabelStore implementation that wraps a `.siesta` bundle."""

    _labels: Labels

    @property
    def query(self):
        """Expose query port backed by the labels."""
        from posetta.io.labels.query import LabelsQuery

        return LabelsQuery(self._labels)

    def videos(self) -> list:
        """Return the tracked videos."""
        return list(self._labels.videos)

    def labels(self) -> list:
        """Return a copy of the labeled frames."""
        return list(self._labels.labels)

    def save(self, path: str) -> None:
        """Save the labels to `path` as .siesta."""
        self._labels.export_h5(path)

    @classmethod
    def load_file(cls, path: str) -> FileLabelStore:
        """Load labels from disk into a store; paths must be manifest-resolved."""
        from posetta.io.labels.model import Labels

        return cls(Labels.load_file(path))


__all__ = ["FileLabelStore", "LabelsFactory", "VideoStub"]
