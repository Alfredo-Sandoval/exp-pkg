"""Video protocol used by the xpkg labels package."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VideoProtocol(Protocol):
    """Structural contract for video objects referenced by labels."""

    filename: str | None
    id: str | None
    label: str | None
    sha256: str | None
    width: int
    height: int
    frames: int
    fps: float
    channels: int
    backend: str
    last_frame_idx: int
    _image_filenames: list[str]

    @property
    def image_filenames(self) -> list[str]: ...

    def get_frame(self, idx: int) -> np.ndarray: ...

    def iter_frames(self) -> Iterator[np.ndarray]: ...

    def close(self) -> None: ...


__all__ = ["VideoProtocol"]
