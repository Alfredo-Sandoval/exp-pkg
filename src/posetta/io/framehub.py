"""FrameHub: shared container leasing and live latest-frame buffering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Event, Lock, RLock
from typing import Any

import numpy as np

from posetta.core.logging_utils import get_logger
from posetta.io.video_backends import _open_pyav_container

logger = get_logger(__name__)


@dataclass(slots=True)
class _FrameEntry:
    """Internal bookkeeping for a borrowed PyAV container."""

    container: Any
    lock: Lock
    refcount: int = 0


@dataclass(frozen=True, slots=True)
class HubFrame:
    """A single frame stored in the hub.

    Attributes:
        camera_id: The unique identifier of the camera.
        frame_id: The monotonically increasing frame identifier.
        timestamp_ns: The capture timestamp in nanoseconds.
        frame_bgr: The raw BGR image data.
    """

    camera_id: str
    frame_id: int
    timestamp_ns: int
    frame_bgr: np.ndarray


class FrameLease:
    """Handle returned by FrameHub to manage a borrowed container."""

    __slots__ = ("_hub", "_released", "container", "lock", "path")

    def __init__(self, hub: FrameHub, path: str, entry: _FrameEntry) -> None:
        self.path = path
        self.container = entry.container
        self.lock = entry.lock
        self._hub = hub
        self._released = False

    def release(self) -> None:
        """Return the borrowed container to the hub."""
        if self._released:
            return
        self._released = True
        self._hub._release(self.path)

    def __enter__(self) -> FrameLease:
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.release()


class FrameHub:
    """Shared hub for PyAV containers and live latest-frame buffering."""

    def __init__(self) -> None:
        self._entries: dict[str, _FrameEntry] = {}
        self._guard = Lock()
        self._lock: RLock = RLock()
        self._cv: Condition = Condition(self._lock)
        self._frames: dict[str, HubFrame] = {}
        self._last_frame_id: dict[str, int] = {}
        self._updated_events: dict[str, Event] = {}
        self._update_counter: int = 0

    def borrow(self, filename: str) -> FrameLease:
        """Return a FrameLease for the given video path."""
        abs_path = str(Path(filename).resolve())
        with self._guard:
            entry = self._entries.get(abs_path)
            if entry is None:
                container = _open_pyav_container(abs_path)
                entry = _FrameEntry(container=container, lock=Lock(), refcount=0)
                self._entries[abs_path] = entry
            entry.refcount += 1
            logger.debug("FrameHub: %s borrowed (refs=%d)", abs_path, entry.refcount)
            return FrameLease(self, abs_path, entry)

    def _release(self, abs_path: str) -> None:
        entry: _FrameEntry | None = None
        with self._guard:
            existing = self._entries.get(abs_path)
            if existing is None:
                return
            existing.refcount -= 1
            logger.debug("FrameHub: %s released (refs=%d)", abs_path, existing.refcount)
            if existing.refcount <= 0:
                entry = existing
                del self._entries[abs_path]
        if entry is not None:
            entry.container.close()

    def reset(self) -> None:
        """Close all containers (used by tests)."""
        items = []
        with self._guard:
            items = list(self._entries.items())
            self._entries.clear()
        for path, entry in items:
            logger.debug("FrameHub: reset closing %s", path)
            entry.container.close()

    def update_frame(
        self, camera_id: str, frame_bgr: np.ndarray, timestamp_ns: int, frame_id: int
    ) -> None:
        """Update the latest frame for a camera.

        Overwrites any previous frame for this camera.

        Args:
            camera_id: The unique identifier of the camera.
            frame_bgr: The raw BGR image data.
            timestamp_ns: The capture timestamp in nanoseconds.
            frame_id: The monotonically increasing frame identifier.

        Raises:
            TypeError: If input types are incorrect.
            ValueError: If frame data is malformed or frame_id is not increasing.
        """
        if frame_bgr.dtype != np.uint8:
            raise ValueError("FrameHub expects uint8 BGR frames")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("FrameHub expects frames with shape (H, W, 3)")
        if not frame_bgr.flags.c_contiguous:
            raise ValueError("FrameHub expects C-contiguous frames")

        with self._lock:
            last_id: int | None = self._last_frame_id.get(camera_id)
            if last_id is not None and frame_id <= last_id:
                raise ValueError(
                    f"FrameId must be monotonically increasing per camera (got {frame_id} after {last_id})"
                )
            self._last_frame_id[camera_id] = frame_id
            frame: HubFrame = HubFrame(
                camera_id=camera_id,
                frame_id=frame_id,
                timestamp_ns=timestamp_ns,
                frame_bgr=frame_bgr,
            )
            self._frames[camera_id] = frame
            self._update_counter += 1
            self._cv.notify_all()
            evt: Event | None = self._updated_events.get(camera_id)
            if evt is not None:
                evt.set()

    def get_latest(self, camera_id: str) -> HubFrame | None:
        """Get the latest frame for a camera.

        Args:
            camera_id: The unique identifier of the camera.

        Returns:
            The latest HubFrame for the camera, or None if no frame exists.
        """
        with self._lock:
            return self._frames.get(camera_id)

    def get_batch(self, camera_ids: list[str]) -> dict[str, HubFrame]:
        """Get latest frames for a list of cameras.

        Args:
            camera_ids: A list of camera identifiers.

        Returns:
            A dictionary mapping camera IDs to their latest HubFrame.
        """
        result: dict[str, HubFrame] = {}
        with self._lock:
            for cid in camera_ids:
                frame = self._frames.get(cid)
                if frame is not None:
                    result[cid] = frame
        return result

    def wait_for_update(self, last_seen: int, *, timeout_s: float) -> int:
        """Block until any camera updates the hub.

        Args:
            last_seen: The previously returned update counter.
            timeout_s: The maximum time to wait in seconds.

        Returns:
            The updated global counter, which is monotonically increasing.
        """
        with self._lock:
            if self._update_counter == last_seen:
                self._cv.wait(timeout=max(0.0, float(timeout_s)))
            return self._update_counter

    def notify_all(self) -> None:
        """Wake any threads waiting on hub updates."""
        with self._lock:
            self._update_counter += 1
            self._cv.notify_all()

    def wait_for_frame(self, camera_id: str, timeout: float = 1.0) -> bool:
        """Wait for a new frame to arrive for the specified camera.

        Args:
            camera_id: The unique identifier of the camera.
            timeout: Maximum time to wait in seconds.

        Returns:
            True if a frame arrived, False if timed out.
        """
        with self._lock:
            evt = self._updated_events.get(camera_id)
            if evt is None:
                evt = Event()
                self._updated_events[camera_id] = evt
            evt.clear()

        return evt.wait(timeout=timeout)

    def register_camera(self, camera_id: str) -> None:
        """Register a camera to track its events.

        Args:
            camera_id: The unique identifier of the camera.
        """
        with self._lock:
            if camera_id not in self._updated_events:
                self._updated_events[camera_id] = Event()
            self._last_frame_id.pop(camera_id, None)

    def unregister_camera(self, camera_id: str) -> None:
        """Cleanup camera resources.

        Args:
            camera_id: The unique identifier of the camera.
        """
        with self._lock:
            self._frames.pop(camera_id, None)
            self._last_frame_id.pop(camera_id, None)
            self._updated_events.pop(camera_id, None)

    def clear_all(self) -> None:
        """Clear all buffered frames (useful for model hotswapping)."""
        with self._lock:
            self._frames.clear()
            self._update_counter += 1
            self._cv.notify_all()


frame_hub = FrameHub()

__all__ = ["FrameHub", "FrameLease", "HubFrame", "frame_hub"]
