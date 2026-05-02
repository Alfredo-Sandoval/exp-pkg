"""Service-bound segmentation mask API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xpkg.model import SegmentationMask
from xpkg.project.segmentation import (
    MaskSaveMode,
    SegmentationFrame,
    VideoSelector,
    clear_project_segmentation_masks,
    load_project_segmentation_frames,
    load_project_segmentation_masks,
    save_project_segmentation_masks,
)


@dataclass(frozen=True, slots=True)
class ProjectSegmentation:
    """Project-bound helpers for saving and loading segmentation masks."""

    project_root: Path

    def load_frames(
        self,
        *,
        video: VideoSelector | None = None,
        frame_index: int | None = None,
        predicted: bool | None = None,
        class_name: str | None = None,
    ) -> list[SegmentationFrame]:
        """Load segmentation masks grouped by video frame."""
        return load_project_segmentation_frames(
            self.project_root,
            video=video,
            frame_index=frame_index,
            predicted=predicted,
            class_name=class_name,
        )

    def load_masks(
        self,
        *,
        frame_index: int,
        video: VideoSelector | None = None,
        predicted: bool | None = None,
        class_name: str | None = None,
    ) -> tuple[SegmentationMask, ...]:
        """Load segmentation masks for one video frame."""
        return load_project_segmentation_masks(
            self.project_root,
            frame_index=frame_index,
            video=video,
            predicted=predicted,
            class_name=class_name,
        )

    def save_masks(
        self,
        *,
        frame_index: int,
        masks: list[SegmentationMask] | tuple[SegmentationMask, ...],
        video: VideoSelector | None = None,
        mode: MaskSaveMode = "replace",
        skeleton_name: str = "segmentation",
    ) -> Path:
        """Save segmentation masks for one video frame."""
        return save_project_segmentation_masks(
            self.project_root,
            frame_index=frame_index,
            masks=masks,
            video=video,
            mode=mode,
            skeleton_name=skeleton_name,
        )

    def clear_masks(
        self,
        *,
        frame_index: int,
        video: VideoSelector | None = None,
    ) -> Path:
        """Remove all segmentation masks from one video frame."""
        return clear_project_segmentation_masks(
            self.project_root,
            frame_index=frame_index,
            video=video,
        )


__all__ = [
    "ProjectSegmentation",
]
