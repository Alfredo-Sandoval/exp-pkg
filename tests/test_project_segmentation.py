from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from xpkg.model import SegmentationMask, Track
from xpkg.project import (
    current_project_state_path,
    load_project_segmentation_masks,
    save_project_segmentation_masks,
)
from xpkg.services import ProjectService


def _write_test_frame(path: Path, value: int = 24) -> None:
    frame = np.full((12, 16, 3), value, dtype=np.uint8)
    ok = cv2.imwrite(path.as_posix(), frame)
    assert ok


def _binary_mask() -> np.ndarray:
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[2:8, 3:11] = 1
    return mask


def test_project_segmentation_service_saves_masks_into_empty_project(
    tmp_path: Path,
) -> None:
    frame_path = tmp_path / "frame.png"
    _write_test_frame(frame_path)
    project = ProjectService.create(tmp_path / "Segmentation Project")
    binary = _binary_mask()
    track = Track(spawned_on=0, name="cell-track")
    mask = SegmentationMask.from_binary_mask(
        binary,
        class_name="cell",
        confidence=0.82,
        track=track,
    )

    state_path = project.segmentation.save_masks(
        frame_index=0,
        video=frame_path,
        masks=[mask],
    )

    assert state_path == current_project_state_path(project.project_root)
    loaded_masks = project.segmentation.load_masks(frame_index=0)
    assert len(loaded_masks) == 1
    assert loaded_masks[0].class_name == "cell"
    assert loaded_masks[0].track is not None
    assert loaded_masks[0].track.name == "cell-track"
    np.testing.assert_array_equal(loaded_masks[0].to_binary_mask(), binary)

    labels = project.load_labels()
    assert len(labels.videos) == 1
    assert labels.skeletons[0].keypoint_names == []
    assert len(labels.labeled_frames) == 1
    assert len(labels.labeled_frames[0].masks) == 1


def test_project_segmentation_append_replace_filter_and_clear(
    tmp_path: Path,
) -> None:
    frame_path = tmp_path / "frame.png"
    _write_test_frame(frame_path)
    project = ProjectService.create(tmp_path / "Mask Edits")
    polygon = SegmentationMask.from_polygon(
        np.array([[0.0, 0.0], [6.0, 0.0], [6.0, 5.0]], dtype=np.float32),
        class_name="body",
    )
    predicted = SegmentationMask.from_binary_mask(_binary_mask(), class_name="body")
    predicted.is_predicted = True

    project.segmentation.save_masks(
        frame_index=0,
        video=frame_path,
        masks=[polygon],
    )
    project.segmentation.save_masks(
        frame_index=0,
        masks=[predicted],
        mode="append",
    )

    body_masks = load_project_segmentation_masks(
        project.project_root,
        frame_index=0,
        class_name="body",
    )
    assert len(body_masks) == 2
    predicted_frames = project.segmentation.load_frames(predicted=True)
    assert len(predicted_frames) == 1
    assert len(predicted_frames[0].masks) == 1
    assert predicted_frames[0].masks[0].is_predicted

    replacement = SegmentationMask.from_polygon(
        np.array([[1.0, 1.0], [4.0, 1.0], [4.0, 4.0]], dtype=np.float32),
        class_name="replacement",
    )
    save_project_segmentation_masks(
        project.project_root,
        frame_index=0,
        masks=[replacement],
    )
    replaced_masks = project.segmentation.load_masks(frame_index=0)
    assert len(replaced_masks) == 1
    assert replaced_masks[0].class_name == "replacement"

    project.segmentation.clear_masks(frame_index=0)
    assert project.segmentation.load_masks(frame_index=0) == ()


def test_project_segmentation_requires_video_when_saving_empty_project(
    tmp_path: Path,
) -> None:
    project = ProjectService.create(tmp_path / "No Video")
    mask = SegmentationMask.from_binary_mask(_binary_mask())

    with pytest.raises(ValueError, match="no videos"):
        project.segmentation.save_masks(frame_index=0, masks=[mask])
