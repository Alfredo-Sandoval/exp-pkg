"""Project payload validation and summary helpers."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _labels_videos_group(project: dict[str, Any]) -> dict[str, Any]:
    labels = project.get("labels")
    if not isinstance(labels, dict):
        raise RuntimeError("Loaded project payload missing labels group")
    videos = labels.get("videos")
    if not isinstance(videos, dict):
        raise RuntimeError("labels group missing videos payload")
    return videos


def _row_count(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape:
        return int(shape[0])
    if isinstance(value, list | tuple):
        return len(value)
    return 0


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape:
        return tuple(int(dim) for dim in shape)
    if isinstance(value, list | tuple):
        if not value:
            return (0,)
        if isinstance(value[0], list | tuple):
            return (len(value), len(value[0]))
        return (len(value),)
    return None


def _summary_count(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _check_frame_group_consistency(group: dict, group_name: str) -> None:
    keys = ("video_index", "frame_index", "num_instances")
    arrays = {key: group[key] for key in keys if key in group}
    if not arrays:
        return
    lengths = {key: _row_count(value) for key, value in arrays.items()}
    values = set(lengths.values())
    if len(values) > 1:
        detail = ", ".join(f"{key}={value}" for key, value in lengths.items())
        raise RuntimeError(f"{group_name} datasets have inconsistent lengths: {detail}")


def validate_loaded_project(project: dict[str, Any]) -> None:
    """Validate a materialized project payload."""

    if not isinstance(project, dict):
        raise TypeError("Loaded project payload is not a dict")

    required_keys = ("labels", "predictions")
    missing = [key for key in required_keys if key not in project]
    if missing:
        raise RuntimeError(f"Missing required keys: {', '.join(missing)}")

    videos = _labels_videos_group(project)
    if "filenames" not in videos:
        raise RuntimeError("videos group missing filenames")
    if "shapes" not in videos:
        raise RuntimeError("videos group missing shapes")

    filenames = list(videos["filenames"])
    shape_tuple = _shape_tuple(videos["shapes"])
    if shape_tuple is None:
        raise RuntimeError("videos group missing shapes")
    if shape_tuple[0] != len(filenames):
        raise RuntimeError("videos.shapes row count does not match filenames length")

    labels = project["labels"]
    predictions = project["predictions"]
    if "frames" in labels:
        _check_frame_group_consistency(labels["frames"], "labels.frames")
    if "frames" in predictions:
        _check_frame_group_consistency(predictions["frames"], "predictions.frames")


@dataclass
class ProjectSummary:
    path: Path
    video_filenames: list[str] = field(default_factory=list)
    video_shapes: tuple[int, ...] | None = None
    label_frames: int = 0
    prediction_frames: int = 0
    schema_version: str | None = None
    created: str | None = None
    modified: str | None = None

    @property
    def n_videos(self) -> int:
        return len(self.video_filenames)

    def print(self, stream: Any = None) -> None:
        if stream is None:
            stream = sys.stdout
        stream.write(f"File: {self.path}\n")
        stream.write(f" videos: {self.n_videos}\n")
        if self.video_shapes is not None:
            stream.write(f"  shapes: {self.video_shapes}\n")
        stream.write(f" labels frames: {self.label_frames}\n")
        stream.write(f" predictions frames: {self.prediction_frames}\n")
        if self.schema_version:
            stream.write(f" schema: {self.schema_version}\n")
        if self.created or self.modified:
            stream.write(f" created: {self.created}  modified: {self.modified}\n")


def summarize_loaded_project(project: dict[str, Any], *, path: Path) -> ProjectSummary:
    videos = _labels_videos_group(project)
    labels = project.get("labels") or {}
    predictions = project.get("predictions") or {}
    metadata = project.get("metadata") or {}

    filenames = list(videos.get("filenames") or [])
    label_frames = _summary_count(metadata.get("n_labels")) or 0
    if label_frames <= 0:
        label_frames = _row_count((labels.get("frames") or {}).get("video_index"))

    prediction_frames = _summary_count(metadata.get("n_predictions_committed")) or 0
    if prediction_frames <= 0:
        prediction_frames = _row_count((predictions.get("frames") or {}).get("frame_index"))

    return ProjectSummary(
        path=path,
        video_filenames=filenames,
        video_shapes=_shape_tuple(videos.get("shapes")),
        label_frames=label_frames,
        prediction_frames=prediction_frames,
        schema_version=metadata.get("schema_version") or metadata.get("version"),
        created=metadata.get("created"),
        modified=metadata.get("modified"),
    )


__all__ = [
    "ProjectSummary",
    "summarize_loaded_project",
    "validate_loaded_project",
]
