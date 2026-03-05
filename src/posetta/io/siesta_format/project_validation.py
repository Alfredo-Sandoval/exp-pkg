"""Project validation and summary helpers for `.siesta` archives."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _check_frame_group_consistency(group: dict, group_name: str) -> None:
    keys = ("video_index", "frame_index", "num_instances")
    arrays = {key: group[key] for key in keys if key in group}
    if not arrays:
        return
    lengths = {key: int(value.shape[0]) for key, value in arrays.items()}
    values = set(lengths.values())
    if len(values) > 1:
        detail = ", ".join(f"{key}={value}" for key, value in lengths.items())
        raise RuntimeError(f"{group_name} datasets have inconsistent lengths: {detail}")


def _validate_payload(project: dict[str, Any]) -> None:
    """Domain validation over a materialized payload (no file I/O)."""
    if not isinstance(project, dict):
        raise TypeError("Loaded project payload is not a dict")

    required_keys = ("videos", "labels", "predictions")
    missing = [key for key in required_keys if key not in project]
    if missing:
        raise RuntimeError(f"Missing required keys: {', '.join(missing)}")

    videos = project["videos"]
    if "filenames" not in videos:
        raise RuntimeError("videos group missing filenames dataset")
    if "shapes" not in videos:
        raise RuntimeError("videos group missing shapes dataset")

    filenames = list(videos["filenames"])
    shapes = videos["shapes"]
    if not filenames:
        raise RuntimeError("Project contains no videos")
    if shapes is None or shapes.shape is None:
        raise RuntimeError("videos group missing shapes dataset")
    if shapes.shape[0] != len(filenames):
        raise RuntimeError("videos.shapes row count does not match filenames length")

    labels = project["labels"]
    preds = project["predictions"]

    if "frames" in labels:
        _check_frame_group_consistency(labels["frames"], "labels.frames")
    if "frames" in preds:
        _check_frame_group_consistency(preds["frames"], "predictions.frames")


@dataclass
class ProjectSummary:
    path: Path
    video_filenames: list[str] = field(default_factory=list)
    video_shapes: tuple[int, ...] | None = None
    label_frames: int = 0
    prediction_frames: int = 0
    schema_version: str | None = None
    siesta_version: str | None = None
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
        if self.schema_version or self.siesta_version:
            stream.write(f" schema: {self.schema_version}  siesta_version: {self.siesta_version}\n")
        if self.created or self.modified:
            stream.write(f" created: {self.created}  modified: {self.modified}\n")


def summarize_project(path: Path) -> ProjectSummary:
    from posetta.io.siesta_format.reader import read_siesta

    project = read_siesta(path, lazy=True)

    videos = project.get("videos") or {}
    labels = project.get("labels") or {}
    preds = project.get("predictions") or {}
    metadata = project.get("metadata") or {}

    filenames = list(videos.get("filenames") or [])
    shapes = videos.get("shapes")

    label_frames = 0
    frames_grp = labels.get("frames") or {}
    video_index = frames_grp.get("video_index")
    if video_index is not None:
        label_frames = int(video_index.shape[0])

    prediction_frames = 0
    preds_frames_grp = preds.get("frames") or {}
    frame_index = preds_frames_grp.get("frame_index")
    if frame_index is not None:
        prediction_frames = int(frame_index.shape[0])

    return ProjectSummary(
        path=path,
        video_filenames=filenames,
        video_shapes=tuple(shapes.shape) if shapes is not None and shapes.shape else None,
        label_frames=label_frames,
        prediction_frames=prediction_frames,
        schema_version=metadata.get("version"),
        siesta_version=metadata.get("siesta_version"),
        created=metadata.get("created"),
        modified=metadata.get("modified"),
    )


def validate_project(path: Path) -> None:
    from posetta.io.siesta_format.reader import read_siesta

    _validate_payload(read_siesta(path, lazy=False))


__all__ = [
    "ProjectSummary",
    "_check_frame_group_consistency",
    "_validate_payload",
    "summarize_project",
    "validate_project",
]
