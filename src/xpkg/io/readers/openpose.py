"""Low-level readers for OpenPose ``--write_json`` body-pose exports."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from xpkg.core.json_utils import load_json_dict
from xpkg.io.readers._common import (
    PoseTrack,
    build_pose_track,
    resolve_node_indices_from_names,
)

_NAT_SORT_RE = re.compile(r"(\d+)")
_POSE_KEY = "pose_keypoints_2d"
_BODY_25_NODE_NAMES: tuple[str, ...] = (
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel",
)
_BODY_25_CONNECTION_NAMES: tuple[tuple[str, str], ...] = (
    ("Neck", "MidHip"),
    ("Neck", "RShoulder"),
    ("Neck", "LShoulder"),
    ("RShoulder", "RElbow"),
    ("RElbow", "RWrist"),
    ("LShoulder", "LElbow"),
    ("LElbow", "LWrist"),
    ("MidHip", "RHip"),
    ("RHip", "RKnee"),
    ("RKnee", "RAnkle"),
    ("MidHip", "LHip"),
    ("LHip", "LKnee"),
    ("LKnee", "LAnkle"),
    ("Neck", "Nose"),
    ("Nose", "REye"),
    ("REye", "REar"),
    ("Nose", "LEye"),
    ("LEye", "LEar"),
    ("LAnkle", "LBigToe"),
    ("LBigToe", "LSmallToe"),
    ("LAnkle", "LHeel"),
    ("RAnkle", "RBigToe"),
    ("RBigToe", "RSmallToe"),
    ("RAnkle", "RHeel"),
)
_BODY_25_INDEX_BY_NAME = {name: idx for idx, name in enumerate(_BODY_25_NODE_NAMES)}
BODY_25_SKELETON_LINKS: tuple[tuple[int, int], ...] = tuple(
    (_BODY_25_INDEX_BY_NAME[start_name], _BODY_25_INDEX_BY_NAME[end_name])
    for start_name, end_name in _BODY_25_CONNECTION_NAMES
)
_BODY_25_VECTOR_LENGTH = len(_BODY_25_NODE_NAMES) * 3


@dataclass(frozen=True)
class OpenPosePerson:
    """One BODY_25 person prediction from an OpenPose frame JSON."""

    coords: np.ndarray
    scores: np.ndarray
    instance_score: float


@dataclass(frozen=True)
class OpenPoseFrame:
    """Decoded BODY_25 people for a single frame JSON file."""

    json_path: Path
    people: tuple[OpenPosePerson, ...]


@dataclass(frozen=True)
class OpenPoseSequence:
    """Decoded BODY_25 frame sequence from an OpenPose JSON directory."""

    node_names: tuple[str, ...]
    frames: tuple[OpenPoseFrame, ...]


def _frame_sort_key(path: Path) -> list[int | str]:
    parts = _NAT_SORT_RE.split(path.stem)
    return [int(part) if part.isdigit() else part for part in parts]


def _resolve_json_dir(path: Path | str) -> Path:
    json_dir = Path(path)
    if not json_dir.exists():
        raise FileNotFoundError(f"OpenPose JSON directory not found: {json_dir}")
    if not json_dir.is_dir():
        raise NotADirectoryError(f"OpenPose JSON input must be a directory: {json_dir}")
    return json_dir


def _json_frame_paths(json_dir: Path) -> list[Path]:
    frame_paths = sorted(
        (path for path in json_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json"),
        key=_frame_sort_key,
    )
    if not frame_paths:
        raise FileNotFoundError(f"No OpenPose JSON frames found in directory: {json_dir}")
    return frame_paths


def _load_people_payload(json_path: Path) -> list[dict[str, Any]]:
    payload = load_json_dict(json_path)
    people = payload.get("people")
    if not isinstance(people, list):
        raise TypeError(f"OpenPose frame JSON must contain a 'people' list: {json_path}")

    decoded: list[dict[str, Any]] = []
    for person_idx, person in enumerate(people):
        if not isinstance(person, dict):
            raise TypeError(
                "OpenPose frame JSON 'people' entries must be objects: "
                f"{json_path} person index {person_idx}"
            )
        decoded.append({str(key): value for key, value in person.items()})
    return decoded


def _coerce_body25_person(
    person: dict[str, Any], *, json_path: Path, person_index: int
) -> OpenPosePerson:
    raw_keypoints = person.get(_POSE_KEY)
    if not isinstance(raw_keypoints, list | tuple):
        raise TypeError(
            "OpenPose person is missing "
            f"{_POSE_KEY!r} array: {json_path} person index {person_index}"
        )

    keypoints = np.asarray(raw_keypoints, dtype=np.float64)
    if keypoints.ndim != 1 or int(keypoints.size) != _BODY_25_VECTOR_LENGTH:
        raise ValueError(
            "OpenPose BODY_25 imports require "
            f"{_BODY_25_VECTOR_LENGTH} {_POSE_KEY} values, got {keypoints.size}: "
            f"{json_path} person index {person_index}"
        )

    reshaped = keypoints.reshape(len(_BODY_25_NODE_NAMES), 3)
    coords = reshaped[:, :2].copy()
    scores = reshaped[:, 2].copy()
    missing_mask = ~np.isfinite(scores) | (scores <= 0.0)
    coords[missing_mask] = np.nan

    valid_scores = scores[np.isfinite(scores) & (scores > 0.0)]
    instance_score = float(valid_scores.mean()) if valid_scores.size else 0.0
    return OpenPosePerson(coords=coords, scores=scores, instance_score=instance_score)


def _read_frame(json_path: Path) -> OpenPoseFrame:
    people_payload = _load_people_payload(json_path)
    people = tuple(
        _coerce_body25_person(person, json_path=json_path, person_index=person_index)
        for person_index, person in enumerate(people_payload)
    )
    return OpenPoseFrame(json_path=json_path, people=people)


def read_sequence(path: Path | str) -> OpenPoseSequence:
    """Read an OpenPose ``--write_json`` directory as a BODY_25 sequence."""

    json_dir = _resolve_json_dir(path)
    frame_paths = _json_frame_paths(json_dir)
    frames = tuple(_read_frame(frame_path) for frame_path in frame_paths)
    return OpenPoseSequence(node_names=_BODY_25_NODE_NAMES, frames=frames)


def read_node_names(path: Path | str) -> list[str]:
    """Return BODY_25 node names for an OpenPose ``--write_json`` directory."""

    # Validate the directory contains OpenPose-style frame payloads before returning names.
    sequence = read_sequence(path)
    del sequence
    return list(_BODY_25_NODE_NAMES)


def read_track(path: Path | str, *, track_index: int) -> PoseTrack:
    """Read one per-frame person slot from an OpenPose ``--write_json`` directory."""

    person_index = int(track_index)
    if person_index < 0:
        raise ValueError(f"OpenPose track_index must be non-negative, got {track_index!r}.")

    sequence = read_sequence(path)
    frame_count = len(sequence.frames)
    node_count = len(sequence.node_names)
    coords = np.full((frame_count, node_count, 2), np.nan, dtype=np.float64)
    scores = np.zeros((frame_count, node_count), dtype=np.float64)
    instance_score = np.zeros((frame_count,), dtype=np.float64)

    for frame_idx, frame in enumerate(sequence.frames):
        if person_index >= len(frame.people):
            continue
        person = frame.people[person_index]
        coords[frame_idx] = person.coords
        scores[frame_idx] = person.scores
        instance_score[frame_idx] = person.instance_score

    return build_pose_track(
        coords=coords,
        scores=scores,
        node_names=sequence.node_names,
        instance_score=instance_score,
        source_label=f"OpenPose JSON directory {Path(path)}",
    )


def resolve_node_indices(
    path: Path | str,
    target_names: Sequence[str],
) -> list[int]:
    """Resolve BODY_25 node names to indices for an OpenPose JSON directory."""

    return resolve_node_indices_from_names(read_node_names(path), target_names)


__all__ = [
    "BODY_25_SKELETON_LINKS",
    "OpenPoseFrame",
    "OpenPosePerson",
    "OpenPoseSequence",
    "PoseTrack",
    "read_node_names",
    "read_sequence",
    "read_track",
    "resolve_node_indices",
]
