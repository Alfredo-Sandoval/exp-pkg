"""Low-level readers for Detectron2 COCO keypoint prediction exports."""

from __future__ import annotations

import json
import re
import warnings
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


@dataclass(frozen=True, slots=True)
class Detectron2KeypointCategory:
    """One COCO keypoint category defined in the dataset JSON."""

    category_id: int
    name: str
    node_names: tuple[str, ...]
    skeleton_links: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class Detectron2Detection:
    """One keypoint detection from Detectron2 COCO results."""

    category_id: int
    score: float
    keypoints: np.ndarray


@dataclass(frozen=True, slots=True)
class Detectron2Frame:
    """One dataset image plus its keypoint detections."""

    image_id: int
    frame_index: int
    image_path: Path
    detections: tuple[Detectron2Detection, ...]


@dataclass(frozen=True, slots=True)
class Detectron2CocoSequence:
    """Decoded Detectron2 COCO keypoint predictions over an image sequence."""

    categories: tuple[Detectron2KeypointCategory, ...]
    frames: tuple[Detectron2Frame, ...]


def _natural_sort_key(path: Path) -> tuple[tuple[int, int | str], ...]:
    normalized = path.as_posix()
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in _NAT_SORT_RE.split(normalized)
        if part
    )


def _require_json_array(value: object, *, field_name: str, path: Path) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"Detectron2 {field_name} must be a JSON array in {path}.")
    return list(value)


def _require_mapping(value: object, *, field_name: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Detectron2 {field_name} must be a JSON object in {path}.")
    return {str(key): item for key, item in value.items()}


def _require_nonempty_string(value: object, *, field_name: str, path: Path) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Detectron2 {field_name} must be a string in {path}.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"Detectron2 {field_name} must be non-empty in {path}.")
    return normalized


def _require_int(value: object, *, field_name: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Detectron2 {field_name} must be an integer in {path}.")
    return int(value)


def _require_number(value: object, *, field_name: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"Detectron2 {field_name} must be numeric in {path}.")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"Detectron2 {field_name} must be finite in {path}.")
    return parsed


def _load_predictions_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    predictions = _require_json_array(payload, field_name="predictions", path=path)
    decoded: list[dict[str, Any]] = []
    for index, item in enumerate(predictions):
        decoded.append(
            _require_mapping(
                item,
                field_name=f"predictions[{index}]",
                path=path,
            )
        )
    return decoded


def _normalize_skeleton_links(
    value: object,
    *,
    node_count: int,
    category_name: str,
    path: Path,
) -> tuple[tuple[int, int], ...]:
    if value is None:
        return tuple()

    raw_links = _require_json_array(
        value,
        field_name=f"categories[{category_name}].skeleton",
        path=path,
    )
    normalized: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for link_index, raw_link in enumerate(raw_links):
        pair = tuple(
            _require_json_array(
                raw_link,
                field_name=f"categories[{category_name}].skeleton[{link_index}]",
                path=path,
            )
        )
        if len(pair) != 2:
            raise ValueError(
                "Detectron2 COCO skeleton links must have exactly two indices in "
                f"{path}; got {pair!r}."
            )
        start = _require_int(
            pair[0],
            field_name=f"categories[{category_name}].skeleton[{link_index}][0]",
            path=path,
        )
        end = _require_int(
            pair[1],
            field_name=f"categories[{category_name}].skeleton[{link_index}][1]",
            path=path,
        )
        if not 1 <= start <= node_count or not 1 <= end <= node_count:
            raise ValueError(
                "Detectron2 COCO skeleton links are 1-based and must stay within the "
                f"node range in {path}; got {(start, end)!r} for {node_count} nodes."
            )
        zero_based = (start - 1, end - 1)
        if zero_based[0] == zero_based[1]:
            raise ValueError(f"Detectron2 skeleton links cannot self-reference in {path}.")
        ordered = (
            zero_based if zero_based[0] < zero_based[1] else (zero_based[1], zero_based[0])
        )
        if ordered in seen:
            continue
        seen.add(ordered)
        normalized.append(ordered)
    return tuple(normalized)


def _load_categories(path: Path) -> tuple[Detectron2KeypointCategory, ...]:
    payload = load_json_dict(path)
    raw_categories = _require_json_array(
        payload.get("categories"),
        field_name="categories",
        path=path,
    )
    categories: list[Detectron2KeypointCategory] = []
    seen_ids: set[int] = set()
    for index, raw_category in enumerate(raw_categories):
        category = _require_mapping(raw_category, field_name=f"categories[{index}]", path=path)
        category_id = _require_int(
            category.get("id"),
            field_name=f"categories[{index}].id",
            path=path,
        )
        if category_id in seen_ids:
            raise ValueError(f"Duplicate Detectron2 category id {category_id} in {path}.")
        seen_ids.add(category_id)

        raw_keypoints = category.get("keypoints")
        if raw_keypoints is None:
            continue
        keypoints = tuple(
            _require_nonempty_string(
                raw_name,
                field_name=f"categories[{index}].keypoints[{keypoint_index}]",
                path=path,
            )
            for keypoint_index, raw_name in enumerate(
                _require_json_array(
                    raw_keypoints,
                    field_name=f"categories[{index}].keypoints",
                    path=path,
                )
            )
        )
        if not keypoints:
            continue
        categories.append(
            Detectron2KeypointCategory(
                category_id=category_id,
                name=_require_nonempty_string(
                    category.get("name"),
                    field_name=f"categories[{index}].name",
                    path=path,
                ),
                node_names=keypoints,
                skeleton_links=_normalize_skeleton_links(
                    category.get("skeleton"),
                    node_count=len(keypoints),
                    category_name=str(category_id),
                    path=path,
                ),
            )
        )

    if not categories:
        raise ValueError(f"No Detectron2 COCO keypoint categories found in {path}.")
    return tuple(categories)


def _load_image_index(dataset_json_path: Path, image_root: Path) -> dict[int, tuple[int, Path]]:
    payload = load_json_dict(dataset_json_path)
    raw_images = _require_json_array(
        payload.get("images"),
        field_name="images",
        path=dataset_json_path,
    )
    images_by_id: dict[int, tuple[int, Path]] = {}
    indexed_images: list[tuple[Path, int]] = []
    for index, raw_image in enumerate(raw_images):
        image = _require_mapping(raw_image, field_name=f"images[{index}]", path=dataset_json_path)
        image_id = _require_int(
            image.get("id"),
            field_name=f"images[{index}].id",
            path=dataset_json_path,
        )
        if image_id in images_by_id:
            raise ValueError(f"Duplicate Detectron2 image id {image_id} in {dataset_json_path}.")
        relative_path = Path(
            _require_nonempty_string(
                image.get("file_name"),
                field_name=f"images[{index}].file_name",
                path=dataset_json_path,
            )
        )
        indexed_images.append((relative_path, image_id))

    for frame_index, (relative_path, image_id) in enumerate(
        sorted(indexed_images, key=lambda item: _natural_sort_key(item[0]))
    ):
        image_path = (image_root / relative_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Detectron2 image file missing on disk: {image_path}")
        images_by_id[image_id] = (frame_index, image_path)

    if not images_by_id:
        raise ValueError(f"Detectron2 dataset JSON contains no images: {dataset_json_path}.")
    return images_by_id


def _coerce_keypoints_array(
    value: object,
    *,
    node_count: int,
    path: Path,
    field_name: str,
) -> np.ndarray:
    if not isinstance(value, list | tuple):
        raise TypeError(f"Detectron2 {field_name} must be an array in {path}.")
    keypoints = np.asarray(value, dtype=np.float64)
    if keypoints.ndim != 1 or int(keypoints.size) != node_count * 3:
        raise ValueError(
            f"Detectron2 {field_name} in {path} must contain {node_count * 3} values, "
            f"got {keypoints.size}."
        )
    reshaped = keypoints.reshape(node_count, 3)
    coords = reshaped[:, :2].copy()
    scores = reshaped[:, 2].copy()
    coords[~np.isfinite(scores) | (scores <= 0.0)] = np.nan
    return np.column_stack((coords, scores))


def _category_by_id(
    categories: tuple[Detectron2KeypointCategory, ...],
) -> dict[int, Detectron2KeypointCategory]:
    return {category.category_id: category for category in categories}


def _resolve_category(
    categories: tuple[Detectron2KeypointCategory, ...],
    *,
    category_id: int | None,
) -> Detectron2KeypointCategory:
    if category_id is None:
        if len(categories) != 1:
            raise ValueError(
                "category_id is required when the Detectron2 dataset contains multiple "
                "keypoint categories."
            )
        return categories[0]

    category_lookup = _category_by_id(categories)
    try:
        return category_lookup[int(category_id)]
    except KeyError as exc:
        raise KeyError(
            f"Detectron2 category_id {category_id} is not defined in the dataset JSON."
        ) from exc


def read_sequence(
    predictions_path: Path | str,
    dataset_json_path: Path | str,
    image_root: Path | str,
) -> Detectron2CocoSequence:
    """Read Detectron2 COCO keypoint predictions plus their dataset/image metadata."""

    resolved_predictions_path = Path(predictions_path).resolve()
    resolved_dataset_json_path = Path(dataset_json_path).resolve()
    resolved_image_root = Path(image_root).resolve()

    if not resolved_predictions_path.is_file():
        raise FileNotFoundError(
            f"Detectron2 predictions JSON not found: {resolved_predictions_path}"
        )
    if not resolved_dataset_json_path.is_file():
        raise FileNotFoundError(
            f"Detectron2 dataset JSON not found: {resolved_dataset_json_path}"
        )
    if not resolved_image_root.is_dir():
        raise NotADirectoryError(
            f"Detectron2 image_root must be a directory: {resolved_image_root}"
        )

    categories = _load_categories(resolved_dataset_json_path)
    categories_by_id = _category_by_id(categories)
    images_by_id = _load_image_index(resolved_dataset_json_path, resolved_image_root)
    predictions = _load_predictions_json(resolved_predictions_path)

    detections_by_image: dict[int, list[tuple[int, Detectron2Detection]]] = {
        image_id: [] for image_id in images_by_id
    }
    for prediction_index, prediction in enumerate(predictions):
        image_id = _require_int(
            prediction.get("image_id"),
            field_name=f"predictions[{prediction_index}].image_id",
            path=resolved_predictions_path,
        )
        if image_id not in images_by_id:
            raise KeyError(
                "Detectron2 prediction references image_id "
                f"{image_id} that is missing from {resolved_dataset_json_path}."
            )

        category_id = _require_int(
            prediction.get("category_id"),
            field_name=f"predictions[{prediction_index}].category_id",
            path=resolved_predictions_path,
        )
        if category_id not in categories_by_id:
            raise KeyError(
                "Detectron2 prediction references category_id "
                f"{category_id} that is not a keypoint category in {resolved_dataset_json_path}."
            )

        category = categories_by_id[category_id]
        detection = Detectron2Detection(
            category_id=category_id,
            score=_require_number(
                prediction.get("score"),
                field_name=f"predictions[{prediction_index}].score",
                path=resolved_predictions_path,
            ),
            keypoints=_coerce_keypoints_array(
                prediction.get("keypoints"),
                node_count=len(category.node_names),
                path=resolved_predictions_path,
                field_name=f"predictions[{prediction_index}].keypoints",
            ),
        )
        detections_by_image[image_id].append((prediction_index, detection))

    frames = tuple(
        Detectron2Frame(
            image_id=image_id,
            frame_index=frame_index,
            image_path=image_path,
            detections=tuple(
                detection
                for _prediction_index, detection in sorted(
                    detections_by_image[image_id],
                    key=lambda item: (-item[1].score, item[0]),
                )
            ),
        )
        for image_id, (frame_index, image_path) in sorted(
            images_by_id.items(),
            key=lambda item: item[1][0],
        )
    )
    return Detectron2CocoSequence(categories=categories, frames=frames)


def read_node_names(
    dataset_json_path: Path | str,
    *,
    category_id: int | None = None,
) -> list[str]:
    """Return keypoint names for one Detectron2 COCO category."""

    categories = _load_categories(Path(dataset_json_path).resolve())
    category = _resolve_category(categories, category_id=category_id)
    return list(category.node_names)


def read_track_count(
    predictions_path: Path | str,
    dataset_json_path: Path | str,
    image_root: Path | str,
    *,
    category_id: int | None = None,
) -> int:
    """Return the maximum per-frame detection slot count for one keypoint category."""

    sequence = read_sequence(predictions_path, dataset_json_path, image_root)
    category = _resolve_category(sequence.categories, category_id=category_id)
    return _track_count_from_sequence(sequence, category=category)


def _track_count_from_sequence(
    sequence: Detectron2CocoSequence,
    *,
    category: Detectron2KeypointCategory,
) -> int:
    return max(
        (
            sum(
                detection.category_id == category.category_id
                for detection in frame.detections
            )
            for frame in sequence.frames
        ),
        default=0,
    )


def _read_track_from_sequence(
    sequence: Detectron2CocoSequence,
    *,
    track_index: int,
    category: Detectron2KeypointCategory,
) -> PoseTrack:
    idx = int(track_index)
    if idx < 0:
        raise ValueError(f"track_index must be >= 0, got {track_index!r}.")

    track_count = _track_count_from_sequence(sequence, category=category)
    if track_count <= idx:
        raise IndexError(
            f"track_index={idx} out of range for Detectron2 predictions with "
            f"track_count={track_count}."
        )

    frame_count = len(sequence.frames)
    node_count = len(category.node_names)
    coords = np.full((frame_count, node_count, 2), np.nan, dtype=np.float64)
    scores = np.full((frame_count, node_count), np.nan, dtype=np.float64)
    instance_score = np.full((frame_count,), np.nan, dtype=np.float64)

    for frame in sequence.frames:
        category_detections = [
            detection
            for detection in frame.detections
            if detection.category_id == category.category_id
        ]
        if len(category_detections) <= idx:
            continue
        detection = category_detections[idx]
        coords[frame.frame_index] = detection.keypoints[:, :2]
        scores[frame.frame_index] = detection.keypoints[:, 2]
        instance_score[frame.frame_index] = detection.score

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if np.isnan(instance_score).all():
            instance_score = np.nanmean(scores, axis=1)

    return build_pose_track(
        coords=coords,
        scores=scores,
        node_names=category.node_names,
        instance_score=instance_score,
        source_label=f"Detectron2 COCO category {category.category_id}",
    )


def read_track(
    predictions_path: Path | str,
    dataset_json_path: Path | str,
    image_root: Path | str,
    *,
    track_index: int,
    category_id: int | None = None,
) -> PoseTrack:
    """Read one per-frame detection slot as a PoseTrack for a Detectron2 category."""

    sequence = read_sequence(predictions_path, dataset_json_path, image_root)
    category = _resolve_category(sequence.categories, category_id=category_id)
    return _read_track_from_sequence(
        sequence,
        track_index=track_index,
        category=category,
    )


def resolve_node_indices(
    dataset_json_path: Path | str,
    *,
    target_names: list[str] | tuple[str, ...],
    category_id: int | None = None,
) -> list[int]:
    """Resolve keypoint names to indices for one Detectron2 COCO category."""

    return resolve_node_indices_from_names(
        read_node_names(dataset_json_path, category_id=category_id),
        target_names,
    )


__all__ = [
    "Detectron2CocoSequence",
    "Detectron2Detection",
    "Detectron2Frame",
    "Detectron2KeypointCategory",
    "PoseTrack",
    "read_node_names",
    "read_sequence",
    "read_track",
    "read_track_count",
    "resolve_node_indices",
]
