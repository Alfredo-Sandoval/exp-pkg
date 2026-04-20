"""Convert Detectron2 COCO keypoint exports into native archives."""

from __future__ import annotations

from pathlib import Path

from xpkg.core.path_registry import ensure_dir, resolve_path
from xpkg.io.archive_format import write_archive
from xpkg.io.converters.converter_helpers import (
    ConversionResult,
    ProgressCallback,
    _emit,
)
from xpkg.io.converters.pose_track_import import (
    build_pose_track_skeleton,
    labels_from_pose_tracks,
)
from xpkg.io.readers.detectron2_coco import (
    Detectron2CocoSequence,
    Detectron2KeypointCategory,
    _read_track_from_sequence,
    _track_count_from_sequence,
    read_sequence,
)
from xpkg.io.video import Video

_DETECTRON2_READ_JSON_MARKER = "DETECTRON2_IMPORT STEP: read_json"
_DETECTRON2_BUILD_LABELS_MARKER = "DETECTRON2_IMPORT STEP: build_labels"
_DETECTRON2_WRITE_ARCHIVE_MARKER = "DETECTRON2_IMPORT STEP: write_archive"
_DETECTRON2_DONE_MARKER = "DETECTRON2_IMPORT DONE"

DETECTRON2_COCO_PROGRESS_MARKERS: tuple[tuple[str, int], ...] = (
    (_DETECTRON2_READ_JSON_MARKER, 10),
    (_DETECTRON2_BUILD_LABELS_MARKER, 60),
    (_DETECTRON2_WRITE_ARCHIVE_MARKER, 80),
    (_DETECTRON2_DONE_MARKER, 100),
)


def _resolve_category(
    sequence: Detectron2CocoSequence,
    *,
    category_id: int | None,
) -> Detectron2KeypointCategory:
    if category_id is None:
        if len(sequence.categories) != 1:
            raise ValueError(
                "category_id is required when the Detectron2 dataset contains multiple "
                "keypoint categories."
            )
        return sequence.categories[0]

    for category in sequence.categories:
        if category.category_id == int(category_id):
            return category
    raise KeyError(f"Detectron2 category_id {category_id} is not defined in the dataset JSON.")


def _sequence_video(sequence: Detectron2CocoSequence) -> Video:
    image_filenames = [frame.image_path.as_posix() for frame in sequence.frames]
    return Video.from_image_filenames(image_filenames)


def _empty_labels(
    *,
    sequence: Detectron2CocoSequence,
    category: Detectron2KeypointCategory,
    skeleton_name: str,
):
    from xpkg.model import Labels

    skeleton = build_pose_track_skeleton(
        category.node_names,
        skeleton_name=skeleton_name,
        skeleton_links=category.skeleton_links,
    )
    labels = Labels(skeletons=[skeleton], videos=[_sequence_video(sequence)])
    labels.update_cache()
    return labels


def _labels_from_sequence(
    sequence: Detectron2CocoSequence,
    *,
    category: Detectron2KeypointCategory,
    skeleton_name: str,
    likelihood_threshold: float,
):
    track_count = _track_count_from_sequence(sequence, category=category)
    if track_count <= 0:
        return _empty_labels(
            sequence=sequence,
            category=category,
            skeleton_name=skeleton_name,
        )

    tracks = [
        _read_track_from_sequence(
            sequence,
            track_index=track_index,
            category=category,
        )
        for track_index in range(track_count)
    ]
    return labels_from_pose_tracks(
        tracks,
        skeleton_name=skeleton_name,
        skeleton_links=category.skeleton_links,
        video=_sequence_video(sequence),
        likelihood_threshold=likelihood_threshold,
    )


def _source_media_roots(sequence: Detectron2CocoSequence) -> list[Path]:
    roots = {frame.image_path.parent.resolve() for frame in sequence.frames}
    return sorted(roots)


def convert_detectron2_coco(
    predictions_path: Path | str,
    dataset_json_path: Path | str,
    image_root: Path | str,
    out_path: Path | str,
    *,
    category_id: int | None = None,
    skeleton_name: str | None = None,
    likelihood_threshold: float = 0.0,
    progress_callback: ProgressCallback | None = None,
) -> ConversionResult:
    """Convert Detectron2 COCO keypoint predictions into a native archive."""

    resolved_predictions_path = resolve_path(predictions_path)
    resolved_dataset_json_path = resolve_path(dataset_json_path)
    resolved_image_root = resolve_path(image_root)
    resolved_out_path = resolve_path(out_path)

    _emit(progress_callback, _DETECTRON2_READ_JSON_MARKER)
    sequence = read_sequence(
        resolved_predictions_path,
        resolved_dataset_json_path,
        resolved_image_root,
    )
    category = _resolve_category(sequence, category_id=category_id)
    resolved_skeleton_name = skeleton_name or category.name
    _emit(
        progress_callback,
        "IMPORT: Found "
        f"{len(sequence.frames)} images, {len(category.node_names)} keypoints, "
        f"category={category.name!r}",
    )

    _emit(progress_callback, _DETECTRON2_BUILD_LABELS_MARKER)
    labels = _labels_from_sequence(
        sequence,
        category=category,
        skeleton_name=resolved_skeleton_name,
        likelihood_threshold=likelihood_threshold,
    )
    labels.validate()

    ensure_dir(resolved_out_path.parent)
    metadata = {
        "source": "detectron2_coco_import",
        "source_predictions": resolved_predictions_path.as_posix(),
        "source_dataset_json": resolved_dataset_json_path.as_posix(),
        "source_image_root": resolved_image_root.as_posix(),
        "source_category_id": category.category_id,
        "source_category_name": category.name,
    }
    _emit(progress_callback, _DETECTRON2_WRITE_ARCHIVE_MARKER)
    write_archive(resolved_out_path, labels, metadata=metadata)
    _emit(progress_callback, _DETECTRON2_DONE_MARKER)

    return ConversionResult(
        source_dir=resolved_predictions_path.parent,
        project_root=resolved_out_path.parent,
        videos=_source_media_roots(sequence),
        archive_path=resolved_out_path,
    )


__all__ = [
    "DETECTRON2_COCO_PROGRESS_MARKERS",
    "convert_detectron2_coco",
]
