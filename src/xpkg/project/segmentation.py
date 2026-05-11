"""Project-state storage for frame-level segmentation masks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from xpkg.io.labels.model import Labels
from xpkg.media.video import Video
from xpkg.pose.annotations import LabeledFrame, SegmentationMask
from xpkg.pose.skeleton import build_keypoint_skeleton
from xpkg.project.layout import resolve_project_root
from xpkg.project.store import save_project_labels

from .._core.path_registry import resolve_path

if TYPE_CHECKING:
    from xpkg.io.labels.video_types import VideoProtocol


MaskSaveMode = Literal["replace", "append"]
type VideoSelector = int | str | Path | Any


@dataclass(frozen=True, slots=True)
class SegmentationFrame:
    """Segmentation masks attached to one project video frame."""

    video_index: int
    frame_index: int
    masks: tuple[SegmentationMask, ...]
    video_id: str = ""
    video_label: str = ""
    video_path: str = ""


def _project_root(path: str | Path) -> Path:
    root = resolve_project_root(path)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {path}")
    return root


def _load_project_labels(project: str | Path) -> tuple[Path, Labels]:
    root = _project_root(project)
    labels = Labels.load_file(root.as_posix())
    return root, labels


def _copy_mask(mask: SegmentationMask) -> SegmentationMask:
    return SegmentationMask.from_dict(mask.to_dict(), track=mask.track)


def _copy_masks(masks: Sequence[SegmentationMask]) -> tuple[SegmentationMask, ...]:
    if isinstance(masks, str | bytes | bytearray):
        raise TypeError("masks must be a sequence of SegmentationMask objects")

    copied: list[SegmentationMask] = []
    for mask in masks:
        if not isinstance(mask, SegmentationMask):
            raise TypeError(
                "masks must contain only SegmentationMask objects; "
                f"got {type(mask).__name__}"
            )
        copied.append(_copy_mask(mask))
    return tuple(copied)


def _video_paths(video: VideoProtocol) -> list[str]:
    paths: list[str] = []
    filename = str(getattr(video, "filename", "") or "").strip()
    if filename:
        paths.append(filename)
    image_filenames = getattr(video, "image_filenames", ())
    if isinstance(image_filenames, Sequence) and not isinstance(
        image_filenames,
        str | bytes | bytearray,
    ):
        paths.extend(str(item) for item in image_filenames if str(item).strip())
    return paths


def _path_values_match(left: str, right: str) -> bool:
    if left == right:
        return True
    if Path(left).name == right or Path(right).name == left:
        return True
    try:
        return resolve_path(left) == resolve_path(right)
    except (OSError, RuntimeError):
        return False


def _video_matches_selector(video: VideoProtocol, selector: str | Path) -> bool:
    raw_selector = str(selector)
    selector_text = raw_selector.strip()
    if not selector_text:
        return False

    for attr in ("id", "label"):
        value = str(getattr(video, attr, "") or "").strip()
        if value and value == selector_text:
            return True

    return any(_path_values_match(path, selector_text) for path in _video_paths(video))


def _find_video_index(labels: Labels, selector: str | Path) -> int | None:
    matches = [
        index
        for index, video in enumerate(labels.videos)
        if _video_matches_selector(video, selector)
    ]
    if len(matches) > 1:
        raise ValueError(f"Video selector {selector!r} matched multiple project videos")
    return matches[0] if matches else None


def _is_video_like(value: object) -> bool:
    return hasattr(value, "filename") and hasattr(value, "image_filenames")


def _coerce_video(value: str | Path | VideoProtocol) -> VideoProtocol:
    if _is_video_like(value):
        return cast("VideoProtocol", value)
    if isinstance(value, str | Path):
        return cast("VideoProtocol", Video.from_filename(Path(value).as_posix()))
    raise TypeError(
        "video must be a video index, path, id, label, or Video-like object; "
        f"got {type(value).__name__}"
    )


def _resolve_video_index(
    labels: Labels,
    video: VideoSelector | None,
    *,
    allow_create: bool,
) -> int:
    if isinstance(video, int):
        if video < 0 or video >= len(labels.videos):
            raise IndexError(f"video index {video} is out of range")
        return int(video)

    if video is None:
        if len(labels.videos) == 1:
            return 0
        if len(labels.videos) == 0:
            raise ValueError(
                "Project has no videos yet; pass video=... when saving "
                "segmentation masks into an empty project"
            )
        raise ValueError("Project has multiple videos; pass video=... to choose one")

    if isinstance(video, str | Path):
        existing_index = _find_video_index(labels, video)
        if existing_index is not None:
            return existing_index
        if not allow_create:
            raise ValueError(f"Video selector {video!r} did not match a project video")

    if not allow_create:
        raise ValueError(f"Video selector {video!r} did not match a project video")

    labels.videos.append(_coerce_video(video))
    return len(labels.videos) - 1


def _ensure_single_skeleton(labels: Labels, *, skeleton_name: str) -> None:
    if len(labels.skeletons) == 0:
        labels.skeletons.append(build_keypoint_skeleton([], name=skeleton_name))
        labels.keypoints = []
        return
    if len(labels.skeletons) != 1:
        raise ValueError(
            "Project segmentation saves require labels with exactly one skeleton"
        )


def _find_frame(
    labels: Labels,
    *,
    video_index: int,
    frame_index: int,
) -> LabeledFrame | None:
    video = labels.videos[video_index]
    for frame in labels.labeled_frames:
        if frame.video is video and int(frame.frame_idx) == int(frame_index):
            return frame
    return None


def _sync_segmentation_tracks(labels: Labels) -> None:
    seen = set(labels.tracks)
    for frame in labels.labeled_frames:
        for mask in getattr(frame, "masks", ()):
            track = mask.track
            if track is None or track in seen:
                continue
            labels.tracks.append(track)
            seen.add(track)
    labels.tracks.sort(key=lambda item: (item.spawned_on, item.name))


def _filter_masks(
    masks: Sequence[SegmentationMask],
    *,
    predicted: bool | None,
    class_name: str | None,
) -> tuple[SegmentationMask, ...]:
    filtered: list[SegmentationMask] = []
    for mask in masks:
        if predicted is not None and bool(mask.is_predicted) is not bool(predicted):
            continue
        if class_name is not None and str(mask.class_name) != str(class_name):
            continue
        filtered.append(_copy_mask(mask))
    return tuple(filtered)


def _frame_result(
    labels: Labels,
    frame: LabeledFrame,
    *,
    masks: tuple[SegmentationMask, ...],
) -> SegmentationFrame:
    video_index = labels.videos.index(frame.video)
    video = labels.videos[video_index]
    paths = _video_paths(video)
    return SegmentationFrame(
        video_index=video_index,
        frame_index=int(frame.frame_idx),
        masks=masks,
        video_id=str(getattr(video, "id", "") or ""),
        video_label=str(getattr(video, "label", "") or ""),
        video_path=paths[0] if paths else "",
    )


def load_project_segmentation_frames(
    project: str | Path,
    *,
    video: VideoSelector | None = None,
    frame_index: int | None = None,
    predicted: bool | None = None,
    class_name: str | None = None,
) -> list[SegmentationFrame]:
    """Load segmentation masks grouped by project video frame."""

    _, labels = _load_project_labels(project)
    selected_video_index: int | None = None
    if video is not None:
        selected_video_index = _resolve_video_index(labels, video, allow_create=False)

    results: list[SegmentationFrame] = []
    for frame in sorted(
        labels.labeled_frames,
        key=lambda item: (labels.videos.index(item.video), int(item.frame_idx)),
    ):
        current_video_index = labels.videos.index(frame.video)
        if selected_video_index is not None and current_video_index != selected_video_index:
            continue
        if frame_index is not None and int(frame.frame_idx) != int(frame_index):
            continue
        masks = _filter_masks(
            getattr(frame, "masks", ()),
            predicted=predicted,
            class_name=class_name,
        )
        if not masks:
            continue
        results.append(_frame_result(labels, frame, masks=masks))
    return results


def load_project_segmentation_masks(
    project: str | Path,
    *,
    frame_index: int,
    video: VideoSelector | None = None,
    predicted: bool | None = None,
    class_name: str | None = None,
) -> tuple[SegmentationMask, ...]:
    """Load segmentation masks for one project video frame."""

    _, labels = _load_project_labels(project)
    video_index = _resolve_video_index(labels, video, allow_create=False)
    frame = _find_frame(labels, video_index=video_index, frame_index=int(frame_index))
    if frame is None:
        return ()
    return _filter_masks(
        getattr(frame, "masks", ()),
        predicted=predicted,
        class_name=class_name,
    )


def save_project_segmentation_masks(
    project: str | Path,
    *,
    frame_index: int,
    masks: Sequence[SegmentationMask],
    video: VideoSelector | None = None,
    mode: MaskSaveMode = "replace",
    skeleton_name: str = "segmentation",
) -> Path:
    """Save segmentation masks for one project video frame."""

    if mode not in {"replace", "append"}:
        raise ValueError("mode must be either 'replace' or 'append'")

    root, labels = _load_project_labels(project)
    _ensure_single_skeleton(labels, skeleton_name=skeleton_name)
    copied_masks = _copy_masks(masks)
    video_index = _resolve_video_index(labels, video, allow_create=True)
    frame = _find_frame(labels, video_index=video_index, frame_index=int(frame_index))

    if frame is None:
        frame = LabeledFrame(
            video=cast(Any, labels.videos[video_index]),
            frame_idx=int(frame_index),
        )
        labels.labeled_frames.append(frame)

    if mode == "replace":
        frame.masks = list(copied_masks)
    else:
        frame.masks = [*frame.masks, *copied_masks]

    _sync_segmentation_tracks(labels)
    labels.validate()
    labels.update_cache()
    return save_project_labels(root, labels)


def clear_project_segmentation_masks(
    project: str | Path,
    *,
    frame_index: int,
    video: VideoSelector | None = None,
) -> Path:
    """Remove all segmentation masks from one project video frame."""

    return save_project_segmentation_masks(
        project,
        frame_index=frame_index,
        masks=(),
        video=video,
        mode="replace",
    )


__all__ = [
    "MaskSaveMode",
    "SegmentationFrame",
    "VideoSelector",
    "clear_project_segmentation_masks",
    "load_project_segmentation_frames",
    "load_project_segmentation_masks",
    "save_project_segmentation_masks",
]
