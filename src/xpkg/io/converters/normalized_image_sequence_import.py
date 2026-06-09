"""Convert normalized image-sequence annotations into project-ready labels."""

from __future__ import annotations

import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from xpkg._core.json_utils import load_json_dict
from xpkg._core.path_registry import ensure_dir, resolve_path
from xpkg.io.converters.result import ConversionResult
from xpkg.io.labels.model import Labels
from xpkg.media.video import Video
from xpkg.pose.annotations import Instance, LabeledFrame, Point
from xpkg.pose.skeleton import Keypoint, Skeleton


@dataclass(frozen=True, slots=True)
class _NormalizedInstance:
    keypoints: tuple[tuple[float | None, float | None, int], ...]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _NormalizedFrame:
    sequence_id: str
    image_path: Path
    frame_id: str
    source_index: int | None
    instances: tuple[_NormalizedInstance, ...]


@dataclass(frozen=True, slots=True)
class _NormalizedPayload:
    schema_version: int
    dataset_key: str
    slice_key: str
    project_name: str
    keypoint_names: tuple[str, ...]
    links: tuple[tuple[int, int], ...]
    frames: tuple[_NormalizedFrame, ...]


def _require_str(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must be non-empty")
    return normalized


def _require_int_or_none(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer or null")
    return int(value)


def _require_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return int(value)


def _require_float_or_none(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be a number or null")
    return float(value)


def _require_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return cast(Mapping[str, object], value)


def _require_sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a sequence")
    return cast(Sequence[object], value)


def _require_schema_version(value: object) -> int:
    version = _require_int(value, field="schema_version")
    if version != 1:
        raise ValueError(f"schema_version must be 1, got {version}")
    return version


def _coerce_keypoint_triplet(
    value: object,
    *,
    field: str,
) -> tuple[float | None, float | None, int]:
    triplet = tuple(_require_sequence(value, field=field))
    if len(triplet) != 3:
        raise TypeError(f"{field} must be a 3-item sequence")
    x_raw, y_raw, visible_raw = triplet
    return (
        _require_float_or_none(x_raw, field=f"{field}.x"),
        _require_float_or_none(y_raw, field=f"{field}.y"),
        _require_int(visible_raw, field=f"{field}.visible"),
    )


def _coerce_link_pair(value: object, *, field: str) -> tuple[int, int]:
    pair = tuple(_require_sequence(value, field=field))
    if len(pair) != 2:
        raise TypeError(f"{field} must be a 2-item sequence")
    return (
        _require_int(pair[0], field=f"{field}[0]"),
        _require_int(pair[1], field=f"{field}[1]"),
    )


def _load_instance(
    payload: object,
    *,
    expected_keypoint_count: int,
    field: str,
) -> _NormalizedInstance:
    payload_map = _require_mapping(payload, field=field)
    keypoints_raw = _require_sequence(payload_map.get("keypoints"), field=f"{field}.keypoints")
    keypoints = tuple(
        _coerce_keypoint_triplet(entry, field=f"{field}.keypoints[{idx}]")
        for idx, entry in enumerate(keypoints_raw)
    )
    if len(keypoints) != expected_keypoint_count:
        raise ValueError(
            f"{field}.keypoints must contain {expected_keypoint_count} entries, "
            f"found {len(keypoints)}"
        )
    metadata_raw = payload_map.get("metadata")
    if metadata_raw is None:
        metadata = {}
    elif isinstance(metadata_raw, Mapping):
        metadata = dict(cast(Mapping[str, Any], metadata_raw))
    else:
        raise TypeError(f"{field}.metadata must be a mapping when provided")
    return _NormalizedInstance(keypoints=keypoints, metadata=metadata)


def _load_frame(
    payload: object,
    *,
    base_dir: Path,
    expected_keypoint_count: int,
    field: str,
) -> _NormalizedFrame:
    payload_map = _require_mapping(payload, field=field)
    image_path_raw = _require_str(payload_map.get("image_path"), field=f"{field}.image_path")
    image_path = Path(image_path_raw)
    if not image_path.is_absolute():
        image_path = (base_dir / image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"{field}.image_path missing on disk: {image_path}")
    instances_raw = _require_sequence(payload_map.get("instances"), field=f"{field}.instances")
    instances = tuple(
        _load_instance(
            entry,
            expected_keypoint_count=expected_keypoint_count,
            field=f"{field}.instances[{idx}]",
        )
        for idx, entry in enumerate(instances_raw)
    )
    return _NormalizedFrame(
        sequence_id=_require_str(payload_map.get("sequence_id"), field=f"{field}.sequence_id"),
        image_path=image_path,
        frame_id=_require_str(payload_map.get("frame_id"), field=f"{field}.frame_id"),
        source_index=_require_int_or_none(
            payload_map.get("source_index"),
            field=f"{field}.source_index",
        ),
        instances=instances,
    )


def _load_payload(annotations_path: Path) -> _NormalizedPayload:
    raw = load_json_dict(annotations_path)
    keypoint_names_raw = _require_sequence(raw.get("keypoint_names"), field="keypoint_names")
    keypoint_names = tuple(
        _require_str(name, field=f"keypoint_names[{idx}]")
        for idx, name in enumerate(keypoint_names_raw)
    )
    links_raw = _require_sequence(raw.get("links", []), field="links")
    links = tuple(
        _coerce_link_pair(entry, field=f"links[{idx}]")
        for idx, entry in enumerate(links_raw)
    )
    frames_raw = _require_sequence(raw.get("frames"), field="frames")
    frames = tuple(
        _load_frame(
            payload,
            base_dir=annotations_path.parent,
            expected_keypoint_count=len(keypoint_names),
            field=f"frames[{idx}]",
        )
        for idx, payload in enumerate(frames_raw)
    )
    return _NormalizedPayload(
        schema_version=_require_schema_version(raw.get("schema_version")),
        dataset_key=_require_str(raw.get("dataset_key"), field="dataset_key"),
        slice_key=_require_str(raw.get("slice_key"), field="slice_key"),
        project_name=_require_str(raw.get("project_name"), field="project_name"),
        keypoint_names=keypoint_names,
        links=links,
        frames=frames,
    )


def _build_skeleton(
    keypoint_names: Sequence[str],
    links: Sequence[tuple[int, int]],
) -> Skeleton:
    keypoints = [Keypoint(id=idx, name=name) for idx, name in enumerate(keypoint_names)]
    return Skeleton(name="benchmark", keypoints=keypoints, links_ids=list(links))


def _copy_sequence_frames(
    frames: Sequence[_NormalizedFrame],
    *,
    project_root: Path,
) -> tuple[Path, list[str], dict[str, int]]:
    sequence_id = frames[0].sequence_id
    target_dir = project_root / "videos" / sequence_id
    ensure_dir(target_dir)
    copied_paths: list[str] = []
    frame_idx_by_id: dict[str, int] = {}
    for frame_idx, frame in enumerate(frames):
        target_path = target_dir / f"{frame_idx:06d}_{frame.image_path.name}"
        if not target_path.exists():
            shutil.copy2(frame.image_path, target_path)
        copied_paths.append(target_path.as_posix())
        frame_idx_by_id[frame.frame_id] = frame_idx
    return target_dir, copied_paths, frame_idx_by_id


def _instance_points(
    instance: _NormalizedInstance,
    *,
    keypoints: Sequence[Keypoint],
) -> dict[str | Keypoint, Point]:
    out: dict[str | Keypoint, Point] = {}
    for idx, (x, y, visible) in enumerate(instance.keypoints):
        if x is None or y is None or visible <= 0:
            continue
        out[keypoints[idx]] = Point(float(x), float(y), visible=True, complete=True)
    return out


def _labels_from_payload(
    payload: _NormalizedPayload,
    *,
    project_root: Path,
) -> tuple[Labels, list[Path]]:
    labels = Labels()
    skeleton = _build_skeleton(payload.keypoint_names, payload.links)
    labels.skeletons = [skeleton]
    sequence_groups: dict[str, list[_NormalizedFrame]] = defaultdict(list)
    for frame in payload.frames:
        sequence_groups[frame.sequence_id].append(frame)

    video_dirs: list[Path] = []
    keypoints = list(skeleton.keypoints)
    for sequence_id in sorted(sequence_groups):
        frames = sequence_groups[sequence_id]
        video_dir, copied_paths, frame_idx_by_id = _copy_sequence_frames(
            frames,
            project_root=project_root,
        )
        video_dirs.append(video_dir)
        video = Video.from_image_filenames(copied_paths)
        for frame in frames:
            labeled_instances = _frame_instances(
                frame.instances,
                keypoints=keypoints,
                skeleton=skeleton,
            )
            if not labeled_instances:
                continue
            labels.append(
                LabeledFrame(
                    video=video,
                    frame_idx=frame_idx_by_id[frame.frame_id],
                    instances=labeled_instances,
                )
            )
    labels.update_cache()
    labels.validate()
    return labels, video_dirs


def _frame_instances(
    instances: Sequence[_NormalizedInstance],
    *,
    keypoints: Sequence[Keypoint],
    skeleton: Skeleton,
) -> list[Instance]:
    out: list[Instance] = []
    for instance in instances:
        points = _instance_points(instance, keypoints=keypoints)
        if points:
            out.append(Instance(skeleton=skeleton, init_points=points))
    return out


def convert_normalized_image_sequence_annotations(
    annotations_json: Path | str,
    out_dir: Path | str,
) -> ConversionResult:
    """Convert a normalized image-sequence JSON payload into project-ready labels."""
    annotations_path = resolve_path(annotations_json)
    project_root = resolve_path(out_dir)
    ensure_dir(project_root)
    payload = _load_payload(annotations_path)
    labels, video_dirs = _labels_from_payload(payload, project_root=project_root)
    metadata = {
        "project_name": payload.project_name,
        "source": "normalized_image_sequence_import",
        "source_annotations": annotations_path.as_posix(),
        "dataset_key": payload.dataset_key,
        "slice_key": payload.slice_key,
    }
    return ConversionResult(
        source_dir=annotations_path.parent,
        project_root=project_root,
        videos=list(video_dirs),
        labels=labels,
        metadata=metadata,
    )


__all__ = ["convert_normalized_image_sequence_annotations"]
