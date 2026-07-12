"""Boundary parser for SLEAP ``.pkg.slp`` frame and label data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import h5py
import numpy as np
import pandas as pd

from xpkg._core.json_utils import parse_json, parse_json_dict
from xpkg._core.path_registry import ensure_dir

_ERR_NO_VIDEOS = "No videos found in SLEAP package"


class SleapPackageFormatError(ValueError):
    """Raised when a SLEAP package cannot form a valid label table."""


@dataclass(frozen=True, slots=True)
class _SkeletonLayout:
    node_ids: tuple[int, ...]
    keypoints: tuple[str, ...]

    @property
    def columns(self) -> list[str]:
        columns = ["frame"]
        for keypoint in self.keypoints:
            columns.extend((f"{keypoint}_x", f"{keypoint}_y"))
        return columns


def _video_groups(hdf: h5py.File) -> dict[str, str]:
    videos: dict[str, str] = {}
    for group in hdf.keys():
        source_video = f"{group}/source_video"
        if not group.startswith("video") or source_video not in hdf:
            continue
        raw_json = hdf[source_video].attrs.get("json", "")
        if not raw_json:
            continue
        metadata = parse_json_dict(cast(str | bytes | bytearray, raw_json))
        backend = metadata.get("backend")
        filename = backend.get("filename") if isinstance(backend, dict) else None
        if isinstance(filename, str) and filename:
            videos[group] = filename
    return videos


def _group_video_indices_from_json(hdf: h5py.File) -> dict[str, int]:
    if "videos_json" not in hdf:
        return {}
    dataset = cast(h5py.Dataset, hdf["videos_json"])
    mapping: dict[str, int] = {}
    for index in range(len(dataset)):
        raw = dataset[index]
        payload_raw = parse_json(bytes(raw) if isinstance(raw, bytes | np.bytes_) else str(raw))
        if not isinstance(payload_raw, dict):
            raise SleapPackageFormatError("videos_json entries must be JSON objects")
        payload = {str(key): value for key, value in payload_raw.items()}
        backend_raw = payload.get("backend")
        backend = (
            {str(key): value for key, value in backend_raw.items()}
            if isinstance(backend_raw, dict)
            else {}
        )
        dataset_path = backend.get("dataset", payload.get("dataset"))
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            raise SleapPackageFormatError("videos_json entry missing backend.dataset")
        group = dataset_path.split("/", 1)[0]
        if not group:
            raise SleapPackageFormatError("videos_json entry missing dataset group")
        mapping[group] = index
    return mapping


def extract_frames(slp_path: str, out_dir: str) -> None:
    output_root = Path(out_dir)
    labeled_root = output_root / "labeled-data"
    ensure_dir(labeled_root)
    with h5py.File(slp_path, "r") as hdf:
        videos = _video_groups(hdf)
        if not videos:
            raise RuntimeError(_ERR_NO_VIDEOS)
        for group, filename in videos.items():
            output_dir = labeled_root / Path(filename).stem
            ensure_dir(output_dir)
            if f"{group}/video" not in hdf:
                continue
            frames = cast(Any, hdf[f"{group}/video"])
            frame_numbers = cast(Any, hdf[f"{group}/frame_numbers"])
            for image_bytes, frame_number in zip(frames, frame_numbers, strict=False):
                from xpkg.media.images import read_rgb_bytes

                rgb = read_rgb_bytes(bytes(image_bytes))
                destination = output_dir / f"img{int(frame_number):08d}.png"
                cv2.imwrite(destination.as_posix(), rgb)


def _skeleton_layout(hdf: h5py.File) -> _SkeletonLayout:
    metadata = parse_json_dict(
        cast(str | bytes | bytearray, hdf["metadata"].attrs.get("json", "{}"))
    )
    raw_nodes = metadata.get("nodes")
    raw_skeletons = metadata.get("skeletons")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise SleapPackageFormatError("SLEAP metadata must contain a non-empty nodes list")
    if not isinstance(raw_skeletons, list) or not raw_skeletons:
        raise SleapPackageFormatError("SLEAP metadata must contain a skeleton")

    names: dict[int, str] = {}
    for node_id, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, dict):
            raise SleapPackageFormatError(f"SLEAP node {node_id} must be an object")
        name = raw_node.get("name")
        if not isinstance(name, str) or not name.strip():
            raise SleapPackageFormatError(f"SLEAP node {node_id} must have a name")
        names[node_id] = name.strip()

    skeleton = raw_skeletons[0]
    if not isinstance(skeleton, dict):
        raise SleapPackageFormatError("SLEAP skeleton must be an object")
    raw_order = skeleton.get("nodes")
    if not raw_order:
        order = tuple(sorted(names))
    elif isinstance(raw_order, list):
        order = _parse_node_order(raw_order, names)
    else:
        raise SleapPackageFormatError("SLEAP skeleton nodes must be a list")
    return _SkeletonLayout(order, tuple(names[node_id] for node_id in order))


def _parse_node_order(raw_order: list[object], names: dict[int, str]) -> tuple[int, ...]:
    order: list[int] = []
    for item in raw_order:
        raw_id = item.get("id") if isinstance(item, dict) else item
        if isinstance(raw_id, dict):
            raw_id = raw_id.get("id")
        if not isinstance(raw_id, int | str) or not str(raw_id).isdigit():
            raise SleapPackageFormatError(f"Invalid SLEAP skeleton node reference: {item!r}")
        node_id = int(raw_id)
        if node_id not in names:
            raise SleapPackageFormatError(f"Unknown SLEAP skeleton node id: {node_id}")
        if node_id in order:
            raise SleapPackageFormatError(f"Duplicate SLEAP skeleton node id: {node_id}")
        order.append(node_id)
    if not order:
        raise SleapPackageFormatError("SLEAP skeleton must reference at least one node")
    return tuple(order)


def _frame_index(frame: Any, fields: set[str]) -> int:
    for field in ("frame_idx", "frame", "frame_id"):
        if field in fields:
            return int(frame[field])
    raise SleapPackageFormatError("SLEAP frames table has no frame index field")


def _frame_id(frame: Any, fields: set[str]) -> int:
    for field in ("frame_id", "frame", "frame_idx"):
        if field in fields:
            return int(frame[field])
    raise SleapPackageFormatError("SLEAP frames table has no frame identifier field")


def _frame_indices_by_video(frames: h5py.Dataset, fields: set[str]) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    if "video" not in fields:
        raise SleapPackageFormatError("SLEAP frames table has no video field")
    for frame in frames:
        result.setdefault(int(frame["video"]), set()).add(_frame_index(frame, fields))
    return result


def _group_frame_indices(hdf: h5py.File, groups: dict[str, str]) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for group in groups:
        path = f"{group}/frame_numbers"
        if path not in hdf:
            raise SleapPackageFormatError(f"SLEAP video group is missing frame_numbers: {group}")
        result[group] = {int(value) for value in cast(h5py.Dataset, hdf[path])}
    return result


def _match_video_groups(
    hdf: h5py.File,
    group_indices: dict[str, set[int]],
    video_indices: dict[int, set[int]],
) -> dict[str, int]:
    mapping = _group_video_indices_from_json(hdf)
    missing = sorted({video for video in mapping.values() if video not in video_indices})
    if missing:
        raise SleapPackageFormatError(f"videos_json indices missing from frames table: {missing}")
    for group, indices in group_indices.items():
        if group in mapping:
            continue
        exact = [video for video, candidate in video_indices.items() if candidate == indices]
        if len(exact) == 1:
            mapping[group] = exact[0]
            continue
        scored = [
            (len(indices & candidate) / len(indices | candidate), video)
            for video, candidate in video_indices.items()
            if indices & candidate
        ]
        if not scored:
            raise SleapPackageFormatError(f"No labeled-frame match for SLEAP group {group}")
        best_score = max(score for score, _video in scored)
        best = [video for score, video in scored if score == best_score]
        if len(best) != 1:
            raise SleapPackageFormatError(f"Ambiguous labeled-frame match for SLEAP group {group}")
        mapping[group] = best[0]
    return mapping


def _empty_table(layout: _SkeletonLayout) -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index(layout.columns))


def _table_from_rows(
    rows: list[list[float | None]], layout: _SkeletonLayout, video_name: str
) -> pd.DataFrame:
    if not rows:
        return _empty_table(layout)
    table = pd.DataFrame(rows, columns=pd.Index(layout.columns))
    table["frame"] = [
        f"labeled-data/{video_name}/img{int(frame):08d}.png" for frame in table["frame"]
    ]
    return table


def _instance_indices(frame: Any, frame_fields: set[str], instances: h5py.Dataset) -> list[int]:
    if {"instance_id_start", "instance_id_end"}.issubset(frame_fields):
        start = int(frame["instance_id_start"])
        end = int(frame["instance_id_end"])
        return list(range(start, end)) if end > start else []
    frame_id = _frame_id(frame, frame_fields)
    instance_fields = set(instances.dtype.names or ())
    if "frame_id" not in instance_fields:
        raise SleapPackageFormatError("SLEAP instances table has no frame_id field")
    return [
        index
        for index in range(len(instances))
        if int(instances[index]["frame_id"]) == frame_id
    ]


def _compact_instance_points(
    instance: Any,
    points: h5py.Dataset,
    point_fields: set[str],
    point_count: int,
) -> tuple[list[float | None], int] | None:
    instance_fields = set(instance.dtype.names or ())
    if "instance_type" in instance_fields and int(instance["instance_type"]) != 0:
        return None
    if not {"point_id_start", "point_id_end"}.issubset(instance_fields):
        raise SleapPackageFormatError("SLEAP instances missing point_id_start/point_id_end")
    start = int(instance["point_id_start"])
    end = int(instance["point_id_end"])
    if end <= start:
        return None
    flat: list[float | None] = []
    valid = 0
    for point in points[start:end]:
        if len(flat) >= point_count * 2:
            break
        x = float(point["x"])
        y = float(point["y"])
        visible = "visible" not in point_fields or bool(point["visible"])
        if visible and np.isfinite(x) and np.isfinite(y):
            flat.extend((x, y))
            valid += 1
        else:
            flat.extend((None, None))
    flat.extend([None] * (point_count * 2 - len(flat)))
    return flat, valid


def _compact_rows(
    frames: h5py.Dataset,
    instances: h5py.Dataset,
    points: h5py.Dataset,
    *,
    video: int,
    allowed_frames: set[int],
    point_count: int,
) -> list[list[float | None]]:
    frame_fields = set(frames.dtype.names or ())
    point_fields = set(points.dtype.names or ())
    rows: list[list[float | None]] = []
    for frame in frames:
        frame_index = _frame_index(frame, frame_fields)
        if int(frame["video"]) != video or frame_index not in allowed_frames:
            continue
        candidates = [
            _compact_instance_points(instances[index], points, point_fields, point_count)
            for index in _instance_indices(frame, frame_fields, instances)
        ]
        available = [candidate for candidate in candidates if candidate is not None]
        best = max(available, key=lambda candidate: candidate[1])[0] if available else None
        rows.append([float(frame_index), *(best or [None] * (point_count * 2))])
    return rows


def _frame_maps(
    frames: h5py.Dataset, *, video: int, allowed_frames: set[int]
) -> tuple[dict[int, int], dict[int, int]]:
    fields = set(frames.dtype.names or ())
    by_id: dict[int, int] = {}
    by_index: dict[int, int] = {}
    for frame in frames:
        frame_index = _frame_index(frame, fields)
        if int(frame["video"]) != video or frame_index not in allowed_frames:
            continue
        frame_id = _frame_id(frame, fields)
        by_id[frame_id] = frame_index
        by_index[frame_index] = frame_id
    return by_id, by_index


def _instances_by_frame(
    instances: h5py.Dataset,
    *,
    video: int,
    frame_map: dict[int, int],
    inverse_map: dict[int, int],
) -> dict[int, list[Any]]:
    fields = set(instances.dtype.names or ())
    grouped: dict[int, list[Any]] = {}
    for instance in instances:
        if "video" in fields and int(instance["video"]) != video:
            continue
        if "frame_id" in fields:
            frame_id = int(instance["frame_id"])
        elif "frame_idx" in fields:
            frame_id = inverse_map.get(int(instance["frame_idx"]), -1)
        elif "frame" in fields:
            frame_id = inverse_map.get(int(instance["frame"]), -1)
        else:
            raise SleapPackageFormatError("SLEAP instances table has no frame field")
        if frame_id in frame_map:
            grouped.setdefault(frame_id, []).append(instance)
    return grouped


def _point_record(point: Any, fields: set[str]) -> dict[str, object]:
    return {field: point[field] for field in fields}


def _candidate_points(
    instance: Any,
    points: h5py.Dataset,
    *,
    video: int,
    frame_id: int,
    frame_index: int,
) -> list[Any]:
    instance_fields = set(instance.dtype.names or ())
    point_fields = set(points.dtype.names or ())
    instance_id = next(
        (
            int(instance[key])
            for key in ("id", "instance", "instance_id", "inst_id")
            if key in instance_fields
        ),
        None,
    )
    instance_key = "instance" if "instance" in point_fields else "instance_id"
    if instance_id is not None and instance_key in point_fields:
        result: list[Any] = []
        for point in points:
            if int(point[instance_key]) != instance_id:
                continue
            if "video" in point_fields and int(point["video"]) != video:
                continue
            if "frame_id" in point_fields and int(point["frame_id"]) != frame_id:
                continue
            if "frame_idx" in point_fields and int(point["frame_idx"]) != frame_index:
                continue
            if "frame" in point_fields and int(point["frame"]) != frame_index:
                continue
            result.append(point)
        if result:
            return result
    if {"point_id_start", "point_id_end"}.issubset(instance_fields):
        start = int(instance["point_id_start"])
        end = int(instance["point_id_end"])
        if 0 <= start <= end <= len(points):
            return list(points[start:end])
    return []


def _points_by_node(
    candidates: list[Any], point_fields: set[str], node_order: tuple[int, ...]
) -> dict[int, dict[str, object]]:
    node_key = "node" if "node" in point_fields else "node_id"
    if node_key in point_fields:
        return {int(point[node_key]): _point_record(point, point_fields) for point in candidates}
    return {
        node_order[index]: _point_record(point, point_fields)
        for index, point in enumerate(candidates[: len(node_order)])
    }


def _fallback_points_by_node(
    points: h5py.Dataset,
    *,
    video: int,
    frame_id: int,
    frame_index: int,
) -> dict[int, dict[str, object]]:
    fields = set(points.dtype.names or ())
    node_key = "node" if "node" in fields else "node_id"
    if node_key not in fields:
        return {}
    instance_key = "instance" if "instance" in fields else "instance_id"
    grouped: dict[int, dict[int, dict[str, object]]] = {}
    for point in points:
        if "video" in fields and int(point["video"]) != video:
            continue
        if "frame_id" in fields and int(point["frame_id"]) != frame_id:
            continue
        if "frame_idx" in fields and int(point["frame_idx"]) != frame_index:
            continue
        if "frame" in fields and int(point["frame"]) != frame_index:
            continue
        instance_id = int(point[instance_key]) if instance_key in fields else 0
        grouped.setdefault(instance_id, {})[int(point[node_key])] = _point_record(point, fields)
    return max(grouped.values(), key=len) if grouped else {}


def _flatten_points(
    points_by_node: dict[int, dict[str, object]],
    node_order: tuple[int, ...],
) -> tuple[list[float | None], int]:
    flat: list[float | None] = []
    valid = 0
    for node_id in node_order:
        point = points_by_node.get(node_id)
        if point is None or point.get("x") is None or point.get("y") is None:
            flat.extend((None, None))
            continue
        x = float(cast(float | int, point["x"]))
        y = float(cast(float | int, point["y"]))
        visible = point.get("visible", point.get("is_visible", True))
        if bool(visible) and np.isfinite(x) and np.isfinite(y):
            flat.extend((x, y))
            valid += 1
        else:
            flat.extend((None, None))
    return flat, valid


def _best_relational_points(
    frame_instances: list[Any],
    points: h5py.Dataset,
    *,
    video: int,
    frame_id: int,
    frame_index: int,
    node_order: tuple[int, ...],
) -> list[float | None] | None:
    point_fields = set(points.dtype.names or ())
    candidates: list[tuple[list[float | None], int]] = []
    for instance in frame_instances:
        fields = set(instance.dtype.names or ())
        if "instance_type" in fields and int(instance["instance_type"]) != 0:
            continue
        raw_points = _candidate_points(
            instance, points, video=video, frame_id=frame_id, frame_index=frame_index
        )
        by_node = _points_by_node(raw_points, point_fields, node_order) if raw_points else {}
        if not by_node:
            by_node = _fallback_points_by_node(
                points, video=video, frame_id=frame_id, frame_index=frame_index
            )
        candidates.append(_flatten_points(by_node, node_order))
    return max(candidates, key=lambda candidate: candidate[1])[0] if candidates else None


def _relational_rows(
    frames: h5py.Dataset,
    instances: h5py.Dataset,
    points: h5py.Dataset,
    *,
    video: int,
    allowed_frames: set[int],
    node_order: tuple[int, ...],
) -> list[list[float | None]]:
    frame_map, inverse_map = _frame_maps(frames, video=video, allowed_frames=allowed_frames)
    grouped = _instances_by_frame(
        instances, video=video, frame_map=frame_map, inverse_map=inverse_map
    )
    rows: list[list[float | None]] = []
    for frame_id, frame_index in frame_map.items():
        best = _best_relational_points(
            grouped.get(frame_id, []),
            points,
            video=video,
            frame_id=frame_id,
            frame_index=frame_index,
            node_order=node_order,
        )
        rows.append([float(frame_index), *(best or [None] * (len(node_order) * 2))])
    return rows


def build_sleap_label_table(slp_path: str, out_dir: str) -> pd.DataFrame:
    """Parse a SLEAP package into the flattened label-table boundary object."""

    ensure_dir(Path(out_dir) / "labeled-data")
    with h5py.File(slp_path, "r") as hdf:
        videos = _video_groups(hdf)
        if not videos:
            raise RuntimeError(_ERR_NO_VIDEOS)
        layout = _skeleton_layout(hdf)
        frames = cast(h5py.Dataset, hdf["frames"])
        points = cast(h5py.Dataset, hdf["points"])
        instances = cast(h5py.Dataset, hdf["instances"])
        frame_fields = set(frames.dtype.names or ())
        video_indices = _frame_indices_by_video(frames, frame_fields)
        group_indices = _group_frame_indices(hdf, videos)
        group_to_video = _match_video_groups(hdf, group_indices, video_indices)
        point_fields = set(points.dtype.names or ())
        compact = {"x", "y"}.issubset(point_fields) and not {
            "instance",
            "instance_id",
            "node",
            "node_id",
            "frame",
            "frame_idx",
            "frame_id",
            "video",
        }.intersection(point_fields)
        tables: list[pd.DataFrame] = []
        for group, filename in videos.items():
            video = group_to_video[group]
            if compact:
                rows = _compact_rows(
                    frames,
                    instances,
                    points,
                    video=video,
                    allowed_frames=group_indices[group],
                    point_count=len(layout.node_ids),
                )
            else:
                rows = _relational_rows(
                    frames,
                    instances,
                    points,
                    video=video,
                    allowed_frames=group_indices[group],
                    node_order=layout.node_ids,
                )
            if rows:
                tables.append(_table_from_rows(rows, layout, Path(filename).stem))
        if not tables:
            return _empty_table(layout)
        return cast(pd.DataFrame, pd.concat(tables, ignore_index=True))


__all__ = ["SleapPackageFormatError", "build_sleap_label_table", "extract_frames"]
