"""JSON serialization helpers for canonical `Labels` payloads."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from xpkg.core.json_utils import load_json_dict, write_json
from xpkg.core.path_registry import ensure_dir, make_path_id

if TYPE_CHECKING:
    from xpkg.core.annotations import Instance
    from xpkg.io.labels.model import Labels, SuggestionFrame
    from xpkg.io.labels.video_types import VideoProtocol

POSETTA_LABELS_JSON_FORMAT = "posetta.labels-json"
POSETTA_LABELS_JSON_VERSION = "2.0.0"


def _sorted_labeled_frames(labels: Labels) -> list[Any]:
    video_lookup = {video: idx for idx, video in enumerate(labels.videos)}
    return sorted(
        labels.labeled_frames,
        key=lambda lf: (video_lookup[lf.video], int(lf.frame_idx)),
    )


def _video_identity(video: VideoProtocol, index: int) -> tuple[str, str]:
    raw_name = str(video.filename or "").strip()
    if raw_name:
        path_id = make_path_id(raw_name, prefix="video")
        return path_id.id, path_id.label

    frames = list(video.image_filenames or [])
    if frames:
        path_id = make_path_id(frames[0], prefix="video")
        return path_id.id, path_id.label

    return f"video_{index}", f"video-{index}"


def _video_payload(labels: Labels) -> dict[str, Any]:
    filenames: list[str] = []
    image_filenames: list[list[str]] = []
    backends: list[str] = []
    sha256_hashes: list[str] = []
    video_ids: list[str] = []
    video_labels: list[str] = []
    shapes = np.zeros((len(labels.videos), 4), dtype=np.int32)

    for index, video in enumerate(labels.videos):
        filenames.append(str(video.filename or ""))
        image_filenames.append(list(video.image_filenames or []))
        backends.append(str(video.backend or "opencv"))
        sha256_hashes.append(str(video.sha256 or ""))
        video_id, video_label = _video_identity(video, index)
        video_ids.append(video.id or video_id)
        video_labels.append(video.label or video_label)
        shapes[index] = (
            int(video.frames),
            int(video.height),
            int(video.width),
            int(video.channels),
        )

    return {
        "resolved_paths": filenames,
        "filenames": filenames,
        "image_filenames": image_filenames,
        "backends": backends,
        "sha256": sha256_hashes,
        "video_ids": video_ids,
        "video_labels": video_labels,
        "shapes": shapes.tolist(),
    }


def _instance_point_payload(
    instance: Instance,
    keypoint_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = instance.get_points_array(copy=False, full=True)
    coords = np.full((keypoint_count, 3), np.nan, dtype=np.float32)
    flags = np.zeros((keypoint_count,), dtype=np.uint8)
    coords[:, 0] = np.asarray(points["x"], dtype=np.float32)
    coords[:, 1] = np.asarray(points["y"], dtype=np.float32)
    visible = np.asarray(points["visible"], dtype=bool)
    coords[~visible, :2] = np.nan
    flags[:] = np.asarray(points["flags"], dtype=np.uint8)
    if "score" in points.dtype.names:
        coords[:, 2] = np.asarray(points["score"], dtype=np.float32)
    else:
        coords[visible, 2] = 1.0
    return coords, flags


def _frame_data_payload(labels: Labels) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = _sorted_labeled_frames(labels)
    keypoint_count = len(labels.skeleton.keypoints)
    max_instances = max((len(frame.instances) for frame in rows), default=1)
    row_count = len(rows)
    video_lookup = {video: idx for idx, video in enumerate(labels.videos)}

    video_index = np.zeros((row_count,), dtype=np.int32)
    frame_index = np.zeros((row_count,), dtype=np.int32)
    num_instances = np.zeros((row_count,), dtype=np.int32)
    keypoints = np.full((row_count, max_instances, keypoint_count, 3), np.nan, dtype=np.float32)
    flags = np.zeros((row_count, max_instances, keypoint_count), dtype=np.uint8)
    track_ids = np.full((row_count, max_instances), -1, dtype=np.int32)

    for row_idx, labeled_frame in enumerate(rows):
        video_index[row_idx] = int(video_lookup[labeled_frame.video])
        frame_index[row_idx] = int(labeled_frame.frame_idx)
        num_instances[row_idx] = int(len(labeled_frame.instances))
        for inst_idx, instance in enumerate(labeled_frame.instances):
            coords, point_flags = _instance_point_payload(instance, keypoint_count)
            keypoints[row_idx, inst_idx] = coords
            flags[row_idx, inst_idx] = point_flags
            if instance.track is not None:
                track_ids[row_idx, inst_idx] = int(instance.track.id)

    frames_payload = {
        "video_index": video_index.tolist(),
        "frame_index": frame_index.tolist(),
        "num_instances": num_instances.tolist(),
    }
    data_payload = {
        "keypoints": keypoints.tolist(),
        "flags": flags.tolist(),
        "track_ids": track_ids.tolist(),
    }
    return frames_payload, data_payload


def _suggestions_payload(
    suggestions: Iterable[SuggestionFrame],
    video_lookup: dict[VideoProtocol, int],
) -> dict[str, Any] | None:
    items = list(suggestions)
    if not items:
        return None

    video_indices: list[int] = []
    frame_indices: list[int] = []
    scores: list[float] = []
    keep_scores = False

    for item in items:
        if item.video not in video_lookup:
            raise ValueError("Suggestion references a video not present in labels.videos")
        video_indices.append(int(video_lookup[item.video]))
        frame_indices.append(int(item.frame_idx))
        if item.score is not None:
            keep_scores = True
            scores.append(float(item.score))
        else:
            scores.append(0.0)

    payload: dict[str, Any] = {
        "video_indices": video_indices,
        "frame_indices": frame_indices,
    }
    if keep_scores:
        payload["scores"] = scores
    return payload


def _tracks_payload(labels: Labels) -> dict[str, Any] | None:
    if not labels.tracks:
        return None

    payload: dict[str, Any] = {}
    for track in sorted(labels.tracks, key=lambda item: (item.spawned_on, item.name)):
        payload[str(int(track.id))] = {
            "spawned_on": int(track.spawned_on),
            "name": str(track.name or f"track-{track.id}"),
        }
    return payload


def labels_to_json_payload(
    labels: Labels,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(labels.skeletons) != 1:
        raise ValueError("Labels JSON export requires exactly one skeleton")

    video_lookup = {video: idx for idx, video in enumerate(labels.videos)}
    frames_payload, data_payload = _frame_data_payload(labels)
    skeleton = labels.skeleton
    symmetry = {
        keypoint.name: keypoint.mirror_partner
        for keypoint in skeleton.keypoints
        if keypoint.mirror_partner
    }
    metadata_payload = dict(metadata or {})
    metadata_payload["preferences"] = dict(labels.preferences)

    payload: dict[str, Any] = {
        "format": POSETTA_LABELS_JSON_FORMAT,
        "version": POSETTA_LABELS_JSON_VERSION,
        "payload": {
            "frames": frames_payload,
            "data": data_payload,
            "videos": _video_payload(labels),
            "metadata": metadata_payload,
            "provenance": dict(labels.provenance),
            "session": dict(labels.session),
            "skeleton": {
                "schema_version": str(skeleton.schema_version),
                "name": str(skeleton.name),
                "names": list(skeleton.keypoint_names),
                "roles": [str(keypoint.role or "") for keypoint in skeleton.keypoints],
                "links": [list(link) for link in skeleton.links_ids],
                "symmetry": symmetry,
            },
        },
    }

    suggestions_payload = _suggestions_payload(labels.suggestions, video_lookup)
    if suggestions_payload is not None:
        payload["payload"]["suggestions"] = suggestions_payload

    tracks_payload = _tracks_payload(labels)
    if tracks_payload is not None:
        payload["payload"]["tracks"] = tracks_payload

    segmentation_payload = _segmentation_payload(labels, video_lookup)
    if segmentation_payload is not None:
        payload["payload"]["segmentation"] = segmentation_payload

    return payload


def _segmentation_payload(
    labels: Labels,
    video_lookup: dict[VideoProtocol, int],
) -> dict[str, Any] | None:
    """Build JSON payload for segmentation masks and ROIs."""
    masks_list: list[dict[str, Any]] = []
    rois_list: list[dict[str, Any]] = []

    sorted_frames = sorted(
        labels.labeled_frames,
        key=lambda lf: (video_lookup.get(lf.video, 0), int(lf.frame_idx)),
    )

    for lf in sorted_frames:
        if not hasattr(lf, "masks") and not hasattr(lf, "rois"):
            continue
        vi = video_lookup.get(lf.video, 0)
        fi = int(lf.frame_idx)
        for mask in getattr(lf, "masks", []):
            d = mask.to_dict()
            d["video_index"] = vi
            d["frame_index"] = fi
            masks_list.append(d)
        for roi in getattr(lf, "rois", []):
            d = roi.to_dict()
            d["video_index"] = vi
            d["frame_index"] = fi
            rois_list.append(d)

    if not masks_list and not rois_list:
        return None

    result: dict[str, Any] = {"version": "1.0.0"}
    if masks_list:
        result["masks"] = masks_list
    if rois_list:
        result["rois"] = rois_list
    return result


def write_labels_json(
    path: str | Path,
    labels: Labels,
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    target = Path(path)
    if target.suffix.lower() != ".json":
        raise ValueError(f"Labels JSON path must use .json suffix, got: {target}")
    ensure_dir(target.parent)
    payload = labels_to_json_payload(labels, metadata=metadata)
    write_json(target, payload, indent=2, sort_keys=False, ensure_ascii=True)
    labels.path = target
    return str(target)


def read_labels_json_payload(path: str | Path) -> dict[str, Any]:
    raw = load_json_dict(path)
    fmt = str(raw.get("format", "")).strip()
    if fmt != POSETTA_LABELS_JSON_FORMAT:
        raise ValueError(
            f"Unsupported labels JSON format {fmt!r}; expected {POSETTA_LABELS_JSON_FORMAT!r}"
        )
    payload = raw.get("payload")
    if not isinstance(payload, dict):
        raise TypeError("Labels JSON payload must contain an object under 'payload'")
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise TypeError("Labels JSON payload keys must be strings")
        out[key] = value
    return out
