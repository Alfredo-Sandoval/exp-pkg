"""Inspect files and folders before importing them into xpkg projects."""

from __future__ import annotations

import importlib
import math
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pandas as pd

from xpkg._core.json_utils import load_json, parse_json
from xpkg.io.readers.pose import read_pose_track
from xpkg.model.metadata import AcquisitionMetadata, DatasetShareMetadata, PoseModelProvenance
from xpkg.model.reporting import DatasetDatasheet, ModelCard
from xpkg.project.artifact import EXPKG_MANIFEST_FILENAME
from xpkg.project.layout import (
    PROJECT_DESCRIPTOR_FILENAME,
    STORE_DIRNAME,
    load_project_descriptor,
    project_summary_path,
    resolve_project_root,
)
from xpkg.project.metadata import (
    ACQUISITION_METADATA_FILENAME,
    DATASET_SHARE_METADATA_FILENAME,
    DATASHEET_FILENAME,
    MODEL_CARD_FILENAME,
    POSE_PROVENANCE_FILENAME,
    PROJECT_METADATA_DIRNAME,
    load_project_acquisition_metadata,
    load_project_dataset_share_metadata,
    load_project_datasheet,
    load_project_model_card,
    load_project_pose_provenance,
    project_acquisition_metadata_path,
    project_dataset_share_metadata_path,
    project_datasheet_path,
    project_model_card_path,
    project_pose_provenance_path,
)
from xpkg.project.summary import snapshot_project_summary


class InspectionKind(StrEnum):
    """Canonical kinds returned by :func:`inspect_path`."""

    UNKNOWN = "unknown"
    DIRECTORY = "directory"
    IMAGE_SEQUENCE = "image_sequence"
    DLC_PROJECT = "dlc_project"
    XPKG_PROJECT = "xpkg_project"
    XPKG_PROJECT_DESCRIPTOR = "xpkg_project_descriptor"
    EXPKG_MANIFEST = "expkg_manifest"
    EXPKG_ARTIFACT = "expkg_artifact"
    POSE_PREDICTIONS = "pose_predictions"
    EVENTS_TABLE = "events_table"
    SIGNALS_TABLE = "signals_table"
    PHOTOMETRY_TABLE = "photometry_table"
    PHOTOMETRY_RECORDING = "photometry_recording"
    CSV = "csv"
    JSON = "json"
    HDF5 = "hdf5"
    VIDEO = "video"
    IMAGE = "image"
    SLEAP_PACKAGE = "sleap_package"


@dataclass(frozen=True, slots=True)
class InspectionWarning:
    """Stable machine-readable warning record for inspection JSON output."""

    code: str
    message: str
    path: str | None = None
    severity: str = "warning"

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable warning record."""
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "severity": self.severity,
        }


@dataclass(frozen=True, slots=True)
class InspectionReport:
    """Structured inspection result for a single file or folder."""

    path: str
    name: str
    suffix: str
    exists: bool
    is_dir: bool
    size_bytes: int | None
    kind: InspectionKind
    likely_importers: tuple[str, ...]
    summary: Mapping[str, Any]
    warnings: tuple[str, ...]
    warning_records: tuple[InspectionWarning, ...] = ()

    @property
    def status(self) -> str:
        return "inspected"

    @property
    def description(self) -> str:
        return self.kind.value.replace("_", " ")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable wire form of this report."""
        return {
            "status": self.status,
            "path": self.path,
            "name": self.name,
            "suffix": self.suffix,
            "exists": self.exists,
            "is_dir": self.is_dir,
            "size_bytes": self.size_bytes,
            "kind": self.kind.value,
            "description": self.description,
            "likely_importers": list(self.likely_importers),
            "summary": dict(self.summary),
            "warnings": list(self.warnings),
            "warning_records": [warning.to_dict() for warning in self.warning_records],
        }

_VIDEO_SUFFIXES = {
    ".avi",
    ".h264",
    ".h265",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
_POSE_CONFIDENCE_THRESHOLD = 0.5
_TIME_COLUMNS = {"time", "timestamp", "timestamps", "time_s", "seconds", "sec"}
_EVENT_COLUMNS = {"event", "kind", "label", "behavior", "start", "start_s", "duration"}
_PHOTOMETRY_HINTS = {"405", "410", "415", "465", "470", "560", "gcamp", "isosbestic", "dff"}

_PROJECT_METADATA_SLOTS: tuple[
    tuple[str, Callable[[str | Path], Path], Callable[[str | Path], object]],
    ...,
] = (
    ("acquisition", project_acquisition_metadata_path, load_project_acquisition_metadata),
    ("dataset_share", project_dataset_share_metadata_path, load_project_dataset_share_metadata),
    ("datasheet", project_datasheet_path, load_project_datasheet),
    ("model_card", project_model_card_path, load_project_model_card),
    ("pose_provenance", project_pose_provenance_path, load_project_pose_provenance),
)

_PROJECT_METADATA_ARCHIVE_SLOTS: tuple[
    tuple[str, str, Callable[[Mapping[str, Any]], object]],
    ...,
] = (
    ("acquisition", ACQUISITION_METADATA_FILENAME, AcquisitionMetadata.from_dict),
    ("dataset_share", DATASET_SHARE_METADATA_FILENAME, DatasetShareMetadata.from_dict),
    ("datasheet", DATASHEET_FILENAME, DatasetDatasheet.from_dict),
    ("model_card", MODEL_CARD_FILENAME, ModelCard.from_dict),
    ("pose_provenance", POSE_PROVENANCE_FILENAME, PoseModelProvenance.from_dict),
)


def _path_payload(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "name": path.name,
        "suffix": path.suffix.lower(),
        "exists": True,
        "is_dir": path.is_dir(),
        "size_bytes": int(stat.st_size) if path.is_file() else None,
    }


def _unique(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items if str(item).strip()))


def _warning_record(
    code: str,
    message: object,
    *,
    path: object | None = None,
    severity: str = "warning",
) -> InspectionWarning:
    path_text = None if path is None else str(path)
    return InspectionWarning(
        code=str(code),
        message=str(message),
        path=path_text,
        severity=str(severity),
    )


def _generic_warning_records(
    warnings: Sequence[str],
    *,
    path: object | None = None,
) -> tuple[InspectionWarning, ...]:
    return tuple(
        _warning_record("inspection_warning", warning, path=path) for warning in warnings
    )


def _coerce_warning_records(
    records: object,
    *,
    fallback_warnings: Sequence[str],
    fallback_path: object | None,
) -> tuple[InspectionWarning, ...]:
    if records is None:
        return _generic_warning_records(fallback_warnings, path=fallback_path)
    if not isinstance(records, Sequence) or isinstance(records, str | bytes | bytearray):
        return _generic_warning_records(fallback_warnings, path=fallback_path)
    out: list[InspectionWarning] = []
    for record in records:
        if isinstance(record, InspectionWarning):
            out.append(record)
            continue
        if not isinstance(record, Mapping):
            continue
        record_map = cast("Mapping[str, Any]", record)
        out.append(
            _warning_record(
                record_map.get("code", "inspection_warning"),
                record_map.get("message", ""),
                path=record_map.get("path"),
                severity=str(record_map.get("severity", "warning")),
            )
        )
    if len(out) != len(fallback_warnings):
        return _generic_warning_records(fallback_warnings, path=fallback_path)
    return tuple(out)


def _read_text_lines(path: Path, *, limit: int = 8) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for _index in range(limit):
            line = handle.readline()
            if not line:
                break
            lines.append(line.rstrip("\n\r"))
    return lines


def _columns_lower(columns: Sequence[object]) -> set[str]:
    return {str(column).strip().lower() for column in columns}


def _inspect_project_dir(path: Path) -> dict[str, Any]:
    project_root = resolve_project_root(path)
    if project_root is None:
        raise FileNotFoundError(f"Not an xpkg project: {path}")
    descriptor = load_project_descriptor(project_root).to_dict()
    summary = snapshot_project_summary(project_root)
    metadata_slots, metadata_warnings, metadata_warning_records = (
        _inspect_project_metadata_slots(project_root)
    )
    inventory_warnings, inventory_warning_records = _project_media_inventory_warnings(
        state_kind=summary.state_kind,
        has_current_state=summary.has_current_state,
        state_summary=summary.state_summary,
        media_items=summary.media,
        summary_path=project_summary_path(project_root),
    )
    media_warnings, media_warning_records = _project_media_warnings(project_root, summary.media)
    summary_warning_records = [
        _warning_record(
            "project_summary_warning",
            warning,
            path=project_summary_path(project_root),
        )
        for warning in summary.warnings
    ]
    return {
        "kind": "xpkg_project",
        "likely_importers": [],
        "summary": {
            "project_id": descriptor["project_id"],
            "title": descriptor["title"],
            "state_kind": summary.state_kind,
            "has_current_state": summary.has_current_state,
            "state_bytes": summary.state_bytes,
            "commit_id": summary.commit_id,
            "modalities": list(summary.modalities),
            "summary_path": str(project_summary_path(project_root)),
            "metadata_slots": metadata_slots,
            "media": _inspect_project_media(project_root, summary.media),
        },
        "warnings": [
            *summary.warnings,
            *metadata_warnings,
            *inventory_warnings,
            *media_warnings,
        ],
        "warning_records": [
            *summary_warning_records,
            *metadata_warning_records,
            *inventory_warning_records,
            *media_warning_records,
        ],
    }


def _inspect_project_metadata_slots(
    project_root: Path,
) -> tuple[dict[str, dict[str, Any]], list[str], list[InspectionWarning]]:
    slots: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    warning_records: list[InspectionWarning] = []
    for name, path_for, load in _PROJECT_METADATA_SLOTS:
        metadata_path = path_for(project_root)
        present = metadata_path.is_file()
        slot: dict[str, Any] = {
            "path": str(metadata_path),
            "present": present,
            "valid": None,
        }
        if present:
            try:
                load(project_root)
            except (OSError, TypeError, ValueError) as exc:
                slot["valid"] = False
                slot["error"] = str(exc)
                message = f"Project metadata slot {name!r} is invalid: {exc}"
                warnings.append(message)
                warning_records.append(
                    _warning_record(
                        "project_metadata_invalid",
                        message,
                        path=metadata_path,
                    )
                )
            else:
                slot["valid"] = True
        slots[name] = slot
    return slots, warnings, warning_records


def _inspect_project_media(
    project_root: Path,
    media_items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    inspected: list[dict[str, Any]] = []
    for item in media_items:
        payload = dict(item)
        media_path = _resolve_project_media_path(project_root, payload.get("path"))
        payload["exists"] = _media_item_exists(media_path, payload.get("kind"))
        if payload.get("kind") == "image_sequence" and media_path is not None:
            payload["current_image_count"] = _image_sequence_file_count(media_path)
        inspected.append(payload)
    return inspected


def _project_media_inventory_warnings(
    *,
    state_kind: object,
    has_current_state: bool,
    state_summary: Mapping[str, Any],
    media_items: Sequence[Mapping[str, Any]],
    summary_path: Path,
) -> tuple[list[str], list[InspectionWarning]]:
    if state_kind != "labels" or not has_current_state or media_items:
        return [], []
    video_count = _optional_non_negative_int(state_summary.get("video_count"))
    if video_count == 0:
        return [], []
    if video_count is None:
        message = (
            "Project media inventory is unavailable for labels state; no "
            "summary-recorded media item(s) are available."
        )
    else:
        message = (
            "Project media inventory is unavailable for labels state with "
            f"{video_count} recorded video(s)."
        )
    return [
        message
    ], [
        _warning_record(
            "project_media_inventory_unavailable",
            message,
            path=summary_path,
        )
    ]


def _project_media_warnings(
    project_root: Path,
    media_items: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[InspectionWarning]]:
    warnings: list[str] = []
    warning_records: list[InspectionWarning] = []
    for item in media_items:
        label = str(item.get("label") or item.get("path") or item.get("index") or "unknown")
        media_path = _resolve_project_media_path(project_root, item.get("path"))
        if not _media_item_exists(media_path, item.get("kind")):
            message = f"Project media item {label!r} is missing: {item.get('path')}"
            warnings.append(message)
            warning_records.append(
                _warning_record(
                    "project_media_missing",
                    message,
                    path=item.get("path"),
                )
            )
            continue
        expected_images = _optional_positive_int(item.get("image_count"))
        if item.get("kind") == "image_sequence" and expected_images is not None:
            current_images = _image_sequence_file_count(media_path)
            if current_images != expected_images:
                message = (
                    f"Project media item {label!r} expected {expected_images} image file(s), "
                    f"found {current_images}."
                )
                warnings.append(message)
                warning_records.append(
                    _warning_record(
                        "project_media_image_count_drift",
                        message,
                        path=item.get("path"),
                    )
                )

        frame_count = _optional_positive_int(item.get("frame_count"))
        if frame_count is None:
            continue
        max_label = _optional_non_negative_int(item.get("max_label_frame_index"))
        if max_label is not None and max_label >= frame_count:
            message = (
                f"Project media item {label!r} has label frame index {max_label} "
                f"outside {frame_count} frame(s)."
            )
            warnings.append(message)
            warning_records.append(
                _warning_record(
                    "project_media_label_frame_out_of_range",
                    message,
                    path=item.get("path"),
                )
            )
        max_prediction = _optional_non_negative_int(item.get("max_prediction_frame_index"))
        if max_prediction is not None and max_prediction >= frame_count:
            message = (
                f"Project media item {label!r} has prediction frame index {max_prediction} "
                f"outside {frame_count} frame(s)."
            )
            warnings.append(message)
            warning_records.append(
                _warning_record(
                    "project_media_prediction_frame_out_of_range",
                    message,
                    path=item.get("path"),
                )
            )
    return warnings, warning_records


def _resolve_project_media_path(project_root: Path, raw_path: object) -> Path | None:
    path_text = str(raw_path or "").strip()
    if not path_text:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else project_root / path


def _media_item_exists(path: Path | None, kind: object) -> bool:
    if path is None:
        return False
    return path.is_dir() if kind == "image_sequence" else path.is_file()


def _image_sequence_file_count(path: Path | None) -> int | None:
    if path is None or not path.is_dir():
        return None
    return sum(1 for candidate in path.iterdir() if candidate.is_file())


def _optional_positive_int(value: object) -> int | None:
    result = _optional_non_negative_int(value)
    if result is None or result <= 0:
        return None
    return result


def _optional_non_negative_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        result = int(cast("str | bytes | bytearray | int | float | bool", value))
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _inspect_directory(path: Path) -> dict[str, Any]:
    if (path / PROJECT_DESCRIPTOR_FILENAME).is_file() and resolve_project_root(path) == path:
        return _inspect_project_dir(path)

    files = [candidate for candidate in path.iterdir() if candidate.is_file()]
    image_files = [candidate for candidate in files if candidate.suffix.lower() in _IMAGE_SUFFIXES]
    summary: dict[str, Any] = {
        "entries": len(list(path.iterdir())),
        "files": len(files),
        "image_files": len(image_files),
    }
    likely_importers: list[str] = []
    kind = "directory"
    if image_files:
        kind = "image_sequence"
        likely_importers.append("image_sequence")
        summary["first_image"] = image_files[0].name
    if (path / "config.yaml").is_file() and (path / "labeled-data").is_dir():
        kind = "dlc_project"
        likely_importers.append("dlc_project")
    return {
        "kind": kind,
        "likely_importers": likely_importers,
        "summary": summary,
        "warnings": [],
    }


def _longest_true_run(mask: np.ndarray) -> int:
    """Return the longest run of consecutive True values along ``mask``."""
    if mask.size == 0:
        return 0
    longest = 0
    current = 0
    for value in mask.tolist():
        if value:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return int(longest)


def _per_keypoint_confidence(
    scores: np.ndarray,
    *,
    node_names: Sequence[str],
    confidence_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute per-keypoint confidence stats and an aggregate summary."""
    if scores.ndim != 2:
        scores = scores.reshape(scores.shape[0], -1)
    n_frames = int(scores.shape[0])
    n_keypoints = int(scores.shape[1])
    per_keypoint: list[dict[str, Any]] = []
    worst_name: str | None = None
    worst_below_fraction = -1.0
    longest_run_overall = 0

    for index in range(n_keypoints):
        column = scores[:, index]
        finite = np.isfinite(column)
        finite_count = int(finite.sum())
        below_mask = (column < confidence_threshold) | (~finite)
        below_threshold = int(below_mask.sum())
        longest_low_run = _longest_true_run(below_mask)
        if finite_count:
            with np.errstate(invalid="ignore"):
                mean_score = float(np.nanmean(column))
                min_score = float(np.nanmin(column))
                max_score = float(np.nanmax(column))
        else:
            mean_score = math.nan
            min_score = math.nan
            max_score = math.nan
        below_fraction = float(below_threshold / n_frames) if n_frames else 0.0
        name = node_names[index] if index < len(node_names) else f"kp_{index}"
        per_keypoint.append(
            {
                "name": str(name),
                "finite_count": finite_count,
                "below_threshold": below_threshold,
                "below_threshold_fraction": below_fraction,
                "longest_low_run": longest_low_run,
                "mean": None if math.isnan(mean_score) else mean_score,
                "min": None if math.isnan(min_score) else min_score,
                "max": None if math.isnan(max_score) else max_score,
            }
        )
        if below_fraction > worst_below_fraction:
            worst_below_fraction = below_fraction
            worst_name = str(name)
        if longest_low_run > longest_run_overall:
            longest_run_overall = longest_low_run

    aggregate: dict[str, Any] = {
        "worst_keypoint": worst_name,
        "worst_below_threshold_fraction": float(worst_below_fraction)
        if worst_below_fraction >= 0.0
        else 0.0,
        "longest_low_run_frames": int(longest_run_overall),
    }
    return per_keypoint, aggregate


def _pose_qc(
    path: Path,
    *,
    software: str,
    file_type: str,
    confidence_threshold: float,
) -> dict[str, Any]:
    track = read_pose_track(path, software=software, file_type=file_type, track_index=0)
    scores = np.asarray(track.scores, dtype=np.float64)
    finite = np.isfinite(scores)
    finite_count = int(finite.sum())
    total = int(scores.size)
    below_threshold = int(((scores < confidence_threshold) & finite).sum())
    with np.errstate(invalid="ignore"):
        mean_score = float(np.nanmean(scores)) if finite_count else math.nan
        min_score = float(np.nanmin(scores)) if finite_count else math.nan
        max_score = float(np.nanmax(scores)) if finite_count else math.nan
    per_keypoint, aggregate = _per_keypoint_confidence(
        scores,
        node_names=track.node_names,
        confidence_threshold=confidence_threshold,
    )
    return {
        "frames": int(track.coords.shape[0]),
        "keypoints": int(track.coords.shape[1]),
        "tracks": 1,
        "node_names": list(track.node_names),
        "confidence": {
            "threshold": float(confidence_threshold),
            "finite_values": finite_count,
            "total_values": total,
            "finite_fraction": float(finite_count / total) if total else 0.0,
            "below_threshold": below_threshold,
            "below_threshold_fraction": float(below_threshold / finite_count)
            if finite_count
            else 0.0,
            "mean": None if math.isnan(mean_score) else mean_score,
            "min": None if math.isnan(min_score) else min_score,
            "max": None if math.isnan(max_score) else max_score,
            "worst_keypoint": aggregate["worst_keypoint"],
            "worst_below_threshold_fraction": aggregate["worst_below_threshold_fraction"],
            "longest_low_run_frames": aggregate["longest_low_run_frames"],
            "per_keypoint": per_keypoint,
        },
    }


def _inspect_csv(path: Path, *, confidence_threshold: float) -> dict[str, Any]:
    lines = _read_text_lines(path)
    lowered = "\n".join(lines).lower()
    likely_importers: list[str] = []
    warnings: list[str] = []
    summary: dict[str, Any] = {"preview_rows": len(lines)}
    kind = "csv"

    if "scorer" in lowered and "bodyparts" in lowered and "likelihood" in lowered:
        kind = "pose_predictions"
        likely_importers.extend(["dlc_csv", "lightning_pose_csv"])
        try:
            summary.update(
                _pose_qc(
                    path,
                    software="DLC",
                    file_type="csv",
                    confidence_threshold=confidence_threshold,
                )
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            warnings.append(f"Could not compute pose QC: {exc}")
        return {
            "kind": kind,
            "likely_importers": _unique(likely_importers),
            "summary": summary,
            "warnings": warnings,
        }

    frame = pd.read_csv(path, nrows=25)
    columns = [str(column) for column in frame.columns]
    column_names = _columns_lower(columns)
    numeric_columns = [
        column for column in columns if pd.api.types.is_numeric_dtype(frame[column])
    ]
    summary.update(
        {
            "columns": columns,
            "numeric_columns": numeric_columns,
            "sample_rows": int(len(frame)),
        }
    )
    if column_names & _TIME_COLUMNS:
        summary["timestamp_available"] = True
        is_events_table = False
        if column_names & _EVENT_COLUMNS:
            kind = "events_table"
            likely_importers.append("events_csv")
            is_events_table = True
        non_time_numeric = [
            column
            for column in numeric_columns
            if str(column).strip().lower() not in _TIME_COLUMNS
        ]
        if non_time_numeric and not is_events_table:
            kind = "signals_table" if kind == "csv" else kind
            likely_importers.append("photometry_csv")
    if any(any(token in column.lower() for token in _PHOTOMETRY_HINTS) for column in columns):
        kind = "photometry_table" if kind == "csv" else kind
        likely_importers.append("photometry_csv")
    if "likelihood" in column_names or any(
        column.endswith("_likelihood") for column in column_names
    ):
        warnings.append("Pose-like likelihood columns are present but this is not a DLC header.")

    return {
        "kind": kind,
        "likely_importers": _unique(likely_importers),
        "summary": summary,
        "warnings": warnings,
    }


def _load_json_payload(path: Path) -> object:
    return load_json(path)


def _object_mapping(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}


def _inspect_json(path: Path, *, confidence_threshold: float) -> dict[str, Any]:
    payload = _load_json_payload(path)
    payload_map = _object_mapping(payload)
    keys = sorted(payload_map or {})
    likely_importers: list[str] = []
    warnings: list[str] = []
    kind = "json"
    summary: dict[str, Any] = {"top_level_keys": keys}

    if payload_map is not None and {"meta_info", "instance_info"}.issubset(keys):
        kind = "pose_predictions"
        likely_importers.append("mmpose_json")
        try:
            summary.update(
                _pose_qc(
                    path,
                    software="MMPose",
                    file_type="json",
                    confidence_threshold=confidence_threshold,
                )
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            warnings.append(f"Could not compute pose QC: {exc}")
    elif payload_map is not None and "frames" in payload_map:
        frames = payload_map.get("frames")
        summary["frames"] = len(frames) if isinstance(frames, list) else None
        if isinstance(frames, list) and any(
            isinstance(frame, Mapping) and "pose_landmarks" in frame for frame in frames[:5]
        ):
            kind = "pose_predictions"
            likely_importers.append("mediapipe_json")
            try:
                summary.update(
                    _pose_qc(
                        path,
                        software="MEDIAPIPE",
                        file_type="json",
                        confidence_threshold=confidence_threshold,
                    )
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                warnings.append(f"Could not compute pose QC: {exc}")
    elif payload_map is not None and payload_map.get("format") == "xpkg-project":
        kind = "xpkg_project_descriptor"
    elif payload_map is not None and payload_map.get("format") == "xpkg-packed-project":
        kind = "expkg_manifest"

    return {
        "kind": kind,
        "likely_importers": _unique(likely_importers),
        "summary": summary,
        "warnings": warnings,
    }


def _h5_keys(path: Path) -> list[str]:
    keys: list[str] = []
    with h5py.File(path, "r") as handle:
        handle.visit(keys.append)
    return keys


def _inspect_h5(path: Path, *, confidence_threshold: float) -> dict[str, Any]:
    keys = _h5_keys(path)
    key_set = set(keys)
    likely_importers: list[str] = []
    warnings: list[str] = []
    kind = "hdf5"
    summary: dict[str, Any] = {"keys": keys[:50], "key_count": len(keys)}

    if {"tracks", "node_names"}.issubset(key_set):
        kind = "pose_predictions"
        likely_importers.append("sleap_h5")
        try:
            summary.update(
                _pose_qc(
                    path,
                    software="SLEAP",
                    file_type="h5",
                    confidence_threshold=confidence_threshold,
                )
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            warnings.append(f"Could not compute pose QC: {exc}")
    elif any("DataAcquisition" in key for key in key_set) or any(
        key.endswith("Signal") or key.endswith("Control") for key in key_set
    ):
        kind = "photometry_recording"
        likely_importers.append("doric_photometry")
    elif path.suffix.lower() in {".h5", ".hdf5"}:
        likely_importers.append("dlc_h5")

    return {
        "kind": kind,
        "likely_importers": _unique(likely_importers),
        "summary": summary,
        "warnings": warnings,
    }


_FPS_DRIFT_PCT_WARN = 1.0
_FRAME_GAP_RATIO_WARN = 1.5


def _video_timing_qc(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Demux PTS via PyAV to estimate dropped frames and FPS drift.

    Returns a ``(timing, warnings)`` pair. ``timing`` is ``None`` when PyAV is
    not installed or the stream lacks a usable timing signal.
    """
    try:
        av: Any = importlib.import_module("av")
    except ImportError:
        return None, []

    warnings: list[str] = []
    timing: dict[str, Any] = {}
    av_error: type[BaseException] = getattr(av, "AVError", OSError)
    try:
        container = av.open(str(path))
    except (OSError, av_error) as exc:
        return None, [f"PyAV could not open this video for timing QC: {exc}"]

    try:
        if not container.streams.video:
            return None, []
        stream = container.streams.video[0]
        time_base = stream.time_base
        declared_rate = stream.average_rate or stream.guessed_rate
        declared_fps = float(declared_rate) if declared_rate is not None else None
        timing["declared_fps"] = declared_fps

        pts_seconds: list[float] = []
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            if time_base is None:
                pts_seconds.append(float(packet.pts))
            else:
                pts_seconds.append(float(packet.pts * time_base))
    finally:
        container.close()

    if len(pts_seconds) < 2:
        return None, []

    pts_array = np.asarray(sorted(pts_seconds), dtype=np.float64)
    deltas = np.diff(pts_array)
    if deltas.size == 0:
        return None, []
    median_dt = float(np.median(deltas))
    if median_dt <= 0.0:
        return None, []

    measured_fps = 1.0 / median_dt
    max_gap_s = float(deltas.max())
    min_gap_s = float(deltas.min())
    gap_ratio_threshold = _FRAME_GAP_RATIO_WARN * median_dt
    dropped_suspects = int((deltas > gap_ratio_threshold).sum())
    timing.update(
        {
            "packet_count": int(pts_array.size),
            "measured_fps": float(measured_fps),
            "median_frame_dt_s": median_dt,
            "min_frame_dt_s": min_gap_s,
            "max_frame_dt_s": max_gap_s,
            "dropped_frame_suspects": dropped_suspects,
        }
    )

    if declared_fps is not None and declared_fps > 0.0:
        drift_pct = abs(measured_fps - declared_fps) / declared_fps * 100.0
        timing["fps_drift_pct"] = float(drift_pct)
        if drift_pct >= _FPS_DRIFT_PCT_WARN:
            warnings.append(
                "Measured FPS deviates from declared FPS by "
                f"{drift_pct:.2f}% (declared={declared_fps:.3f}, "
                f"measured={measured_fps:.3f})."
            )
    if dropped_suspects > 0:
        warnings.append(
            f"Detected {dropped_suspects} inter-frame gap(s) longer than "
            f"{_FRAME_GAP_RATIO_WARN:g}x the median dt; possible dropped frames."
        )
    return timing, warnings


def _inspect_video(path: Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        return {
            "kind": "video",
            "likely_importers": [],
            "summary": {},
            "warnings": [f"OpenCV is not available for video metadata: {exc}"],
        }

    capture = cv2.VideoCapture(str(path))
    warnings: list[str] = []
    if not capture.isOpened():
        warnings.append("OpenCV could not open this video.")
        return {
            "kind": "video",
            "likely_importers": [],
            "summary": {},
            "warnings": warnings,
        }
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if frame_count <= 0:
        warnings.append("Frame count is missing or zero.")
    if fps <= 0.0:
        warnings.append("FPS is missing or zero.")
    summary: dict[str, Any] = {
        "frames": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_s": float(frame_count / fps) if frame_count > 0 and fps > 0.0 else None,
    }
    timing, timing_warnings = _video_timing_qc(path)
    if timing is not None:
        summary["timing"] = timing
    warnings.extend(timing_warnings)
    return {
        "kind": "video",
        "likely_importers": [],
        "summary": summary,
        "warnings": warnings,
    }


def _inspect_image(path: Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        return {
            "kind": "image",
            "likely_importers": [],
            "summary": {},
            "warnings": [f"OpenCV is not available for image metadata: {exc}"],
        }
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return {
            "kind": "image",
            "likely_importers": [],
            "summary": {},
            "warnings": ["OpenCV could not read this image."],
        }
    return {
        "kind": "image",
        "likely_importers": [],
        "summary": {
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "channels": int(image.shape[2]) if image.ndim == 3 else 1,
        },
        "warnings": [],
    }


def _inspect_expkg(path: Path) -> dict[str, Any]:
    warnings: list[str] = []
    warning_records: list[InspectionWarning] = []
    summary: dict[str, Any] = {}
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        summary["members"] = len(names)
        metadata_slots, metadata_warnings, metadata_warning_records = (
            _inspect_expkg_metadata_slots(archive, names)
        )
        summary["metadata_slots"] = metadata_slots
        warnings.extend(metadata_warnings)
        warning_records.extend(metadata_warning_records)
        if EXPKG_MANIFEST_FILENAME in names:
            manifest = parse_json(archive.read(EXPKG_MANIFEST_FILENAME))
            manifest_map = _object_mapping(manifest)
            if manifest_map is not None:
                summary["manifest"] = {
                    "format": manifest_map.get("format"),
                    "artifact_schema_version": manifest_map.get("artifact_schema_version"),
                    "media": manifest_map.get("media"),
                }
        else:
            message = f"Missing {EXPKG_MANIFEST_FILENAME}."
            warnings.append(message)
            warning_records.append(
                _warning_record(
                    "packed_project_manifest_missing",
                    message,
                    path=EXPKG_MANIFEST_FILENAME,
                )
            )
    return {
        "kind": "expkg_artifact",
        "likely_importers": [],
        "summary": summary,
        "warnings": warnings,
        "warning_records": warning_records,
    }


def _inspect_expkg_metadata_slots(
    archive: zipfile.ZipFile,
    names: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], list[str], list[InspectionWarning]]:
    name_set = set(names)
    slots: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    warning_records: list[InspectionWarning] = []
    for name, filename, load in _PROJECT_METADATA_ARCHIVE_SLOTS:
        member_path = f"{STORE_DIRNAME}/{PROJECT_METADATA_DIRNAME}/{filename}"
        present = member_path in name_set
        slot: dict[str, Any] = {
            "path": member_path,
            "present": present,
            "valid": None,
        }
        if present:
            try:
                payload = parse_json(archive.read(member_path))
                payload_map = _object_mapping(payload)
                if payload_map is None:
                    raise TypeError("metadata payload must be a JSON object")
                load(payload_map)
            except (KeyError, OSError, TypeError, ValueError) as exc:
                slot["valid"] = False
                slot["error"] = str(exc)
                message = f"Packed project metadata slot {name!r} is invalid: {exc}"
                warnings.append(message)
                warning_records.append(
                    _warning_record(
                        "packed_project_metadata_invalid",
                        message,
                        path=member_path,
                    )
                )
            else:
                slot["valid"] = True
        slots[name] = slot
    return slots, warnings, warning_records


def inspect_path(
    path: str | Path,
    *,
    confidence_threshold: float = _POSE_CONFIDENCE_THRESHOLD,
) -> InspectionReport:
    """Inspect a file or folder and return a structured :class:`InspectionReport`."""

    threshold = float(confidence_threshold)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"confidence_threshold must be in [0, 1], got {threshold}.")
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if resolved.is_dir():
        details = _inspect_directory(resolved)
    else:
        suffix = resolved.suffix.lower()
        if suffix == ".csv":
            details = _inspect_csv(resolved, confidence_threshold=threshold)
        elif suffix == ".json":
            details = _inspect_json(resolved, confidence_threshold=threshold)
        elif suffix in {".h5", ".hdf5"}:
            details = _inspect_h5(resolved, confidence_threshold=threshold)
        elif suffix == ".slp":
            details = {
                "kind": "sleap_package",
                "likely_importers": ["sleap_package"],
                "summary": {},
                "warnings": [],
            }
        elif suffix == ".expkg":
            details = _inspect_expkg(resolved)
        elif suffix in _VIDEO_SUFFIXES:
            details = _inspect_video(resolved)
        elif suffix in _IMAGE_SUFFIXES:
            details = _inspect_image(resolved)
        else:
            details = {
                "kind": "unknown",
                "likely_importers": [],
                "summary": {},
                "warnings": ["No xpkg importer was inferred from this path."],
            }

    importers = cast(Sequence[str], details["likely_importers"])
    summary_value = details["summary"]
    warnings = cast(Sequence[str], details["warnings"])
    summary = dict(summary_value) if isinstance(summary_value, Mapping) else {}
    payload = _path_payload(resolved)
    warning_records = _coerce_warning_records(
        details.get("warning_records"),
        fallback_warnings=tuple(str(item) for item in warnings),
        fallback_path=payload["path"],
    )
    return InspectionReport(
        path=payload["path"],
        name=payload["name"],
        suffix=payload["suffix"],
        exists=payload["exists"],
        is_dir=payload["is_dir"],
        size_bytes=payload["size_bytes"],
        kind=InspectionKind(str(details["kind"])),
        likely_importers=tuple(str(item) for item in importers),
        summary=summary,
        warnings=tuple(str(item) for item in warnings),
        warning_records=warning_records,
    )


__all__ = ["InspectionKind", "InspectionReport", "InspectionWarning", "inspect_path"]
