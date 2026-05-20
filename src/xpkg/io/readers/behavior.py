"""Readers for source-neutral behavior-label outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from xpkg.model import BehaviorFrameLabel, BehaviorInterval, BehaviorLabels

TimeUnit = Literal["s", "sec", "second", "seconds", "ms", "millisecond", "milliseconds"]

KNOWN_BEHAVIOR_SOURCE_TYPES: tuple[str, ...] = (
    "behavior_csv",
    "behavior_events_json",
    "asoid",
    "boris",
    "bsoid",
    "deepethogram",
    "jaaba",
    "keypoint_moseq",
    "simba",
    "vame",
)

_START_TIME_CANDIDATES = (
    "start_s",
    "start_seconds",
    "startTimeSec",
    "onset_s",
    "onset",
    "start",
    "time",
    "timestamp",
)
_END_TIME_CANDIDATES = (
    "end_s",
    "end_seconds",
    "endTimeSec",
    "offset_s",
    "offset",
    "stop",
    "end",
)
_DURATION_CANDIDATES = ("duration_s", "duration_seconds", "durationSec", "duration")
_LABEL_CANDIDATES = (
    "label",
    "behavior",
    "behaviour",
    "behavior_label",
    "behaviour_label",
    "class",
    "class_name",
    "state",
    "motif",
    "syllable",
    "prediction",
    "name",
)
_FRAME_CANDIDATES = ("frame", "frame_index", "frame_idx", "source_frame")
_START_FRAME_CANDIDATES = ("start_frame", "startFrame", "start_frame_index", "onset_frame")
_END_FRAME_CANDIDATES = ("end_frame", "endFrame", "end_frame_index", "offset_frame")
_SCORE_CANDIDATES = ("score", "probability", "prob", "likelihood", "confidence_score")
_CONFIDENCE_CANDIDATES = ("confidence", "confidence_label", "quality")
_SOURCE_ID_CANDIDATES = ("id", "event_id", "behaviorEventId", "source_id", "uuid")


def read_behavior_events_json(
    path: str | Path,
    *,
    source_type: str = "behavior_events_json",
    media_path: str | Path | None = None,
) -> BehaviorLabels:
    """Read a behavior annotation JSON with a top-level ``behaviorEvents`` list."""

    source_path = Path(path)
    payload = _read_json_object(source_path)
    raw_events = payload.get("behaviorEvents")
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, str | bytes):
        raise ValueError("Behavior JSON must contain a behaviorEvents list.")
    raw_metadata = payload.get("metadata")
    source_metadata = _mapping_or_empty(raw_metadata, name="behavior JSON metadata")
    resolved_media_path = _media_path_from_metadata(source_metadata, media_path)
    intervals = tuple(_interval_from_json_event(event) for event in raw_events)
    metadata = _behavior_json_metadata(payload, source_metadata, source_path)
    return BehaviorLabels(
        source_type=source_type,
        intervals=intervals,
        media_path=resolved_media_path,
        annotator=_optional_text(source_metadata.get("annotatorName")),
        metadata=metadata,
    )


def read_behavior_events_csv(
    path: str | Path,
    *,
    source_type: str = "behavior_csv",
    media_path: str | Path | None = None,
    label_column: str | None = None,
    start_column: str | None = None,
    end_column: str | None = None,
    duration_column: str | None = None,
    frame_column: str | None = None,
    start_frame_column: str | None = None,
    end_frame_column: str | None = None,
    score_column: str | None = None,
    confidence_column: str | None = None,
    source_id_column: str | None = None,
    time_unit: TimeUnit = "s",
    max_mb: float | None = None,
) -> BehaviorLabels:
    """Read behavior interval or per-frame labels from a flexible CSV export."""

    source_path = Path(path)
    frame, size_bytes = _read_csv(source_path, max_mb=max_mb)
    if frame.empty:
        raise ValueError(f"Behavior CSV '{source_path}' is empty.")
    columns = _BehaviorColumns.resolve(
        frame,
        label_column=label_column,
        start_column=start_column,
        end_column=end_column,
        duration_column=duration_column,
        frame_column=frame_column,
        start_frame_column=start_frame_column,
        end_frame_column=end_frame_column,
        score_column=score_column,
        confidence_column=confidence_column,
        source_id_column=source_id_column,
    )
    scale = _time_scale(time_unit)
    intervals = _csv_intervals(frame, columns, scale=scale)
    frame_labels = _csv_frame_labels(frame, columns) if not intervals else ()
    return BehaviorLabels(
        source_type=source_type,
        intervals=intervals,
        frame_labels=frame_labels,
        media_path=None if media_path is None else Path(media_path).as_posix(),
        metadata={
            "source": {
                "type": source_type,
                "path": str(source_path),
                "size_bytes": size_bytes,
            }
        },
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Behavior JSON '{path}' is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Behavior JSON '{path}' must contain a JSON object.")
    return payload


def _mapping_or_empty(value: object, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    return {str(key): item for key, item in value.items()}


def _media_path_from_metadata(
    metadata: Mapping[str, Any],
    media_path: str | Path | None,
) -> str | None:
    if media_path is not None:
        return Path(media_path).as_posix()
    video_file = _optional_text(metadata.get("videoFileName"))
    return video_file


def _behavior_json_metadata(
    payload: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    metadata = dict(source_metadata)
    metadata["source"] = {"type": "behavior_events_json", "path": str(source_path)}
    for key, value in (
        ("body_part_count", _optional_len(payload.get("bodyParts"))),
        ("frame_annotation_count", _optional_len(payload.get("frameAnnotations"))),
        ("skeleton_connection_count", _optional_len(payload.get("skeletonConnections"))),
    ):
        if value is not None:
            metadata[key] = value
    return metadata


def _optional_len(value: object) -> int | None:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return len(value)
    return None


def _interval_from_json_event(payload: object) -> BehaviorInterval:
    if not isinstance(payload, Mapping):
        raise ValueError("behaviorEvents entries must be JSON objects.")
    fields = {str(key): value for key, value in payload.items()}
    start_s = _required_float(fields, "startTimeSec")
    end_s = _event_end_seconds(fields, start_s)
    metadata = _json_event_metadata(fields)
    return BehaviorInterval(
        label=_required_text(fields, "label"),
        start_s=start_s,
        end_s=end_s,
        confidence=_optional_text(fields.get("confidence")),
        source_id=_optional_text(fields.get("behaviorEventId")),
        metadata=metadata,
    )


def _event_end_seconds(payload: Mapping[str, Any], start_s: float) -> float:
    if "endTimeSec" in payload:
        return _required_float(payload, "endTimeSec")
    if "durationSec" in payload:
        return start_s + _required_float(payload, "durationSec")
    return start_s


def _json_event_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("notes", "durationSec"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            metadata[key] = value
    return metadata


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = _optional_text(payload.get(key))
    if value is None:
        raise ValueError(f"Behavior event field {key!r} is required.")
    return value


def _required_float(payload: Mapping[str, Any], key: str) -> float:
    if key not in payload:
        raise ValueError(f"Behavior event field {key!r} is required.")
    value = float(payload[key])
    if not np.isfinite(value):
        raise ValueError(f"Behavior event field {key!r} must be finite.")
    return value


def _read_csv(path: Path, *, max_mb: float | None) -> tuple[pd.DataFrame, int]:
    size_bytes = path.stat().st_size
    if max_mb is not None:
        max_bytes = int(float(max_mb) * 1024 * 1024)
        if max_bytes <= 0:
            raise ValueError(f"max_mb must be positive when provided, got {max_mb!r}.")
        if size_bytes > max_bytes:
            raise ValueError(f"Behavior CSV '{path}' exceeds max load size ({max_mb} MB).")
    return pd.read_csv(path), size_bytes


def _column_by_name(frame: pd.DataFrame, name: str) -> str:
    names = {str(column).lower(): str(column) for column in frame.columns}
    key = str(name).lower()
    if key not in names:
        raise ValueError(
            f"Column {name!r} was not found. Available columns: {list(frame.columns)}."
        )
    return names[key]


def _first_matching_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    names = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        match = names.get(candidate.lower())
        if match is not None:
            return match
    return None


def _resolve_column(
    frame: pd.DataFrame,
    explicit: str | None,
    candidates: Sequence[str],
) -> str | None:
    if explicit is not None:
        return _column_by_name(frame, explicit)
    return _first_matching_column(frame, candidates)


def _time_scale(unit: TimeUnit) -> float:
    normalized = unit.lower()
    if normalized in {"s", "sec", "second", "seconds"}:
        return 1.0
    if normalized in {"ms", "millisecond", "milliseconds"}:
        return 0.001
    raise ValueError(f"Unsupported time_unit {unit!r}; expected seconds or milliseconds.")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    missing = pd.isna(value)
    if isinstance(missing, bool | np.bool_):
        return bool(missing)
    return False


def _optional_text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    coerced = float(value)
    if not np.isfinite(coerced):
        raise ValueError(f"Expected finite numeric value, got {coerced!r}.")
    return coerced


def _optional_int(value: Any) -> int | None:
    if _is_missing(value):
        return None
    coerced = int(value)
    if coerced < 0:
        raise ValueError(f"Expected non-negative frame index, got {coerced}.")
    return coerced


class _BehaviorColumns:
    def __init__(
        self,
        *,
        label: str,
        start: str | None,
        end: str | None,
        duration: str | None,
        frame: str | None,
        start_frame: str | None,
        end_frame: str | None,
        score: str | None,
        confidence: str | None,
        source_id: str | None,
    ) -> None:
        self.label = label
        self.start = start
        self.end = end
        self.duration = duration
        self.frame = frame
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.score = score
        self.confidence = confidence
        self.source_id = source_id

    @classmethod
    def resolve(cls, frame: pd.DataFrame, **columns: str | None) -> _BehaviorColumns:
        label = _resolve_column(frame, columns["label_column"], _LABEL_CANDIDATES)
        if label is None:
            raise ValueError("Behavior CSV must include a behavior label column.")
        resolved = cls(
            label=label,
            start=_resolve_column(frame, columns["start_column"], _START_TIME_CANDIDATES),
            end=_resolve_column(frame, columns["end_column"], _END_TIME_CANDIDATES),
            duration=_resolve_column(frame, columns["duration_column"], _DURATION_CANDIDATES),
            frame=_resolve_column(frame, columns["frame_column"], _FRAME_CANDIDATES),
            start_frame=_resolve_column(
                frame,
                columns["start_frame_column"],
                _START_FRAME_CANDIDATES,
            ),
            end_frame=_resolve_column(frame, columns["end_frame_column"], _END_FRAME_CANDIDATES),
            score=_resolve_column(frame, columns["score_column"], _SCORE_CANDIDATES),
            confidence=_resolve_column(
                frame,
                columns["confidence_column"],
                _CONFIDENCE_CANDIDATES,
            ),
            source_id=_resolve_column(frame, columns["source_id_column"], _SOURCE_ID_CANDIDATES),
        )
        if resolved.start is None and resolved.start_frame is None and resolved.frame is None:
            raise ValueError("Behavior CSV must include time or frame index columns.")
        return resolved


def _csv_intervals(
    frame: pd.DataFrame,
    columns: _BehaviorColumns,
    *,
    scale: float,
) -> tuple[BehaviorInterval, ...]:
    if columns.start is None and columns.start_frame is None:
        return ()
    rows: list[BehaviorInterval] = []
    for index in range(len(frame)):
        start_s = _optional_scaled_float(frame, columns.start, index, scale)
        end_s = _csv_end_seconds(frame, columns, index, start_s, scale)
        start_frame = _optional_int_from_frame(frame, columns.start_frame, index)
        end_frame = _csv_end_frame(frame, columns, index, start_frame)
        rows.append(_csv_interval(frame, columns, index, start_s, end_s, start_frame, end_frame))
    return tuple(rows)


def _csv_interval(
    frame: pd.DataFrame,
    columns: _BehaviorColumns,
    index: int,
    start_s: float | None,
    end_s: float | None,
    start_frame: int | None,
    end_frame: int | None,
) -> BehaviorInterval:
    return BehaviorInterval(
        label=_required_row_text(frame, columns.label, index),
        start_s=start_s,
        end_s=end_s,
        start_frame=start_frame,
        end_frame=end_frame,
        score=_optional_float_from_frame(frame, columns.score, index),
        confidence=_optional_text_from_frame(frame, columns.confidence, index),
        source_id=_optional_text_from_frame(frame, columns.source_id, index),
        metadata={"row_index": index},
    )


def _csv_frame_labels(
    frame: pd.DataFrame,
    columns: _BehaviorColumns,
) -> tuple[BehaviorFrameLabel, ...]:
    frame_column = columns.frame or columns.start_frame
    if frame_column is None:
        return ()
    return tuple(
        BehaviorFrameLabel(
            frame_index=_required_row_int(frame, frame_column, index),
            label=_required_row_text(frame, columns.label, index),
            score=_optional_float_from_frame(frame, columns.score, index),
            confidence=_optional_text_from_frame(frame, columns.confidence, index),
            source_id=_optional_text_from_frame(frame, columns.source_id, index),
            metadata={"row_index": index},
        )
        for index in range(len(frame))
    )


def _optional_scaled_float(
    frame: pd.DataFrame,
    column: str | None,
    index: int,
    scale: float,
) -> float | None:
    value = _optional_float_from_frame(frame, column, index)
    if value is None:
        return None
    return value * scale


def _csv_end_seconds(
    frame: pd.DataFrame,
    columns: _BehaviorColumns,
    index: int,
    start_s: float | None,
    scale: float,
) -> float | None:
    end_s = _optional_scaled_float(frame, columns.end, index, scale)
    if end_s is not None:
        return end_s
    duration_s = _optional_scaled_float(frame, columns.duration, index, scale)
    if start_s is not None and duration_s is not None:
        return start_s + duration_s
    return start_s


def _csv_end_frame(
    frame: pd.DataFrame,
    columns: _BehaviorColumns,
    index: int,
    start_frame: int | None,
) -> int | None:
    end_frame = _optional_int_from_frame(frame, columns.end_frame, index)
    if end_frame is not None:
        return end_frame
    return start_frame


def _optional_float_from_frame(frame: pd.DataFrame, column: str | None, index: int) -> float | None:
    if column is None:
        return None
    return _optional_float(frame[column].iloc[index])


def _optional_int_from_frame(frame: pd.DataFrame, column: str | None, index: int) -> int | None:
    if column is None:
        return None
    return _optional_int(frame[column].iloc[index])


def _optional_text_from_frame(frame: pd.DataFrame, column: str | None, index: int) -> str | None:
    if column is None:
        return None
    return _optional_text(frame[column].iloc[index])


def _required_row_text(frame: pd.DataFrame, column: str, index: int) -> str:
    value = _optional_text(frame[column].iloc[index])
    if value is None:
        raise ValueError(f"Behavior label column {column!r} contains an empty value.")
    return value


def _required_row_int(frame: pd.DataFrame, column: str, index: int) -> int:
    value = _optional_int(frame[column].iloc[index])
    if value is None:
        raise ValueError(f"Behavior frame column {column!r} contains an empty value.")
    return value


__all__ = [
    "KNOWN_BEHAVIOR_SOURCE_TYPES",
    "read_behavior_events_csv",
    "read_behavior_events_json",
]
