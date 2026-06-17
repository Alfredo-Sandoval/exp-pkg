"""Readers for source-neutral behavior-label outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from xpkg.io.readers._columns import (
    column_by_name,
    first_matching_column,
    resolve_column,
)
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
_BSOID_LABEL_CANDIDATES = (
    # B-SOiD writes the label column literally as "B-SOiD labels" in every
    # standard export (per-frame, SVM-classifier, and run-length tables); it
    # must come first so real exports resolve before the generic fallbacks.
    "B-SOiD labels",
    "B-SOiD label",
    "behavior",
    "label",
    "labels",
    "prediction",
    "class",
    "cluster",
    "cluster_id",
    "bsoid_label",
    "bsoid_class",
)
# B-SOiD run-length / bout tables express bout onsets in frames, not seconds.
_BSOID_START_FRAME_CANDIDATES = ("Start time (frames)", "start time (frames)")
_BSOID_RUN_LENGTH_CANDIDATES = ("Run lengths", "run lengths", "Run length")
_BORIS_LABEL_CANDIDATES = ("Behavior", "behavior")
_BORIS_START_TIME_CANDIDATES = ("Start (seconds)", "Start (s)", "start")
_BORIS_END_TIME_CANDIDATES = ("Stop (seconds)", "Stop (s)", "stop")
_BORIS_DURATION_CANDIDATES = ("Duration (seconds)", "Duration (s)", "duration")
# Modern BORIS aggregated exports use "Media file name"; legacy/v7 exports and
# the aggregated golden fixtures use the singular "Media file"; the tabular
# export uses "Media file path".
_BORIS_MEDIA_CANDIDATES = (
    "Media file name",
    "Media file",
    "Media file path",
    "media_file_name",
)
# BORIS tabular exports encode events as START/STOP/POINT rows in a Status
# column with a single Time column, rather than Start/Stop pairs. The Time,
# Status and Behavior columns are resolved by their exact (case-insensitive)
# names; Subject and Behavioral category are optional.
_BORIS_SUBJECT_CANDIDATES = ("Subject", "subject")
_BORIS_CATEGORY_CANDIDATES = ("Behavioral category", "behavioral category")
_BORIS_SOURCE_ID_CANDIDATES = ("Observation id", "observation")
_SIMBA_FRAME_CANDIDATES = (
    "frame",
    "Frame",
    "frame_index",
    "Frame_index",
    "frame_idx",
    "Frame_idx",
    "frame_number",
    "Frame_number",
)
_SIMBA_TIME_CANDIDATES = (
    "time",
    "Time",
    "timestamp",
    "Timestamp",
    "time_s",
    "Time_s",
    "seconds",
    "Seconds",
)
_SIMBA_PROBABILITY_PREFIX = "Probability_"
_SIMBA_CONFIDENCE_PREFIXES = ("Confidence_", "confidence_")
_KEYPOINT_MOSEQ_FRAME_CANDIDATES = (
    "frame_index",
    "frame",
    "frame_idx",
    "Frame",
)
_KEYPOINT_MOSEQ_TIME_CANDIDATES = (
    "time_s",
    "time",
    "timestamp",
    "seconds",
)
_KEYPOINT_MOSEQ_LABEL_CANDIDATES = (
    "syllable",
    "syllable_id",
    "motif",
    "motif_id",
    "state",
    "z",
)
_KEYPOINT_MOSEQ_RECORDING_CANDIDATES = (
    "recording_name",
    "recording",
    "name",
    "video_name",
    "video",
)
_KEYPOINT_MOSEQ_SCORE_CANDIDATES = (
    "score",
    "probability",
    "prob",
    "likelihood",
    "posterior_probability",
    "syllable_probability",
    "motif_probability",
    "confidence_score",
)
_KEYPOINT_MOSEQ_CONFIDENCE_CANDIDATES = (
    "confidence",
    "confidence_label",
    "quality",
)
_KEYPOINT_MOSEQ_UNCERTAINTY_CANDIDATES = (
    "uncertainty",
    "syllable_uncertainty",
    "motif_uncertainty",
    "entropy",
    "marginal_entropy",
)


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


def read_boris_csv(
    path: str | Path,
    *,
    media_path: str | Path | None = None,
    max_mb: float | None = None,
) -> BehaviorLabels:
    """Read BORIS event CSV exports as behavior intervals.

    BORIS produces two incompatible event layouts. The default *tabular* export
    ("Export events > Tabular events") writes a metadata preamble above the data
    header and encodes each event as a START/STOP/POINT row in a ``Status``
    column with a single ``Time`` column; START and STOP rows for the same
    subject+behavior are paired into intervals, and POINT rows become
    instantaneous events. The *aggregated* export writes one row per interval
    with ``Start (s)``/``Stop (s)`` columns and no preamble. Both are supported.
    """

    source_path = Path(path)
    size_bytes = _csv_size_bytes(source_path, max_mb=max_mb)
    tabular_header = _detect_boris_tabular_header(source_path)
    if tabular_header is not None:
        frame = pd.read_csv(source_path, skiprows=tabular_header)
        export_format = "tabular_events_csv"
        intervals = _boris_tabular_intervals(frame)
    else:
        frame = pd.read_csv(source_path)
        export_format = "aggregated_events_csv"
        columns = _BehaviorColumns.resolve(
            frame,
            label_column=first_matching_column(frame, _BORIS_LABEL_CANDIDATES),
            start_column=first_matching_column(frame, _BORIS_START_TIME_CANDIDATES),
            end_column=first_matching_column(frame, _BORIS_END_TIME_CANDIDATES),
            duration_column=first_matching_column(frame, _BORIS_DURATION_CANDIDATES),
            frame_column=None,
            start_frame_column=None,
            end_frame_column=None,
            score_column=None,
            confidence_column=None,
            source_id_column=first_matching_column(frame, _BORIS_SOURCE_ID_CANDIDATES),
        )
        intervals = _csv_intervals(frame, columns, scale=1.0)
    if frame.empty:
        raise ValueError(f"BORIS CSV '{source_path}' is empty.")
    if not intervals:
        raise ValueError("BORIS CSV must include interval start columns.")
    return BehaviorLabels(
        source_type="boris",
        intervals=intervals,
        media_path=_boris_media_path(frame, media_path),
        metadata={
            "source": {
                "type": "boris",
                "path": str(source_path),
                "size_bytes": size_bytes,
                "format": export_format,
            }
        },
    )


def _detect_boris_tabular_header(path: Path) -> int | None:
    """Return the 0-based header-row index of a BORIS *tabular* export.

    The tabular export precedes its data header with a variable metadata
    preamble; the header row is the first line carrying both a ``Behavior`` and
    a ``Status`` field. Aggregated exports (header on row 0, with Start/Stop
    columns and no ``Status``) return ``None`` so the caller parses them as a
    plain one-row-per-interval table.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, line in enumerate(handle):
            if index > 500:
                break
            cells = {cell.strip().strip('"').lower() for cell in line.rstrip("\r\n").split(",")}
            if "behavior" in cells and "status" in cells:
                return index
    return None


def _boris_tabular_intervals(frame: pd.DataFrame) -> tuple[BehaviorInterval, ...]:
    """Pair BORIS tabular START/STOP rows into intervals; POINT rows stay points.

    STATE behaviors emit a START row then a STOP row for the same
    subject+behavior; they are paired in arrival order (a state is either on or
    off, so at most one is open at a time). POINT behaviors emit a single row
    that becomes a zero-duration interval.
    """
    time_column = column_by_name(frame, "Time")
    status_column = column_by_name(frame, "Status")
    behavior_column = column_by_name(frame, "Behavior")
    subject_column = first_matching_column(frame, _BORIS_SUBJECT_CANDIDATES)
    category_column = first_matching_column(frame, _BORIS_CATEGORY_CANDIDATES)

    open_starts: dict[tuple[str, str], list[tuple[float, dict[str, Any]]]] = {}
    intervals: list[BehaviorInterval] = []
    for index in range(len(frame)):
        status = _optional_text_from_frame(frame, status_column, index)
        behavior = _optional_text_from_frame(frame, behavior_column, index)
        time_s = _optional_float_from_frame(frame, time_column, index)
        if status is None or behavior is None or time_s is None:
            continue
        subject = _optional_text_from_frame(frame, subject_column, index)
        category = _optional_text_from_frame(frame, category_column, index)
        metadata: dict[str, Any] = {"source_format": "tabular_events_csv"}
        if subject is not None:
            metadata["subject"] = subject
        if category is not None:
            metadata["behavioral_category"] = category
        key = (subject or "", behavior)

        normalized = status.strip().upper()
        if normalized == "POINT":
            metadata["behavior_type"] = "POINT"
            intervals.append(
                BehaviorInterval(
                    label=behavior,
                    start_s=time_s,
                    end_s=time_s,
                    source_id=subject,
                    metadata=metadata,
                )
            )
        elif normalized == "START":
            metadata["behavior_type"] = "STATE"
            open_starts.setdefault(key, []).append((time_s, metadata))
        elif normalized == "STOP":
            pending = open_starts.get(key)
            if not pending:
                # STOP without a matching START — skip rather than fabricate.
                continue
            start_s, start_metadata = pending.pop(0)
            intervals.append(
                BehaviorInterval(
                    label=behavior,
                    start_s=start_s,
                    end_s=time_s,
                    source_id=subject,
                    metadata=start_metadata,
                )
            )
    intervals.sort(key=lambda interval: (interval.start_s or 0.0, interval.label))
    return tuple(intervals)


def read_bsoid_csv(
    path: str | Path,
    *,
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
    """Read B-SOiD CSV outputs as imported behavior labels.

    B-SOiD exports are commonly shared as framewise cluster or behavior-label
    CSVs, or as derived bout tables. This reader reuses the flexible behavior
    CSV parser while preserving the source type as imported B-SOiD output.
    """

    source_path = Path(path)
    frame, size_bytes = _read_csv(source_path, max_mb=max_mb)
    if frame.empty:
        raise ValueError(f"B-SOiD CSV '{source_path}' is empty.")
    resolved_label_column = label_column or first_matching_column(frame, _BSOID_LABEL_CANDIDATES)
    if resolved_label_column is None:
        raise ValueError(
            "B-SOiD CSV must include a label column (B-SOiD writes 'B-SOiD labels'). "
            "The multi-row SVM-classifier header variant is not yet supported; "
            "pass label_column= explicitly for nonstandard exports."
        )

    # B-SOiD's run-length bout table has its own frame columns: "Start time
    # (frames)" + "Run lengths". Detect them only by those literal names.
    runlen_start_col = first_matching_column(frame, _BSOID_START_FRAME_CANDIDATES)
    run_length_col = first_matching_column(frame, _BSOID_RUN_LENGTH_CANDIDATES)
    if runlen_start_col is not None and run_length_col is not None:
        intervals = _bsoid_runlen_intervals(
            frame, resolved_label_column, runlen_start_col, run_length_col
        )
        if not intervals:
            raise ValueError("B-SOiD run-length CSV contained no bouts.")
        return BehaviorLabels(
            source_type="bsoid",
            intervals=intervals,
            media_path=None if media_path is None else Path(media_path).as_posix(),
            metadata=_bsoid_metadata(
                source_path, size_bytes, "bsoid_runlen_csv", resolved_label_column
            ),
        )

    # Any explicit time/frame column (generic bout or framewise tables) defers
    # to the flexible parser, which resolves start/end/frame columns itself.
    has_explicit_columns = (
        start_column is not None
        or frame_column is not None
        or start_frame_column is not None
        or end_frame_column is not None
        or first_matching_column(frame, _START_TIME_CANDIDATES) is not None
        or first_matching_column(frame, _FRAME_CANDIDATES) is not None
        or first_matching_column(frame, _START_FRAME_CANDIDATES) is not None
    )
    if has_explicit_columns:
        labels = read_behavior_events_csv(
            source_path,
            source_type="bsoid",
            media_path=media_path,
            label_column=resolved_label_column,
            start_column=start_column,
            end_column=end_column,
            duration_column=duration_column,
            frame_column=frame_column,
            start_frame_column=start_frame_column,
            end_frame_column=end_frame_column,
            score_column=score_column,
            confidence_column=confidence_column,
            source_id_column=source_id_column,
            time_unit=time_unit,
            max_mb=max_mb,
        )
        metadata = dict(labels.metadata)
        source = dict(metadata["source"])
        source["format"] = "bsoid_csv"
        source["label_column"] = resolved_label_column
        metadata["source"] = source
        return replace(labels, metadata=metadata)

    # Per-frame label export: one cluster label per frame, indexed by row order
    # (B-SOiD's leading index column is the frame number written via index=True).
    frame_labels = _bsoid_framewise_labels(frame, resolved_label_column)
    if not frame_labels:
        raise ValueError("B-SOiD CSV contained no per-frame labels.")
    return BehaviorLabels(
        source_type="bsoid",
        frame_labels=frame_labels,
        media_path=None if media_path is None else Path(media_path).as_posix(),
        metadata=_bsoid_metadata(
            source_path, size_bytes, "bsoid_labels_csv", resolved_label_column
        ),
    )


def _bsoid_metadata(
    source_path: Path, size_bytes: int, export_format: str, label_column: str
) -> dict[str, Any]:
    return {
        "source": {
            "type": "bsoid",
            "path": str(source_path),
            "size_bytes": size_bytes,
            "format": export_format,
            "label_column": label_column,
        }
    }


def _cluster_label_text(value: Any) -> str:
    """Render a B-SOiD cluster id as a clean string ("4", not "4.0")."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _bsoid_framewise_labels(
    frame: pd.DataFrame, label_column: str
) -> tuple[BehaviorFrameLabel, ...]:
    labels: list[BehaviorFrameLabel] = []
    for index in range(len(frame)):
        value = frame[label_column].iloc[index]
        if _is_missing(value):
            continue
        labels.append(BehaviorFrameLabel(frame_index=index, label=_cluster_label_text(value)))
    return tuple(labels)


def _bsoid_runlen_intervals(
    frame: pd.DataFrame,
    label_column: str,
    start_frame_column: str,
    run_length_column: str,
) -> tuple[BehaviorInterval, ...]:
    intervals: list[BehaviorInterval] = []
    for index in range(len(frame)):
        value = frame[label_column].iloc[index]
        start = _optional_int_from_frame(frame, start_frame_column, index)
        run = _optional_int_from_frame(frame, run_length_column, index)
        if _is_missing(value) or start is None:
            continue
        end = start + run - 1 if run else start
        intervals.append(
            BehaviorInterval(
                label=_cluster_label_text(value),
                start_frame=start,
                end_frame=max(end, start),
            )
        )
    return tuple(intervals)


def read_simba_csv(
    path: str | Path,
    *,
    media_path: str | Path | None = None,
    frame_column: str | None = None,
    time_column: str | None = None,
    behavior_columns: Sequence[str] | None = None,
    time_unit: TimeUnit = "s",
    probability_threshold: float | None = None,
    max_mb: float | None = None,
) -> BehaviorLabels:
    """Read SimBA framewise classifier CSV outputs as external behavior labels.

    SimBA machine-result and validation CSVs commonly include per-frame columns
    named ``Probability_<classifier>`` plus optional binary classifier columns.
    This reader preserves those outputs as imported frame labels and row
    metadata; it does not threshold probabilities unless
    ``probability_threshold`` is explicitly provided.
    """

    source_path = Path(path)
    frame, size_bytes = _read_csv(source_path, max_mb=max_mb)
    if frame.empty:
        raise ValueError(f"SimBA CSV '{source_path}' is empty.")
    if probability_threshold is not None:
        threshold = float(probability_threshold)
        if not np.isfinite(threshold):
            raise ValueError("probability_threshold must be finite when provided.")
    else:
        threshold = None
    simba_columns = _SimbaColumns.resolve(
        frame,
        frame_column=frame_column,
        time_column=time_column,
        behavior_columns=behavior_columns,
    )
    frame_labels = _simba_frame_labels(
        frame,
        simba_columns,
        time_scale=_time_scale(time_unit),
        probability_threshold=threshold,
    )
    if not frame_labels:
        raise ValueError("SimBA CSV did not contain any behavior labels to import.")
    return BehaviorLabels(
        source_type="simba",
        frame_labels=frame_labels,
        media_path=None if media_path is None else Path(media_path).as_posix(),
        metadata={
            "source": {
                "type": "simba",
                "path": str(source_path),
                "size_bytes": size_bytes,
                "format": "framewise_classifier_csv",
                "frame_column": simba_columns.frame,
                "time_column": simba_columns.time,
                "time_unit": time_unit,
                "classifier_labels": [item.label for item in simba_columns.behaviors],
                "behavior_columns": [
                    item.behavior for item in simba_columns.behaviors if item.behavior is not None
                ],
                "probability_columns": [
                    item.probability
                    for item in simba_columns.behaviors
                    if item.probability is not None
                ],
            }
        },
    )


def read_keypoint_moseq_syllables_csv(
    path: str | Path,
    *,
    media_path: str | Path | None = None,
    frame_column: str | None = None,
    time_column: str | None = None,
    syllable_column: str | None = None,
    recording_column: str | None = None,
    score_column: str | None = None,
    confidence_column: str | None = None,
    uncertainty_columns: Sequence[str] | None = None,
    recording_name: str | None = None,
    time_unit: TimeUnit = "s",
    max_mb: float | None = None,
) -> BehaviorLabels:
    """Read Keypoint-MoSeq syllable CSV outputs as imported frame labels.

    ``keypoint_moseq.io.save_results_as_csv`` writes one row per frame with a
    ``syllable`` column and no explicit frame column, while analysis dataframes
    can include ``frame_index`` and recording ``name`` columns. This reader
    keeps the syllable or motif assignment as a frame label and preserves the
    remaining Keypoint-MoSeq columns as row metadata.
    """

    source_path = Path(path)
    frame, size_bytes = _read_csv(source_path, max_mb=max_mb)
    if frame.empty:
        raise ValueError(f"Keypoint-MoSeq syllables CSV '{source_path}' is empty.")
    columns = _KeypointMoseqColumns.resolve(
        frame,
        frame_column=frame_column,
        time_column=time_column,
        syllable_column=syllable_column,
        recording_column=recording_column,
        score_column=score_column,
        confidence_column=confidence_column,
        uncertainty_columns=uncertainty_columns,
    )
    resolved_recording = _keypoint_moseq_recording_name(
        frame,
        columns.recording,
        explicit=recording_name,
        fallback=source_path.stem,
    )
    frame_labels = _keypoint_moseq_frame_labels(
        frame,
        columns,
        recording_name=resolved_recording,
        time_scale=_time_scale(time_unit),
    )
    if not frame_labels:
        raise ValueError("Keypoint-MoSeq syllables CSV did not contain labels to import.")
    return BehaviorLabels(
        source_type="keypoint_moseq",
        frame_labels=frame_labels,
        media_path=None if media_path is None else Path(media_path).as_posix(),
        metadata={
            "source": {
                "type": "keypoint_moseq",
                "path": str(source_path),
                "size_bytes": size_bytes,
                "format": "syllable_csv",
                "frame_column": columns.frame,
                "frame_index_source": "column" if columns.frame is not None else "row_index",
                "time_column": columns.time,
                "time_unit": time_unit,
                "syllable_column": columns.label,
                "recording_column": columns.recording,
                "recording_name": resolved_recording,
                "score_column": columns.score,
                "confidence_column": columns.confidence,
                "uncertainty_columns": list(columns.uncertainty),
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


def _boris_media_path(frame: pd.DataFrame, media_path: str | Path | None) -> str | None:
    if media_path is not None:
        return Path(media_path).as_posix()
    media_column = first_matching_column(frame, _BORIS_MEDIA_CANDIDATES)
    if media_column is None or frame.empty:
        return None
    return _optional_text(frame[media_column].iloc[0])


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


def _csv_size_bytes(path: Path, *, max_mb: float | None) -> int:
    size_bytes = path.stat().st_size
    if max_mb is not None:
        max_bytes = int(float(max_mb) * 1024 * 1024)
        if max_bytes <= 0:
            raise ValueError(f"max_mb must be positive when provided, got {max_mb!r}.")
        if size_bytes > max_bytes:
            raise ValueError(f"Behavior CSV '{path}' exceeds max load size ({max_mb} MB).")
    return size_bytes


def _read_csv(path: Path, *, max_mb: float | None) -> tuple[pd.DataFrame, int]:
    size_bytes = _csv_size_bytes(path, max_mb=max_mb)
    return pd.read_csv(path), size_bytes


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


class _SimbaBehaviorColumns:
    def __init__(
        self,
        *,
        label: str,
        behavior: str | None,
        probability: str | None,
        confidence: str | None,
    ) -> None:
        self.label = label
        self.behavior = behavior
        self.probability = probability
        self.confidence = confidence


class _SimbaColumns:
    def __init__(
        self,
        *,
        frame: str | None,
        time: str | None,
        behaviors: tuple[_SimbaBehaviorColumns, ...],
    ) -> None:
        self.frame = frame
        self.time = time
        self.behaviors = behaviors

    @classmethod
    def resolve(
        cls,
        frame: pd.DataFrame,
        *,
        frame_column: str | None,
        time_column: str | None,
        behavior_columns: Sequence[str] | None,
    ) -> _SimbaColumns:
        resolved_frame = resolve_column(frame, frame_column, _SIMBA_FRAME_CANDIDATES)
        resolved_time = resolve_column(frame, time_column, _SIMBA_TIME_CANDIDATES)
        behaviors = _resolve_simba_behavior_columns(frame, behavior_columns)
        if not behaviors:
            raise ValueError(
                "SimBA CSV must include Probability_<classifier> columns or explicit "
                "behavior_columns."
            )
        return cls(frame=resolved_frame, time=resolved_time, behaviors=behaviors)


def _resolve_simba_behavior_columns(
    frame: pd.DataFrame,
    behavior_columns: Sequence[str] | None,
) -> tuple[_SimbaBehaviorColumns, ...]:
    if behavior_columns is not None:
        return tuple(
            _simba_behavior_from_column(frame, column_by_name(frame, column))
            for column in behavior_columns
        )
    probability_columns = _simba_probability_columns(frame)
    return tuple(_simba_behavior_from_probability(frame, column) for column in probability_columns)


def _simba_probability_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    columns: list[str] = []
    for column in frame.columns:
        column_name = str(column)
        if column_name.lower().startswith(_SIMBA_PROBABILITY_PREFIX.lower()):
            columns.append(column_name)
    return tuple(columns)


def _simba_behavior_from_column(frame: pd.DataFrame, column: str) -> _SimbaBehaviorColumns:
    label = _simba_label_from_behavior_column(column)
    return _SimbaBehaviorColumns(
        label=label,
        behavior=column,
        probability=_case_insensitive_column(frame, f"{_SIMBA_PROBABILITY_PREFIX}{label}"),
        confidence=_simba_confidence_column(frame, label),
    )


def _simba_behavior_from_probability(frame: pd.DataFrame, column: str) -> _SimbaBehaviorColumns:
    label = _simba_label_from_probability_column(column)
    behavior_column = _case_insensitive_column(frame, label)
    return _SimbaBehaviorColumns(
        label=label,
        behavior=behavior_column,
        probability=column,
        confidence=_simba_confidence_column(frame, label),
    )


def _simba_label_from_probability_column(column: str) -> str:
    if column.lower().startswith(_SIMBA_PROBABILITY_PREFIX.lower()):
        label = column[len(_SIMBA_PROBABILITY_PREFIX) :]
        if label:
            return label
    return column


def _simba_label_from_behavior_column(column: str) -> str:
    if column.lower().startswith(_SIMBA_PROBABILITY_PREFIX.lower()):
        return _simba_label_from_probability_column(column)
    return column


def _simba_confidence_column(frame: pd.DataFrame, label: str) -> str | None:
    for prefix in _SIMBA_CONFIDENCE_PREFIXES:
        match = _case_insensitive_column(frame, f"{prefix}{label}")
        if match is not None:
            return match
    for suffix in ("_confidence", "_Confidence"):
        match = _case_insensitive_column(frame, f"{label}{suffix}")
        if match is not None:
            return match
    return None


def _case_insensitive_column(frame: pd.DataFrame, name: str) -> str | None:
    names = {str(column).lower(): str(column) for column in frame.columns}
    return names.get(name.lower())


def _simba_frame_labels(
    frame: pd.DataFrame,
    columns: _SimbaColumns,
    *,
    time_scale: float,
    probability_threshold: float | None,
) -> tuple[BehaviorFrameLabel, ...]:
    labels: list[BehaviorFrameLabel] = []
    for index in range(len(frame)):
        frame_index = _simba_frame_index(frame, columns.frame, index)
        for behavior in columns.behaviors:
            if not _simba_row_has_label(
                frame,
                behavior,
                index,
                probability_threshold=probability_threshold,
            ):
                continue
            labels.append(
                BehaviorFrameLabel(
                    frame_index=frame_index,
                    label=behavior.label,
                    score=_optional_float_from_frame(frame, behavior.probability, index),
                    confidence=_optional_text_from_frame(frame, behavior.confidence, index),
                    metadata=_simba_row_metadata(
                        frame,
                        columns,
                        behavior,
                        index,
                        time_scale=time_scale,
                    ),
                )
            )
    return tuple(labels)


def _simba_frame_index(frame: pd.DataFrame, column: str | None, index: int) -> int:
    if column is None:
        return index
    return _required_row_int(frame, column, index)


def _simba_row_has_label(
    frame: pd.DataFrame,
    behavior: _SimbaBehaviorColumns,
    index: int,
    *,
    probability_threshold: float | None,
) -> bool:
    if behavior.behavior is not None:
        return _simba_behavior_present(frame[behavior.behavior].iloc[index])
    if probability_threshold is None:
        return _optional_float_from_frame(frame, behavior.probability, index) is not None
    score = _optional_float_from_frame(frame, behavior.probability, index)
    return score is not None and score >= probability_threshold


def _simba_behavior_present(value: Any) -> bool:
    if _is_missing(value):
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "present"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "absent"}:
            return False
    try:
        coerced = _optional_float(value)
        return coerced is not None and coerced > 0
    except (TypeError, ValueError):
        return bool(value)


def _simba_row_metadata(
    frame: pd.DataFrame,
    columns: _SimbaColumns,
    behavior: _SimbaBehaviorColumns,
    index: int,
    *,
    time_scale: float,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"row_index": index}
    time_s = _optional_float_from_frame(frame, columns.time, index)
    if time_s is not None:
        metadata["time_s"] = time_s * time_scale
    if columns.frame is None:
        metadata["frame_index_source"] = "row_index"
    metadata["source_label"] = behavior.label
    for key, value in (
        ("source_behavior_column", behavior.behavior),
        ("source_probability_column", behavior.probability),
        ("source_confidence_column", behavior.confidence),
    ):
        if value is not None:
            metadata[key] = value

    used_columns = {
        column
        for column in (
            columns.frame,
            columns.time,
            behavior.behavior,
            behavior.probability,
            behavior.confidence,
        )
        if column is not None
    }
    for column in frame.columns:
        column_name = str(column)
        if column_name in used_columns:
            continue
        value = frame[column].iloc[index]
        if _is_missing(value):
            continue
        metadata[column_name] = _json_scalar(value)
    return metadata


class _KeypointMoseqColumns:
    def __init__(
        self,
        *,
        frame: str | None,
        time: str | None,
        label: str,
        recording: str | None,
        score: str | None,
        confidence: str | None,
        uncertainty: tuple[str, ...],
    ) -> None:
        self.frame = frame
        self.time = time
        self.label = label
        self.recording = recording
        self.score = score
        self.confidence = confidence
        self.uncertainty = uncertainty

    @classmethod
    def resolve(
        cls,
        frame: pd.DataFrame,
        *,
        frame_column: str | None,
        time_column: str | None,
        syllable_column: str | None,
        recording_column: str | None,
        score_column: str | None,
        confidence_column: str | None,
        uncertainty_columns: Sequence[str] | None,
    ) -> _KeypointMoseqColumns:
        label = resolve_column(frame, syllable_column, _KEYPOINT_MOSEQ_LABEL_CANDIDATES)
        if label is None:
            raise ValueError(
                "Keypoint-MoSeq syllables CSV must include a syllable or motif column."
            )
        return cls(
            frame=resolve_column(frame, frame_column, _KEYPOINT_MOSEQ_FRAME_CANDIDATES),
            time=resolve_column(frame, time_column, _KEYPOINT_MOSEQ_TIME_CANDIDATES),
            label=label,
            recording=resolve_column(
                frame,
                recording_column,
                _KEYPOINT_MOSEQ_RECORDING_CANDIDATES,
            ),
            score=resolve_column(frame, score_column, _KEYPOINT_MOSEQ_SCORE_CANDIDATES),
            confidence=resolve_column(
                frame,
                confidence_column,
                _KEYPOINT_MOSEQ_CONFIDENCE_CANDIDATES,
            ),
            uncertainty=_resolve_keypoint_moseq_uncertainty_columns(
                frame,
                uncertainty_columns,
            ),
        )

    def used_names(self) -> set[str]:
        return {
            str(column)
            for column in (
                self.frame,
                self.time,
                self.label,
                self.recording,
                self.score,
                self.confidence,
            )
            if column is not None
        }


def _resolve_keypoint_moseq_uncertainty_columns(
    frame: pd.DataFrame,
    uncertainty_columns: Sequence[str] | None,
) -> tuple[str, ...]:
    if uncertainty_columns is not None:
        return tuple(column_by_name(frame, column) for column in uncertainty_columns)
    return tuple(
        column
        for column in (
            first_matching_column(frame, (candidate,))
            for candidate in _KEYPOINT_MOSEQ_UNCERTAINTY_CANDIDATES
        )
        if column is not None
    )


def _keypoint_moseq_recording_name(
    frame: pd.DataFrame,
    column: str | None,
    *,
    explicit: str | None,
    fallback: str,
) -> str | None:
    if explicit is not None:
        return _optional_text(explicit)
    if column is None:
        return fallback
    values = {
        text
        for text in (_optional_text(value) for value in frame[column].to_numpy())
        if text is not None
    }
    if len(values) == 1:
        return next(iter(values))
    return None


def _keypoint_moseq_frame_labels(
    frame: pd.DataFrame,
    columns: _KeypointMoseqColumns,
    *,
    recording_name: str | None,
    time_scale: float,
) -> tuple[BehaviorFrameLabel, ...]:
    labels: list[BehaviorFrameLabel] = []
    for index in range(len(frame)):
        labels.append(
            BehaviorFrameLabel(
                frame_index=_keypoint_moseq_frame_index(frame, columns.frame, index),
                label=_keypoint_moseq_label(frame[columns.label].iloc[index], columns.label),
                score=_optional_float_from_frame(frame, columns.score, index),
                confidence=_optional_text_from_frame(frame, columns.confidence, index),
                metadata=_keypoint_moseq_row_metadata(
                    frame,
                    columns,
                    index,
                    recording_name=recording_name,
                    time_scale=time_scale,
                ),
            )
        )
    return tuple(labels)


def _keypoint_moseq_frame_index(frame: pd.DataFrame, column: str | None, index: int) -> int:
    if column is None:
        return index
    return _required_row_int(frame, column, index)


def _keypoint_moseq_label(value: Any, column: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"Keypoint-MoSeq syllable column {column!r} contains an empty value.")
    prefix = "motif" if "motif" in column.lower() else "syllable"
    if text.lower().startswith(("syllable", "motif")):
        return text
    try:
        numeric = float(text)
    except ValueError:
        return text
    if np.isfinite(numeric) and numeric.is_integer():
        return f"{prefix}_{int(numeric)}"
    return text


def _keypoint_moseq_row_metadata(
    frame: pd.DataFrame,
    columns: _KeypointMoseqColumns,
    index: int,
    *,
    recording_name: str | None,
    time_scale: float,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"row_index": index}
    if columns.frame is None:
        metadata["frame_index_source"] = "row_index"
    recording = _optional_text_from_frame(frame, columns.recording, index) or recording_name
    if recording is not None:
        metadata["recording_name"] = recording
    time_s = _optional_float_from_frame(frame, columns.time, index)
    if time_s is not None:
        metadata["time_s"] = time_s * time_scale
    metadata["source_label_column"] = columns.label
    metadata["source_label"] = _json_scalar(frame[columns.label].iloc[index])
    for key, column in (
        ("source_score_column", columns.score),
        ("source_confidence_column", columns.confidence),
    ):
        if column is not None:
            metadata[key] = column
            value = frame[column].iloc[index]
            if not _is_missing(value):
                metadata[key.removesuffix("_column")] = _json_scalar(value)
    if columns.uncertainty:
        metadata["source_uncertainty_columns"] = list(columns.uncertainty)

    used_columns = columns.used_names()
    for column in frame.columns:
        column_name = str(column)
        if column_name in used_columns:
            continue
        value = frame[column].iloc[index]
        if _is_missing(value):
            continue
        metadata[column_name] = _json_scalar(value)
    return metadata


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
        label = resolve_column(frame, columns["label_column"], _LABEL_CANDIDATES)
        if label is None:
            raise ValueError("Behavior CSV must include a behavior label column.")
        resolved = cls(
            label=label,
            start=resolve_column(frame, columns["start_column"], _START_TIME_CANDIDATES),
            end=resolve_column(frame, columns["end_column"], _END_TIME_CANDIDATES),
            duration=resolve_column(frame, columns["duration_column"], _DURATION_CANDIDATES),
            frame=resolve_column(frame, columns["frame_column"], _FRAME_CANDIDATES),
            start_frame=resolve_column(
                frame,
                columns["start_frame_column"],
                _START_FRAME_CANDIDATES,
            ),
            end_frame=resolve_column(frame, columns["end_frame_column"], _END_FRAME_CANDIDATES),
            score=resolve_column(frame, columns["score_column"], _SCORE_CANDIDATES),
            confidence=resolve_column(
                frame,
                columns["confidence_column"],
                _CONFIDENCE_CANDIDATES,
            ),
            source_id=resolve_column(frame, columns["source_id_column"], _SOURCE_ID_CANDIDATES),
        )
        if resolved.start is None and resolved.start_frame is None and resolved.frame is None:
            raise ValueError("Behavior CSV must include time or frame index columns.")
        return resolved

    def used_names(self) -> set[str]:
        return {
            str(column)
            for column in (
                self.label,
                self.start,
                self.end,
                self.duration,
                self.frame,
                self.start_frame,
                self.end_frame,
                self.score,
                self.confidence,
                self.source_id,
            )
            if column is not None
        }


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
        metadata=_csv_row_metadata(frame, columns, index),
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
            metadata=_csv_row_metadata(frame, columns, index),
        )
        for index in range(len(frame))
    )


def _csv_row_metadata(
    frame: pd.DataFrame,
    columns: _BehaviorColumns,
    index: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"row_index": index}
    used_columns = columns.used_names()
    for column in frame.columns:
        column_name = str(column)
        if column_name in used_columns:
            continue
        value = frame[column].iloc[index]
        if _is_missing(value):
            continue
        metadata[column_name] = _json_scalar(value)
    return metadata


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


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
    "read_boris_csv",
    "read_bsoid_csv",
    "read_behavior_events_csv",
    "read_behavior_events_json",
    "read_keypoint_moseq_syllables_csv",
    "read_simba_csv",
]
