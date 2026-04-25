"""Readers for Vicon CSV and C3D recordings."""

from __future__ import annotations

import importlib
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from xpkg.model.vicon import (
    ViconAdditionalPointData,
    ViconAnalogData,
    ViconCamera,
    ViconEvent,
    ViconMarkerModel,
    ViconRecording,
)


def canonical_marker_label(name: str) -> str:
    """Return a marker label without any Vicon subject prefix."""
    label = str(name).strip()
    if ":" in label:
        label = label.split(":", 1)[1]
    return label


def normalize_marker_name(name: str) -> str:
    """Normalize marker labels for case-insensitive schema matching."""
    return canonical_marker_label(name).lower()


def _normalize_source_label(name: str) -> str:
    return str(name).strip().lower()


def _unique_names(names: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        canonical = canonical_marker_label(name)
        normalized = normalize_marker_name(canonical)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(canonical)
    return tuple(ordered)


def _display_marker_names(source_labels: Sequence[str]) -> tuple[str, ...]:
    canonical_counts = Counter(normalize_marker_name(label) for label in source_labels)
    display_names: list[str] = []
    for source_label in source_labels:
        canonical = canonical_marker_label(source_label)
        if canonical_counts[normalize_marker_name(source_label)] > 1:
            display_names.append(str(source_label).strip())
        else:
            display_names.append(canonical)
    return tuple(display_names)


MOUSE_MARKER_NAMES: tuple[str, ...] = (
    "center",
    "R_crest",
    "L_crest",
    "R_hip",
    "R_knee",
    "R_ankle",
    "R_foot",
    "L_hip",
    "L_knee",
    "L_ankle",
    "L_foot",
    "L_shoulder",
    "L_elbow",
    "L_paw",
    "R_shoulder",
    "R_elbow",
    "R_paw",
)

MOUSE_EDGES: tuple[tuple[str, str], ...] = (
    ("center", "R_crest"),
    ("center", "L_crest"),
    ("R_crest", "L_crest"),
    ("R_crest", "R_hip"),
    ("R_hip", "R_knee"),
    ("R_knee", "R_ankle"),
    ("R_ankle", "R_foot"),
    ("L_crest", "L_hip"),
    ("L_hip", "L_knee"),
    ("L_knee", "L_ankle"),
    ("L_ankle", "L_foot"),
    ("center", "R_shoulder"),
    ("R_shoulder", "R_elbow"),
    ("R_elbow", "R_paw"),
    ("center", "L_shoulder"),
    ("L_shoulder", "L_elbow"),
    ("L_elbow", "L_paw"),
)

PLUG_IN_GAIT_MARKER_NAMES: tuple[str, ...] = (
    "LFHD",
    "RFHD",
    "LBHD",
    "RBHD",
    "C7",
    "T10",
    "CLAV",
    "STRN",
    "LSHO",
    "LELB",
    "LWRA",
    "LWRB",
    "LFIN",
    "RSHO",
    "RELB",
    "RWRA",
    "RWRB",
    "RFIN",
    "LASI",
    "RASI",
    "LPSI",
    "RPSI",
    "LTHI",
    "LKNE",
    "LTIB",
    "LANK",
    "LHEE",
    "LTOE",
    "RTHI",
    "RKNE",
    "RTIB",
    "RANK",
    "RHEE",
    "RTOE",
)

PLUG_IN_GAIT_EDGES: tuple[tuple[str, str], ...] = (
    ("LFHD", "RFHD"),
    ("LFHD", "LBHD"),
    ("RFHD", "RBHD"),
    ("LBHD", "RBHD"),
    ("C7", "CLAV"),
    ("C7", "STRN"),
    ("T10", "CLAV"),
    ("T10", "STRN"),
    ("CLAV", "STRN"),
    ("CLAV", "LSHO"),
    ("LSHO", "LELB"),
    ("LELB", "LWRA"),
    ("LELB", "LWRB"),
    ("LWRA", "LWRB"),
    ("LWRA", "LFIN"),
    ("LWRB", "LFIN"),
    ("CLAV", "RSHO"),
    ("RSHO", "RELB"),
    ("RELB", "RWRA"),
    ("RELB", "RWRB"),
    ("RWRA", "RWRB"),
    ("RWRA", "RFIN"),
    ("RWRB", "RFIN"),
    ("LASI", "RASI"),
    ("LASI", "LPSI"),
    ("LASI", "RPSI"),
    ("RASI", "LPSI"),
    ("RASI", "RPSI"),
    ("LPSI", "RPSI"),
    ("LASI", "LTHI"),
    ("LASI", "LKNE"),
    ("LPSI", "LKNE"),
    ("LTHI", "LKNE"),
    ("LKNE", "LTIB"),
    ("LKNE", "LANK"),
    ("LTIB", "LANK"),
    ("LANK", "LHEE"),
    ("LANK", "LTOE"),
    ("LHEE", "LTOE"),
    ("RASI", "RTHI"),
    ("RASI", "RKNE"),
    ("RPSI", "RKNE"),
    ("RTHI", "RKNE"),
    ("RKNE", "RTIB"),
    ("RKNE", "RANK"),
    ("RTIB", "RANK"),
    ("RANK", "RHEE"),
    ("RANK", "RTOE"),
    ("RHEE", "RTOE"),
)

KNOWN_MODELS: tuple[ViconMarkerModel, ...] = (
    ViconMarkerModel(
        name="mouse_vicon_17",
        display_name="Mouse Vicon 17",
        marker_names=MOUSE_MARKER_NAMES,
        edges=MOUSE_EDGES,
        source="detected",
    ),
    ViconMarkerModel(
        name="plug_in_gait_fullbody",
        display_name="Plug-in Gait Full Body",
        marker_names=PLUG_IN_GAIT_MARKER_NAMES,
        edges=PLUG_IN_GAIT_EDGES,
        source="detected",
    ),
)


def _parse_float_cell(cell: str) -> float:
    stripped = str(cell).strip()
    if not stripped:
        return float("nan")
    return float(stripped)


def _slugify(value: str) -> str:
    slug = value.strip().lower().replace(" ", "_").replace("-", "_")
    return "".join(ch for ch in slug if ch.isalnum() or ch == "_").strip("_") or "vicon_model"


def _fallback_label(raw_label: str, *, prefix: str, ordinal: int) -> str:
    label = str(raw_label).strip()
    return label if label else f"{prefix}_{ordinal}"


def _normalize_event_strings(values: Sequence[object]) -> tuple[str, ...]:
    return tuple(str(value).strip() for value in values)


def _event_type_from_label(label: str) -> str:
    normalized = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    slug = "".join(ch for ch in normalized if ch.isalnum() or ch == "_").strip("_")
    return slug or "event"


def _event_side_from_context(context: str) -> str | None:
    normalized = str(context).strip().lower()
    if normalized in {"left", "right"}:
        return normalized
    return None


def _match_known_model(marker_names: Sequence[str]) -> ViconMarkerModel | None:
    observed = {normalize_marker_name(name) for name in marker_names}
    for model in KNOWN_MODELS:
        expected = {normalize_marker_name(name) for name in model.marker_names}
        if expected.issubset(observed):
            return model
    return None


def _match_partial_model(
    marker_names: Sequence[str],
    *,
    min_fraction: float = 0.75,
    min_markers: int = 8,
) -> ViconMarkerModel | None:
    observed = {normalize_marker_name(name) for name in marker_names}
    best_model: ViconMarkerModel | None = None
    best_fraction = 0.0
    best_count = 0

    for model in KNOWN_MODELS:
        expected = [normalize_marker_name(name) for name in model.marker_names]
        if not expected:
            continue
        matched_count = sum(1 for name in expected if name in observed)
        fraction = matched_count / len(expected)
        if matched_count > best_count or (matched_count == best_count and fraction > best_fraction):
            best_model = model
            best_fraction = fraction
            best_count = matched_count

    if best_model is None:
        return None
    if best_fraction < min_fraction or best_count < min_markers:
        return None
    return best_model


def _detected_marker_cloud(marker_names: Sequence[str]) -> ViconMarkerModel:
    seen: set[str] = set()
    ordered: list[str] = []
    for marker_name in marker_names:
        label = str(marker_name).strip()
        normalized = _normalize_source_label(label)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(label)
    return ViconMarkerModel(
        name="marker_cloud",
        display_name="Marker Cloud",
        marker_names=tuple(ordered),
        source="detected",
    )


def _project_model_to_observed(
    model: ViconMarkerModel,
    observed_marker_names: Sequence[str],
) -> ViconMarkerModel:
    observed = {normalize_marker_name(marker_name) for marker_name in observed_marker_names}
    projected_markers = tuple(
        marker_name
        for marker_name in model.marker_names
        if normalize_marker_name(marker_name) in observed
    )
    projected_edges = tuple(
        (parent, child)
        for parent, child in model.edges
        if normalize_marker_name(parent) in observed and normalize_marker_name(child) in observed
    )
    return ViconMarkerModel(
        name=model.name,
        display_name=model.display_name,
        marker_names=projected_markers,
        edges=projected_edges,
        source=model.source,
    )


def select_marker_labels(
    source_labels: Sequence[str],
    *,
    preferred_model: ViconMarkerModel | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], ViconMarkerModel]:
    """Choose a stable marker subset and model from raw source labels."""

    canonical_names: list[str] = []
    aligned_source_labels: list[str] = []
    seen_sources: set[str] = set()

    for source_label in source_labels:
        source = str(source_label).strip()
        if not source:
            continue
        normalized_source = _normalize_source_label(source)
        if normalized_source in seen_sources:
            continue
        seen_sources.add(normalized_source)
        canonical = canonical_marker_label(source)
        canonical_names.append(canonical)
        aligned_source_labels.append(source)

    if not aligned_source_labels:
        raise ValueError("No source marker labels were provided.")

    marker_names = _display_marker_names(aligned_source_labels)
    canonical_counts = Counter(normalize_marker_name(label) for label in aligned_source_labels)
    if any(count > 1 for count in canonical_counts.values()):
        source_marker_names = tuple(aligned_source_labels)
        return (
            source_marker_names,
            tuple(aligned_source_labels),
            _detected_marker_cloud(source_marker_names),
        )

    lookup = {
        normalize_marker_name(source): (marker_name, source)
        for marker_name, source in zip(marker_names, aligned_source_labels, strict=True)
    }

    active_model = preferred_model
    if active_model is None:
        active_model = _match_known_model(canonical_names)
    if active_model is None:
        active_model = _match_partial_model(canonical_names)
    if active_model is not None:
        ordered_markers: list[str] = []
        ordered_sources: list[str] = []
        for marker_name in active_model.marker_names:
            resolved = lookup.get(normalize_marker_name(marker_name))
            if resolved is None:
                continue
            ordered_markers.append(resolved[0])
            ordered_sources.append(resolved[1])
        if ordered_markers:
            return (
                tuple(ordered_markers),
                tuple(ordered_sources),
                _project_model_to_observed(active_model, ordered_markers),
            )

    return marker_names, tuple(aligned_source_labels), _detected_marker_cloud(marker_names)


def parse_vsk(path: str | Path) -> ViconMarkerModel:
    """Parse a Vicon VSK file into marker-model metadata."""

    path = Path(path)
    root = ET.parse(path).getroot()

    markers_parent = root.find(".//MarkerSet/Markers")
    sticks_parent = root.find(".//MarkerSet/Sticks")

    marker_names = _unique_names(
        [
            marker.attrib["NAME"]
            for marker in ([] if markers_parent is None else list(markers_parent.findall("Marker")))
            if marker.attrib.get("NAME")
        ]
    )

    seen_edges: set[tuple[str, str]] = set()
    edges: list[tuple[str, str]] = []
    if sticks_parent is not None:
        for stick in sticks_parent.findall("Stick"):
            marker_1 = canonical_marker_label(stick.attrib.get("MARKER1", ""))
            marker_2 = canonical_marker_label(stick.attrib.get("MARKER2", ""))
            if not marker_1 or not marker_2:
                continue
            normalized_key = (normalize_marker_name(marker_1), normalize_marker_name(marker_2))
            if normalized_key in seen_edges:
                continue
            seen_edges.add(normalized_key)
            edges.append((marker_1, marker_2))

    model_name = root.attrib.get("MODEL", path.stem)
    return ViconMarkerModel(
        name=f"vsk_{_slugify(model_name)}",
        display_name=f"{model_name} (VSK)",
        marker_names=marker_names,
        edges=tuple(edges),
        source="vsk",
    )


def find_associated_vsk(path: str | Path) -> Path | None:
    """Find the best-matching VSK file near a Vicon recording."""

    path = Path(path)
    direct = path.with_suffix(".vsk")
    if direct.exists():
        return direct

    sibling_vsks = sorted(path.parent.glob("*.vsk"))
    if len(sibling_vsks) == 1:
        return sibling_vsks[0]
    if not sibling_vsks:
        return None

    stem_normalized = normalize_marker_name(path.stem)
    for candidate in sibling_vsks:
        if normalize_marker_name(candidate.stem) in stem_normalized:
            return candidate
    return sibling_vsks[0]


def parse_xcp(path: str | Path) -> tuple[ViconCamera, ...]:
    """Parse a Vicon XCP file into camera metadata."""

    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()

    cameras: list[ViconCamera] = []
    for cam_el in root.findall("Camera"):
        sensor_size_str = cam_el.get("SENSOR_SIZE", "0 0")
        sensor_width, sensor_height = sensor_size_str.split()
        keyframe = cam_el.find(".//KeyFrame")
        if keyframe is None:
            continue

        position = np.array(
            [float(value) for value in keyframe.get("POSITION", "0 0 0").split()],
            dtype=np.float64,
        )
        orientation = np.array(
            [float(value) for value in keyframe.get("ORIENTATION", "0 0 0 1").split()],
            dtype=np.float64,
        )
        cameras.append(
            ViconCamera(
                device_id=int(cam_el.get("DEVICEID", "0")),
                user_id=int(cam_el.get("USERID", "0")),
                sensor=cam_el.get("SENSOR", ""),
                position=position,
                orientation=orientation,
                focal_length=float(keyframe.get("FOCAL_LENGTH", "0")),
                image_error=float(keyframe.get("IMAGE_ERROR", "0")),
                world_error=float(keyframe.get("WORLD_ERROR", "0")),
                sensor_size=(int(sensor_width), int(sensor_height)),
            )
        )

    return tuple(sorted(cameras, key=lambda camera: camera.user_id))


def _load_c3d_module() -> Any:
    try:
        return importlib.import_module("c3d")
    except ImportError as exc:
        raise ImportError(
            "Reading .c3d files requires the runtime dependency `c3d`. "
            "Reinstall the package or run `make env`."
        ) from exc


def _associated_vsk(path: Path) -> tuple[ViconMarkerModel | None, Path | None]:
    vsk_path = find_associated_vsk(path)
    if vsk_path is None:
        return None, None
    return parse_vsk(vsk_path), vsk_path


def _associated_cameras(path: Path) -> tuple[tuple[ViconCamera, ...], Path | None]:
    xcp_path = path.with_suffix(".xcp")
    if not xcp_path.exists():
        return (), None
    return parse_xcp(xcp_path), xcp_path


def _resolve_label_indices(
    labels: list[str],
    selected_labels: tuple[str, ...],
) -> list[int]:
    indices: list[int] = []
    used_indices: set[int] = set()
    for selected_label in selected_labels:
        for idx, label in enumerate(labels):
            if idx in used_indices:
                continue
            if label == selected_label:
                indices.append(idx)
                used_indices.add(idx)
                break
        else:
            raise KeyError(f"Point label {selected_label!r} not found in C3D labels {labels}.")
    return indices


def _read_analog_labels(reader: Any) -> tuple[str, ...]:
    try:
        raw_labels = reader.analog_labels
    except AttributeError:
        return ()
    return tuple(
        _fallback_label(label, prefix="ANALOG", ordinal=idx + 1)
        for idx, label in enumerate(raw_labels)
    )


def _c3d_event_group(reader: Any) -> Any | None:
    try:
        return reader.get("EVENT")
    except KeyError:
        return None


def _c3d_event_string_values(group: Any, key: str, *, used: int) -> tuple[str, ...]:
    param = group.get(key)
    if param is None:
        raise ValueError(f"C3D EVENT group is missing {key}.")
    values = np.asarray(param.string_array, dtype=object).reshape(-1)
    return _normalize_event_strings(tuple(values[:used]))


def _c3d_event_time_seconds(group: Any, *, used: int) -> np.ndarray:
    param = group.get("TIMES")
    if param is None:
        raise ValueError("C3D EVENT group is missing TIMES.")
    values = np.asarray(param.float_array, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(
            "Expected EVENT:TIMES to be a 2D array with frame/time pairs, "
            f"got {values.shape}."
        )
    if values.size != used * 2:
        raise ValueError(
            "Expected EVENT:TIMES to encode exactly two values per event, "
            f"got shape {values.shape} for used={used}."
        )
    paired_values = values.reshape(-1).reshape(used, 2)
    if paired_values.shape != (used, 2):
        raise ValueError(
            "Expected EVENT:TIMES to reshape to (n_events, 2), "
            f"got {paired_values.shape} from {values.shape}."
        )
    return paired_values[:, 1]


def _c3d_event_subject_labels(group: Any, *, used: int) -> tuple[str | None, ...]:
    param = group.get("SUBJECTS")
    if param is None:
        return (None,) * used
    values = np.asarray(param.string_array, dtype=object).reshape(-1)
    normalized = _normalize_event_strings(tuple(values[:used]))
    return tuple(value or None for value in normalized)


def _read_c3d_events(
    reader: Any,
    *,
    fps: int,
    frame_offset: int,
    n_frames: int,
    path: Path,
) -> tuple[ViconEvent, ...]:
    group = _c3d_event_group(reader)
    if group is None:
        return ()

    used_param = group.get("USED")
    if used_param is None:
        raise ValueError(f"Recording {path} has an EVENT group without EVENT:USED.")
    used = int(used_param.int16_value)
    if used < 0:
        raise ValueError(f"Recording {path} has invalid EVENT:USED={used}.")
    if used == 0:
        return ()

    contexts = _c3d_event_string_values(group, "CONTEXTS", used=used)
    labels = _c3d_event_string_values(group, "LABELS", used=used)
    times_seconds = _c3d_event_time_seconds(group, used=used)
    subject_labels = _c3d_event_subject_labels(group, used=used)
    if not (len(contexts) == len(labels) == len(times_seconds) == len(subject_labels) == used):
        raise ValueError(
            "C3D EVENT arrays are inconsistent: "
            f"used={used}, contexts={len(contexts)}, labels={len(labels)}, "
            f"times={len(times_seconds)}, subjects={len(subject_labels)}."
        )

    last_source_frame = frame_offset + n_frames - 1
    events: list[ViconEvent] = []
    for context, label, time_seconds, subject_label in zip(
        contexts,
        labels,
        times_seconds,
        subject_labels,
        strict=True,
    ):
        source_frame = int(round(float(time_seconds) * float(fps)))
        frame = source_frame - frame_offset
        if frame < 0 or frame >= n_frames:
            raise ValueError(
                f"Event {context!r} / {label!r} at source_frame={source_frame} falls outside "
                f"recording frame range {frame_offset}..{last_source_frame}."
            )
        events.append(
            ViconEvent(
                context=context,
                label=label,
                frame=frame,
                source_frame=source_frame,
                time_seconds=float(time_seconds),
                event_type=_event_type_from_label(label),
                side=_event_side_from_context(context),
                subject_label=subject_label,
            )
        )
    return tuple(events)


def read_vicon_csv(path: str | Path) -> ViconRecording:
    """Read a Vicon Nexus CSV trajectory export."""

    path = Path(path)
    with path.open(encoding="utf-8-sig") as handle:
        lines = handle.readlines()

    if len(lines) < 6:
        raise ValueError(f"CSV too short ({len(lines)} lines): {path}")

    fps = int(float(lines[1].strip()))
    raw_labels = [label.strip() for label in lines[2].rstrip("\r\n").split(",")]
    preferred_model, vsk_path = _associated_vsk(path)
    marker_names, source_marker_labels, model = select_marker_labels(
        raw_labels,
        preferred_model=preferred_model,
    )
    source_column_indices = [raw_labels.index(label) for label in source_marker_labels]
    n_expected_cols = max(source_column_indices) + 3 if source_column_indices else 2

    rows: list[list[float]] = []
    frame_offset = 0
    for line in lines[5:]:
        parts = line.rstrip("\r\n").split(",")
        if len(parts) < 2:
            continue
        try:
            frame = int(float(parts[0].strip()))
        except ValueError:
            continue
        if not rows:
            frame_offset = frame

        if len(parts) < n_expected_cols:
            parts = parts + [""] * (n_expected_cols - len(parts))

        coords: list[float] = []
        for marker_idx in source_column_indices:
            coords.extend(_parse_float_cell(value) for value in parts[marker_idx : marker_idx + 3])
        rows.append(coords)

    if not rows:
        raise ValueError(f"No data rows parsed from {path}.")

    flat = np.asarray(rows, dtype=np.float64)
    positions = flat.reshape(flat.shape[0], len(marker_names), 3)
    marker_valid = np.asarray(np.isfinite(positions).all(axis=2), dtype=bool)
    positions = positions.copy()
    positions[~marker_valid] = np.nan
    cameras, xcp_path = _associated_cameras(path)

    return ViconRecording(
        path=path,
        source_type="csv",
        fps=fps,
        marker_names=marker_names,
        source_marker_labels=source_marker_labels,
        positions=positions,
        marker_valid=marker_valid,
        frame_offset=frame_offset,
        cameras=cameras,
        model=model,
        xcp_path=xcp_path,
        vsk_path=vsk_path,
    )


def read_vicon_c3d(path: str | Path) -> ViconRecording:
    """Read a Vicon C3D recording."""

    c3d = _load_c3d_module()
    path = Path(path)
    preferred_model, vsk_path = _associated_vsk(path)

    with path.open("rb") as handle:
        reader = c3d.Reader(handle)
        raw_labels = [
            _fallback_label(label, prefix="POINT", ordinal=idx + 1)
            for idx, label in enumerate(reader.point_labels)
        ]
        marker_names, source_marker_labels, model = select_marker_labels(
            raw_labels,
            preferred_model=preferred_model,
        )
        if not marker_names:
            raise ValueError(f"No point labels found in {path}.")

        selected_indices = _resolve_label_indices(raw_labels, source_marker_labels)
        selected_index_set = set(selected_indices)
        additional_pairs = [
            (idx, label)
            for idx, label in enumerate(raw_labels)
            if idx not in selected_index_set
        ]
        additional_indices = [idx for idx, _label in additional_pairs]
        additional_labels = tuple(label for _idx, label in additional_pairs)
        analog_labels = _read_analog_labels(reader)

        positions_by_frame: list[np.ndarray] = []
        valid_by_frame: list[np.ndarray] = []
        additional_by_frame: list[np.ndarray] = []
        analog_by_frame: list[np.ndarray] = []
        frame_offset: int | None = None

        for frame_no, points, analog in reader.read_frames():
            if frame_offset is None:
                frame_offset = int(frame_no)

            point_frame = np.asarray(points, dtype=np.float64)
            selected_points = point_frame[selected_indices]
            xyz = selected_points[:, :3]
            residual = selected_points[:, 3]
            marker_valid = np.asarray(np.isfinite(xyz).all(axis=1) & (residual >= 0), dtype=bool)
            xyz = xyz.copy()
            xyz[~marker_valid] = np.nan

            positions_by_frame.append(xyz)
            valid_by_frame.append(marker_valid)

            if additional_indices:
                extra_points = point_frame[additional_indices].copy()
                extra_residual = extra_points[:, 3]
                extra_valid = np.isfinite(extra_points[:, :3]).all(axis=1) & (extra_residual >= 0)
                extra_points[~extra_valid, :3] = np.nan
                additional_by_frame.append(extra_points)

            if analog_labels:
                analog_frame = np.asarray(analog, dtype=np.float64)
                if analog_frame.ndim == 1:
                    analog_frame = analog_frame[:, np.newaxis]
                analog_by_frame.append(analog_frame.T.copy())

    if not positions_by_frame:
        raise ValueError(f"No point frames parsed from {path}.")

    frame_offset_value = 0 if frame_offset is None else frame_offset
    fps = int(round(float(reader.header.frame_rate)))
    positions = np.stack(positions_by_frame, axis=0)
    marker_valid = np.stack(valid_by_frame, axis=0)
    analog_data = None
    if analog_by_frame:
        analog_data = ViconAnalogData(
            fps=int(round(float(reader.analog_rate))),
            samples_per_frame=int(reader.analog_per_frame),
            channel_names=analog_labels,
            values=np.concatenate(analog_by_frame, axis=0),
        )
    additional_points = None
    if additional_by_frame:
        additional_points = ViconAdditionalPointData(
            labels=additional_labels,
            values=np.stack(additional_by_frame, axis=0),
        )
    events = _read_c3d_events(
        reader,
        fps=fps,
        frame_offset=frame_offset_value,
        n_frames=positions.shape[0],
        path=path,
    )

    cameras, xcp_path = _associated_cameras(path)
    return ViconRecording(
        path=path,
        source_type="c3d",
        fps=fps,
        marker_names=marker_names,
        source_marker_labels=source_marker_labels,
        positions=positions,
        marker_valid=marker_valid,
        frame_offset=frame_offset_value,
        events=events,
        analog=analog_data,
        additional_points=additional_points,
        cameras=cameras,
        model=model,
        xcp_path=xcp_path,
        vsk_path=vsk_path,
    )


def read_vicon_recording(path: str | Path) -> ViconRecording:
    """Read a Vicon recording from a CSV or C3D source path."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_vicon_csv(path)
    if suffix == ".c3d":
        return read_vicon_c3d(path)
    raise ValueError(f"Unsupported Vicon recording format: {path.suffix or '<none>'}.")


__all__ = [
    "ViconAdditionalPointData",
    "ViconAnalogData",
    "ViconCamera",
    "ViconEvent",
    "ViconMarkerModel",
    "ViconRecording",
    "canonical_marker_label",
    "find_associated_vsk",
    "normalize_marker_name",
    "parse_vsk",
    "parse_xcp",
    "read_vicon_c3d",
    "read_vicon_csv",
    "read_vicon_recording",
    "select_marker_labels",
]
