"""JSON codecs for canonical ``ViconRecording`` payloads."""

from __future__ import annotations

import base64
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from xpkg.core.json_utils import load_json_dict
from xpkg.model.vicon import (
    ViconAdditionalPointData,
    ViconAnalogData,
    ViconCamera,
    ViconEvent,
    ViconMarkerModel,
    ViconRecording,
)

XPKG_VICON_JSON_FORMAT = "xpkg.vicon-recording-json"
XPKG_VICON_JSON_VERSION = "1.1.0"


def _serialize_path(path: str | Path | None, *, source_root: Path | None = None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if source_root is not None:
        try:
            return candidate.resolve().relative_to(source_root.resolve()).as_posix()
        except ValueError:
            return candidate.resolve().as_posix()
    return candidate.as_posix()


def _deserialize_path(raw: Any, *, source_root: Path | None = None) -> Path | None:
    if raw is None:
        return None
    path = Path(str(raw))
    if path.is_absolute() or source_root is None:
        return path
    return (source_root / path).resolve()


def _encode_array(array: np.ndarray) -> dict[str, Any]:
    values = np.ascontiguousarray(np.asarray(array))
    compressed = zlib.compress(values.tobytes())
    return {
        "dtype": values.dtype.str,
        "shape": list(values.shape),
        "codec": "base64+zlib",
        "data": base64.b64encode(compressed).decode("ascii"),
    }


def _decode_array(payload: Any, *, name: str) -> np.ndarray:
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must be encoded as a mapping.")
    raw_dtype = payload.get("dtype")
    raw_shape = payload.get("shape")
    raw_data = payload.get("data")
    if not isinstance(raw_dtype, str) or not isinstance(raw_shape, list) or not isinstance(
        raw_data, str
    ):
        raise TypeError(f"{name} is missing one of dtype, shape, or data.")

    dtype = np.dtype(raw_dtype)
    shape = tuple(int(item) for item in raw_shape)
    compressed = base64.b64decode(raw_data.encode("ascii"))
    data = zlib.decompress(compressed)
    return np.frombuffer(data, dtype=dtype).copy().reshape(shape)


def _camera_to_payload(camera: ViconCamera) -> dict[str, Any]:
    return {
        "device_id": int(camera.device_id),
        "user_id": int(camera.user_id),
        "sensor": str(camera.sensor),
        "position": np.asarray(camera.position, dtype=np.float64).tolist(),
        "orientation": np.asarray(camera.orientation, dtype=np.float64).tolist(),
        "focal_length": float(camera.focal_length),
        "image_error": float(camera.image_error),
        "world_error": float(camera.world_error),
        "sensor_size": [int(camera.sensor_size[0]), int(camera.sensor_size[1])],
    }


def _camera_from_payload(payload: Any) -> ViconCamera:
    if not isinstance(payload, dict):
        raise TypeError("camera payload must be a mapping.")
    raw_sensor_size = payload.get("sensor_size", [0, 0])
    if isinstance(raw_sensor_size, list | tuple):
        sensor_width = int(raw_sensor_size[0])
        sensor_height = int(raw_sensor_size[1])
    else:
        sensor_width = 0
        sensor_height = 0
    return ViconCamera(
        device_id=int(payload.get("device_id", 0)),
        user_id=int(payload.get("user_id", 0)),
        sensor=str(payload.get("sensor", "")),
        position=np.asarray(payload.get("position", [0.0, 0.0, 0.0]), dtype=np.float64),
        orientation=np.asarray(
            payload.get("orientation", [0.0, 0.0, 0.0, 1.0]),
            dtype=np.float64,
        ),
        focal_length=float(payload.get("focal_length", 0.0)),
        image_error=float(payload.get("image_error", 0.0)),
        world_error=float(payload.get("world_error", 0.0)),
        sensor_size=(sensor_width, sensor_height),
    )


def _event_to_payload(event: ViconEvent) -> dict[str, Any]:
    return {
        "context": event.context,
        "label": event.label,
        "frame": int(event.frame),
        "source_frame": int(event.source_frame),
        "time_seconds": float(event.time_seconds),
        "event_type": event.event_type,
        "side": event.side,
        "subject_label": event.subject_label,
    }


def _event_from_payload(payload: Any) -> ViconEvent:
    if not isinstance(payload, dict):
        raise TypeError("event payload must be a mapping.")
    raw_side = payload.get("side")
    raw_subject = payload.get("subject_label")
    return ViconEvent(
        context=str(payload.get("context", "")),
        label=str(payload.get("label", "")),
        frame=int(payload.get("frame", 0)),
        source_frame=int(payload.get("source_frame", 0)),
        time_seconds=float(payload.get("time_seconds", 0.0)),
        event_type=str(payload.get("event_type", "")),
        side=None if raw_side is None else str(raw_side),
        subject_label=None if raw_subject is None else str(raw_subject),
    )


def _model_to_payload(model: ViconMarkerModel | None) -> dict[str, Any] | None:
    if model is None:
        return None
    return {
        "name": model.name,
        "display_name": model.display_name,
        "marker_names": list(model.marker_names),
        "edges": [list(edge) for edge in model.edges],
        "source": model.source,
    }


def _model_from_payload(payload: Any) -> ViconMarkerModel | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise TypeError("model payload must be a mapping.")
    raw_edges = payload.get("edges") or []
    return ViconMarkerModel(
        name=str(payload.get("name", "")),
        display_name=str(payload.get("display_name", "")),
        marker_names=tuple(str(item) for item in payload.get("marker_names") or []),
        edges=tuple((str(edge[0]), str(edge[1])) for edge in raw_edges),
        source=str(payload.get("source", "detected")),
    )


def _analog_to_payload(analog: ViconAnalogData | None) -> dict[str, Any] | None:
    if analog is None:
        return None
    return {
        "fps": int(analog.fps),
        "samples_per_frame": int(analog.samples_per_frame),
        "channel_names": list(analog.channel_names),
        "values": _encode_array(analog.values),
    }


def _analog_from_payload(payload: Any) -> ViconAnalogData | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise TypeError("analog payload must be a mapping.")
    return ViconAnalogData(
        fps=int(payload.get("fps", 0)),
        samples_per_frame=int(payload.get("samples_per_frame", 0)),
        channel_names=tuple(str(item) for item in payload.get("channel_names") or []),
        values=_decode_array(payload.get("values"), name="analog.values"),
    )


def _additional_points_to_payload(
    additional_points: ViconAdditionalPointData | None,
) -> dict[str, Any] | None:
    if additional_points is None:
        return None
    return {
        "labels": list(additional_points.labels),
        "values": _encode_array(additional_points.values),
    }


def _additional_points_from_payload(payload: Any) -> ViconAdditionalPointData | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise TypeError("additional_points payload must be a mapping.")
    return ViconAdditionalPointData(
        labels=tuple(str(item) for item in payload.get("labels") or []),
        values=_decode_array(payload.get("values"), name="additional_points.values"),
    )


def vicon_recording_to_json_payload(
    recording: ViconRecording,
    *,
    metadata: dict[str, Any] | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Serialize a ``ViconRecording`` to a JSON document."""

    payload = {
        "path": _serialize_path(recording.path, source_root=source_root),
        "source_type": recording.source_type,
        "fps": int(recording.fps),
        "frame_offset": int(recording.frame_offset),
        "marker_names": list(recording.marker_names),
        "source_marker_labels": list(recording.source_marker_labels),
        "positions": _encode_array(recording.positions),
        "marker_valid": _encode_array(recording.marker_valid),
        "events": [_event_to_payload(event) for event in recording.events],
        "analog": _analog_to_payload(recording.analog),
        "additional_points": _additional_points_to_payload(recording.additional_points),
        "cameras": [_camera_to_payload(camera) for camera in recording.cameras],
        "model": _model_to_payload(recording.model),
        "xcp_path": _serialize_path(recording.xcp_path, source_root=source_root),
        "vsk_path": _serialize_path(recording.vsk_path, source_root=source_root),
        "metadata": dict(metadata or {}),
    }
    return {
        "format": XPKG_VICON_JSON_FORMAT,
        "version": XPKG_VICON_JSON_VERSION,
        "payload": payload,
    }


def _coerce_vicon_json_payload(document_or_payload: Any) -> dict[str, Any]:
    if not isinstance(document_or_payload, dict):
        raise TypeError("Vicon JSON input must be a mapping.")

    if "format" in document_or_payload:
        fmt = str(document_or_payload.get("format", "")).strip()
        if fmt != XPKG_VICON_JSON_FORMAT:
            raise ValueError(
                f"Unsupported Vicon JSON format {fmt!r}; expected {XPKG_VICON_JSON_FORMAT!r}."
            )
        payload = document_or_payload.get("payload")
        if not isinstance(payload, dict):
            raise TypeError("Vicon JSON document must contain an object under 'payload'.")
    else:
        payload = document_or_payload

    out: dict[str, Any] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise TypeError("Vicon JSON payload keys must be strings.")
        out[key] = value
    return out


def read_vicon_json_payload(path: str | Path) -> dict[str, Any]:
    """Read a serialized Vicon payload from disk."""

    return _coerce_vicon_json_payload(load_json_dict(path))


def vicon_recording_from_json_payload(
    document_or_payload: dict[str, Any],
    *,
    source_root: Path | None = None,
) -> ViconRecording:
    """Hydrate ``ViconRecording`` from a JSON payload or full JSON document."""

    payload = _coerce_vicon_json_payload(document_or_payload)
    raw_cameras = payload.get("cameras") or []
    raw_events = payload.get("events") or []
    return ViconRecording(
        path=_deserialize_path(payload.get("path"), source_root=source_root)
        or Path("recording"),
        source_type=str(payload.get("source_type", "")),
        fps=int(payload.get("fps", 0)),
        marker_names=tuple(str(item) for item in payload.get("marker_names") or []),
        source_marker_labels=tuple(
            str(item) for item in payload.get("source_marker_labels") or []
        ),
        positions=_decode_array(payload.get("positions"), name="positions"),
        marker_valid=_decode_array(payload.get("marker_valid"), name="marker_valid"),
        frame_offset=int(payload.get("frame_offset", 0)),
        events=tuple(_event_from_payload(item) for item in raw_events),
        analog=_analog_from_payload(payload.get("analog")),
        additional_points=_additional_points_from_payload(payload.get("additional_points")),
        cameras=tuple(_camera_from_payload(item) for item in raw_cameras),
        model=_model_from_payload(payload.get("model")),
        xcp_path=_deserialize_path(payload.get("xcp_path"), source_root=source_root),
        vsk_path=_deserialize_path(payload.get("vsk_path"), source_root=source_root),
    )


__all__ = [
    "XPKG_VICON_JSON_FORMAT",
    "XPKG_VICON_JSON_VERSION",
    "read_vicon_json_payload",
    "vicon_recording_from_json_payload",
    "vicon_recording_to_json_payload",
]
