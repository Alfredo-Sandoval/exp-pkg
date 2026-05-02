"""Direct reader for pyPhotometry PPD recordings."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from xpkg.model import (
    Event,
    EventTable,
    PhotometryChannel,
    PhotometryRecording,
    RecordingSession,
    SignalChannel,
    Timeline,
    TimeSeries,
)


def _sampling_rate(header: dict[str, Any]) -> float:
    for key in ("sampling_rate", "sample_rate", "SamplingRate", "fs", "Fs"):
        if key in header:
            value = float(header[key])
            if np.isfinite(value) and value > 0.0:
                return value
            break
    raise ValueError("Sampling rate missing or invalid in PPD header.")


def _channel_count(header: dict[str, Any]) -> int:
    value = header.get("n_analog_channels", header.get("n_analog_signals", 2))
    count = int(value)
    if count <= 0:
        raise ValueError(f"n_analog_channels must be positive, got {count}.")
    return count


def _volts_per_division(header: dict[str, Any], channel_count: int) -> np.ndarray | None:
    if "volts_per_division" not in header:
        return None
    raw = header["volts_per_division"]
    if isinstance(raw, list | tuple):
        values = np.asarray(raw, dtype=np.float64)
        if values.size != channel_count:
            raise ValueError(
                "volts_per_division length must match n_analog_channels: "
                f"{values.size} vs {channel_count}."
            )
        return values.reshape((1, channel_count))
    return np.full((1, channel_count), float(raw), dtype=np.float64)


def _read_ppd_words(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    with path.open("rb") as handle:
        header_len_bytes = handle.read(2)
        if len(header_len_bytes) != 2:
            raise RuntimeError("Invalid .ppd: missing header length.")
        (header_len,) = struct.unpack("<H", header_len_bytes)
        header_json = handle.read(header_len)
        if len(header_json) != header_len:
            raise RuntimeError("Invalid .ppd: incomplete header JSON.")
        try:
            header = json.loads(header_json.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON header in .ppd file: {path}") from exc
        payload = handle.read()
    if len(payload) % 2:
        payload = payload[:-1]
    return header, np.frombuffer(payload, dtype="<u2")


def _split_channels(words: np.ndarray, channel_count: int) -> tuple[np.ndarray, np.ndarray]:
    complete_words = (words.size // channel_count) * channel_count
    if complete_words == 0:
        raise ValueError("PPD file contains no complete samples.")
    rows = words[:complete_words].reshape((-1, channel_count))
    analog = (rows >> 1).astype(np.float64)
    digital = (rows & 0x1).astype(np.float64)
    return analog, digital


def _rising_edges(bits: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    if bits.size < 2:
        return np.asarray([], dtype=np.float64)
    high = bits.astype(bool)
    indices = np.where((~high[:-1]) & high[1:])[0] + 1
    return indices.astype(np.float64) / sample_rate_hz


def _digital_events(digital: np.ndarray, sample_rate_hz: float, path: Path) -> EventTable:
    events: list[Event] = []
    for index in range(digital.shape[1]):
        label = f"digital_{index + 1}"
        for time_s in _rising_edges(digital[:, index], sample_rate_hz):
            events.append(
                Event(
                    kind="ttl",
                    start_s=float(time_s),
                    label=label,
                    metadata={"source": {"type": "pyphotometry_ppd", "path": str(path)}},
                )
            )
    return EventTable.from_events(events)


def read_pyphotometry_ppd(path: str | Path) -> RecordingSession:
    """Read a pyPhotometry ``.ppd`` file into a session-level xpkg object."""

    source_path = Path(path)
    header, words = _read_ppd_words(source_path)
    sample_rate_hz = _sampling_rate(header)
    channel_count = _channel_count(header)
    analog, digital = _split_channels(words, channel_count)
    scale = _volts_per_division(header, channel_count)
    unit = "V" if scale is not None else "raw"
    if scale is not None:
        analog = analog * scale

    timeline = Timeline.from_sample_rate(
        n_samples=analog.shape[0],
        sample_rate_hz=sample_rate_hz,
    )
    analog_names = tuple(f"analog_{index + 1}" for index in range(channel_count))
    photometry = PhotometryRecording(
        series=TimeSeries(
            values=analog,
            channels=tuple(PhotometryChannel(name=name, unit=unit) for name in analog_names),
            timeline=timeline,
            name="photometry",
            provenance={"source": {"type": "pyphotometry_ppd", "path": str(source_path)}},
        ),
        signal_channel=analog_names[0],
        reference_channel=analog_names[1] if len(analog_names) > 1 else None,
        metadata={"header": dict(header), "source_type": "pyphotometry_ppd"},
    )
    digital_names = tuple(f"digital_{index + 1}" for index in range(channel_count))
    digital_series = TimeSeries(
        values=digital,
        channels=tuple(SignalChannel(name=name, unit="state") for name in digital_names),
        timeline=timeline,
        name="digital",
        provenance={"source": {"type": "pyphotometry_ppd", "path": str(source_path)}},
    )
    return RecordingSession(
        session_id=source_path.stem,
        signals={"photometry": photometry, "digital": digital_series},
        events=_digital_events(digital, sample_rate_hz, source_path),
        metadata={
            "source": {"type": "pyphotometry_ppd", "path": str(source_path)},
            "ppd_header": dict(header),
        },
    )


__all__ = ["read_pyphotometry_ppd"]
