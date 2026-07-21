"""Direct reader for pyPhotometry PPD recordings."""

from __future__ import annotations

import struct
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from xpkg._core.json_utils import load_json_dict, parse_json_dict
from xpkg.io.readers._discovery import find_first_file
from xpkg.model import (
    Event,
    EventTable,
    PhotometryChannel,
    PhotometryRecording,
    RecordingSession,
    SessionEventStream,
    SessionSignal,
    SignalChannel,
    Timeline,
    TimeSeries,
)


def _sampling_rate_with_source(header: dict[str, Any]) -> tuple[float, str]:
    for key in ("sampling_rate", "sample_rate", "SamplingRate", "fs", "Fs"):
        if key in header:
            value = float(header[key])
            if np.isfinite(value) and value > 0.0:
                return value, f"header.{key}"
            break
    raise ValueError("Sampling rate missing or invalid in PPD header.")


def _channel_count(header: dict[str, Any]) -> int:
    value = header.get("n_analog_channels", header.get("n_analog_signals", 2))
    count = int(value)
    if count <= 0:
        raise ValueError(f"n_analog_channels must be positive, got {count}.")
    return count


def _digital_count(header: dict[str, Any], analog_count: int) -> int:
    for key in ("n_digital_channels", "n_digital_signals"):
        if key in header:
            count = int(header[key])
            if count < 0:
                raise ValueError(f"{key} must be non-negative, got {count}.")
            if count > analog_count:
                raise ValueError(
                    f"{key} cannot exceed the number of analog channels used to encode "
                    f"digital bits: {count} vs {analog_count}."
                )
            return count
    raise ValueError("PPD header is missing n_digital_channels or n_digital_signals.")


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
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("volts_per_division values must be positive and finite.")
        return values.reshape((1, channel_count))
    value = float(raw)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("volts_per_division must be positive and finite.")
    return np.full((1, channel_count), value, dtype=np.float64)


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
            header = parse_json_dict(header_json)
        except UnicodeDecodeError:
            header = {
                "sampling_rate": int.from_bytes(header_json[32:34], "little"),
                "n_analog_channels": 2,
                "n_digital_signals": 2,
                "volts_per_division": [0.00010122, 0.00010122],
                "mode": "legacy",
                "version": "0",
                "header_format": "legacy_binary",
            }
        except JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON header in .ppd file: {path}") from exc
        payload = handle.read()
    if len(payload) % 2:
        raise ValueError("Invalid .ppd: payload byte length must be divisible by 2.")
    return header, np.frombuffer(payload, dtype="<u2")


def is_pyphotometry_ppd_file(path: str | Path) -> bool:
    """Return whether ``path`` has a pyPhotometry PPD header envelope."""

    source_path = Path(path)
    if not source_path.is_file() or source_path.suffix.lower() != ".ppd":
        return False
    try:
        with source_path.open("rb") as handle:
            header_len_bytes = handle.read(2)
            if len(header_len_bytes) != 2:
                return False
            (header_len,) = struct.unpack("<H", header_len_bytes)
            header_json = handle.read(header_len)
    except OSError:
        return False
    if len(header_json) != header_len:
        return False
    try:
        parse_json_dict(header_json)
    except UnicodeDecodeError:
        return len(header_json) >= 34
    except JSONDecodeError:
        return False
    return True


def find_first_pyphotometry_ppd_file(path: str | Path) -> Path | None:
    """Return the first pyPhotometry PPD file under ``path``."""

    return find_first_file(path, is_pyphotometry_ppd_file)


def _version_tuple(value: object) -> tuple[int, ...]:
    parts: list[int] = []
    for part in str(value).split("."):
        digits = "".join(char for char in part if char.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _uses_pulsed_v11_layout(header: dict[str, Any]) -> bool:
    mode = str(header.get("mode", "")).lower()
    if "pulsed" not in mode:
        return False
    version = _version_tuple(header.get("version", "0"))
    return version >= (1, 1)


def _split_old_layout(
    words: np.ndarray,
    channel_count: int,
    digital_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if words.size == 0:
        raise ValueError("PPD file contains no complete samples.")
    if words.size % channel_count:
        raise ValueError(
            "PPD old-layout payload word count must be divisible by "
            f"n_analog_channels ({channel_count}); got {words.size} words."
        )
    rows = words.reshape((-1, channel_count))
    analog = (rows >> 1).astype(np.float64)
    digital = (rows[:, :digital_count] & 0x1).astype(np.float64)
    return analog, digital


def _split_pulsed_v11_layout(
    words: np.ndarray,
    channel_count: int,
    digital_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_width = 2 * channel_count
    if words.size == 0:
        raise ValueError("PPD file contains no complete pulsed-mode samples.")
    if words.size % row_width:
        raise ValueError(
            "PPD pulsed-mode payload word count must be divisible by "
            f"2 * n_analog_channels ({row_width}); got {words.size} words."
        )
    rows = words.reshape((-1, row_width))
    raw_led_on = (rows[:, 0::2] >> 1).astype(np.float64)
    raw_baseline = (rows[:, 1::2] >> 1).astype(np.float64)
    analog = raw_led_on - raw_baseline
    digital = (rows[:, 0::2][:, :digital_count] & 0x1).astype(np.float64)
    return analog, digital, raw_led_on, raw_baseline


def _clipping_mask(
    raw_analog: np.ndarray,
    header: dict[str, Any],
    scale: np.ndarray | None,
) -> np.ndarray | None:
    if "ADC_max_value" not in header:
        return None
    if scale is None:
        raise ValueError("PPD ADC_max_value requires volts_per_division for clipping detection.")
    maximum = float(header["ADC_max_value"])
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("ADC_max_value must be positive and finite.")
    return raw_analog * scale >= maximum


def _rising_edges(bits: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    if bits.size < 2:
        return np.asarray([], dtype=np.float64)
    high = bits.astype(bool)
    indices = np.where((~high[:-1]) & high[1:])[0] + 1
    return indices.astype(np.float64) / sample_rate_hz


def _require_binary_digital(digital: np.ndarray, *, source: str) -> np.ndarray:
    values = np.asarray(digital, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{source} digital data must have shape (samples, channels).")
    invalid = values[(values != 0.0) & (values != 1.0)]
    if invalid.size:
        examples = sorted({float(value) for value in invalid[:5]})
        raise ValueError(f"{source} digital values must be 0 or 1; got {examples}.")
    return values


def _digital_events(
    digital: np.ndarray,
    sample_rate_hz: float,
    path: Path,
    *,
    source_type: str,
) -> EventTable:
    digital = _require_binary_digital(digital, source="pyPhotometry")
    events: list[Event] = []
    for index in range(digital.shape[1]):
        label = f"digital_{index + 1}"
        for time_s in _rising_edges(digital[:, index], sample_rate_hz):
            events.append(
                Event(
                    event_id=f"{source_type}-{len(events):06d}",
                    kind="ttl",
                    start_s=float(time_s),
                    label=label,
                    metadata={"source": {"type": source_type, "path": str(path)}},
                )
            )
    return EventTable.from_events(events)


def read_pyphotometry_ppd(path: str | Path) -> RecordingSession:
    """Read a pyPhotometry ``.ppd`` file into a session-level xpkg object."""

    source_path = Path(path)
    header, words = _read_ppd_words(source_path)
    sample_rate_hz, sample_rate_source = _sampling_rate_with_source(header)
    channel_count = _channel_count(header)
    digital_count = _digital_count(header, channel_count)
    extra_signals: dict[str, TimeSeries] = {}
    if _uses_pulsed_v11_layout(header):
        analog, digital, raw_led_on, raw_baseline = _split_pulsed_v11_layout(
            words,
            channel_count,
            digital_count,
        )
        clipping_source = raw_led_on
    else:
        analog, digital = _split_old_layout(words, channel_count, digital_count)
        raw_led_on = None
        raw_baseline = None
        clipping_source = analog
    scale = _volts_per_division(header, channel_count)
    clipping = _clipping_mask(clipping_source, header, scale)
    unit = "V" if scale is not None else "raw"
    if scale is not None:
        analog = analog * scale
        if raw_led_on is not None:
            raw_led_on = raw_led_on * scale
        if raw_baseline is not None:
            raw_baseline = raw_baseline * scale

    timeline = Timeline.from_sample_rate(
        n_samples=analog.shape[0],
        sample_rate_hz=sample_rate_hz,
    )
    analog_names = tuple(f"analog_{index + 1}" for index in range(channel_count))
    if clipping is not None:
        extra_signals["clipping"] = TimeSeries(
            values=clipping,
            channels=tuple(SignalChannel(name=name, unit="bool") for name in analog_names),
            timeline=timeline,
            name="pyphotometry_clipping",
            provenance={"source": {"type": "pyphotometry_ppd", "path": str(source_path)}},
        )
    if raw_led_on is not None and raw_baseline is not None:
        extra_signals["raw_led_on"] = TimeSeries(
            values=raw_led_on,
            channels=tuple(SignalChannel(name=name, unit=unit) for name in analog_names),
            timeline=timeline,
            name="pyphotometry_raw_led_on",
            provenance={"source": {"type": "pyphotometry_ppd", "path": str(source_path)}},
        )
        extra_signals["raw_baseline"] = TimeSeries(
            values=raw_baseline,
            channels=tuple(SignalChannel(name=name, unit=unit) for name in analog_names),
            timeline=timeline,
            name="pyphotometry_raw_baseline",
            provenance={"source": {"type": "pyphotometry_ppd", "path": str(source_path)}},
        )
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
        metadata={
            "header": dict(header),
            "source_type": "pyphotometry_ppd",
            "sampling_rate_hz": sample_rate_hz,
            "sampling_rate_source": sample_rate_source,
            "event_label_scheme": "digital_channels",
        },
    )
    digital_names = tuple(f"digital_{index + 1}" for index in range(digital_count))
    digital_series = TimeSeries(
        values=digital,
        channels=tuple(SignalChannel(name=name, unit="state") for name in digital_names),
        timeline=timeline,
        name="digital",
        provenance={"source": {"type": "pyphotometry_ppd", "path": str(source_path)}},
    )
    return RecordingSession(
        session_id=source_path.stem,
        signals=(
            SessionSignal("photometry", photometry),
            SessionSignal("digital", digital_series),
            *(SessionSignal(name, signal) for name, signal in extra_signals.items()),
        ),
        event_streams=(
            SessionEventStream(
                "digital",
                _digital_events(
                    digital,
                    sample_rate_hz,
                    source_path,
                    source_type="pyphotometry_ppd",
                ),
            ),
        ),
        metadata={
            "source": {"type": "pyphotometry_ppd", "path": str(source_path)},
            "ppd_header": dict(header),
            "sampling_rate_hz": sample_rate_hz,
            "sampling_rate_source": sample_rate_source,
            "event_label_scheme": "digital_channels",
        },
    )


def _load_settings(settings_path: Path | None, *, required: bool = False) -> dict[str, Any]:
    if settings_path is None:
        return {}
    if not settings_path.is_file():
        if required:
            raise FileNotFoundError(f"pyPhotometry settings_path was not found: {settings_path}")
        return {}
    try:
        return load_json_dict(settings_path)
    except JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON settings file: {settings_path}") from exc


def _matching_columns(frame: pd.DataFrame, prefix: str) -> tuple[str, ...]:
    columns: list[str] = []
    for column in frame.columns:
        text = str(column).strip()
        normalized = text.lower().replace("_", "")
        if normalized.startswith(prefix):
            columns.append(text)
    return tuple(columns)


def is_pyphotometry_csv(path: str | Path) -> bool:
    """Return whether ``path`` has pyPhotometry-style Analog CSV columns."""

    source_path = Path(path)
    if not source_path.is_file() or source_path.suffix.lower() != ".csv":
        return False
    try:
        frame = pd.read_csv(source_path, nrows=0, skipinitialspace=True)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError):
        return False
    return bool(_matching_columns(frame, "analog"))


def find_first_pyphotometry_csv(path: str | Path) -> Path | None:
    """Return the first pyPhotometry CSV export under ``path``."""

    return find_first_file(path, is_pyphotometry_csv)


def read_pyphotometry_csv(
    path: str | Path,
    *,
    settings_path: str | Path | None = None,
    sample_rate_hz: float | None = None,
    volts_per_division: float | None = None,
) -> RecordingSession:
    """Read a pyPhotometry CSV export and optional JSON settings sidecar."""

    source_path = Path(path)
    explicit_settings_path = settings_path is not None
    sidecar_path = (
        Path(settings_path) if settings_path is not None else source_path.with_suffix(".json")
    )
    settings = _load_settings(sidecar_path, required=explicit_settings_path)
    header = dict(settings)
    if sample_rate_hz is not None:
        header["sampling_rate"] = float(sample_rate_hz)
    if volts_per_division is not None:
        header["volts_per_division"] = float(volts_per_division)
    rate, sample_rate_source = _sampling_rate_with_source(header)

    # pyPhotometry's acquisition GUI writes the header as ", ".join(channels),
    # so real exports carry a space after each delimiter ("Analog1, Analog2, ...").
    # skipinitialspace lets the parsed column names match the stripped names used
    # for channel matching below; it is a no-op for comma-only headers.
    frame = pd.read_csv(source_path, skipinitialspace=True)
    if frame.empty:
        raise ValueError(f"pyPhotometry CSV '{source_path}' is empty.")
    analog_columns = _matching_columns(frame, "analog")
    digital_columns = _matching_columns(frame, "digital")
    if not analog_columns:
        raise ValueError(f"pyPhotometry CSV '{source_path}' has no Analog columns.")
    analog = (
        frame.loc[:, analog_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    )
    channel_count = len(analog_columns)
    scale = _volts_per_division(header, channel_count)
    unit = "V" if scale is not None else "raw"
    if scale is not None:
        analog = analog * scale

    timeline = Timeline.from_sample_rate(n_samples=analog.shape[0], sample_rate_hz=rate)
    analog_names = tuple(f"analog_{index + 1}" for index in range(channel_count))
    photometry = PhotometryRecording(
        series=TimeSeries(
            values=analog,
            channels=tuple(PhotometryChannel(name=name, unit=unit) for name in analog_names),
            timeline=timeline,
            name="photometry",
            provenance={"source": {"type": "pyphotometry_csv", "path": str(source_path)}},
        ),
        signal_channel=analog_names[0],
        reference_channel=analog_names[1] if len(analog_names) > 1 else None,
        metadata={
            "header": header,
            "source_type": "pyphotometry_csv",
            "sampling_rate_hz": rate,
            "sampling_rate_source": sample_rate_source,
            "event_label_scheme": "digital_channels",
        },
    )

    signals: dict[str, TimeSeries | PhotometryRecording] = {"photometry": photometry}
    events = EventTable()
    if digital_columns:
        digital = (
            frame.loc[:, digital_columns]
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(dtype=np.float64)
        )
        digital = _require_binary_digital(digital, source="pyPhotometry CSV")
        digital_names = tuple(f"digital_{index + 1}" for index in range(digital.shape[1]))
        signals["digital"] = TimeSeries(
            values=digital,
            channels=tuple(SignalChannel(name=name, unit="state") for name in digital_names),
            timeline=timeline,
            name="digital",
            provenance={"source": {"type": "pyphotometry_csv", "path": str(source_path)}},
        )
        events = _digital_events(
            digital,
            rate,
            source_path,
            source_type="pyphotometry_csv",
        )

    return RecordingSession(
        session_id=source_path.stem,
        signals=tuple(SessionSignal(name, signal) for name, signal in signals.items()),
        event_streams=(SessionEventStream("digital", events),),
        metadata={
            "source": {"type": "pyphotometry_csv", "path": str(source_path)},
            "settings_path": str(sidecar_path) if sidecar_path.is_file() else None,
            "ppd_header": header,
            "sampling_rate_hz": rate,
            "sampling_rate_source": sample_rate_source,
            "event_label_scheme": "digital_channels",
        },
    )


__all__ = [
    "find_first_pyphotometry_csv",
    "find_first_pyphotometry_ppd_file",
    "is_pyphotometry_csv",
    "is_pyphotometry_ppd_file",
    "read_pyphotometry_csv",
    "read_pyphotometry_ppd",
]
