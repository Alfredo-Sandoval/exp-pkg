"""ABF1/ABF2 reader for whole-cell patch-clamp recordings.

This module wraps :mod:`pyabf` rather than reaching for ``neo.io.AxonIO`` so the
pCLAMP-specific protocol metadata (epoch table, levels, command waveform,
units) is preserved as recorded. Units are normalized to mV/pA at the model
boundary, mode is auto-detected from channel units, and every conversion is
captured in the recording's ``conversion_log``.

``pyabf`` is an optional dependency; install ``exp-pkg[ephys]`` to pick it up.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from xpkg.model import (
    EphysRecording,
    RecordingMode,
    SignalChannel,
    StimulusEpoch,
    Sweep,
    SweepSet,
    Timeline,
    TimeSeries,
)
from xpkg.model.ephys import (
    detect_recording_mode,
    is_current_unit,
    is_voltage_unit,
    normalize_signal_units,
)


def _pyabf_module(module: Any | None) -> Any:
    if module is not None:
        return module
    try:
        return importlib.import_module("pyabf")
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise ModuleNotFoundError(
            "ABF support requires the optional 'pyabf' package. Install "
            "exp-pkg[ephys] or install pyabf in the active environment."
        ) from exc


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _channel_layout(abf: Any) -> tuple[list[int], list[str], list[str]]:
    channel_list: list[int] = list(getattr(abf, "channelList", [0]))
    raw_names = list(getattr(abf, "adcNames", []))
    raw_units = list(getattr(abf, "adcUnits", []))
    names: list[str] = []
    units: list[str] = []
    for index, channel in enumerate(channel_list):
        name = (
            str(raw_names[index]).strip()
            if index < len(raw_names) and str(raw_names[index]).strip()
            else f"IN_{int(channel)}"
        )
        unit = (
            str(raw_units[index]).strip()
            if index < len(raw_units) and str(raw_units[index]).strip()
            else ""
        )
        names.append(name)
        units.append(unit)
    if len({*names}) != len(names):
        deduped: list[str] = []
        for base in names:
            candidate = base
            suffix = 0
            while candidate in deduped:
                suffix += 1
                candidate = f"{base}_{suffix}"
            deduped.append(candidate)
        names = deduped
    return channel_list, names, units


def _auto_channel_roles(
    *,
    names: Sequence[str],
    units: Sequence[str],
    mode: str,
) -> dict[str, str]:
    """Apply the documented role-detection rule to a channel layout."""
    roles: dict[str, str] = {}
    pairs = list(zip(names, units, strict=True))
    voltage_channels = [name for name, unit in pairs if is_voltage_unit(unit)]
    current_channels = [name for name, unit in pairs if is_current_unit(unit)]

    if mode == "current_clamp":
        electrode_pool = voltage_channels
        monitor_pool = current_channels
    elif mode == "voltage_clamp":
        electrode_pool = current_channels
        monitor_pool = voltage_channels
    else:
        electrode_pool = []
        monitor_pool = []

    if electrode_pool:
        roles[electrode_pool[0]] = "electrode"
        for extra in electrode_pool[1:]:
            roles[extra] = "auxiliary"
    if monitor_pool:
        roles[monitor_pool[0]] = "stimulus_monitor"
        for extra in monitor_pool[1:]:
            roles[extra] = "auxiliary"
    for name, unit in zip(names, units, strict=True):
        if name in roles:
            continue
        if is_voltage_unit(unit) or is_current_unit(unit):
            roles[name] = "auxiliary"
        else:
            roles[name] = "unknown"
    return roles


def _detect_mode_from_layout(units: Sequence[str]) -> RecordingMode:
    voltage_count = sum(1 for unit in units if is_voltage_unit(unit))
    current_count = sum(1 for unit in units if is_current_unit(unit))
    if voltage_count >= 1 and current_count == 0:
        return "current_clamp"
    if current_count >= 1 and voltage_count == 0:
        return "voltage_clamp"
    if voltage_count >= 1 and current_count >= 1:
        # Both kinds present: the first channel is the electrode by pyABF
        # convention (channelList[0]). Use its unit to disambiguate.
        first_unit = next((unit for unit in units if unit), "")
        if is_voltage_unit(first_unit):
            return "current_clamp"
        if is_current_unit(first_unit):
            return "voltage_clamp"
        return "unknown"
    return "unknown"


def _load_sweep_values(
    abf: Any,
    *,
    sweep_index: int,
    channel_list: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    columns: list[np.ndarray] = []
    times: np.ndarray | None = None
    for channel in channel_list:
        abf.setSweep(sweepNumber=int(sweep_index), channel=int(channel))
        sweep_y = np.asarray(abf.sweepY, dtype=np.float64)
        columns.append(sweep_y)
        if times is None:
            times = np.asarray(abf.sweepX, dtype=np.float64)
    if times is None:
        raise ValueError(f"ABF sweep {sweep_index} has no channels.")
    values = np.column_stack(columns)
    return values, times


def _epochs_for_sweep(
    abf: Any,
    *,
    sweep_index: int,
    channel: int,
    sample_rate_hz: float,
) -> tuple[StimulusEpoch, ...]:
    abf.setSweep(sweepNumber=int(sweep_index), channel=int(channel))
    epochs_obj = getattr(abf, "sweepEpochs", None)
    if epochs_obj is None:
        return ()
    types = list(getattr(epochs_obj, "types", []) or [])
    levels = list(getattr(epochs_obj, "levels", []) or [])
    p1s = list(getattr(epochs_obj, "p1s", []) or [])
    p2s = list(getattr(epochs_obj, "p2s", []) or [])
    if not types or sample_rate_hz <= 0:
        return ()
    level_unit = str(getattr(abf, "sweepUnitsC", "")).strip()
    if not level_unit:
        # Fall back to DAC units if available.
        dac_units = list(getattr(abf, "dacUnits", []) or [])
        level_unit = str(dac_units[0]).strip() if dac_units else ""
    epochs: list[StimulusEpoch] = []
    for index, kind in enumerate(types):
        try:
            start_sample = int(p1s[index])
            end_sample = int(p2s[index])
        except (IndexError, TypeError, ValueError):
            continue
        start_s = float(start_sample) / float(sample_rate_hz)
        duration_s = float(max(0, end_sample - start_sample)) / float(sample_rate_hz)
        try:
            level = float(levels[index])
        except (IndexError, TypeError, ValueError):
            level = 0.0
        epochs.append(
            StimulusEpoch(
                index=index,
                kind=str(kind).strip() or f"epoch_{index}",
                start_s=start_s,
                duration_s=duration_s,
                level=level,
                level_unit=level_unit,
                metadata={
                    "p1_sample": start_sample,
                    "p2_sample": end_sample,
                },
            )
        )
    return tuple(epochs)


def _resolve_overrides(
    *,
    auto: Mapping[str, str],
    override: Mapping[str, str] | None,
    available: Sequence[str],
) -> tuple[dict[str, str], dict[str, str]]:
    final: dict[str, str] = dict(auto)
    applied: dict[str, str] = {}
    if not override:
        return final, applied
    available_set = set(available)
    for key, value in override.items():
        channel = str(key).strip()
        if channel not in available_set:
            raise ValueError(
                f"channel_roles override references unknown channel {key!r}; "
                f"available channels are {list(available)}."
            )
        role_text = str(value).strip()
        if not role_text:
            raise ValueError(f"channel_roles override for {key!r} must be non-empty.")
        applied[channel] = role_text
        final[channel] = role_text
    return final, applied


def read_abf(
    path: str | Path,
    *,
    channel_roles: Mapping[str, str] | None = None,
    units: Mapping[str, str] | None = None,
    mode: RecordingMode | None = None,
    name: str = "ephys",
    pyabf_module: Any | None = None,
) -> EphysRecording:
    """Read an ABF1/ABF2 patch-clamp recording into an :class:`EphysRecording`.

    ``channel_roles``, ``units``, and ``mode`` accept overrides; each applied
    override is recorded in the recording's metadata so the choice is
    auditable.
    """

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"ABF file not found: {source_path}.")
    module = _pyabf_module(pyabf_module)
    abf = module.ABF(str(source_path))

    channel_list, channel_names, recorded_units = _channel_layout(abf)
    sample_rate_hz = float(getattr(abf, "dataRate", 0.0) or getattr(abf, "sampleRate", 0.0) or 0.0)
    if sample_rate_hz <= 0:
        raise ValueError(f"ABF file {source_path} reports a non-positive sample rate.")

    # User-provided unit overrides are applied before mode detection so the
    # rules see the units the caller actually believes are correct.
    effective_units = list(recorded_units)
    unit_overrides: dict[str, str] = {}
    if units is not None:
        available = set(channel_names)
        for key, value in units.items():
            channel = str(key).strip()
            if channel not in available:
                raise ValueError(
                    f"units override references unknown channel {key!r}; "
                    f"available channels are {channel_names}."
                )
            replacement = str(value).strip()
            unit_overrides[channel] = replacement
            effective_units[channel_names.index(channel)] = replacement

    tentative_mode = _detect_mode_from_layout(effective_units)
    auto_roles = _auto_channel_roles(
        names=channel_names, units=effective_units, mode=tentative_mode
    )
    final_roles, role_overrides = _resolve_overrides(
        auto=auto_roles, override=channel_roles, available=channel_names
    )
    detected_mode = detect_recording_mode(
        channel_roles=final_roles,
        channel_units=dict(zip(channel_names, effective_units, strict=True)),
    )
    if mode is not None:
        resolved_mode = str(mode).strip()
    elif detected_mode != "unknown":
        resolved_mode = detected_mode
    else:
        resolved_mode = tentative_mode

    sweep_count = int(getattr(abf, "sweepCount", 1) or 1)
    sweeps: list[Sweep] = []
    cumulative_log: list[dict[str, Any]] = []
    for sweep_index in range(sweep_count):
        values, times = _load_sweep_values(
            abf, sweep_index=sweep_index, channel_list=channel_list
        )
        timeline = Timeline(timestamps_s=times)
        raw_channels = tuple(
            SignalChannel(
                name=name,
                unit=unit,
                metadata={"adc_index": int(channel_list[index])},
            )
            for index, (name, unit) in enumerate(
                zip(channel_names, effective_units, strict=True)
            )
        )
        scaled_values, normalized_channels, log_entries = normalize_signal_units(
            values, raw_channels, roles=final_roles
        )
        cumulative_log.extend(
            {**entry, "sweep_index": int(sweep_index)} for entry in log_entries
        )
        series = TimeSeries(
            values=scaled_values,
            channels=normalized_channels,
            timeline=timeline,
            name=f"{name}_sweep_{int(sweep_index)}",
            provenance={
                "source": {
                    "type": "abf",
                    "path": str(source_path),
                },
                "sweep_index": int(sweep_index),
            },
        )
        epoch_channel = int(channel_list[0])
        epochs = _epochs_for_sweep(
            abf,
            sweep_index=sweep_index,
            channel=epoch_channel,
            sample_rate_hz=sample_rate_hz,
        )
        sweeps.append(
            Sweep(
                index=int(sweep_index),
                series=series,
                epochs=epochs,
                sweep_start_s=float(timeline.start_s),
            )
        )

    sweep_set = SweepSet.from_sweeps(sweeps)

    parser_version = str(getattr(module, "__version__", "unknown"))
    abf_version = str(getattr(abf, "abfVersionString", "")).strip() or None
    protocol_path = str(getattr(abf, "protocolPath", "")).strip() or None
    protocol = str(getattr(abf, "protocol", "")).strip() or None
    stat = source_path.stat()
    provenance = {
        "source": {
            "type": "abf",
            "path": str(source_path),
            "size_bytes": int(stat.st_size),
            "mtime": float(stat.st_mtime),
            "sha256": _file_hash(source_path),
        },
        "parser": {"name": "pyabf", "version": parser_version},
        "abf_version": abf_version,
        "protocol": protocol,
        "protocol_path": protocol_path,
    }

    metadata: dict[str, Any] = {
        "source_type": "abf",
        "name": name,
        "n_sweeps": sweep_set.n_sweeps,
        "channel_indices": [int(channel) for channel in channel_list],
        "recorded_units": list(recorded_units),
        "unit_overrides": unit_overrides,
        "role_overrides": role_overrides,
    }

    return EphysRecording(
        sweeps=sweep_set,
        mode=cast(RecordingMode, resolved_mode),
        channel_roles=final_roles,
        conversion_log=tuple(cumulative_log),
        provenance=provenance,
        metadata=metadata,
    )


__all__ = ["read_abf"]
