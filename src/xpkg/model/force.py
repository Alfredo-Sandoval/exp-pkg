"""Source-neutral force-plate payload objects."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _metadata_pairs(
    items: Sequence[tuple[str, str]],
    *,
    field_name: str,
) -> tuple[tuple[str, str], ...]:
    pairs = tuple((str(key), str(value)) for key, value in items)
    if any(not key for key, _value in pairs):
        raise ValueError(f"{field_name} keys cannot be empty.")
    return pairs


def _force_vector_array(
    value: object,
    *,
    field_name: str,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != expected_shape:
        raise ValueError(f"{field_name} must have shape {expected_shape}, got {array.shape}.")
    return array


def _force_axis_convention(
    items: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    pairs = _metadata_pairs(items, field_name="force.axis_convention")
    convention = {axis.lower(): direction.lower() for axis, direction in pairs}
    expected = {"x": "forward", "y": "lateral", "z": "up"}
    if convention != expected:
        raise ValueError(
            "Force plate axis_convention must explicitly be "
            "x=forward, y=lateral, z=up for the current force feature bank; "
            f"got {dict(pairs)!r}."
        )
    return pairs


@dataclass(frozen=True, slots=True)
class ForcePlateData:
    """Ground-reaction force and moment samples from one recording."""

    sample_times_s: np.ndarray
    force_xyz_N: np.ndarray  # noqa: N815
    plate_names: tuple[str, ...]
    valid_mask: np.ndarray
    sample_rate_hz: float
    units: tuple[tuple[str, str], ...]
    axis_convention: tuple[tuple[str, str], ...]
    provenance: tuple[tuple[str, str], ...]
    moment_xyz_Nm: np.ndarray | None = None  # noqa: N815
    cop_xyz_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        sample_times_s = np.asarray(self.sample_times_s, dtype=np.float64)
        force_xyz_n = np.asarray(self.force_xyz_N, dtype=np.float64)
        plate_names = tuple(str(name) for name in self.plate_names)
        valid_mask = np.asarray(self.valid_mask, dtype=bool)
        sample_rate_hz = float(self.sample_rate_hz)
        units = _metadata_pairs(self.units, field_name="force.units")
        axis_convention = _force_axis_convention(self.axis_convention)
        provenance = _metadata_pairs(self.provenance, field_name="force.provenance")

        if sample_times_s.ndim != 1:
            raise ValueError(
                "ForcePlateData.sample_times_s must have shape (samples,), "
                f"got {sample_times_s.shape}."
            )
        if not np.isfinite(sample_times_s).all():
            raise ValueError("ForcePlateData.sample_times_s must be finite.")
        if not np.all(np.diff(sample_times_s) > 0.0):
            raise ValueError("ForcePlateData.sample_times_s must be strictly increasing.")
        if force_xyz_n.ndim != 3 or force_xyz_n.shape[2] != 3:
            raise ValueError(
                "ForcePlateData.force_xyz_N must have shape (samples, plates, 3), "
                f"got {force_xyz_n.shape}."
            )
        if force_xyz_n.shape[0] != sample_times_s.shape[0]:
            raise ValueError(
                "ForcePlateData.sample_times_s length must match force samples, "
                f"got {sample_times_s.shape[0]} vs {force_xyz_n.shape[0]}."
            )
        if len(plate_names) != force_xyz_n.shape[1]:
            raise ValueError(
                "ForcePlateData.plate_names length must match force plate axis, "
                f"got {len(plate_names)} vs {force_xyz_n.shape[1]}."
            )
        if len(set(plate_names)) != len(plate_names):
            raise ValueError("ForcePlateData.plate_names must be unique.")
        if valid_mask.shape != force_xyz_n.shape[:2]:
            raise ValueError(
                "ForcePlateData.valid_mask shape must match force sample/plate axes, "
                f"got {valid_mask.shape} vs {force_xyz_n.shape[:2]}."
            )
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
            raise ValueError(
                f"ForcePlateData.sample_rate_hz must be positive, got {sample_rate_hz}."
            )

        vector_shape: tuple[int, int, int] = (
            int(force_xyz_n.shape[0]),
            int(force_xyz_n.shape[1]),
            int(force_xyz_n.shape[2]),
        )
        moment_xyz_nm = None
        if self.moment_xyz_Nm is not None:
            moment_xyz_nm = _force_vector_array(
                self.moment_xyz_Nm,
                field_name="moment_xyz_Nm",
                expected_shape=vector_shape,
            )
        cop_xyz_m = None
        if self.cop_xyz_m is not None:
            cop_xyz_m = _force_vector_array(
                self.cop_xyz_m,
                field_name="cop_xyz_m",
                expected_shape=vector_shape,
            )

        object.__setattr__(self, "sample_times_s", sample_times_s)
        object.__setattr__(self, "force_xyz_N", force_xyz_n)
        object.__setattr__(self, "plate_names", plate_names)
        object.__setattr__(self, "valid_mask", valid_mask)
        object.__setattr__(self, "sample_rate_hz", sample_rate_hz)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "axis_convention", axis_convention)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "moment_xyz_Nm", moment_xyz_nm)
        object.__setattr__(self, "cop_xyz_m", cop_xyz_m)

    @property
    def n_samples(self) -> int:
        return int(self.force_xyz_N.shape[0])

    @property
    def n_plates(self) -> int:
        return int(self.force_xyz_N.shape[1])


__all__ = ["ForcePlateData"]
