"""Vicon force-platform extraction into source-neutral force payloads."""

from __future__ import annotations

import numpy as np

from xpkg.model import ForcePlateData, ViconRecording

_SUPPORTED_AXIS_CONVENTION: tuple[tuple[str, str], ...] = (
    ("x", "forward"),
    ("y", "lateral"),
    ("z", "up"),
)
_TYPE_2_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("Force", "Fx"),
    ("Force", "Fy"),
    ("Force", "Fz"),
    ("Moment", "Mx"),
    ("Moment", "My"),
    ("Moment", "Mz"),
)
_METERS_PER_MILLIMETER = 0.001
MIN_COP_VERTICAL_FORCE_N = 1.0


def _require_axis_convention(
    axis_convention: tuple[tuple[str, str], ...] | None,
) -> tuple[tuple[str, str], ...]:
    if axis_convention is None:
        raise ValueError("Vicon force mapping requires an explicit axis_convention.")
    normalized = tuple(
        (str(axis).lower(), str(direction).lower())
        for axis, direction in axis_convention
    )
    if normalized != _SUPPORTED_AXIS_CONVENTION:
        raise ValueError(
            "Unsupported Vicon force axis_convention; expected "
            "x=forward, y=lateral, z=up."
        )
    return _SUPPORTED_AXIS_CONVENTION


def _expected_label(kind: str, component: str, *, plate_number: int) -> str:
    return f"{kind}.{component}{plate_number}".lower()


def _validate_channel_label(label: str, kind: str, component: str, *, plate_number: int) -> None:
    expected = _expected_label(kind, component, plate_number=plate_number)
    normalized = str(label).strip().lower()
    if normalized != expected:
        raise ValueError(
            "Vicon FORCE_PLATFORM.CHANNEL mapped to unexpected analog label: "
            f"expected {expected!r}, got {label!r}."
        )


def _validate_force_unit(unit: str, *, label: str) -> None:
    if str(unit).strip().lower() != "n":
        raise ValueError(f"Vicon force channel {label!r} must use unit 'N', got {unit!r}.")


def _moment_scale_to_nm(unit: str, *, label: str) -> float:
    normalized = str(unit).strip().lower().replace(" ", "")
    if normalized == "nmm":
        return 0.001
    if normalized in {"nm", "n*m"}:
        return 1.0
    raise ValueError(
        f"Vicon moment channel {label!r} must use unit 'Nmm' or 'N*m', got {unit!r}."
    )


def _channel_indices_from_metadata(
    channels: np.ndarray,
    *,
    n_channels: int,
    plate_number: int,
) -> np.ndarray:
    indices = np.asarray(channels, dtype=np.int64) - 1
    if indices.shape != (6,):
        raise ValueError(
            f"FORCE_PLATFORM.CHANNEL for plate {plate_number} must contain 6 channels, "
            f"got shape {indices.shape}."
        )
    if np.any(indices < 0) or np.any(indices >= n_channels):
        raise ValueError(
            f"FORCE_PLATFORM.CHANNEL for plate {plate_number} contains channel indices "
            f"outside analog channel range 1..{n_channels}."
        )
    if len(set(int(index) for index in indices)) != 6:
        raise ValueError(f"FORCE_PLATFORM.CHANNEL for plate {plate_number} contains duplicates.")
    return indices


def _extract_type_2_plate_values(
    recording: ViconRecording,
    *,
    plate_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    analog = recording.analog
    force_platform = recording.force_platform
    if analog is None or force_platform is None:
        raise ValueError("Vicon force mapping requires analog data and FORCE_PLATFORM metadata.")

    plate_number = plate_index + 1
    indices = _channel_indices_from_metadata(
        force_platform.channels[plate_index],
        n_channels=analog.n_channels,
        plate_number=plate_number,
    )
    force_columns: list[np.ndarray] = []
    moment_columns: list[np.ndarray] = []
    for component_index, (kind, component) in enumerate(_TYPE_2_COMPONENTS):
        channel_index = int(indices[component_index])
        label = analog.channel_names[channel_index]
        _validate_channel_label(label, kind, component, plate_number=plate_number)
        unit = analog.channel_units[channel_index]
        values = analog.values[:, channel_index]
        if kind == "Force":
            _validate_force_unit(unit, label=label)
            force_columns.append(values)
        else:
            moment_columns.append(values * _moment_scale_to_nm(unit, label=label))
    return np.column_stack(force_columns), np.column_stack(moment_columns)


def _unit_vector(vector: np.ndarray, *, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Vicon force-platform {name} vector is degenerate.")
    return np.asarray(vector, dtype=np.float64) / norm


def _plate_surface_frame(corners_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corners = np.asarray(corners_m, dtype=np.float64)
    if corners.shape != (4, 3):
        raise ValueError(
            f"Vicon force-platform corners must have shape (4, 3), got {corners.shape}."
        )
    if not np.isfinite(corners).all():
        raise ValueError("Vicon force-platform corners must be finite.")

    surface_origin_m = np.mean(corners, axis=0)
    positive_x_edge_midpoint = 0.5 * (corners[0] + corners[3])
    negative_x_edge_midpoint = 0.5 * (corners[1] + corners[2])
    positive_y_edge_midpoint = 0.5 * (corners[0] + corners[1])
    negative_y_edge_midpoint = 0.5 * (corners[2] + corners[3])
    x_axis = _unit_vector(
        positive_x_edge_midpoint - negative_x_edge_midpoint,
        name="surface x-axis",
    )
    y_span = positive_y_edge_midpoint - negative_y_edge_midpoint
    z_axis = _unit_vector(np.cross(x_axis, y_span), name="surface normal")
    y_axis = _unit_vector(np.cross(z_axis, x_axis), name="surface y-axis")
    return surface_origin_m, x_axis, y_axis


def _type_2_plate_cop_xyz_m(
    *,
    force_xyz_n: np.ndarray,
    moment_xyz_nm: np.ndarray,
    origin_mm: np.ndarray,
    corners_mm: np.ndarray,
) -> np.ndarray:
    """Compute AMTI/Kistler type-2 surface COP in global lab meters.

    Forces are N, moments are N*m, and FORCE_PLATFORM geometry is C3D mm. The
    type-2 formula uses the platform origin offset before projecting the surface
    COP from plate-local coordinates through the corner-derived surface frame.
    """

    origin_m = np.asarray(origin_mm, dtype=np.float64) * _METERS_PER_MILLIMETER
    if origin_m.shape != (3,):
        raise ValueError(f"Vicon FORCE_PLATFORM.ORIGIN must have shape (3,), got {origin_m.shape}.")
    if not np.isfinite(origin_m).all():
        raise ValueError("Vicon FORCE_PLATFORM.ORIGIN values must be finite.")
    surface_origin_m, x_axis, y_axis = _plate_surface_frame(
        np.asarray(corners_mm, dtype=np.float64) * _METERS_PER_MILLIMETER
    )

    force = np.asarray(force_xyz_n, dtype=np.float64)
    moment = np.asarray(moment_xyz_nm, dtype=np.float64)
    if force.shape != moment.shape or force.ndim != 2 or force.shape[1] != 3:
        raise ValueError(
            "Vicon type-2 COP requires force and moment arrays with shape (samples, 3), "
            f"got {force.shape} and {moment.shape}."
        )

    cop_xyz_m = np.full(force.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(force).all(axis=1) & np.isfinite(moment).all(axis=1)
    loaded = finite & (np.abs(force[:, 2]) >= MIN_COP_VERTICAL_FORCE_N)
    if not np.any(loaded):
        return cop_xyz_m

    fx = force[loaded, 0]
    fy = force[loaded, 1]
    fz = force[loaded, 2]
    mx = moment[loaded, 0]
    my = moment[loaded, 1]
    x0, y0, z0 = origin_m
    cop_x_local_m = (-my - fx * z0) / fz + x0
    cop_y_local_m = (mx - fy * z0) / fz + y0
    cop_xyz_m[loaded] = (
        surface_origin_m
        + cop_x_local_m[:, np.newaxis] * x_axis
        + cop_y_local_m[:, np.newaxis] * y_axis
    )
    return cop_xyz_m


def build_force_plate_data_from_vicon_recording(
    recording: ViconRecording,
    *,
    axis_convention: tuple[tuple[str, str], ...] | None,
) -> ForcePlateData:
    """Build source-neutral force-plate data from explicit Vicon C3D metadata."""

    resolved_axis_convention = _require_axis_convention(axis_convention)
    if recording.source_type != "c3d":
        raise ValueError("Vicon force mapping requires recording.source_type == 'c3d'.")
    if recording.analog is None:
        raise ValueError("Vicon force mapping requires analog data.")
    if recording.force_platform is None:
        raise ValueError("Vicon force mapping requires typed FORCE_PLATFORM metadata.")

    force_platform = recording.force_platform
    unsupported_types = tuple(
        plate_type for plate_type in force_platform.plate_types if plate_type != 2
    )
    if unsupported_types:
        raise ValueError(
            "Unsupported FORCE_PLATFORM.TYPE values for Vicon force mapping: "
            f"{unsupported_types}; only type 2 is supported."
        )

    plate_forces: list[np.ndarray] = []
    plate_moments: list[np.ndarray] = []
    plate_cops: list[np.ndarray] = []
    for plate_index in range(force_platform.used):
        force_xyz_n, moment_xyz_nm = _extract_type_2_plate_values(
            recording,
            plate_index=plate_index,
        )
        plate_forces.append(force_xyz_n)
        plate_moments.append(moment_xyz_nm)
        plate_cops.append(
            _type_2_plate_cop_xyz_m(
                force_xyz_n=force_xyz_n,
                moment_xyz_nm=moment_xyz_nm,
                origin_mm=force_platform.origins[plate_index],
                corners_mm=force_platform.corners[plate_index],
            )
        )

    force_xyz_n = np.stack(plate_forces, axis=1)
    moment_xyz_nm = np.stack(plate_moments, axis=1)
    cop_xyz_m = np.stack(plate_cops, axis=1)
    sample_times_s = np.arange(recording.analog.n_samples, dtype=np.float64) / float(
        recording.analog.fps
    )
    valid_mask = np.asarray(
        np.isfinite(force_xyz_n).all(axis=2) & np.isfinite(moment_xyz_nm).all(axis=2),
        dtype=bool,
    )
    return ForcePlateData(
        sample_times_s=sample_times_s,
        force_xyz_N=force_xyz_n,
        plate_names=tuple(f"plate_{index + 1}" for index in range(force_platform.used)),
        valid_mask=valid_mask,
        sample_rate_hz=float(recording.analog.fps),
        units=(("force", "N"), ("moment", "N*m")),
        axis_convention=resolved_axis_convention,
        provenance=(
            *force_platform.provenance,
            ("mapper", "xpkg.io.readers.vicon.force.build_force_plate_data_from_vicon_recording"),
        ),
        moment_xyz_Nm=moment_xyz_nm,
        cop_xyz_m=cop_xyz_m,
    )


__all__ = ["build_force_plate_data_from_vicon_recording"]
