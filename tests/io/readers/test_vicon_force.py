from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xpkg.model import ViconAnalogData, ViconForcePlatformMetadata, ViconRecording

_AXIS_CONVENTION = (("x", "forward"), ("y", "lateral"), ("z", "up"))
_MILLIMETERS_PER_METER = 1000.0


def _recording(
    *,
    analog: ViconAnalogData | None,
    force_platform: ViconForcePlatformMetadata | None,
    source_type: str = "c3d",
) -> ViconRecording:
    return ViconRecording(
        path=Path("trial.c3d"),
        source_type=source_type,
        fps=100,
        marker_names=("LASI",),
        source_marker_labels=("LASI",),
        positions=np.zeros((2, 1, 3), dtype=np.float64),
        marker_valid=np.ones((2, 1), dtype=bool),
        frame_offset=100,
        analog=analog,
        force_platform=force_platform,
    )


def _analog() -> ViconAnalogData:
    values = np.array(
        [
            [1.0, 2.0, 100.0, 1000.0, 2000.0, 3000.0],
            [2.0, 3.0, 110.0, 1100.0, 2100.0, 3100.0],
            [3.0, 4.0, 120.0, 1200.0, 2200.0, 3200.0],
            [4.0, 5.0, 130.0, 1300.0, 2300.0, 3300.0],
        ],
        dtype=np.float64,
    )
    return ViconAnalogData(
        fps=1000,
        samples_per_frame=2,
        channel_names=(
            "Force.Fx1",
            "Force.Fy1",
            "Force.Fz1",
            "Moment.Mx1",
            "Moment.My1",
            "Moment.Mz1",
        ),
        values=values,
        channel_units=("N", "N", "N", "Nmm", "Nmm", "Nmm"),
    )


def _analog_from_force_moment(
    force_xyz_n: np.ndarray,
    moment_xyz_nm: np.ndarray,
) -> ViconAnalogData:
    force = np.asarray(force_xyz_n, dtype=np.float64)
    moment_nmm = np.asarray(moment_xyz_nm, dtype=np.float64) * _MILLIMETERS_PER_METER
    return ViconAnalogData(
        fps=1000,
        samples_per_frame=1,
        channel_names=(
            "Force.Fx1",
            "Force.Fy1",
            "Force.Fz1",
            "Moment.Mx1",
            "Moment.My1",
            "Moment.Mz1",
        ),
        values=np.column_stack((force, moment_nmm)),
        channel_units=("N", "N", "N", "Nmm", "Nmm", "Nmm"),
    )


def _surface_corners_mm(
    *,
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    x_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    y_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> np.ndarray:
    center = np.asarray(center_m, dtype=np.float64)
    x = np.asarray(x_axis, dtype=np.float64)
    y = np.asarray(y_axis, dtype=np.float64)
    half_x_m = 0.30
    half_y_m = 0.20
    corners_m = np.array(
        [
            center + half_x_m * x + half_y_m * y,
            center - half_x_m * x + half_y_m * y,
            center - half_x_m * x - half_y_m * y,
            center + half_x_m * x - half_y_m * y,
        ],
        dtype=np.float64,
    )
    return corners_m[np.newaxis, :, :] * _MILLIMETERS_PER_METER


def _force_platform(
    *,
    plate_type: int = 2,
    channels: np.ndarray | None = None,
    corners: np.ndarray | None = None,
    origins: np.ndarray | None = None,
) -> ViconForcePlatformMetadata:
    return ViconForcePlatformMetadata(
        used=1,
        plate_types=(plate_type,),
        channels=np.asarray([[1, 2, 3, 4, 5, 6]] if channels is None else channels, dtype=np.int64),
        corners=_surface_corners_mm() if corners is None else np.asarray(corners, dtype=np.float64),
        origins=np.zeros((1, 3), dtype=np.float64)
        if origins is None
        else np.asarray(origins, dtype=np.float64),
        provenance=(("source_path", "trial.c3d"), ("reader_version", "test")),
    )


def test_force_plate_data_validates_source_neutral_shapes() -> None:
    from xpkg.model import ForcePlateData

    force_xyz_n = np.zeros((6, 2, 3), dtype=float)
    force_xyz_n[:, 0, 2] = 650.0
    force_xyz_n[:, 1, 2] = 120.0
    moment_xyz_nm = np.ones((6, 2, 3), dtype=float)

    force = ForcePlateData(
        sample_times_s=np.arange(6, dtype=float) / 1000.0,
        force_xyz_N=force_xyz_n,
        plate_names=("plate_1", "plate_2"),
        valid_mask=np.ones((6, 2), dtype=bool),
        sample_rate_hz=1000.0,
        units=(("force", "N"), ("moment", "N*m")),
        axis_convention=_AXIS_CONVENTION,
        provenance=(("source", "synthetic"),),
        moment_xyz_Nm=moment_xyz_nm,
        cop_xyz_m=np.zeros((6, 2, 3), dtype=float),
    )

    assert force.n_samples == 6
    assert force.n_plates == 2
    assert force.force_xyz_N.shape == (6, 2, 3)
    assert force.moment_xyz_Nm is not None
    assert force.cop_xyz_m is not None


def test_force_plate_data_rejects_mismatched_shapes() -> None:
    from xpkg.model import ForcePlateData

    with pytest.raises(ValueError, match="sample_times_s length must match force samples"):
        ForcePlateData(
            sample_times_s=np.arange(5, dtype=float),
            force_xyz_N=np.zeros((6, 2, 3), dtype=float),
            plate_names=("plate_1", "plate_2"),
            valid_mask=np.ones((6, 2), dtype=bool),
            sample_rate_hz=1000.0,
            units=(("force", "N"),),
            axis_convention=_AXIS_CONVENTION,
            provenance=(("source", "synthetic"),),
        )

    with pytest.raises(ValueError, match="moment_xyz_Nm must have shape"):
        ForcePlateData(
            sample_times_s=np.arange(6, dtype=float),
            force_xyz_N=np.zeros((6, 2, 3), dtype=float),
            plate_names=("plate_1", "plate_2"),
            valid_mask=np.ones((6, 2), dtype=bool),
            sample_rate_hz=1000.0,
            units=(("force", "N"),),
            axis_convention=_AXIS_CONVENTION,
            provenance=(("source", "synthetic"),),
            moment_xyz_Nm=np.zeros((6, 1, 3), dtype=float),
        )


def test_build_force_plate_data_from_vicon_recording_maps_type_2_channels() -> None:
    from xpkg.io.readers import build_force_plate_data_from_vicon_recording

    force = build_force_plate_data_from_vicon_recording(
        _recording(analog=_analog(), force_platform=_force_platform()),
        axis_convention=_AXIS_CONVENTION,
    )

    assert force.plate_names == ("plate_1",)
    assert force.force_xyz_N.shape == (4, 1, 3)
    np.testing.assert_allclose(force.force_xyz_N[:, 0, 2], [100.0, 110.0, 120.0, 130.0])
    assert force.moment_xyz_Nm is not None
    np.testing.assert_allclose(force.moment_xyz_Nm[0, 0], [1.0, 2.0, 3.0])
    assert force.cop_xyz_m is not None
    assert force.cop_xyz_m.shape == (4, 1, 3)
    assert force.valid_mask.tolist() == [[True], [True], [True], [True]]
    np.testing.assert_allclose(force.sample_times_s, [0.0, 0.001, 0.002, 0.003])


def test_type_2_force_plate_cop_recovers_known_global_surface_point() -> None:
    from xpkg.io.readers import build_force_plate_data_from_vicon_recording

    target_local_m = np.array([0.12, -0.08, 0.0], dtype=np.float64)
    origin_m = np.array([0.01, -0.02, -0.04], dtype=np.float64)
    force_row_n = np.array([20.0, -30.0, 500.0], dtype=np.float64)
    moment_row_nm = np.array(
        [
            (target_local_m[1] - origin_m[1]) * force_row_n[2]
            + force_row_n[1] * origin_m[2],
            -(target_local_m[0] - origin_m[0]) * force_row_n[2]
            - force_row_n[0] * origin_m[2],
            0.0,
        ],
        dtype=np.float64,
    )
    force_xyz_n = np.repeat(force_row_n[np.newaxis, :], 2, axis=0)
    moment_xyz_nm = np.repeat(moment_row_nm[np.newaxis, :], 2, axis=0)
    force = build_force_plate_data_from_vicon_recording(
        _recording(
            analog=_analog_from_force_moment(force_xyz_n, moment_xyz_nm),
            force_platform=_force_platform(
                corners=_surface_corners_mm(
                    center_m=(1.0, 2.0, 0.30),
                    x_axis=(0.0, 1.0, 0.0),
                    y_axis=(-1.0, 0.0, 0.0),
                ),
                origins=origin_m[np.newaxis, :] * _MILLIMETERS_PER_METER,
            ),
        ),
        axis_convention=_AXIS_CONVENTION,
    )

    assert force.cop_xyz_m is not None
    np.testing.assert_allclose(force.cop_xyz_m[:, 0], [[1.08, 2.12, 0.30]] * 2, atol=1e-12)


def test_type_2_force_plate_cop_is_nan_for_low_vertical_force() -> None:
    from xpkg.io.readers import build_force_plate_data_from_vicon_recording

    force = build_force_plate_data_from_vicon_recording(
        _recording(
            analog=_analog_from_force_moment(
                np.array([[0.0, 0.0, 0.0], [5.0, 3.0, 0.5]], dtype=np.float64),
                np.zeros((2, 3), dtype=np.float64),
            ),
            force_platform=_force_platform(),
        ),
        axis_convention=_AXIS_CONVENTION,
    )

    assert force.cop_xyz_m is not None
    assert force.cop_xyz_m.shape == (2, 1, 3)
    assert np.isnan(force.cop_xyz_m).all()


def test_force_platform_metadata_is_required() -> None:
    from xpkg.io.readers import build_force_plate_data_from_vicon_recording

    with pytest.raises(ValueError, match="FORCE_PLATFORM metadata"):
        build_force_plate_data_from_vicon_recording(
            _recording(analog=_analog(), force_platform=None),
            axis_convention=_AXIS_CONVENTION,
        )


def test_unsupported_force_platform_type_fails() -> None:
    from xpkg.io.readers import build_force_plate_data_from_vicon_recording

    with pytest.raises(ValueError, match="only type 2 is supported"):
        build_force_plate_data_from_vicon_recording(
            _recording(analog=_analog(), force_platform=_force_platform(plate_type=1)),
            axis_convention=_AXIS_CONVENTION,
        )


def test_channel_indices_out_of_range_fail() -> None:
    from xpkg.io.readers import build_force_plate_data_from_vicon_recording

    with pytest.raises(ValueError, match="outside analog channel range"):
        build_force_plate_data_from_vicon_recording(
            _recording(
                analog=_analog(),
                force_platform=_force_platform(channels=np.array([[1, 2, 3, 4, 5, 99]])),
            ),
            axis_convention=_AXIS_CONVENTION,
        )


def test_wrong_force_unit_fails() -> None:
    from xpkg.io.readers import build_force_plate_data_from_vicon_recording

    analog = ViconAnalogData(
        fps=1000,
        samples_per_frame=2,
        channel_names=_analog().channel_names,
        values=_analog().values,
        channel_units=("V", "N", "N", "Nmm", "Nmm", "Nmm"),
    )

    with pytest.raises(ValueError, match="must use unit 'N'"):
        build_force_plate_data_from_vicon_recording(
            _recording(analog=analog, force_platform=_force_platform()),
            axis_convention=_AXIS_CONVENTION,
        )


def test_mapped_channel_label_mismatch_fails() -> None:
    from xpkg.io.readers import build_force_plate_data_from_vicon_recording

    analog = ViconAnalogData(
        fps=1000,
        samples_per_frame=2,
        channel_names=(
            "Force.Fx1",
            "Force.Fy1",
            "Force.Fz1",
            "Moment.Mx1",
            "Moment.My1",
            "Moment.Mz2",
        ),
        values=_analog().values,
        channel_units=("N", "N", "N", "Nmm", "Nmm", "Nmm"),
    )

    with pytest.raises(ValueError, match="unexpected analog label"):
        build_force_plate_data_from_vicon_recording(
            _recording(analog=analog, force_platform=_force_platform()),
            axis_convention=_AXIS_CONVENTION,
        )


def test_missing_axis_convention_fails() -> None:
    from xpkg.io.readers import build_force_plate_data_from_vicon_recording

    with pytest.raises(ValueError, match="requires an explicit axis_convention"):
        build_force_plate_data_from_vicon_recording(
            _recording(analog=_analog(), force_platform=_force_platform()),
            axis_convention=None,
        )
