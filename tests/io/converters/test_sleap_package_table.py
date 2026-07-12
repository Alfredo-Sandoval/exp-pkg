from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from xpkg.io.converters.sleap_helpers import (
    SleapPackageFormatError,
    build_sleap_label_table,
)


def _write_compact_package(
    path: Path,
    *,
    node_count: int,
    frame_count: int,
    node_order: list[int] | None = None,
) -> None:
    order = node_order if node_order is not None else list(range(node_count))
    metadata = {
        "nodes": [{"name": f"kp_{node_id}"} for node_id in range(node_count)],
        "skeletons": [{"nodes": [{"id": {"id": node_id}} for node_id in order]}],
    }
    frame_dtype = np.dtype(
        [
            ("frame_id", "<i8"),
            ("video", "<i8"),
            ("frame_idx", "<i8"),
            ("instance_id_start", "<i8"),
            ("instance_id_end", "<i8"),
        ]
    )
    frames = np.zeros(frame_count, dtype=frame_dtype)
    frames["frame_id"] = np.arange(frame_count)
    frames["frame_idx"] = np.arange(frame_count)
    frames["instance_id_start"] = np.arange(frame_count)
    frames["instance_id_end"] = np.arange(1, frame_count + 1)

    instance_dtype = np.dtype(
        [
            ("frame_id", "<i8"),
            ("instance_type", "<i8"),
            ("point_id_start", "<i8"),
            ("point_id_end", "<i8"),
        ]
    )
    instances = np.zeros(frame_count, dtype=instance_dtype)
    instances["frame_id"] = np.arange(frame_count)
    instances["point_id_start"] = np.arange(frame_count) * node_count
    instances["point_id_end"] = np.arange(1, frame_count + 1) * node_count

    point_dtype = [("x", "<f8"), ("y", "<f8"), ("visible", "?")]
    points = np.zeros(frame_count * node_count, dtype=point_dtype)
    for frame_index in range(frame_count):
        for offset, node_id in enumerate(order):
            point_index = frame_index * node_count + offset
            x = frame_index * 100 + node_id
            points[point_index] = (x, x + 0.5, True)

    with h5py.File(path, "w") as hdf:
        video = hdf.create_group("video0")
        source_video = video.create_group("source_video")
        source_video.attrs["json"] = json.dumps({"backend": {"filename": "session.mp4"}})
        video.create_dataset("frame_numbers", data=np.arange(frame_count))
        metadata_group = hdf.create_group("metadata")
        metadata_group.attrs["json"] = json.dumps(metadata)
        video_json = json.dumps({"backend": {"dataset": "video0/video"}}).encode()
        hdf.create_dataset("videos_json", data=np.asarray([video_json]))
        hdf.create_dataset("frames", data=frames)
        hdf.create_dataset("instances", data=instances)
        hdf.create_dataset("points", data=points)


def _write_relational_package(path: Path) -> None:
    metadata = {
        "nodes": [{"name": "nose"}, {"name": "tail"}],
        "skeletons": [{"nodes": [0, 1]}],
    }
    frames = np.asarray(
        [(7, 0, 3)],
        dtype=[("frame_id", "<i8"), ("video", "<i8"), ("frame_idx", "<i8")],
    )
    instances = np.asarray(
        [(10, 7, 0), (11, 7, 0)],
        dtype=[("id", "<i8"), ("frame_id", "<i8"), ("instance_type", "<i8")],
    )
    point_dtype = [
        ("x", "<f8"),
        ("y", "<f8"),
        ("visible", "?"),
        ("instance", "<i8"),
        ("node_id", "<i8"),
        ("frame_id", "<i8"),
        ("video", "<i8"),
    ]
    points = np.asarray(
        [
            (99.0, 99.5, True, 10, 0, 7, 0),
            (1.0, 1.5, True, 11, 0, 7, 0),
            (2.0, 2.5, True, 11, 1, 7, 0),
        ],
        dtype=point_dtype,
    )
    with h5py.File(path, "w") as hdf:
        video = hdf.create_group("video0")
        source_video = video.create_group("source_video")
        source_video.attrs["json"] = json.dumps({"backend": {"filename": "session.mp4"}})
        video.create_dataset("frame_numbers", data=np.asarray([3]))
        metadata_group = hdf.create_group("metadata")
        metadata_group.attrs["json"] = json.dumps(metadata)
        video_json = json.dumps({"backend": {"dataset": "video0/video"}}).encode()
        hdf.create_dataset("videos_json", data=np.asarray([video_json]))
        hdf.create_dataset("frames", data=frames)
        hdf.create_dataset("instances", data=instances)
        hdf.create_dataset("points", data=points)


@settings(max_examples=24, deadline=None)
@given(node_count=st.integers(min_value=1, max_value=6), frame_count=st.integers(1, 5))
def test_compact_sleap_package_round_trips_coordinates(
    tmp_path: Path,
    node_count: int,
    frame_count: int,
) -> None:
    package = tmp_path / f"labels-{node_count}-{frame_count}.pkg.slp"
    _write_compact_package(package, node_count=node_count, frame_count=frame_count)

    table = build_sleap_label_table(package.as_posix(), tmp_path.as_posix())

    assert list(table.columns) == [
        "frame",
        *(
            field
            for node_id in range(node_count)
            for field in (f"kp_{node_id}_x", f"kp_{node_id}_y")
        ),
    ]
    assert len(table) == frame_count
    for frame_index, row in table.iterrows():
        assert row["frame"] == f"labeled-data/session/img{frame_index:08d}.png"
        for node_id in range(node_count):
            assert row[f"kp_{node_id}_x"] == frame_index * 100 + node_id
            assert row[f"kp_{node_id}_y"] == frame_index * 100 + node_id + 0.5


def test_sleap_package_respects_declared_node_order(tmp_path: Path) -> None:
    package = tmp_path / "ordered.pkg.slp"
    _write_compact_package(package, node_count=3, frame_count=1, node_order=[2, 0, 1])

    table = build_sleap_label_table(package.as_posix(), tmp_path.as_posix())

    assert list(table.columns) == [
        "frame",
        "kp_2_x",
        "kp_2_y",
        "kp_0_x",
        "kp_0_y",
        "kp_1_x",
        "kp_1_y",
    ]
    assert table.iloc[0].tolist() == [
        "labeled-data/session/img00000000.png",
        2.0,
        2.5,
        0.0,
        0.5,
        1.0,
        1.5,
    ]


def test_relational_sleap_package_selects_instance_with_most_valid_points(
    tmp_path: Path,
) -> None:
    package = tmp_path / "relational.pkg.slp"
    _write_relational_package(package)

    table = build_sleap_label_table(package.as_posix(), tmp_path.as_posix())

    assert table.iloc[0].tolist() == [
        "labeled-data/session/img00000003.png",
        1.0,
        1.5,
        2.0,
        2.5,
    ]


def test_sleap_package_rejects_duplicate_node_links(tmp_path: Path) -> None:
    package = tmp_path / "duplicate.pkg.slp"
    _write_compact_package(package, node_count=2, frame_count=1, node_order=[0, 0])

    with pytest.raises(SleapPackageFormatError, match="Duplicate SLEAP skeleton node id: 0"):
        build_sleap_label_table(package.as_posix(), tmp_path.as_posix())
