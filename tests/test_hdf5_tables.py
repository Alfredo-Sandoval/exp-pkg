from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest

from xpkg.api import (
    read_hdf5_table,
    read_hdf5_table_group,
    write_hdf5_table,
    write_hdf5_table_group,
)


def test_hdf5_table_round_trips_columns_and_dtypes(tmp_path) -> None:
    path = tmp_path / "tables.h5"
    df = pd.DataFrame(
        {
            "label": ["a", "b"],
            "count": np.asarray([1, 2], dtype=np.int64),
            "value": np.asarray([1.25, 2.5], dtype=np.float64),
            "flag": [True, False],
        }
    )

    write_hdf5_table(path=path, group_name="features/control", df=df)
    restored = read_hdf5_table(path=path, group_name="features/control")

    assert restored.columns.tolist() == ["label", "count", "value", "flag"]
    assert restored["count"].to_numpy().dtype == np.dtype("int64")
    assert restored["value"].to_numpy().dtype == np.dtype("float64")
    assert restored["flag"].to_numpy().dtype == np.dtype("bool")
    pd.testing.assert_frame_equal(restored, df)


def test_hdf5_table_group_replaces_existing_table(tmp_path) -> None:
    path = tmp_path / "tables.h5"
    with h5py.File(path, "w") as handle:
        write_hdf5_table_group(handle, "hmm/states", pd.DataFrame({"state": [1, 2]}))
        write_hdf5_table_group(handle, "hmm/states", pd.DataFrame({"state": [3]}))
        restored = read_hdf5_table_group(handle, "hmm/states")

    pd.testing.assert_frame_equal(restored, pd.DataFrame({"state": [3]}))


def test_hdf5_table_rejects_missing_values(tmp_path) -> None:
    path = tmp_path / "tables.h5"
    df = pd.DataFrame({"bad": [1.0, np.nan]})

    with pytest.raises(ValueError, match="missing values are not representable"):
        write_hdf5_table(path=path, group_name="features/control", df=df)


def test_hdf5_table_requires_column_metadata(tmp_path) -> None:
    path = tmp_path / "tables.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("table")
        group.create_dataset("value", data=np.asarray([1.0]))

    with h5py.File(path, "r") as handle:
        with pytest.raises(KeyError, match="columns"):
            read_hdf5_table_group(handle, "table")
