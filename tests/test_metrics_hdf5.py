from __future__ import annotations

import h5py
import pandas as pd
import pytest

from xpkg.io.metrics_hdf5 import (
    MetricsError,
    MetricsReadError,
    MetricStore,
    MissingMetricsGroupError,
    MissingMetricsTableError,
    _compute_chunk_shape,
    _compute_string_chunk,
    _partition_columns,
    has_metrics_group,
    iter_tables,
    read_table,
    write_table,
    write_table_to_handle,
)


def test_metrics_group_detection(tmp_path) -> None:
    path = tmp_path / "metrics.h5"
    assert has_metrics_group(path) is False
    with h5py.File(path, "w") as handle:
        handle.create_group("metrics")
    assert has_metrics_group(path) is True


def test_metrics_table_roundtrip_mixed_columns(tmp_path) -> None:
    path = tmp_path / "metrics.h5"
    df = pd.DataFrame({"epoch": [1, 2], "loss": [0.5, 0.25], "split": ["train", "val"]})

    write_table(path, "training", df)
    result = read_table(path, "training")

    assert list(result.columns) == ["epoch", "loss", "split"]
    assert len(result) == 2
    assert list(result["split"]) == ["train", "val"]


def test_metrics_append_and_replace(tmp_path) -> None:
    path = tmp_path / "metrics.h5"
    write_table(path, "data", pd.DataFrame({"value": [1, 2]}))
    write_table(path, "data", pd.DataFrame({"value": [3]}), mode="append")
    assert list(read_table(path, "data")["value"]) == [1.0, 2.0, 3.0]

    write_table(path, "data", pd.DataFrame({"value": [4]}), mode="replace")
    assert list(read_table(path, "data")["value"]) == [4.0]


def test_write_table_to_handle_uses_open_file(tmp_path) -> None:
    path = tmp_path / "metrics.h5"
    with h5py.File(path, "w") as handle:
        write_table_to_handle(handle, "labels", pd.DataFrame({"name": ["nose"]}))
        assert "labels" in handle["metrics"]

    assert list(read_table(path, "labels")["name"]) == ["nose"]


def test_missing_metrics_errors(tmp_path) -> None:
    path = tmp_path / "metrics.h5"
    with h5py.File(path, "w"):
        pass

    with pytest.raises(MissingMetricsGroupError):
        read_table(path, "training")
    with pytest.raises(MissingMetricsGroupError):
        list(iter_tables(path))

    with h5py.File(path, "a") as handle:
        handle.create_group("metrics")
    with pytest.raises(MissingMetricsTableError):
        read_table(path, "training")


def test_iter_tables_and_store_helpers(tmp_path) -> None:
    path = tmp_path / "metrics.h5"
    store = MetricStore(path)
    store.write_table("a", pd.DataFrame({"value": [1]}))
    store.write_table("b", pd.DataFrame({"value": [2]}))

    assert store.has_metrics_group() is True
    assert set(store.iter_tables()) == {"a", "b"}
    assert len(store.read_table("a")) == 1


def test_chunk_and_partition_contracts() -> None:
    assert _compute_chunk_shape(100, 10) is not None
    assert _compute_chunk_shape(0, 10) is None
    assert _compute_string_chunk(100) is not None
    assert _compute_string_chunk(0) is None

    partitioned = _partition_columns(pd.DataFrame({"a": [1], "name": ["nose"]}))
    assert partitioned.numeric_names == ["a"]
    assert partitioned.numeric_values.shape == (1, 1)
    assert "name" in partitioned.string_columns


def test_metric_store_static_contracts() -> None:
    assert MetricStore._normalize_prediction_errors(
        pd.DataFrame({"bodypart": ["nose"], "error": [1.5]})
    ) is not None
    assert MetricStore._normalize_prediction_errors(pd.DataFrame()) is None
    assert MetricStore._coerce_frame_index("file.png#frame=42") == 42
    assert MetricStore._coerce_frame_index("123") == 123
    assert MetricStore._coerce_float(3.14) == pytest.approx(3.14)
    assert MetricStore._coerce_int(3.7) == 4
    assert MetricStore._table_has_epoch_v2(pd.DataFrame({"epoch": [1]})) is True
    assert MetricStore._match_column(["train_loss"], ("train", "loss")) == "train_loss"

    with pytest.raises(TypeError):
        MetricStore._coerce_float(None)
    with pytest.raises(ValueError):
        MetricStore._coerce_frame_index("invalid")


def test_public_error_hierarchy() -> None:
    assert issubclass(MissingMetricsGroupError, MetricsError)
    assert issubclass(MissingMetricsTableError, MetricsError)
    assert issubclass(MetricsReadError, MetricsError)
