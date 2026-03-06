from __future__ import annotations

import posetta
from posetta.adapters import (
    convert_dlc_csv,
    convert_dlc_h5,
    convert_dlc_project,
    convert_sleap_package,
)
from posetta.formats import (
    LazyDatasetHandle,
    MaxInstancesExceededError,
    PredictionAppendItem,
    SerializerPredictedInstance,
    append_predictions_siesta,
    merge_predictions_siesta,
    read_metrics_table,
    read_siesta,
    summarize_project,
    update_labels_siesta,
    validate_project,
    write_metrics_table,
    write_siesta,
)


def test_public_exports_are_callable() -> None:
    assert posetta.__version__
    assert posetta.adapters is not None
    assert posetta.formats is not None
    assert LazyDatasetHandle is not None
    assert PredictionAppendItem is not None
    assert SerializerPredictedInstance is not None
    assert MaxInstancesExceededError is not None
    assert callable(append_predictions_siesta)
    assert callable(merge_predictions_siesta)
    assert callable(read_metrics_table)
    assert callable(read_siesta)
    assert callable(summarize_project)
    assert callable(update_labels_siesta)
    assert callable(validate_project)
    assert callable(write_metrics_table)
    assert callable(write_siesta)
    assert callable(convert_dlc_csv)
    assert callable(convert_dlc_h5)
    assert callable(convert_dlc_project)
    assert callable(convert_sleap_package)
