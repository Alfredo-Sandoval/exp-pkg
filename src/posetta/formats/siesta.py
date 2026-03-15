"""Public format exports for the native `.siesta` archive format."""

from __future__ import annotations

from posetta.io.siesta_format import (
    LazyDatasetHandle,
    MaxInstancesExceededError,
    PredictionAppendItem,
    SerializerPredictedInstance,
    append_predictions_siesta,
    merge_predictions_siesta,
    read_siesta,
    summarize_project,
    update_labels_siesta,
    validate_project,
    write_siesta,
)
from posetta.io.siesta_format.metrics_hdf5 import (
    read_table as read_metrics_table,
)
from posetta.io.siesta_format.metrics_hdf5 import (
    write_table as write_metrics_table,
)

__all__ = [
    "LazyDatasetHandle",
    "MaxInstancesExceededError",
    "PredictionAppendItem",
    "SerializerPredictedInstance",
    "append_predictions_siesta",
    "merge_predictions_siesta",
    "read_metrics_table",
    "read_siesta",
    "summarize_project",
    "update_labels_siesta",
    "validate_project",
    "write_metrics_table",
    "write_siesta",
]
