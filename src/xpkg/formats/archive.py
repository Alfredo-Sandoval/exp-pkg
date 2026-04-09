"""Compatibility exports for canonical `.xpkg` archives and legacy aliases."""

from __future__ import annotations

from xpkg.io.archive_format import (
    LazyDatasetHandle,
    MaxInstancesExceededError,
    PredictionAppendItem,
    SerializerPredictedInstance,
    append_predictions_archive,
    merge_predictions_archive,
    read_archive,
    summarize_project,
    update_labels_archive,
    validate_project,
    write_archive,
)
from xpkg.io.archive_format.metrics_hdf5 import (
    read_table as read_metrics_table,
)
from xpkg.io.archive_format.metrics_hdf5 import (
    write_table as write_metrics_table,
)

__all__ = [
    "LazyDatasetHandle",
    "MaxInstancesExceededError",
    "PredictionAppendItem",
    "SerializerPredictedInstance",
    "append_predictions_archive",
    "merge_predictions_archive",
    "read_metrics_table",
    "read_archive",
    "summarize_project",
    "update_labels_archive",
    "validate_project",
    "write_metrics_table",
    "write_archive",
]
