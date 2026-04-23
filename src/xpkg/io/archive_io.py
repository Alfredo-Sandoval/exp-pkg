"""Canonical archive IO surface for `.xpkg` compatibility archives."""

from __future__ import annotations

from xpkg.io.archive_format import (
    LazyDatasetHandle,
    MaxInstancesExceededError,
    PredictionAppendItem,
    SerializerPredictedInstance,
    append_predictions_archive,
    merge_predictions_archive,
    read_archive,
    update_labels_archive,
    write_archive,
)
from xpkg.io.archive_format import (
    summarize_project as summarize_archive,
)
from xpkg.io.archive_format import (
    validate_project as validate_archive,
)
from xpkg.io.archive_format.metrics_hdf5 import (
    read_table as read_metrics_table,
)
from xpkg.io.archive_format.metrics_hdf5 import (
    write_table as write_metrics_table,
)
from xpkg.io.project_metadata import load_archive_metadata_field, save_archive_metadata_field

__all__ = [
    "LazyDatasetHandle",
    "MaxInstancesExceededError",
    "PredictionAppendItem",
    "SerializerPredictedInstance",
    "append_predictions_archive",
    "load_archive_metadata_field",
    "merge_predictions_archive",
    "read_archive",
    "read_metrics_table",
    "save_archive_metadata_field",
    "summarize_archive",
    "update_labels_archive",
    "validate_archive",
    "write_archive",
    "write_metrics_table",
]
