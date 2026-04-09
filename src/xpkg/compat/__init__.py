"""Compatibility surface for canonical `.xpkg` archives and legacy aliases."""

from __future__ import annotations

from xpkg.io.archive_io import (
    LazyDatasetHandle,
    MaxInstancesExceededError,
    PredictionAppendItem,
    SerializerPredictedInstance,
    append_predictions_archive,
    merge_predictions_archive,
    read_archive,
    read_metrics_table,
    summarize_archive,
    update_labels_archive,
    validate_archive,
    write_archive,
    write_metrics_table,
)
from xpkg.io.archive_store import (
    ArchiveStore,
    create_archive_store,
    create_xpkg_store,
    open_archive_store,
)

read_xpkg = read_archive
write_xpkg = write_archive
update_labels_xpkg = update_labels_archive
append_predictions_xpkg = append_predictions_archive
merge_predictions_xpkg = merge_predictions_archive
summarize_xpkg = summarize_archive
validate_xpkg = validate_archive
read_sta = read_archive
write_sta = write_archive
update_labels_sta = update_labels_archive
append_predictions_sta = append_predictions_archive
merge_predictions_sta = merge_predictions_archive
summarize_sta = summarize_archive
validate_sta = validate_archive

summarize_project = summarize_archive
validate_project = validate_archive
create_store_from_archive = create_archive_store
create_store_from_xpkg = create_xpkg_store
create_store_from_sta = create_archive_store
open_store = open_archive_store

__all__ = [
    "ArchiveStore",
    "LazyDatasetHandle",
    "MaxInstancesExceededError",
    "PredictionAppendItem",
    "SerializerPredictedInstance",
    "append_predictions_archive",
    "append_predictions_sta",
    "append_predictions_xpkg",
    "create_archive_store",
    "create_store_from_archive",
    "create_store_from_sta",
    "create_store_from_xpkg",
    "create_xpkg_store",
    "merge_predictions_archive",
    "merge_predictions_sta",
    "merge_predictions_xpkg",
    "open_archive_store",
    "open_store",
    "read_metrics_table",
    "read_archive",
    "read_sta",
    "read_xpkg",
    "summarize_archive",
    "summarize_project",
    "summarize_sta",
    "summarize_xpkg",
    "update_labels_archive",
    "update_labels_sta",
    "update_labels_xpkg",
    "validate_archive",
    "validate_project",
    "validate_sta",
    "validate_xpkg",
    "write_metrics_table",
    "write_archive",
    "write_sta",
    "write_xpkg",
]
