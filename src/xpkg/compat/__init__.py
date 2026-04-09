"""Compatibility surface for canonical `.xpkg` archives and legacy aliases."""

from __future__ import annotations

from xpkg.formats.siesta import (
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
from xpkg.formats.siesta_store import (
    SiestaStore,
    create_store_from_archive,
    create_store_from_sta,
    create_store_from_xpkg,
    open_store,
)

read_xpkg = read_siesta
write_xpkg = write_siesta
update_labels_xpkg = update_labels_siesta
append_predictions_xpkg = append_predictions_siesta
merge_predictions_xpkg = merge_predictions_siesta
summarize_xpkg = summarize_project
validate_xpkg = validate_project
read_sta = read_siesta
write_sta = write_siesta
update_labels_sta = update_labels_siesta
append_predictions_sta = append_predictions_siesta
merge_predictions_sta = merge_predictions_siesta
summarize_sta = summarize_project
validate_sta = validate_project
ArchiveStore = SiestaStore
create_archive_store = create_store_from_archive
open_archive_store = open_store

__all__ = [
    "ArchiveStore",
    "LazyDatasetHandle",
    "MaxInstancesExceededError",
    "PredictionAppendItem",
    "SerializerPredictedInstance",
    "SiestaStore",
    "append_predictions_siesta",
    "append_predictions_sta",
    "append_predictions_xpkg",
    "create_archive_store",
    "create_store_from_archive",
    "create_store_from_sta",
    "create_store_from_xpkg",
    "merge_predictions_siesta",
    "merge_predictions_sta",
    "merge_predictions_xpkg",
    "open_archive_store",
    "open_store",
    "read_metrics_table",
    "read_siesta",
    "read_sta",
    "read_xpkg",
    "summarize_project",
    "summarize_sta",
    "summarize_xpkg",
    "update_labels_siesta",
    "update_labels_sta",
    "update_labels_xpkg",
    "validate_project",
    "validate_sta",
    "validate_xpkg",
    "write_metrics_table",
    "write_siesta",
    "write_sta",
    "write_xpkg",
]
