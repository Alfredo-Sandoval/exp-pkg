"""Read-only helpers for canonical `.xpkg` archives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py

from xpkg.io.archive_format.project_validation import (
    ProjectSummary,
    summarize_project,
    validate_project,
)
from xpkg.io.archive_format.reader_core import (
    LazyArchiveHandle,
    LazyDatasetHandle,
    _looks_like_iso_timestamp,
    build_common_reader_state,
    read_archive_with_assembler,
)
from xpkg.io.archive_format.shared import _looks_like_int
from xpkg.io.archive_format.tracks_hdf5 import read_tracks_group

__all__ = [
    "LazyDatasetHandle",
    "LazyArchiveHandle",
    "ProjectSummary",
    "_looks_like_int",
    "_looks_like_iso_timestamp",
    "read_archive",
    "summarize_project",
    "validate_project",
]


def _assemble_result(
    handle: h5py.File,
    path: Path,
    archive_root: Path,
    lazy_read: bool,
) -> dict[str, Any]:
    common = build_common_reader_state(
        handle,
        path=path,
        archive_root=archive_root,
        lazy_read=lazy_read,
    )

    from xpkg.io.archive_format.segmentation_hdf5 import read_segmentation_group

    tracks_by_id = read_tracks_group(handle)
    common.result["labels"]["tracks"] = tracks_by_id
    common.metadata["preferences"] = common.preferences_override
    common.result["labels"]["metadata"]["preferences"] = common.preferences_override
    common.result["segmentation"] = read_segmentation_group(handle, tracks_by_id=tracks_by_id)
    return common.result


def read_archive(
    path: Path,
    *,
    lazy: bool = False,
) -> dict[str, Any]:
    """Load a canonical `.xpkg` archive from disk."""

    return read_archive_with_assembler(path, lazy=lazy, assemble_result=_assemble_result)
