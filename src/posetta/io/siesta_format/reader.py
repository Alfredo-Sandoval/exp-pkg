"""Read-only helpers for `.sta` archives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py

from posetta.io.siesta_format.project_validation import (
    ProjectSummary,
    summarize_project,
    validate_project,
)
from posetta.io.siesta_format.reader_core import (
    LazyDatasetHandle,
    LazySiestaHandle,
    _looks_like_iso_timestamp,
    build_common_reader_state,
    read_siesta_with_assembler,
)
from posetta.io.siesta_format.shared import _looks_like_int

__all__ = [
    "LazyDatasetHandle",
    "LazySiestaHandle",
    "ProjectSummary",
    "_looks_like_int",
    "_looks_like_iso_timestamp",
    "read_siesta",
    "summarize_project",
    "validate_project",
]


def _assemble_result(
    handle: h5py.File,
    path: Path,
    bundle_root: Path,
    lazy_read: bool,
) -> dict[str, Any]:
    common = build_common_reader_state(
        handle,
        path=path,
        bundle_root=bundle_root,
        lazy_read=lazy_read,
    )

    from posetta.io.siesta_format.segmentation_hdf5 import read_segmentation_group

    common.metadata["preferences"] = common.preferences_override
    common.result["labels"]["metadata"]["preferences"] = common.preferences_override
    common.result["segmentation"] = read_segmentation_group(handle)
    return common.result


def read_siesta(
    path: Path,
    *,
    lazy: bool = False,
) -> dict[str, Any]:
    """Load a `.sta` project archive from disk.

    Args:
        path: Path to the `.sta` file.
        lazy: If True, return lazy dataset handles instead of materializing arrays.
            The return payload includes ``h5_handle`` (LazySiestaHandle), and the
            caller must close it after materializing lazy datasets.

    Returns:
        dict: Project data including videos, labels, predictions, and metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
    """

    return read_siesta_with_assembler(path, lazy=lazy, assemble_result=_assemble_result)
