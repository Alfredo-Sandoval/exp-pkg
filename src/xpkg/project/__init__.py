"""Curated project contract entry points for xpkg projects.

This module exposes stable project lifecycle, validation, descriptor, storage,
segmentation, artifact, calibration, and exchange helpers. Use
``ProjectService`` for import orchestration across pose, calibration, and
motion formats.
"""

from __future__ import annotations

from xpkg.io.labels.json_format import read_labels_json_payload, write_labels_json
from xpkg.project.artifact import (
    pack_project,
    unpack_project,
    validate_artifact,
    validate_expkg,
    validate_project,
)
from xpkg.project.artifacts import (
    ArtifactFile,
    ArtifactIndexEntry,
    ArtifactManifest,
    ArtifactOutputSpec,
    FigureArtifact,
    delete_project_artifact,
    list_project_artifact_index,
    list_project_artifacts,
    list_project_figures,
    load_project_artifact,
    load_project_figure,
    rebuild_project_artifact_index,
    save_project_artifact,
    save_project_figure,
    validate_project_artifact,
    validate_project_artifacts,
    validate_project_figure,
    validate_project_figures,
)
from xpkg.project.calibration import (
    list_project_calibrations,
    load_project_calibration,
    save_project_calibration,
)
from xpkg.project.inspection import ProjectInspection, inspect_project
from xpkg.project.layout import (
    ProjectDescriptor,
    default_expkg_path,
    is_project_root,
    load_project_descriptor,
    write_project_descriptor,
)

# Path-level metadata helpers for callers that need a function-level seam.
# ``ProjectService.metadata`` remains the normal service-bound path.
from xpkg.project.metadata import (
    load_project_acquisition_metadata as load_project_acquisition_metadata,
)
from xpkg.project.metadata import (
    load_project_dataset_share_metadata as load_project_dataset_share_metadata,
)
from xpkg.project.metadata import (
    load_project_datasheet as load_project_datasheet,
)
from xpkg.project.metadata import (
    load_project_metadata_field as load_project_metadata_field,
)
from xpkg.project.metadata import (
    load_project_model_card as load_project_model_card,
)
from xpkg.project.metadata import (
    load_project_pose_provenance as load_project_pose_provenance,
)
from xpkg.project.metadata import (
    save_project_acquisition_metadata as save_project_acquisition_metadata,
)
from xpkg.project.metadata import (
    save_project_dataset_share_metadata as save_project_dataset_share_metadata,
)
from xpkg.project.metadata import (
    save_project_datasheet as save_project_datasheet,
)
from xpkg.project.metadata import (
    save_project_metadata_field as save_project_metadata_field,
)
from xpkg.project.metadata import (
    save_project_model_card as save_project_model_card,
)
from xpkg.project.metadata import (
    save_project_pose_provenance as save_project_pose_provenance,
)
from xpkg.project.segmentation import (
    SegmentationFrame,
    clear_project_segmentation_masks,
    load_project_segmentation_frames,
    load_project_segmentation_masks,
    save_project_segmentation_masks,
)
from xpkg.project.store import (
    current_project_state_path,
    init_project,
    load_project_payload,
    load_project_vicon_recording,
    save_project_labels,
)

# State-metadata helpers for callers that need a function-level seam.
# ``ProjectService.load_state_metadata`` / ``save_state_metadata`` remain the
# normal service-bound path.
from xpkg.project.store import (
    load_project_metadata as load_project_metadata,
)
from xpkg.project.store import (
    save_project_metadata as save_project_metadata,
)
from xpkg.project.summary import (
    ProjectSummaryIndex,
    labels_state_summary,
    load_project_summary,
    refresh_project_summary,
    vicon_state_summary,
)

# Curated stable public surface.
#
# This list is the supported public API of ``xpkg.project``. Two families are
# intentionally excluded from it: package-level format importers (use
# ``ProjectService.import_pose`` / ``import_calibration`` / ``import_motion``),
# and the private-store layout details -- the ``.xpkg/`` directory and filename
# constants and the ``project_*`` path helpers. Those layout names remain
# importable from their submodules (``xpkg.project.layout`` and friends) for
# internal use, but are not part of the public contract: the on-disk layout may
# change, so downstream code should locate project files through
# ``ProjectService`` rather than hard-coding these names.
__all__ = [
    # Typed dataclasses and frame records
    "ArtifactFile",
    "ArtifactIndexEntry",
    "ArtifactManifest",
    "ArtifactOutputSpec",
    "FigureArtifact",
    "ProjectDescriptor",
    "ProjectInspection",
    "ProjectSummaryIndex",
    "SegmentationFrame",
    # Project lifecycle
    "init_project",
    "pack_project",
    "unpack_project",
    "validate_project",
    "validate_expkg",
    "validate_artifact",
    "inspect_project",
    "is_project_root",
    "default_expkg_path",
    # Descriptor I/O
    "load_project_descriptor",
    "write_project_descriptor",
    # Artifact registry
    "delete_project_artifact",
    "list_project_artifacts",
    "list_project_artifact_index",
    "load_project_artifact",
    "save_project_artifact",
    "validate_project_artifact",
    "validate_project_artifacts",
    "rebuild_project_artifact_index",
    "list_project_figures",
    "load_project_figure",
    "save_project_figure",
    "validate_project_figure",
    "validate_project_figures",
    # Segmentation
    "load_project_segmentation_frames",
    "load_project_segmentation_masks",
    "save_project_segmentation_masks",
    "clear_project_segmentation_masks",
    # Labels and payload
    "current_project_state_path",
    "save_project_labels",
    "load_project_payload",
    "load_project_vicon_recording",
    "load_project_summary",
    "refresh_project_summary",
    "labels_state_summary",
    "vicon_state_summary",
    # Metadata storage
    "load_project_acquisition_metadata",
    "load_project_dataset_share_metadata",
    "load_project_datasheet",
    "load_project_metadata",
    "load_project_metadata_field",
    "load_project_model_card",
    "load_project_pose_provenance",
    "save_project_acquisition_metadata",
    "save_project_dataset_share_metadata",
    "save_project_datasheet",
    "save_project_metadata",
    "save_project_metadata_field",
    "save_project_model_card",
    "save_project_pose_provenance",
    # Calibration storage
    "list_project_calibrations",
    "load_project_calibration",
    "save_project_calibration",
    # Labels JSON exchange (used by xpkg.adapters)
    "read_labels_json_payload",
    "write_labels_json",
]
