"""Curated project contract entry points for xpkg projects.

This module exposes stable project lifecycle, validation, descriptor, storage,
segmentation, artifact, calibration, and exchange helpers. Use
``ProjectService`` for import orchestration across pose, calibration, and
motion formats.
"""

from __future__ import annotations

from xpkg.io.labels.json_format import read_labels_json_payload, write_labels_json
from xpkg.project.artifact import (
    EXPKG_MANIFEST_FILENAME,
    pack_project,
    unpack_project,
    validate_artifact,
    validate_expkg,
    validate_project,
)
from xpkg.project.artifacts import (
    ARTIFACT_INDEX_FILENAME,
    ARTIFACT_MANIFEST_FILENAME,
    ARTIFACT_SCHEMA_VERSION,
    FIGURE_ARTIFACT_SCHEMA_VERSION,
    FIGURE_ARTIFACT_TYPE,
    FIGURE_MANIFEST_FILENAME,
    FIGURES_DIRNAME,
    ArtifactFile,
    ArtifactIndexEntry,
    ArtifactManifest,
    ArtifactOutputSpec,
    FigureArtifact,
    artifact_kind_dir,
    delete_project_artifact,
    list_project_artifact_index,
    list_project_artifacts,
    list_project_figures,
    load_project_artifact,
    load_project_figure,
    project_artifact_index_path,
    project_artifact_root,
    project_artifact_type_root,
    project_figure_root,
    project_figures_root,
    rebuild_project_artifact_index,
    save_project_artifact,
    save_project_figure,
    validate_project_artifact,
    validate_project_artifacts,
    validate_project_figure,
    validate_project_figures,
)
from xpkg.project.calibration import (
    CALIBRATION_FILENAME,
    CALIBRATION_SOURCE_DIRNAME,
    CALIBRATIONS_DIRNAME,
    list_project_calibrations,
    load_project_calibration,
    project_calibration_path,
    project_calibration_root,
    project_calibration_source_root,
    project_calibrations_root,
    save_project_calibration,
)
from xpkg.project.inspection import ProjectInspection, inspect_project
from xpkg.project.layout import (
    ARTIFACTS_DIRNAME,
    EXPKG_SUFFIX,
    INDEXES_DIRNAME,
    PROJECT_DESCRIPTOR_FILENAME,
    PROJECT_SUMMARY_FILENAME,
    ProjectDescriptor,
    default_expkg_path,
    is_project_root,
    load_project_descriptor,
    project_artifacts_root,
    project_descriptor_path,
    project_exports_root,
    project_indexes_root,
    project_media_root,
    project_state_root,
    project_store_root,
    project_summary_path,
    resolve_project_root,
    write_project_descriptor,
)
from xpkg.project.metadata import (
    ACQUISITION_METADATA_FILENAME,
    DATASET_SHARE_METADATA_FILENAME,
    DATASHEET_FILENAME,
    MODEL_CARD_FILENAME,
    POSE_PROVENANCE_FILENAME,
    PROJECT_METADATA_DIRNAME,
    project_acquisition_metadata_path,
    project_dataset_share_metadata_path,
    project_datasheet_path,
    project_metadata_root,
    project_model_card_path,
    project_pose_provenance_path,
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
    PROJECT_SUMMARY_SCHEMA_VERSION,
    ProjectSummaryIndex,
    labels_state_summary,
    load_project_summary,
    refresh_project_summary,
    vicon_state_summary,
)

# Curated stable public surface.
#
# Package-level format importers are intentionally not re-exported here; use
# ``ProjectService.import_pose``, ``import_calibration``, or ``import_motion``.
__all__ = [
    # Constants and filenames
    "ACQUISITION_METADATA_FILENAME",
    "ARTIFACTS_DIRNAME",
    "ARTIFACT_INDEX_FILENAME",
    "ARTIFACT_MANIFEST_FILENAME",
    "ARTIFACT_SCHEMA_VERSION",
    "CALIBRATION_FILENAME",
    "CALIBRATION_SOURCE_DIRNAME",
    "CALIBRATIONS_DIRNAME",
    "DATASET_SHARE_METADATA_FILENAME",
    "DATASHEET_FILENAME",
    "EXPKG_MANIFEST_FILENAME",
    "EXPKG_SUFFIX",
    "FIGURE_ARTIFACT_SCHEMA_VERSION",
    "FIGURE_ARTIFACT_TYPE",
    "FIGURE_MANIFEST_FILENAME",
    "FIGURES_DIRNAME",
    "INDEXES_DIRNAME",
    "MODEL_CARD_FILENAME",
    "POSE_PROVENANCE_FILENAME",
    "PROJECT_DESCRIPTOR_FILENAME",
    "PROJECT_METADATA_DIRNAME",
    "PROJECT_SUMMARY_FILENAME",
    "PROJECT_SUMMARY_SCHEMA_VERSION",
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
    # Path helpers
    "resolve_project_root",
    "current_project_state_path",
    "project_descriptor_path",
    "project_store_root",
    "project_state_root",
    "project_media_root",
    "project_exports_root",
    "project_metadata_root",
    "project_indexes_root",
    "project_summary_path",
    "project_acquisition_metadata_path",
    "project_dataset_share_metadata_path",
    "project_datasheet_path",
    "project_model_card_path",
    "project_pose_provenance_path",
    "project_artifacts_root",
    "project_artifact_type_root",
    "project_artifact_root",
    "project_artifact_index_path",
    "project_figures_root",
    "project_figure_root",
    "project_calibration_path",
    "project_calibration_root",
    "project_calibration_source_root",
    "project_calibrations_root",
    "artifact_kind_dir",
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
