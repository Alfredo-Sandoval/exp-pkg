"""Project contract entry points for free-function integrations.

This module defines the free-function form of the project contract: project
creation, import, validation, metadata, segmentation, artifact registry, and
portable ``.expkg`` packing. New integrations should prefer ``ProjectService``
for lifecycle orchestration, while these functions remain public for explicit
function-level callers.
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
from xpkg.project.calibration import (
    import_anipose_calibration_project as import_anipose_calibration_project,
)
from xpkg.project.inspection import ProjectInspection, inspect_project
from xpkg.project.layout import (
    ARTIFACTS_DIRNAME,
    EXPKG_SUFFIX,
    PROJECT_DESCRIPTOR_FILENAME,
    ProjectDescriptor,
    default_expkg_path,
    is_project_root,
    load_project_descriptor,
    project_artifacts_root,
    project_descriptor_path,
    project_exports_root,
    project_media_root,
    project_state_root,
    project_store_root,
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

# Path-level metadata helpers re-exported for callers that need a function-level
# seam; deliberately omitted from ``__all__`` since ``ProjectService.metadata``
# is the public path. Redundant aliases mark these as intentional re-exports.
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

# Path-level import and metadata-state helpers re-exported for callers that
# need a function-level seam; deliberately omitted from ``__all__`` since
# ``ProjectService.import_pose`` / ``import_calibration`` / ``import_motion``
# and ``ProjectService.load_state_metadata`` / ``save_state_metadata`` are the
# public path. Redundant aliases mark these as intentional re-exports.
from xpkg.project.store import (
    import_dlc_csv_project as import_dlc_csv_project,
)
from xpkg.project.store import (
    import_dlc_h5_project as import_dlc_h5_project,
)
from xpkg.project.store import (
    import_dlc_project_directory as import_dlc_project_directory,
)
from xpkg.project.store import (
    import_lightning_pose_csv_project as import_lightning_pose_csv_project,
)
from xpkg.project.store import (
    import_mediapipe_pose_landmarks_json_project as import_mediapipe_pose_landmarks_json_project,
)
from xpkg.project.store import (
    import_mmpose_topdown_json_project as import_mmpose_topdown_json_project,
)
from xpkg.project.store import (
    import_sleap_h5_project as import_sleap_h5_project,
)
from xpkg.project.store import (
    import_sleap_package_project as import_sleap_package_project,
)
from xpkg.project.store import (
    import_vicon_c3d_project as import_vicon_c3d_project,
)
from xpkg.project.store import (
    import_vicon_csv_project as import_vicon_csv_project,
)
from xpkg.project.store import (
    import_vicon_project as import_vicon_project,
)
from xpkg.project.store import (
    load_project_metadata as load_project_metadata,
)
from xpkg.project.store import (
    save_project_metadata as save_project_metadata,
)

# Curated stable public surface.
#
# The verbose path-level helpers ``import_*_project`` and ``save/load_project_*``
# (typed metadata slots and the state-bound metadata dict) are intentionally
# omitted: ``ProjectService.import_pose`` / ``import_calibration`` /
# ``import_motion`` and ``ProjectService.metadata`` / ``load_state_metadata``
# are the public path. The free functions remain importable from this module
# for callers that need a function-level seam, but they are not part of the
# stable surface.
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
    "MODEL_CARD_FILENAME",
    "POSE_PROVENANCE_FILENAME",
    "PROJECT_DESCRIPTOR_FILENAME",
    "PROJECT_METADATA_DIRNAME",
    # Typed dataclasses and frame records
    "ArtifactFile",
    "ArtifactIndexEntry",
    "ArtifactManifest",
    "ArtifactOutputSpec",
    "FigureArtifact",
    "ProjectDescriptor",
    "ProjectInspection",
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
    # Calibration storage
    "list_project_calibrations",
    "load_project_calibration",
    "save_project_calibration",
    # Labels JSON exchange (used by xpkg.adapters)
    "read_labels_json_payload",
    "write_labels_json",
]
