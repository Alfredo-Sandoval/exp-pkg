"""Read and write project metadata slots."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xpkg.model.metadata import (
    AcquisitionMetadata,
    DatasetShareMetadata,
    PoseModelProvenance,
)
from xpkg.model.reporting import DatasetDatasheet, ModelCard
from xpkg.project.layout import project_store_root, resolve_project_root
from xpkg.project.store import load_project_metadata, save_project_metadata

from .._core.json_utils import load_json_dict, write_json
from .._core.path_registry import ensure_dir

PROJECT_METADATA_DIRNAME = "metadata"
ACQUISITION_METADATA_FILENAME = "acquisition.json"
DATASET_SHARE_METADATA_FILENAME = "dataset_share.json"
POSE_PROVENANCE_FILENAME = "pose_provenance.json"
DATASHEET_FILENAME = "datasheet.json"
MODEL_CARD_FILENAME = "model_card.json"


def _require_field_name(field: str) -> str:
    normalized = str(field).strip()
    if not normalized:
        raise ValueError("metadata field name must be a non-empty string")
    return normalized


def _require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    payload: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings")
        payload[key] = item
    return payload


def _require_project_root(project: str | Path) -> Path:
    root = resolve_project_root(project)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")
    return root


def _acquisition_metadata(value: AcquisitionMetadata | Mapping[str, Any]) -> AcquisitionMetadata:
    if isinstance(value, AcquisitionMetadata):
        return value
    if isinstance(value, Mapping):
        return AcquisitionMetadata.from_dict(value)
    raise TypeError(f"acquisition metadata must be AcquisitionMetadata or mapping, got {value!r}.")


def _dataset_share_metadata(
    value: DatasetShareMetadata | Mapping[str, Any],
) -> DatasetShareMetadata:
    if isinstance(value, DatasetShareMetadata):
        return value
    if isinstance(value, Mapping):
        return DatasetShareMetadata.from_dict(value)
    raise TypeError(
        f"dataset share metadata must be DatasetShareMetadata or mapping, got {value!r}."
    )


def project_metadata_root(project: str | Path) -> Path:
    """Return the project-managed metadata directory under ``.xpkg/``."""

    return project_store_root(_require_project_root(project)) / PROJECT_METADATA_DIRNAME


def project_acquisition_metadata_path(project: str | Path) -> Path:
    """Return the canonical acquisition metadata JSON path for a project."""

    return project_metadata_root(project) / ACQUISITION_METADATA_FILENAME


def project_dataset_share_metadata_path(project: str | Path) -> Path:
    """Return the canonical dataset sharing metadata JSON path for a project."""

    return project_metadata_root(project) / DATASET_SHARE_METADATA_FILENAME


def save_project_acquisition_metadata(
    project: str | Path,
    acquisition: AcquisitionMetadata | Mapping[str, Any],
) -> Path:
    """Write project-level acquisition metadata to ``.xpkg/metadata/acquisition.json``."""

    metadata = _acquisition_metadata(acquisition)
    target_path = project_acquisition_metadata_path(project)
    ensure_dir(target_path.parent)
    write_json(target_path, metadata.to_dict())
    return target_path


def load_project_acquisition_metadata(project: str | Path) -> AcquisitionMetadata | None:
    """Load project-level acquisition metadata when present."""

    metadata_path = project_acquisition_metadata_path(project)
    if not metadata_path.is_file():
        return None
    return AcquisitionMetadata.from_dict(load_json_dict(metadata_path))


def save_project_dataset_share_metadata(
    project: str | Path,
    dataset_share: DatasetShareMetadata | Mapping[str, Any],
) -> Path:
    """Write project-level dataset sharing metadata to ``.xpkg/metadata/dataset_share.json``."""

    metadata = _dataset_share_metadata(dataset_share)
    target_path = project_dataset_share_metadata_path(project)
    ensure_dir(target_path.parent)
    write_json(target_path, metadata.to_dict())
    return target_path


def load_project_dataset_share_metadata(project: str | Path) -> DatasetShareMetadata | None:
    """Load project-level dataset sharing metadata when present."""

    metadata_path = project_dataset_share_metadata_path(project)
    if not metadata_path.is_file():
        return None
    return DatasetShareMetadata.from_dict(load_json_dict(metadata_path))


def _pose_model_provenance(
    value: PoseModelProvenance | Mapping[str, Any],
) -> PoseModelProvenance:
    if isinstance(value, PoseModelProvenance):
        return value
    if isinstance(value, Mapping):
        return PoseModelProvenance.from_dict(value)
    raise TypeError(
        f"pose provenance must be PoseModelProvenance or mapping, got {value!r}."
    )


def _dataset_datasheet(value: DatasetDatasheet | Mapping[str, Any]) -> DatasetDatasheet:
    if isinstance(value, DatasetDatasheet):
        return value
    if isinstance(value, Mapping):
        return DatasetDatasheet.from_dict(value)
    raise TypeError(
        f"datasheet must be DatasetDatasheet or mapping, got {value!r}."
    )


def _model_card(value: ModelCard | Mapping[str, Any]) -> ModelCard:
    if isinstance(value, ModelCard):
        return value
    if isinstance(value, Mapping):
        return ModelCard.from_dict(value)
    raise TypeError(f"model card must be ModelCard or mapping, got {value!r}.")


def project_pose_provenance_path(project: str | Path) -> Path:
    """Return the canonical pose-model provenance JSON path for a project."""

    return project_metadata_root(project) / POSE_PROVENANCE_FILENAME


def save_project_pose_provenance(
    project: str | Path,
    provenance: PoseModelProvenance | Mapping[str, Any],
) -> Path:
    """Write project-level pose-model provenance to ``.xpkg/metadata/pose_provenance.json``."""

    record = _pose_model_provenance(provenance)
    target_path = project_pose_provenance_path(project)
    ensure_dir(target_path.parent)
    write_json(target_path, record.to_dict())
    return target_path


def load_project_pose_provenance(project: str | Path) -> PoseModelProvenance | None:
    """Load project-level pose-model provenance when present."""

    metadata_path = project_pose_provenance_path(project)
    if not metadata_path.is_file():
        return None
    return PoseModelProvenance.from_dict(load_json_dict(metadata_path))


def project_datasheet_path(project: str | Path) -> Path:
    """Return the canonical datasheet (Gebru et al. 2021) JSON path for a project."""

    return project_metadata_root(project) / DATASHEET_FILENAME


def save_project_datasheet(
    project: str | Path,
    datasheet: DatasetDatasheet | Mapping[str, Any],
) -> Path:
    """Write project-level datasheet metadata to ``.xpkg/metadata/datasheet.json``."""

    record = _dataset_datasheet(datasheet)
    target_path = project_datasheet_path(project)
    ensure_dir(target_path.parent)
    write_json(target_path, record.to_dict())
    return target_path


def load_project_datasheet(project: str | Path) -> DatasetDatasheet | None:
    """Load project-level datasheet metadata when present."""

    metadata_path = project_datasheet_path(project)
    if not metadata_path.is_file():
        return None
    return DatasetDatasheet.from_dict(load_json_dict(metadata_path))


def project_model_card_path(project: str | Path) -> Path:
    """Return the canonical model-card (Mitchell et al. 2019) JSON path for a project."""

    return project_metadata_root(project) / MODEL_CARD_FILENAME


def save_project_model_card(
    project: str | Path,
    model_card: ModelCard | Mapping[str, Any],
) -> Path:
    """Write project-level model-card metadata to ``.xpkg/metadata/model_card.json``."""

    record = _model_card(model_card)
    target_path = project_model_card_path(project)
    ensure_dir(target_path.parent)
    write_json(target_path, record.to_dict())
    return target_path


def load_project_model_card(project: str | Path) -> ModelCard | None:
    """Load project-level model-card metadata when present."""

    metadata_path = project_model_card_path(project)
    if not metadata_path.is_file():
        return None
    return ModelCard.from_dict(load_json_dict(metadata_path))


def load_project_metadata_field(path: str | Path, field: str) -> dict[str, Any] | None:
    """Load one mapping-valued metadata field from the current project head."""
    metadata = load_project_metadata(path)
    if metadata is None:
        return None
    raw_value = metadata.get(_require_field_name(field))
    if raw_value is None:
        return None
    return _require_mapping(raw_value, name=f"project_metadata.{field}")


def save_project_metadata_field(
    path: str | Path,
    field: str,
    value: Mapping[str, Any],
    *,
    reason: str,
) -> Path:
    """Persist one mapping-valued metadata field onto the current project head."""
    attr_key = _require_field_name(field)
    metadata = load_project_metadata(path) or {}
    metadata[attr_key] = _require_mapping(value, name=f"project_metadata.{attr_key}")
    return save_project_metadata(path, metadata, reason=reason)


__all__ = [
    "ACQUISITION_METADATA_FILENAME",
    "DATASET_SHARE_METADATA_FILENAME",
    "DATASHEET_FILENAME",
    "MODEL_CARD_FILENAME",
    "POSE_PROVENANCE_FILENAME",
    "PROJECT_METADATA_DIRNAME",
    "load_project_acquisition_metadata",
    "load_project_dataset_share_metadata",
    "load_project_datasheet",
    "load_project_metadata_field",
    "load_project_model_card",
    "load_project_pose_provenance",
    "project_acquisition_metadata_path",
    "project_dataset_share_metadata_path",
    "project_datasheet_path",
    "project_metadata_root",
    "project_model_card_path",
    "project_pose_provenance_path",
    "save_project_acquisition_metadata",
    "save_project_dataset_share_metadata",
    "save_project_datasheet",
    "save_project_metadata_field",
    "save_project_model_card",
    "save_project_pose_provenance",
]
