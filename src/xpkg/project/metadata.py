"""Read and write project metadata slots."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xpkg._core.json_utils import load_json_dict, write_json
from xpkg._core.path_registry import ensure_dir
from xpkg.model.reporting import DatasetDatasheet, ModelCard
from xpkg.project.layout import (
    project_store_root,
)
from xpkg.project.layout import (
    require_project_root as _require_project_root,
)
from xpkg.project.store import load_project_metadata, save_project_metadata

PROJECT_METADATA_DIRNAME = "metadata"
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


def project_metadata_root(project: str | Path) -> Path:
    """Return the project-managed metadata directory under ``.xpkg/``."""

    return project_store_root(_require_project_root(project)) / PROJECT_METADATA_DIRNAME


def _refresh_project_summary(project: str | Path) -> None:
    from xpkg.project.summary import refresh_project_summary

    refresh_project_summary(project)


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
    _refresh_project_summary(project)
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
    _refresh_project_summary(project)
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
    "DATASHEET_FILENAME",
    "MODEL_CARD_FILENAME",
    "PROJECT_METADATA_DIRNAME",
    "load_project_datasheet",
    "load_project_metadata_field",
    "load_project_model_card",
    "project_datasheet_path",
    "project_metadata_root",
    "project_model_card_path",
    "save_project_datasheet",
    "save_project_metadata_field",
    "save_project_model_card",
]
