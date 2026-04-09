"""Workspace descriptor and layout helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from xpkg.core.path_registry import ensure_dir, resolve_path
from xpkg.io.archive_format.shared import CANONICAL_BUNDLE_SUFFIX, LEGACY_BUNDLE_SUFFIXES

PROJECT_DESCRIPTOR_FILENAME = "PROJECT.json"
EXPKG_SUFFIX = ".expkg"
STORE_DIRNAME = ".xpkg"
STORE_STATE_DIRNAME = "state"
CURRENT_SNAPSHOT_FILENAME = "current.json"
CURRENT_ARCHIVE_FILENAME = f"current{CANONICAL_BUNDLE_SUFFIX}"
SUPPORTED_CURRENT_ARCHIVE_FILENAMES = (
    CURRENT_ARCHIVE_FILENAME,
    *(f"current{suffix}" for suffix in LEGACY_BUNDLE_SUFFIXES),
)
MEDIA_DIRNAME = "Media"
EXPORTS_DIRNAME = "Exports"
PackMode = Literal["portable", "snapshot"]


def _now_utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class ProjectDescriptor:
    """Public xpkg workspace descriptor."""

    title: str
    project_id: str
    created_at: str
    updated_at: str
    format: str = "xpkg-project"
    project_schema_version: int = 1
    layout_version: int = 1
    store_path: str = STORE_DIRNAME
    media_root: str = MEDIA_DIRNAME
    exports_root: str = EXPORTS_DIRNAME
    default_pack_mode: PackMode = "portable"

    @classmethod
    def new(
        cls,
        *,
        title: str,
        project_id: str | None = None,
        default_pack_mode: PackMode = "portable",
    ) -> ProjectDescriptor:
        timestamp = _now_utc_iso()
        return cls(
            title=title,
            project_id=project_id or str(uuid4()),
            created_at=timestamp,
            updated_at=timestamp,
            default_pack_mode=default_pack_mode,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectDescriptor:
        required = {
            "format",
            "project_schema_version",
            "layout_version",
            "title",
            "project_id",
            "created_at",
            "updated_at",
            "store_path",
            "media_root",
            "exports_root",
            "default_pack_mode",
        }
        missing = sorted(required.difference(data))
        if missing:
            raise ValueError(f"PROJECT.json missing required field(s): {', '.join(missing)}")
        raw_pack_mode = str(data["default_pack_mode"])
        if raw_pack_mode == "portable":
            default_pack_mode: PackMode = "portable"
        elif raw_pack_mode == "snapshot":
            default_pack_mode = "snapshot"
        else:
            raise ValueError(
                "default_pack_mode must be 'portable' or 'snapshot', "
                f"got {raw_pack_mode!r}"
            )
        descriptor = cls(
            format=str(data["format"]),
            project_schema_version=int(data["project_schema_version"]),
            layout_version=int(data["layout_version"]),
            title=str(data["title"]),
            project_id=str(data["project_id"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            store_path=str(data["store_path"]),
            media_root=str(data["media_root"]),
            exports_root=str(data["exports_root"]),
            default_pack_mode=default_pack_mode,
        )
        descriptor.validate()
        return descriptor

    def validate(self) -> None:
        if self.format != "xpkg-project":
            raise ValueError(f"Unsupported PROJECT.json format: {self.format!r}")
        if int(self.project_schema_version) != 1:
            raise ValueError(
                f"Unsupported project schema version: {self.project_schema_version!r}"
            )
        if int(self.layout_version) != 1:
            raise ValueError(f"Unsupported layout version: {self.layout_version!r}")
        if not self.title.strip():
            raise ValueError("PROJECT.json title cannot be empty")
        if not self.project_id.strip():
            raise ValueError("PROJECT.json project_id cannot be empty")
        if self.store_path != STORE_DIRNAME:
            raise ValueError(f"Unsupported store_path: {self.store_path!r}")
        if self.media_root != MEDIA_DIRNAME:
            raise ValueError(f"Unsupported media_root: {self.media_root!r}")
        if self.exports_root != EXPORTS_DIRNAME:
            raise ValueError(f"Unsupported exports_root: {self.exports_root!r}")
        if self.default_pack_mode not in {"portable", "snapshot"}:
            raise ValueError(
                "default_pack_mode must be 'portable' or 'snapshot', "
                f"got {self.default_pack_mode!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "format": self.format,
            "project_schema_version": int(self.project_schema_version),
            "layout_version": int(self.layout_version),
            "title": self.title,
            "project_id": self.project_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "store_path": self.store_path,
            "media_root": self.media_root,
            "exports_root": self.exports_root,
            "default_pack_mode": self.default_pack_mode,
        }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _candidate_workspace_root(path: str | Path) -> Path:
    return resolve_path(path)


def project_descriptor_path(path: str | Path) -> Path:
    root = resolve_workspace_root(path)
    if root is None:
        candidate = _candidate_workspace_root(path)
        if candidate.name == PROJECT_DESCRIPTOR_FILENAME:
            return candidate
        return candidate / PROJECT_DESCRIPTOR_FILENAME
    return root / PROJECT_DESCRIPTOR_FILENAME


def resolve_workspace_root(path: str | Path) -> Path | None:
    candidate = _candidate_workspace_root(path)
    if candidate.is_file() and candidate.name == PROJECT_DESCRIPTOR_FILENAME:
        return candidate.parent
    if candidate.is_dir() and (candidate / PROJECT_DESCRIPTOR_FILENAME).is_file():
        return candidate
    return None


def is_workspace_root(path: str | Path) -> bool:
    return resolve_workspace_root(path) is not None


def load_project_descriptor(path: str | Path) -> ProjectDescriptor:
    descriptor_path = project_descriptor_path(path)
    if not descriptor_path.is_file():
        raise FileNotFoundError(f"PROJECT.json not found: {descriptor_path}")
    data = json.loads(descriptor_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"PROJECT.json must contain an object: {descriptor_path}")
    return ProjectDescriptor.from_dict(data)


def write_project_descriptor(path: str | Path, descriptor: ProjectDescriptor) -> Path:
    root = resolve_workspace_root(path) or _candidate_workspace_root(path)
    descriptor.validate()
    descriptor_path = root / PROJECT_DESCRIPTOR_FILENAME
    _write_json(descriptor_path, descriptor.to_dict())
    return descriptor_path


def workspace_store_root(path: str | Path) -> Path:
    root = resolve_workspace_root(path) or _candidate_workspace_root(path)
    try:
        descriptor = load_project_descriptor(root)
        store_name = descriptor.store_path
    except FileNotFoundError:
        store_name = STORE_DIRNAME
    return root / store_name


def workspace_state_root(path: str | Path) -> Path:
    return workspace_store_root(path) / STORE_STATE_DIRNAME


def workspace_current_snapshot_path(path: str | Path) -> Path:
    return workspace_state_root(path) / CURRENT_SNAPSHOT_FILENAME


def workspace_media_root(path: str | Path) -> Path:
    root = resolve_workspace_root(path) or _candidate_workspace_root(path)
    try:
        descriptor = load_project_descriptor(root)
        media_name = descriptor.media_root
    except FileNotFoundError:
        media_name = MEDIA_DIRNAME
    return root / media_name


def workspace_exports_root(path: str | Path) -> Path:
    root = resolve_workspace_root(path) or _candidate_workspace_root(path)
    try:
        descriptor = load_project_descriptor(root)
        exports_name = descriptor.exports_root
    except FileNotFoundError:
        exports_name = EXPORTS_DIRNAME
    return root / exports_name


def default_expkg_path(path: str | Path) -> Path:
    root = resolve_workspace_root(path) or _candidate_workspace_root(path)
    return workspace_exports_root(root) / f"{root.name}{EXPKG_SUFFIX}"


__all__ = [
    "CANONICAL_BUNDLE_SUFFIX",
    "CURRENT_ARCHIVE_FILENAME",
    "CURRENT_SNAPSHOT_FILENAME",
    "EXPORTS_DIRNAME",
    "EXPKG_SUFFIX",
    "LEGACY_BUNDLE_SUFFIXES",
    "MEDIA_DIRNAME",
    "PROJECT_DESCRIPTOR_FILENAME",
    "PackMode",
    "ProjectDescriptor",
    "STORE_DIRNAME",
    "STORE_STATE_DIRNAME",
    "SUPPORTED_CURRENT_ARCHIVE_FILENAMES",
    "_candidate_workspace_root",
    "_now_utc_iso",
    "default_expkg_path",
    "is_workspace_root",
    "load_project_descriptor",
    "project_descriptor_path",
    "resolve_workspace_root",
    "workspace_current_snapshot_path",
    "workspace_exports_root",
    "workspace_media_root",
    "workspace_state_root",
    "workspace_store_root",
    "write_project_descriptor",
]
