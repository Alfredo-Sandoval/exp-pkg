"""Project descriptor schema and canonical managed-path layout."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from xpkg._core.json_utils import load_json_dict, write_json
from xpkg._core.path_registry import ensure_dir, resolve_path
from xpkg._core.time import now_utc_iso

PROJECT_DESCRIPTOR_FILENAME = "PROJECT.json"
EXPKG_SUFFIX = ".expkg"
STORE_DIRNAME = ".xpkg"
STORE_STATE_DIRNAME = "state"
ARTIFACTS_DIRNAME = "artifacts"
INDEXES_DIRNAME = "indexes"
PROJECT_SUMMARY_FILENAME = "project_summary.json"
CURRENT_STATE_FILENAME = "current.json"
MEDIA_DIRNAME = "Media"
EXPORTS_DIRNAME = "Exports"

_PROJECT_DESCRIPTOR_FIELDS = {
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
}
_UUID_PATTERN = (
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}"
)
_ULID_PATTERN = r"[0-9A-HJKMNP-TV-Z]{26}"
_PROJECT_ID_PATTERN = re.compile(rf"^(?:{_UUID_PATTERN}|{_ULID_PATTERN})$")


@dataclass(slots=True)
class ProjectDescriptor:
    """Public xpkg project descriptor."""

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

    @classmethod
    def new(
        cls,
        *,
        title: str,
        project_id: str | None = None,
    ) -> ProjectDescriptor:
        timestamp = now_utc_iso(drop_microseconds=True)
        return cls(
            title=title,
            project_id=project_id or str(uuid4()),
            created_at=timestamp,
            updated_at=timestamp,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectDescriptor:
        missing = sorted(_PROJECT_DESCRIPTOR_FIELDS.difference(data))
        if missing:
            raise ValueError(f"PROJECT.json missing required field(s): {', '.join(missing)}")
        unsupported = sorted(set(data).difference(_PROJECT_DESCRIPTOR_FIELDS))
        if unsupported:
            raise ValueError(
                "PROJECT.json contains unsupported field(s): "
                + ", ".join(unsupported)
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
        if _PROJECT_ID_PATTERN.fullmatch(self.project_id) is None:
            raise ValueError("PROJECT.json project_id must be a UUID or ULID")
        if self.store_path != STORE_DIRNAME:
            raise ValueError(f"Unsupported store_path: {self.store_path!r}")
        if self.media_root != MEDIA_DIRNAME:
            raise ValueError(f"Unsupported media_root: {self.media_root!r}")
        if self.exports_root != EXPORTS_DIRNAME:
            raise ValueError(f"Unsupported exports_root: {self.exports_root!r}")

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
        }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    write_json(path, payload, indent=2, sort_keys=False)


def _candidate_project_root(path: str | Path) -> Path:
    return resolve_path(path)


def project_descriptor_path(path: str | Path) -> Path:
    root = resolve_project_root(path)
    if root is None:
        candidate = _candidate_project_root(path)
        if candidate.name == PROJECT_DESCRIPTOR_FILENAME:
            return candidate
        return candidate / PROJECT_DESCRIPTOR_FILENAME
    return root / PROJECT_DESCRIPTOR_FILENAME


def resolve_project_root(path: str | Path) -> Path | None:
    candidate = _candidate_project_root(path)
    if candidate.is_file() and candidate.name == PROJECT_DESCRIPTOR_FILENAME:
        return candidate.parent
    if candidate.is_dir():
        for root in (candidate, *candidate.parents):
            if (root / PROJECT_DESCRIPTOR_FILENAME).is_file():
                return root
    return None


def is_project_root(path: str | Path) -> bool:
    return resolve_project_root(path) is not None


def require_project_root(path: str | Path) -> Path:
    """Return the owning project root or raise for a non-project path."""
    root = resolve_project_root(path)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {path}")
    return root


def load_project_descriptor(path: str | Path) -> ProjectDescriptor:
    descriptor_path = project_descriptor_path(path)
    if not descriptor_path.is_file():
        raise FileNotFoundError(f"PROJECT.json not found: {descriptor_path}")
    data = load_json_dict(descriptor_path)
    return ProjectDescriptor.from_dict(data)


def write_project_descriptor(path: str | Path, descriptor: ProjectDescriptor) -> Path:
    root = resolve_project_root(path) or _candidate_project_root(path)
    descriptor.validate()
    descriptor_path = root / PROJECT_DESCRIPTOR_FILENAME
    _write_json(descriptor_path, descriptor.to_dict())
    return descriptor_path


def project_store_root(path: str | Path) -> Path:
    root = resolve_project_root(path) or _candidate_project_root(path)
    try:
        descriptor = load_project_descriptor(root)
        store_name = descriptor.store_path
    except FileNotFoundError:
        store_name = STORE_DIRNAME
    return root / store_name


def project_state_root(path: str | Path) -> Path:
    return project_store_root(path) / STORE_STATE_DIRNAME


def project_artifacts_root(path: str | Path) -> Path:
    return project_store_root(path) / ARTIFACTS_DIRNAME


def project_indexes_root(path: str | Path) -> Path:
    return project_store_root(path) / INDEXES_DIRNAME


def project_summary_path(path: str | Path) -> Path:
    return project_indexes_root(path) / PROJECT_SUMMARY_FILENAME


def project_current_state_path(path: str | Path) -> Path:
    return project_state_root(path) / CURRENT_STATE_FILENAME


def project_media_root(path: str | Path) -> Path:
    root = resolve_project_root(path) or _candidate_project_root(path)
    try:
        descriptor = load_project_descriptor(root)
        media_name = descriptor.media_root
    except FileNotFoundError:
        media_name = MEDIA_DIRNAME
    return root / media_name


def project_exports_root(path: str | Path) -> Path:
    root = resolve_project_root(path) or _candidate_project_root(path)
    try:
        descriptor = load_project_descriptor(root)
        exports_name = descriptor.exports_root
    except FileNotFoundError:
        exports_name = EXPORTS_DIRNAME
    return root / exports_name


def default_expkg_path(path: str | Path) -> Path:
    root = resolve_project_root(path) or _candidate_project_root(path)
    return project_exports_root(root) / f"{root.name}{EXPKG_SUFFIX}"


__all__ = [
    "ARTIFACTS_DIRNAME",
    "CURRENT_STATE_FILENAME",
    "EXPORTS_DIRNAME",
    "EXPKG_SUFFIX",
    "INDEXES_DIRNAME",
    "MEDIA_DIRNAME",
    "PROJECT_DESCRIPTOR_FILENAME",
    "PROJECT_SUMMARY_FILENAME",
    "ProjectDescriptor",
    "STORE_DIRNAME",
    "STORE_STATE_DIRNAME",
    "_candidate_project_root",
    "default_expkg_path",
    "is_project_root",
    "load_project_descriptor",
    "project_descriptor_path",
    "resolve_project_root",
    "project_current_state_path",
    "project_artifacts_root",
    "project_exports_root",
    "project_indexes_root",
    "project_media_root",
    "project_summary_path",
    "project_state_root",
    "project_store_root",
    "require_project_root",
    "write_project_descriptor",
]
