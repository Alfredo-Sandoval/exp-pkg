from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StorePaths:
    """Canonical path layout for a siesta_store root."""

    root: Path

    @property
    def superblock_a(self) -> Path:
        return self.root / "superblock.a.json"

    @property
    def superblock_b(self) -> Path:
        return self.root / "superblock.b.json"

    @property
    def lock_file(self) -> Path:
        return self.root / "LOCK"

    @property
    def journal_dir(self) -> Path:
        return self.root / "journal"

    @property
    def active_journal(self) -> Path:
        return self.journal_dir / "active.json"

    @property
    def commits_dir(self) -> Path:
        return self.root / "commits"

    @property
    def objects_dir(self) -> Path:
        return self.root / "objects"

    @property
    def workspace_dir(self) -> Path:
        return self.root / "workspace"

    @property
    def snapshots_dir(self) -> Path:
        return self.root / "snapshots"

    def commit_dir(self, generation: int) -> Path:
        return self.commits_dir / f"{int(generation):012d}"

    def commit_json(self, generation: int) -> Path:
        return self.commit_dir(generation) / "commit.json"

    def object_path(self, object_id: str, *, ext: str) -> Path:
        normalized_ext = ext if ext.startswith(".") else f".{ext}"
        key = object_id.replace("obj_", "")
        a = key[:2] if len(key) >= 2 else "00"
        b = key[2:4] if len(key) >= 4 else "00"
        return self.objects_dir / a / b / f"{object_id}{normalized_ext}"
