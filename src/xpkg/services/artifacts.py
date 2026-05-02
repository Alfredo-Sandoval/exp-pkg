"""Service-bound generic artifact registry API."""

from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

from xpkg.project.artifacts import (
    ArtifactIndexEntry,
    ArtifactManifest,
    ArtifactOutputSpec,
    list_project_artifact_index,
    list_project_artifacts,
    load_project_artifact,
    rebuild_project_artifact_index,
    save_project_artifact,
    validate_project_artifact,
    validate_project_artifacts,
)


@dataclass(frozen=True, slots=True)
class ProjectArtifacts:
    """Project-bound helpers for generic artifact manifests and indexes."""

    project_root: Path

    def register(
        self,
        *,
        artifact_id: str,
        artifact_type: str,
        outputs: ArtifactOutputSpec,
        title: str = "",
        inputs: Sequence[str | Path] = (),
        producer: Mapping[str, Any] | None = None,
        stats: Sequence[str | Path] = (),
        metadata: Mapping[str, Any] | None = None,
        namespace: str | None = None,
        overwrite: bool = True,
    ) -> ArtifactManifest:
        """Copy artifact outputs into the project and write a manifest."""
        return save_project_artifact(
            self.project_root,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            outputs=outputs,
            title=title,
            inputs=inputs,
            producer=producer,
            stats=stats,
            metadata=metadata,
            namespace=namespace,
            overwrite=overwrite,
        )

    def load(
        self,
        artifact_id: str,
        *,
        kind: str | None = None,
        namespace: str | None = None,
    ) -> ArtifactManifest:
        """Load one saved artifact manifest."""
        return load_project_artifact(
            self.project_root,
            artifact_id,
            artifact_type=kind,
            namespace=namespace,
        )

    def list(
        self,
        *,
        kind: str | None = None,
        namespace: str | None = None,
    ) -> builtins.list[ArtifactManifest]:
        """List saved artifact manifests."""
        return list_project_artifacts(
            self.project_root,
            artifact_type=kind,
            namespace=namespace,
        )

    def index(
        self,
        *,
        kind: str | None = None,
        namespace: str | None = None,
    ) -> builtins.list[ArtifactIndexEntry]:
        """List the compact project artifact index."""
        return list_project_artifact_index(
            self.project_root,
            artifact_type=kind,
            namespace=namespace,
        )

    @overload
    def validate(
        self,
        artifact_id: str,
        *,
        kind: str | None = None,
        namespace: str | None = None,
    ) -> ArtifactManifest: ...

    @overload
    def validate(
        self,
        artifact_id: None = None,
        *,
        kind: str | None = None,
        namespace: str | None = None,
    ) -> builtins.list[ArtifactManifest]: ...

    def validate(
        self,
        artifact_id: str | None = None,
        *,
        kind: str | None = None,
        namespace: str | None = None,
    ) -> ArtifactManifest | builtins.list[ArtifactManifest]:
        """Validate one saved artifact, or every matching artifact when omitted."""
        if artifact_id is None:
            return validate_project_artifacts(
                self.project_root,
                artifact_type=kind,
                namespace=namespace,
            )
        return validate_project_artifact(
            self.project_root,
            artifact_id,
            artifact_type=kind,
            namespace=namespace,
        )

    def rebuild_index(self) -> builtins.list[ArtifactIndexEntry]:
        """Rebuild the project-wide artifact index from manifests."""
        return rebuild_project_artifact_index(self.project_root)


__all__ = [
    "ProjectArtifacts",
]
