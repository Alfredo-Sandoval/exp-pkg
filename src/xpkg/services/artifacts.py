"""Service-bound generic artifact registry API."""

from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

from xpkg.workspace.artifacts import (
    ArtifactIndexEntry,
    ArtifactManifest,
    ArtifactOutputSpec,
    list_workspace_artifact_index,
    list_workspace_artifacts,
    load_workspace_artifact,
    rebuild_workspace_artifact_index,
    save_workspace_artifact,
    validate_workspace_artifact,
    validate_workspace_artifacts,
)


@dataclass(frozen=True, slots=True)
class WorkspaceArtifacts:
    """Workspace-bound helpers for generic artifact manifests and indexes."""

    workspace_root: Path

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
        """Copy artifact outputs into the workspace and write a manifest."""
        return save_workspace_artifact(
            self.workspace_root,
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
        return load_workspace_artifact(
            self.workspace_root,
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
        return list_workspace_artifacts(
            self.workspace_root,
            artifact_type=kind,
            namespace=namespace,
        )

    def index(
        self,
        *,
        kind: str | None = None,
        namespace: str | None = None,
    ) -> builtins.list[ArtifactIndexEntry]:
        """List the compact workspace artifact index."""
        return list_workspace_artifact_index(
            self.workspace_root,
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
            return validate_workspace_artifacts(
                self.workspace_root,
                artifact_type=kind,
                namespace=namespace,
            )
        return validate_workspace_artifact(
            self.workspace_root,
            artifact_id,
            artifact_type=kind,
            namespace=namespace,
        )

    def rebuild_index(self) -> builtins.list[ArtifactIndexEntry]:
        """Rebuild the workspace-wide artifact index from manifests."""
        return rebuild_workspace_artifact_index(self.workspace_root)


__all__ = [
    "WorkspaceArtifacts",
]
