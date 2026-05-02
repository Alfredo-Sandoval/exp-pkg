"""Service-bound convenience API for figure artifacts."""

from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

from xpkg.project.artifacts import (
    FigureArtifact,
    FigureOutputSpec,
    list_project_figures,
    load_project_figure,
    save_project_figure,
    validate_project_figure,
    validate_project_figures,
)


@dataclass(frozen=True, slots=True)
class ProjectFigures:
    """Service-bound access to figure artifact manifests for one project."""

    project_root: Path

    def save(
        self,
        *,
        figure_id: str,
        outputs: FigureOutputSpec,
        title: str = "",
        inputs: Sequence[str | Path] = (),
        producer: Mapping[str, Any] | None = None,
        stats: Sequence[str | Path] = (),
        metadata: Mapping[str, Any] | None = None,
        namespace: str | None = None,
        overwrite: bool = True,
    ) -> FigureArtifact:
        """Copy figure outputs into the project and write a manifest."""
        return save_project_figure(
            self.project_root,
            figure_id=figure_id,
            outputs=outputs,
            title=title,
            inputs=inputs,
            producer=producer,
            stats=stats,
            metadata=metadata,
            namespace=namespace,
            overwrite=overwrite,
        )

    def load(self, figure_id: str, *, namespace: str | None = None) -> FigureArtifact:
        """Load one saved figure manifest."""
        return load_project_figure(self.project_root, figure_id, namespace=namespace)

    def list(self, *, namespace: str | None = None) -> builtins.list[FigureArtifact]:
        """List saved figure manifests."""
        return list_project_figures(self.project_root, namespace=namespace)

    @overload
    def validate(self, figure_id: str, *, namespace: str | None = None) -> FigureArtifact: ...

    @overload
    def validate(
        self,
        figure_id: None = None,
        *,
        namespace: str | None = None,
    ) -> builtins.list[FigureArtifact]: ...

    def validate(
        self,
        figure_id: str | None = None,
        *,
        namespace: str | None = None,
    ) -> FigureArtifact | builtins.list[FigureArtifact]:
        """Validate one saved figure, or every saved figure when omitted."""
        if figure_id is None:
            return validate_project_figures(self.project_root, namespace=namespace)
        return validate_project_figure(
            self.project_root,
            figure_id,
            namespace=namespace,
        )


__all__ = [
    "ProjectFigures",
]
