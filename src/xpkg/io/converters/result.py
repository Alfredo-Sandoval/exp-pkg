"""Conversion result envelope returned by every external-format importer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ConversionResult:
    """Outcome of converting an external data format into project state."""

    source_dir: Path
    project_root: Path
    videos: list[Path]
    labels: Any
    metadata: dict[str, Any]


__all__ = ["ConversionResult"]
