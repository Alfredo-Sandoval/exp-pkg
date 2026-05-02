"""Normalized polygon sidecar-label adapters for segmentation datasets.

This module implements a small text convention from scratch: one label file per
image, one object per line, class index followed by normalized polygon
coordinates. It does not depend on any model runtime or training package.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from xpkg.segmentation.coco import mask_to_coco_polygons
from xpkg.segmentation.model import MaskType, SegmentationMask


@dataclass(frozen=True, slots=True)
class NormalizedPolygonLabel:
    """One normalized polygon sidecar-label row."""

    class_index: int
    vertices: np.ndarray

    def to_pixel_vertices(self, *, image_width: int, image_height: int) -> np.ndarray:
        """Scale normalized vertices into pixel coordinates."""

        width, height = _image_size(image_width=image_width, image_height=image_height)
        scale = np.array([float(width), float(height)], dtype=np.float32)
        return (self.vertices * scale).astype(np.float32)


def read_normalized_polygon_rows(
    path: str | Path,
    *,
    allow_out_of_bounds: bool = False,
) -> list[NormalizedPolygonLabel]:
    """Read normalized polygon label rows from a sidecar text file."""

    target = Path(path)
    if not target.exists():
        return []
    rows: list[NormalizedPolygonLabel] = []
    raw_lines = target.read_text(encoding="utf-8").splitlines()
    for line_number, raw_line in enumerate(raw_lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        rows.append(
            _parse_label_line(
                line,
                line_number=line_number,
                allow_out_of_bounds=allow_out_of_bounds,
            )
        )
    return rows


def write_normalized_polygon_rows(
    path: str | Path,
    rows: Sequence[NormalizedPolygonLabel],
    *,
    precision: int = 6,
) -> Path:
    """Write normalized polygon label rows to a sidecar text file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [_format_label_row(row, precision=precision) for row in rows]
    text = "\n".join(lines)
    if text:
        text += "\n"
    target.write_text(text, encoding="utf-8")
    return target


def read_normalized_polygon_labels(
    path: str | Path,
    *,
    image_width: int,
    image_height: int,
    class_names: Mapping[int, str] | Sequence[str] | None = None,
    allow_out_of_bounds: bool = False,
    is_predicted: bool = False,
) -> list[SegmentationMask]:
    """Read normalized polygon labels as xpkg segmentation masks."""

    rows = read_normalized_polygon_rows(
        path,
        allow_out_of_bounds=allow_out_of_bounds,
    )
    masks: list[SegmentationMask] = []
    for row_index, row in enumerate(rows):
        masks.append(
            SegmentationMask.from_polygon(
                row.to_pixel_vertices(image_width=image_width, image_height=image_height),
                class_name=_class_name(class_names, row.class_index),
                instance_ref=row_index,
                is_predicted=is_predicted,
            )
        )
    return masks


def write_normalized_polygon_labels(
    path: str | Path,
    masks: Sequence[SegmentationMask],
    *,
    image_width: int,
    image_height: int,
    class_name_to_id: Mapping[str, int] | None = None,
    allow_raster_to_polygon: bool = False,
    precision: int = 6,
) -> Path:
    """Write xpkg masks as normalized polygon sidecar labels.

    Raster/RLE masks require contour extraction and are rejected unless
    ``allow_raster_to_polygon`` is true.
    """

    width, height = _image_size(image_width=image_width, image_height=image_height)
    rows: list[NormalizedPolygonLabel] = []
    for mask in masks:
        class_index = _class_index(mask, class_name_to_id)
        polygons = _mask_polygons(mask, allow_raster_to_polygon=allow_raster_to_polygon)
        for polygon in polygons:
            rows.append(
                NormalizedPolygonLabel(
                    class_index=class_index,
                    vertices=_normalize_vertices(
                        polygon,
                        image_width=width,
                        image_height=height,
                    ),
                )
            )
    return write_normalized_polygon_rows(path, rows, precision=precision)


def read_normalized_polygon_dataset_yaml(path: str | Path) -> dict[str, Any]:
    """Read a dataset YAML file used with normalized polygon sidecar labels."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Dataset YAML must contain a mapping.")
    return dict(payload)


def write_normalized_polygon_dataset_yaml(
    path: str | Path,
    *,
    names: Mapping[int, str] | Sequence[str],
    train: str = "images/train",
    val: str = "images/val",
    test: str | None = None,
    dataset_path: str = ".",
) -> Path:
    """Write a small dataset YAML for normalized polygon sidecar labels."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "path": dataset_path,
        "train": train,
        "val": val,
        "names": _names_payload(names),
    }
    if test is not None:
        payload["test"] = test
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def _parse_label_line(
    line: str,
    *,
    line_number: int,
    allow_out_of_bounds: bool,
) -> NormalizedPolygonLabel:
    parts = line.split()
    if len(parts) < 7:
        raise ValueError(
            f"Line {line_number} must contain class index plus at least three points."
        )
    if len(parts[1:]) % 2 != 0:
        raise ValueError(f"Line {line_number} has an odd number of coordinate values.")
    try:
        class_index = int(parts[0])
    except ValueError as exc:
        raise ValueError(f"Line {line_number} has a non-integer class index.") from exc
    if class_index < 0:
        raise ValueError(f"Line {line_number} class index must be non-negative.")
    try:
        values = np.asarray([float(value) for value in parts[1:]], dtype=np.float32)
    except ValueError as exc:
        raise ValueError(f"Line {line_number} contains non-numeric coordinates.") from exc
    vertices = values.reshape((-1, 2))
    if vertices.shape[0] < 3:
        raise ValueError(f"Line {line_number} must contain at least three points.")
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f"Line {line_number} contains non-finite coordinates.")
    if not allow_out_of_bounds and (
        np.any(vertices < 0.0) or np.any(vertices > 1.0)
    ):
        raise ValueError(f"Line {line_number} contains coordinates outside [0, 1].")
    return NormalizedPolygonLabel(class_index=class_index, vertices=vertices)


def _format_label_row(row: NormalizedPolygonLabel, *, precision: int) -> str:
    values = [str(int(row.class_index))]
    values.extend(_format_float(value, precision=precision) for value in row.vertices.reshape(-1))
    return " ".join(values)


def _format_float(value: float, *, precision: int) -> str:
    text = f"{float(value):.{int(precision)}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _image_size(*, image_width: int, image_height: int) -> tuple[int, int]:
    width = int(image_width)
    height = int(image_height)
    if width <= 0 or height <= 0:
        raise ValueError("image_width and image_height must be positive.")
    return width, height


def _class_name(
    class_names: Mapping[int, str] | Sequence[str] | None,
    class_index: int,
) -> str:
    if class_names is None:
        return str(class_index)
    if isinstance(class_names, Mapping):
        mapping = cast("Mapping[int, str]", class_names)
        value = mapping.get(class_index)
        return str(class_index if value is None else value)
    if 0 <= class_index < len(class_names):
        return str(class_names[class_index])
    return str(class_index)


def _class_index(
    mask: SegmentationMask,
    class_name_to_id: Mapping[str, int] | None,
) -> int:
    clean_name = str(mask.class_name).strip()
    if class_name_to_id is not None and clean_name in class_name_to_id:
        return int(class_name_to_id[clean_name])
    if clean_name.isdigit():
        return int(clean_name)
    raise ValueError(
        f"Mask class {clean_name!r} is missing from class_name_to_id and is not numeric."
    )


def _mask_polygons(
    mask: SegmentationMask,
    *,
    allow_raster_to_polygon: bool,
) -> list[np.ndarray]:
    if mask.mask_type == MaskType.POLYGON:
        if mask.polygon_vertices is None:
            return []
        if len(mask.polygon_vertices) > 1:
            raise ValueError("Sidecar polygon labels cannot represent polygon holes.")
        return [np.asarray(mask.polygon_vertices[0], dtype=np.float32)]
    if not allow_raster_to_polygon:
        raise ValueError(
            "Raster/RLE masks require lossy contour extraction; pass "
            "allow_raster_to_polygon=True to export them."
        )
    return [
        np.asarray(values, dtype=np.float32).reshape((-1, 2))
        for values in mask_to_coco_polygons(mask.to_binary_mask())
    ]


def _normalize_vertices(
    vertices: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    arr = np.asarray(vertices, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Polygon vertices must have shape (N, 2), got {arr.shape}.")
    if arr.shape[0] < 3:
        raise ValueError("Polygon rows require at least three vertices.")
    scale = np.array([float(image_width), float(image_height)], dtype=np.float32)
    normalized = arr / scale
    if np.any(normalized < 0.0) or np.any(normalized > 1.0):
        raise ValueError("Polygon vertices normalize outside [0, 1].")
    return normalized.astype(np.float32)


def _names_payload(names: Mapping[int, str] | Sequence[str]) -> dict[int, str]:
    if isinstance(names, Mapping):
        mapping = cast("Mapping[int, str]", names)
        return {
            int(key): str(value)
            for key, value in sorted(mapping.items(), key=lambda item: int(item[0]))
        }
    return {index: str(value) for index, value in enumerate(names)}


__all__ = [
    "NormalizedPolygonLabel",
    "read_normalized_polygon_dataset_yaml",
    "read_normalized_polygon_labels",
    "read_normalized_polygon_rows",
    "write_normalized_polygon_dataset_yaml",
    "write_normalized_polygon_labels",
    "write_normalized_polygon_rows",
]
