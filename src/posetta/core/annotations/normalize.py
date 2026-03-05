"""Normalization helpers for point-like payloads."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from posetta.core.annotations.points import (
    CanonicalPoint,
    NormalizedPoint,
    Point,
    PointCtor,
    PredictedPoint,
)

PointData = CanonicalPoint | Mapping[str, Any]


def _flags_from_mapping(point: Mapping[str, Any]) -> int:
    raw_flags = point["flags"] if "flags" in point else 0
    return int(raw_flags) & 0xFF


def _point_from_mapping(point: Mapping[str, Any]) -> NormalizedPoint:
    if "x" not in point or "y" not in point:
        raise KeyError("Point mapping requires 'x' and 'y' entries.")

    visible = bool(point["visible"]) if "visible" in point else True
    complete = bool(point["complete"]) if "complete" in point else False
    flags_val = _flags_from_mapping(point)

    if "score" in point:
        return PredictedPoint(
            x=float(point["x"]),
            y=float(point["y"]),
            visible=visible,
            complete=complete,
            score=float(point["score"]),
            flags=flags_val,
        )

    return PointCtor(
        x=float(point["x"]),
        y=float(point["y"]),
        visible=visible,
        complete=complete,
        flags=flags_val,
    )


def normalize_point_like(point: PointData) -> NormalizedPoint:
    """Normalize point-like payloads into the canonical Point types."""
    if isinstance(point, Point | PredictedPoint):
        return point
    if isinstance(point, Mapping):
        normalized: dict[str, Any] = {}
        for key, value in point.items():
            if not isinstance(key, str):
                raise TypeError("Point mapping keys must be strings.")
            normalized[key] = value
        return _point_from_mapping(normalized)
    raise TypeError("Point data must be a Point, PredictedPoint, or coordinate mapping.")


def normalize_points_sequence(points: Iterable[PointData]) -> tuple[NormalizedPoint, ...]:
    """Normalize an iterable of point-like payloads into a tuple of Points."""
    return tuple(normalize_point_like(point) for point in points)


__all__ = ["PointData", "normalize_point_like", "normalize_points_sequence"]
