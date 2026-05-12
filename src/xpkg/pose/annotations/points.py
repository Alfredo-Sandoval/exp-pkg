"""Point primitives for pose annotations and segmentation prompts."""

from __future__ import annotations

import math
from enum import IntFlag
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np


class KPFlag(IntFlag):
    """Bit flags for per-keypoint state.

    Flags are stored natively in the point record's ``flags`` field. Visibility
    answers whether a point should be shown/used as a coordinate. ``NO_TRAIN``
    independently records that a labeled point should be excluded by downstream
    training code that honors :attr:`Point.include_in_training`.
    """

    NONE = 0
    OCCLUDED = 1 << 0
    NO_TRAIN = 1 << 1
    INTERP = 1 << 2
    LOCKED = 1 << 3


class CanonicalPoint(Protocol):
    """Canonical point interface shared by user and predicted points."""

    x: float
    y: float
    visible: bool
    complete: bool
    flags: int


class CanonicalPredictedPoint(CanonicalPoint, Protocol):
    score: float


class Point(np.record):
    """Basic 2D point record with visibility/flags metadata."""

    dtype: Any = np.dtype(
        [("x", "f8"), ("y", "f8"), ("visible", "?"), ("complete", "?"), ("flags", "u1")]
    )

    def __new__(
        cls,
        x: float = math.nan,
        y: float = math.nan,
        visible: bool = True,
        complete: bool = False,
        flags: int = 0,
    ) -> Point:
        """Instantiate a numpy record point, defaulting to NaN/visible."""
        val = PointArray(1)
        val[0] = (x, y, visible, complete, int(flags) & 0xFF)
        return cast(Point, val[0])

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def isnan(self) -> bool:
        """Return True if either coordinate is NaN using field indexing.

        Avoid attribute-style field access on numpy.record to sidestep
        `records.record.__getattribute__` during GC on some platforms.
        """
        return bool(np.isnan(self["x"]) or np.isnan(self["y"]))

    def numpy(self) -> np.ndarray:
        """Return the coordinate pair as a numpy array.

        Returns:
            np.ndarray: Array of shape (2,) containing [x, y].
        """
        return np.array([self["x"], self["y"]])

    def xy_or_none(self) -> tuple[float, float] | None:
        """Return ``(x, y)`` for labeled points, otherwise ``None``.

        Missing points are represented by NaN coordinates, not ``None`` values.
        Use this helper when branching on a single point's presence.
        """
        if self.isnan():
            return None
        return float(self["x"]), float(self["y"])

    @property
    def is_labeled(self) -> bool:
        """Return True when the point has concrete coordinates."""
        return not self.isnan()

    def has(self, flag: KPFlag) -> bool:
        """Check if a flag is set."""
        return bool(int(self["flags"]) & flag)

    def set_flag(self, flag: KPFlag, on: bool) -> None:
        """Set or unset a flag."""
        curr = int(self["flags"]) & 0xFF
        if on:
            curr |= int(flag)
        else:
            curr &= ~int(flag)
        self["flags"] = curr

    @property
    def include_in_training(self) -> bool:
        """Return True if this point should contribute to loss.

        Included when labeled and not marked NO_TRAIN.
        """
        return self.is_labeled and not self.has(KPFlag.NO_TRAIN)

    if TYPE_CHECKING:

        def __init__(
            self,
            x: float = math.nan,
            y: float = math.nan,
            visible: bool = True,
            complete: bool = False,
            flags: int = 0,
        ) -> None: ...


class PredictedPoint(Point):
    """Point record augmented with confidence + tracking metadata."""

    dtype = np.dtype(
        [
            ("x", "f8"),
            ("y", "f8"),
            ("visible", "?"),
            ("complete", "?"),
            ("score", "f8"),
            ("flags", "u1"),
        ]
    )

    def __new__(
        cls,
        x: float = math.nan,
        y: float = math.nan,
        visible: bool = True,
        complete: bool = False,
        score: float = 0.0,
        flags: int = 0,
    ) -> PredictedPoint:
        """Instantiate a predicted point with optional score/flags."""
        val = PredictedPointArray(1)
        val[0] = (x, y, visible, complete, score, int(flags) & 0xFF)
        return cast(PredictedPoint, val[0])

    @classmethod
    def from_point(cls, point: Point, score: float = 0.0) -> PredictedPoint:
        """Upgrade a Point to PredictedPoint.

        Uses dictionary-style access for all fields to avoid numpy.record's
        attribute shadowing (.flags attribute vs dtype field).
        """
        x = float(point["x"])
        y = float(point["y"])
        vis = bool(point["visible"])
        comp = bool(point["complete"])
        fval = int(point["flags"])

        d: dict[str, Any] = {
            "x": x,
            "y": y,
            "visible": vis,
            "complete": comp,
            "score": float(score),
            "flags": int(fval) & 0xFF,
        }
        return cls(**d)

    if TYPE_CHECKING:

        def __init__(
            self,
            x: float = math.nan,
            y: float = math.nan,
            visible: bool = True,
            complete: bool = False,
            score: float = 0.0,
            flags: int = 0,
        ) -> None: ...


class PointArray(np.recarray):
    """Array wrapper for Point records with helpful constructors."""

    _record_type = Point

    def __new__(
        cls,
        shape,
        buf=None,
        offset=0,
        strides=None,
        **_unused,
    ) -> PointArray:
        """Allocate a new `PointArray` view with the provided shape."""
        dtype = cls._record_type.dtype
        if dtype is not None:
            descr = np.dtype(dtype)
        else:
            raise NotImplementedError("Record dtype must be defined for Point/PredictedPoint")

        if buf is None:
            self = np.ndarray.__new__(cls, shape, (cls._record_type, descr))
        else:
            self = np.ndarray.__new__(
                cls,
                shape,
                (cls._record_type, descr),
                buffer=buf,
                offset=offset,
                strides=strides,
            )
        return self

    def __array_finalize__(self, obj):
        if obj is None:
            return
        super().__array_finalize__(obj)

    @classmethod
    def make_default(cls, size: int) -> PointArray:
        """Allocate a default `PointArray` pre-filled with zero points."""
        p = cls(size)
        p[:] = cls._record_type()
        return p

    @classmethod
    def from_array(cls, a: PointArray) -> PointArray:
        """Copy an existing `PointArray` into a new instance."""
        v = cls.make_default(len(a))
        v["x"] = a["x"]
        v["y"] = a["y"]
        v["visible"] = a["visible"]
        v["complete"] = a["complete"]
        v["flags"] = a["flags"]
        return v


class PredictedPointArray(PointArray):
    """Array wrapper for predicted points (confidence + flags)."""

    _record_type = PredictedPoint

    @classmethod
    def to_array(cls, a: PredictedPointArray) -> PointArray:
        """Convert predicted points into plain `PointArray` entries."""
        v = PointArray.make_default(len(a))
        v["x"] = a["x"]
        v["y"] = a["y"]
        v["visible"] = a["visible"]
        v["complete"] = a["complete"]
        v["flags"] = a["flags"]
        return v


PointCtor: type[Point] = Point

NormalizedPoint = Point | PredictedPoint

__all__ = [
    "CanonicalPoint",
    "CanonicalPredictedPoint",
    "KPFlag",
    "NormalizedPoint",
    "Point",
    "PointArray",
    "PointCtor",
    "PredictedPoint",
    "PredictedPointArray",
]
