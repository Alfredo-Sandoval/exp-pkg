"""Region-level annotation structures: ROI, segmentation masks, and prompt provenance."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xpkg.pose.annotations.instances import Track


class MaskType(enum.IntEnum):
    """Encoding type for a segmentation mask."""

    POLYGON = 0
    RLE = 1


class PromptType(enum.IntEnum):
    """How a segmentation mask was produced."""

    NONE = 0
    BOX = 1
    POINT = 2
    TEXT = 3
    POSE = 4


@dataclass
class SegmentationPrompt:
    """Records what produced a segmentation mask (provenance)."""

    prompt_type: PromptType = PromptType.NONE
    box: tuple[float, float, float, float] | None = None
    points: list[tuple[float, float, int]] | None = None
    text: str = ""
    model_id: str = ""
    backend: str = ""


@dataclass
class ROI:
    """Axis-aligned bounding box region of interest.

    Coordinates are in pixel space, XYXY format (x1, y1, x2, y2).
    """

    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str = ""
    confidence: float = float("nan")
    track: Track | None = None
    instance_ref: int = -1
    is_predicted: bool = False

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_array(self) -> np.ndarray:
        """Return [x1, y1, x2, y2] as a float32 array."""
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    def to_dict(self) -> dict:
        d: dict = {
            "x1": float(self.x1),
            "y1": float(self.y1),
            "x2": float(self.x2),
            "y2": float(self.y2),
        }
        if self.class_name:
            d["class_name"] = self.class_name
        if not np.isnan(self.confidence):
            d["confidence"] = float(self.confidence)
        if self.track is not None:
            d["track_id"] = int(self.track.id)
        if self.instance_ref >= 0:
            d["instance_ref"] = int(self.instance_ref)
        if self.is_predicted:
            d["is_predicted"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict, track: Track | None = None) -> ROI:
        return cls(
            x1=float(d["x1"]),
            y1=float(d["y1"]),
            x2=float(d["x2"]),
            y2=float(d["y2"]),
            class_name=str(d.get("class_name", "")),
            confidence=float(d.get("confidence", float("nan"))),
            track=track,
            instance_ref=int(d.get("instance_ref", -1)),
            is_predicted=bool(d.get("is_predicted", False)),
        )


@dataclass
class SegmentationMask:
    """A segmentation mask: polygon vertices or run-length encoded raster.

    For polygon masks:
        polygon_vertices is a list of (N, 2) float32 arrays.
        The first array is the exterior ring; subsequent arrays are holes.

    For RLE masks:
        rle_counts is a uint32 array of run lengths.
        rle_start is the initial pixel value (0 or 1).
        rle_height, rle_width define the raster dimensions.
    """

    mask_type: MaskType = MaskType.POLYGON
    polygon_vertices: list[np.ndarray] | None = None
    rle_counts: np.ndarray | None = None
    rle_start: int = 0
    rle_height: int = 0
    rle_width: int = 0
    class_name: str = ""
    confidence: float = float("nan")
    track: Track | None = None
    instance_ref: int = -1
    is_predicted: bool = False
    prompt: SegmentationPrompt | None = None

    def is_polygon(self) -> bool:
        return self.mask_type == MaskType.POLYGON

    def is_rle(self) -> bool:
        return self.mask_type == MaskType.RLE

    def to_binary_mask(self) -> np.ndarray:
        """Decode to a (height, width) uint8 binary mask.

        For polygon masks, rasterizes using scanline fill.
        For RLE masks, decodes run lengths.
        """
        if self.mask_type == MaskType.RLE:
            if self.rle_counts is None:
                raise ValueError("RLE mask is missing run-length counts.")
            return rle_decode(
                self.rle_counts, self.rle_start, self.rle_height, self.rle_width
            )
        if self.mask_type == MaskType.POLYGON:
            raise NotImplementedError(
                "Polygon rasterization requires height/width context; "
                "use rasterize_polygon() instead."
            )
        raise ValueError(f"Unknown mask type: {self.mask_type}")

    @property
    def bounding_box(self) -> np.ndarray:
        """Compute tight XYXY bounding box from mask geometry."""
        if self.mask_type == MaskType.POLYGON and self.polygon_vertices:
            all_pts = np.concatenate(self.polygon_vertices, axis=0)
            mins = np.min(all_pts, axis=0)
            maxs = np.max(all_pts, axis=0)
            return np.array([mins[0], mins[1], maxs[0], maxs[1]], dtype=np.float32)
        if self.mask_type == MaskType.RLE and self.rle_counts is not None:
            mask = self.to_binary_mask()
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any():
                return np.array([0, 0, 0, 0], dtype=np.float32)
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            return np.array([x1, y1, x2 + 1, y2 + 1], dtype=np.float32)
        return np.array([0, 0, 0, 0], dtype=np.float32)

    @property
    def vertex_count(self) -> int:
        """Total number of polygon vertices across all rings."""
        if self.polygon_vertices is None:
            return 0
        return sum(v.shape[0] for v in self.polygon_vertices)

    def to_dict(self) -> dict:
        d: dict = {"mask_type": self.mask_type.name.lower()}
        if self.mask_type == MaskType.POLYGON and self.polygon_vertices is not None:
            d["polygon_vertices"] = [v.tolist() for v in self.polygon_vertices]
        elif self.mask_type == MaskType.RLE and self.rle_counts is not None:
            d["rle"] = {
                "counts": self.rle_counts.tolist(),
                "start": int(self.rle_start),
                "height": int(self.rle_height),
                "width": int(self.rle_width),
            }
        if self.class_name:
            d["class_name"] = self.class_name
        if not np.isnan(self.confidence):
            d["confidence"] = float(self.confidence)
        if self.track is not None:
            d["track_id"] = int(self.track.id)
        if self.instance_ref >= 0:
            d["instance_ref"] = int(self.instance_ref)
        if self.is_predicted:
            d["is_predicted"] = True
        if self.prompt is not None and self.prompt.prompt_type != PromptType.NONE:
            d["prompt"] = _prompt_to_dict(self.prompt)
        return d

    @classmethod
    def from_dict(cls, d: dict, track: Track | None = None) -> SegmentationMask:
        mask_type = MaskType[str(d.get("mask_type", "polygon")).upper()]
        polygon_vertices = None
        rle_counts = None
        rle_start = 0
        rle_height = 0
        rle_width = 0
        if mask_type == MaskType.POLYGON and "polygon_vertices" in d:
            polygon_vertices = [
                np.asarray(ring, dtype=np.float32) for ring in d["polygon_vertices"]
            ]
        elif mask_type == MaskType.RLE and "rle" in d:
            rle_data = d["rle"]
            rle_counts = np.asarray(rle_data["counts"], dtype=np.uint32)
            rle_start = int(rle_data.get("start", 0))
            rle_height = int(rle_data["height"])
            rle_width = int(rle_data["width"])

        prompt = None
        if "prompt" in d and d["prompt"] is not None:
            prompt = _prompt_from_dict(d["prompt"])

        return cls(
            mask_type=mask_type,
            polygon_vertices=polygon_vertices,
            rle_counts=rle_counts,
            rle_start=rle_start,
            rle_height=rle_height,
            rle_width=rle_width,
            class_name=str(d.get("class_name", "")),
            confidence=float(d.get("confidence", float("nan"))),
            track=track,
            instance_ref=int(d.get("instance_ref", -1)),
            is_predicted=bool(d.get("is_predicted", False)),
            prompt=prompt,
        )

    @classmethod
    def from_polygon(
        cls,
        vertices: list[np.ndarray] | np.ndarray,
        *,
        class_name: str = "",
        confidence: float = float("nan"),
        track: Track | None = None,
        instance_ref: int = -1,
    ) -> SegmentationMask:
        """Create a polygon mask from vertex arrays."""
        if isinstance(vertices, np.ndarray):
            vertices = [vertices]
        return cls(
            mask_type=MaskType.POLYGON,
            polygon_vertices=[np.asarray(v, dtype=np.float32) for v in vertices],
            class_name=class_name,
            confidence=confidence,
            track=track,
            instance_ref=instance_ref,
        )

    @classmethod
    def from_rle(
        cls,
        counts: np.ndarray | list[int],
        start: int,
        height: int,
        width: int,
        *,
        class_name: str = "",
        confidence: float = float("nan"),
        track: Track | None = None,
        instance_ref: int = -1,
    ) -> SegmentationMask:
        """Create an RLE mask from run-length data."""
        return cls(
            mask_type=MaskType.RLE,
            rle_counts=np.asarray(counts, dtype=np.uint32),
            rle_start=int(start),
            rle_height=int(height),
            rle_width=int(width),
            class_name=class_name,
            confidence=confidence,
            track=track,
            instance_ref=instance_ref,
        )

    @classmethod
    def from_binary_mask(
        cls,
        mask: np.ndarray,
        *,
        class_name: str = "",
        confidence: float = float("nan"),
        track: Track | None = None,
        instance_ref: int = -1,
    ) -> SegmentationMask:
        """Create an RLE mask from a (height, width) binary array."""
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D binary mask, got shape {mask.shape}")
        counts, start = rle_encode(mask)
        return cls(
            mask_type=MaskType.RLE,
            rle_counts=counts,
            rle_start=start,
            rle_height=int(mask.shape[0]),
            rle_width=int(mask.shape[1]),
            class_name=class_name,
            confidence=confidence,
            track=track,
            instance_ref=instance_ref,
        )


# ---------------------------------------------------------------------------
# RLE encode / decode
# ---------------------------------------------------------------------------


def rle_encode(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Encode a 2D binary mask as run-length counts.

    Returns:
        (counts, start) where counts is uint32 run lengths and start is 0 or 1.
    """
    flat = mask.ravel().astype(np.uint8, copy=False)
    if flat.size == 0:
        return np.zeros(0, dtype=np.uint32), 0
    start = int(flat[0])
    diffs = np.diff(flat)
    change_indices = np.flatnonzero(diffs)
    boundaries = np.concatenate([[0], change_indices + 1, [flat.size]])
    counts = np.diff(boundaries).astype(np.uint32)
    return counts, start


def rle_decode(
    counts: np.ndarray | list[int],
    start: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Decode RLE run-length counts into a (height, width) binary mask."""
    counts_arr = np.asarray(counts, dtype=np.uint32)
    total_pixels = height * width
    if counts_arr.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    actual_total = int(counts_arr.sum())
    if actual_total != total_pixels:
        raise ValueError(
            f"RLE counts sum to {actual_total} but expected {total_pixels} "
            f"({height}x{width})"
        )
    flat = np.empty(total_pixels, dtype=np.uint8)
    current_val = int(start)
    offset = 0
    for count in counts_arr:
        c = int(count)
        flat[offset : offset + c] = current_val
        offset += c
        current_val = 1 - current_val
    return flat.reshape(height, width)


# ---------------------------------------------------------------------------
# Prompt serialization helpers
# ---------------------------------------------------------------------------


def _prompt_to_dict(prompt: SegmentationPrompt) -> dict:
    d: dict = {"type": prompt.prompt_type.name.lower()}
    if prompt.box is not None:
        d["box"] = list(prompt.box)
    if prompt.points is not None:
        d["points"] = [list(p) for p in prompt.points]
    if prompt.text:
        d["text"] = prompt.text
    if prompt.model_id:
        d["model_id"] = prompt.model_id
    if prompt.backend:
        d["backend"] = prompt.backend
    return d


def _prompt_from_dict(d: dict) -> SegmentationPrompt:
    pt = PromptType[str(d.get("type", "none")).upper()]
    box = None
    if "box" in d and d["box"] is not None:
        b = d["box"]
        box = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    points = None
    if "points" in d and d["points"] is not None:
        points = [(float(p[0]), float(p[1]), int(p[2])) for p in d["points"]]
    return SegmentationPrompt(
        prompt_type=pt,
        box=box,
        points=points,
        text=str(d.get("text", "")),
        model_id=str(d.get("model_id", "")),
        backend=str(d.get("backend", "")),
    )


__all__ = [
    "MaskType",
    "PromptType",
    "ROI",
    "SegmentationMask",
    "SegmentationPrompt",
    "rle_decode",
    "rle_encode",
]
