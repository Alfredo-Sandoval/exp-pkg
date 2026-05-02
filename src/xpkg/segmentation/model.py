"""Segmentation masks, ROIs, prompts, and frame-level result models."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from xpkg.segmentation.rle import (
    decode_mask_rle,
    rle_decode,
    rle_encode,
)

if TYPE_CHECKING:
    from xpkg.pose.annotations.instances import Track


class MaskType(enum.IntEnum):
    """Encoding type for a segmentation mask."""

    POLYGON = 0
    RLE = 1
    MASK_REF = 2


class PromptType(enum.IntEnum):
    """How a segmentation mask was produced."""

    NONE = 0
    BOX = 1
    POINT = 2
    TEXT = 3
    POSE = 4


@dataclass
class SegmentationPrompt:
    """Records what produced a segmentation mask."""

    prompt_type: PromptType = PromptType.NONE
    box: tuple[float, float, float, float] | None = None
    points: list[tuple[float, float, int]] | None = None
    text: str = ""
    model_id: str = ""
    backend: str = ""


@dataclass
class ROI:
    """Axis-aligned bounding box region of interest.

    Coordinates are in pixel space, XYXY format: ``x1, y1, x2, y2``.
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
        """Return ``[x1, y1, x2, y2]`` as a float32 array."""

        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "x1": float(self.x1),
            "y1": float(self.y1),
            "x2": float(self.x2),
            "y2": float(self.y2),
        }
        if self.class_name:
            payload["class_name"] = self.class_name
        if not np.isnan(self.confidence):
            payload["confidence"] = float(self.confidence)
        if self.track is not None:
            payload["track_id"] = int(self.track.id)
        if self.instance_ref >= 0:
            payload["instance_ref"] = int(self.instance_ref)
        if self.is_predicted:
            payload["is_predicted"] = True
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any], track: Track | None = None) -> ROI:
        return cls(
            x1=float(payload["x1"]),
            y1=float(payload["y1"]),
            x2=float(payload["x2"]),
            y2=float(payload["y2"]),
            class_name=str(payload.get("class_name", "")),
            confidence=float(payload.get("confidence", float("nan"))),
            track=track,
            instance_ref=int(payload.get("instance_ref", -1)),
            is_predicted=bool(payload.get("is_predicted", False)),
        )


@dataclass
class SegmentationMask:
    """A segmentation mask encoded as polygon vertices, RLE, or an artifact ref."""

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
    mask_id: str = ""
    artifact_ref: str = ""
    mask_path: str = ""

    def is_polygon(self) -> bool:
        return self.mask_type == MaskType.POLYGON

    def is_rle(self) -> bool:
        return self.mask_type == MaskType.RLE

    def is_mask_ref(self) -> bool:
        return self.mask_type == MaskType.MASK_REF

    def to_binary_mask(
        self,
        *,
        height: int | None = None,
        width: int | None = None,
    ) -> np.ndarray:
        """Decode or rasterize this mask into a ``uint8`` binary array."""

        if self.mask_type == MaskType.RLE:
            if self.rle_counts is None:
                raise ValueError("RLE mask is missing run-length counts.")
            return rle_decode(
                self.rle_counts,
                self.rle_start,
                self.rle_height,
                self.rle_width,
            )
        if self.mask_type == MaskType.POLYGON:
            if height is None or width is None:
                raise ValueError("Polygon rasterization requires height and width.")
            if self.polygon_vertices is None:
                return np.zeros((height, width), dtype=np.uint8)
            return rasterize_polygon(self.polygon_vertices, height=height, width=width)
        if self.mask_type == MaskType.MASK_REF:
            raise ValueError("Mask references must be loaded through xpkg.segmentation.images.")
        raise ValueError(f"Unknown mask type: {self.mask_type}")

    def to_rle_payload(self) -> dict[str, Any]:
        """Return the canonical xpkg RLE payload for this mask."""

        if self.mask_type != MaskType.RLE:
            raise ValueError("Only RLE masks can be serialized directly as RLE payloads.")
        if self.rle_counts is None:
            raise ValueError("RLE mask is missing run-length counts.")
        return {
            "encoding": "xpkg.rle.v1",
            "size": [int(self.rle_height), int(self.rle_width)],
            "order": "C",
            "start": int(self.rle_start),
            "counts": [int(value) for value in self.rle_counts.tolist()],
        }

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
        return sum(vertices.shape[0] for vertices in self.polygon_vertices)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"mask_type": self.mask_type.name.lower()}
        if self.mask_type == MaskType.POLYGON and self.polygon_vertices is not None:
            payload["polygon_vertices"] = [
                vertices.tolist() for vertices in self.polygon_vertices
            ]
        elif self.mask_type == MaskType.RLE and self.rle_counts is not None:
            payload["rle"] = self.to_rle_payload() | {
                "height": int(self.rle_height),
                "width": int(self.rle_width),
            }
        elif self.mask_type == MaskType.MASK_REF:
            if self.mask_path:
                payload["mask_path"] = self.mask_path
            if self.artifact_ref:
                payload["artifact_ref"] = self.artifact_ref

        if self.class_name:
            payload["class_name"] = self.class_name
        if not np.isnan(self.confidence):
            payload["confidence"] = float(self.confidence)
        if self.track is not None:
            payload["track_id"] = int(self.track.id)
        if self.instance_ref >= 0:
            payload["instance_ref"] = int(self.instance_ref)
        if self.is_predicted:
            payload["is_predicted"] = True
        if self.prompt is not None and self.prompt.prompt_type != PromptType.NONE:
            payload["prompt"] = _prompt_to_dict(self.prompt)
        if self.mask_id:
            payload["mask_id"] = self.mask_id
        if self.artifact_ref and self.mask_type != MaskType.MASK_REF:
            payload["artifact_ref"] = self.artifact_ref
        if self.mask_path and self.mask_type != MaskType.MASK_REF:
            payload["mask_path"] = self.mask_path
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        track: Track | None = None,
    ) -> SegmentationMask:
        mask_type = MaskType[str(payload.get("mask_type", "polygon")).upper()]
        polygon_vertices = None
        rle_counts = None
        rle_start = 0
        rle_height = 0
        rle_width = 0
        if mask_type == MaskType.POLYGON and "polygon_vertices" in payload:
            polygon_vertices = _coerce_polygon_vertices(payload["polygon_vertices"])
        elif mask_type == MaskType.RLE and "rle" in payload:
            rle_data = payload["rle"]
            rle_counts = np.asarray(rle_data["counts"], dtype=np.uint32)
            rle_start = int(rle_data.get("start", 0))
            if "size" in rle_data:
                size = rle_data["size"]
                rle_height = int(size[0])
                rle_width = int(size[1])
            else:
                rle_height = int(rle_data["height"])
                rle_width = int(rle_data["width"])
            rle_decode(rle_counts, rle_start, rle_height, rle_width)

        prompt = None
        if "prompt" in payload and payload["prompt"] is not None:
            prompt = _prompt_from_dict(payload["prompt"])

        return cls(
            mask_type=mask_type,
            polygon_vertices=polygon_vertices,
            rle_counts=rle_counts,
            rle_start=rle_start,
            rle_height=rle_height,
            rle_width=rle_width,
            class_name=str(payload.get("class_name", "")),
            confidence=float(payload.get("confidence", float("nan"))),
            track=track,
            instance_ref=int(payload.get("instance_ref", -1)),
            is_predicted=bool(payload.get("is_predicted", False)),
            prompt=prompt,
            mask_id=str(payload.get("mask_id", "")),
            artifact_ref=str(payload.get("artifact_ref", "")),
            mask_path=str(payload.get("mask_path", "")),
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
        is_predicted: bool = False,
        prompt: SegmentationPrompt | None = None,
        mask_id: str = "",
        artifact_ref: str = "",
        mask_path: str = "",
    ) -> SegmentationMask:
        """Create a polygon mask from vertex arrays."""

        return cls(
            mask_type=MaskType.POLYGON,
            polygon_vertices=_coerce_polygon_vertices(vertices),
            class_name=class_name,
            confidence=confidence,
            track=track,
            instance_ref=instance_ref,
            is_predicted=is_predicted,
            prompt=prompt,
            mask_id=mask_id,
            artifact_ref=artifact_ref,
            mask_path=mask_path,
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
        is_predicted: bool = False,
        prompt: SegmentationPrompt | None = None,
        mask_id: str = "",
        artifact_ref: str = "",
        mask_path: str = "",
    ) -> SegmentationMask:
        """Create an RLE mask from row-major run-length data."""

        counts_arr = np.asarray(counts, dtype=np.uint32)
        rle_decode(counts_arr, int(start), int(height), int(width))
        return cls(
            mask_type=MaskType.RLE,
            rle_counts=counts_arr,
            rle_start=int(start),
            rle_height=int(height),
            rle_width=int(width),
            class_name=class_name,
            confidence=confidence,
            track=track,
            instance_ref=instance_ref,
            is_predicted=is_predicted,
            prompt=prompt,
            mask_id=mask_id,
            artifact_ref=artifact_ref,
            mask_path=mask_path,
        )

    @classmethod
    def from_rle_payload(
        cls,
        payload: dict[str, Any],
        *,
        class_name: str = "",
        confidence: float = float("nan"),
        track: Track | None = None,
        instance_ref: int = -1,
        is_predicted: bool = False,
        prompt: SegmentationPrompt | None = None,
        mask_id: str = "",
        artifact_ref: str = "",
        mask_path: str = "",
    ) -> SegmentationMask:
        """Create a mask from canonical xpkg RLE payload."""

        decoded = decode_mask_rle(payload)
        return cls.from_binary_mask(
            decoded,
            class_name=class_name,
            confidence=confidence,
            track=track,
            instance_ref=instance_ref,
            is_predicted=is_predicted,
            prompt=prompt,
            mask_id=mask_id,
            artifact_ref=artifact_ref,
            mask_path=mask_path,
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
        is_predicted: bool = False,
        prompt: SegmentationPrompt | None = None,
        mask_id: str = "",
        artifact_ref: str = "",
        mask_path: str = "",
    ) -> SegmentationMask:
        """Create an RLE mask from a 2D binary array."""

        if np.asarray(mask).ndim != 2:
            raise ValueError(f"Expected 2D binary mask, got shape {np.asarray(mask).shape}")
        counts, start = rle_encode(mask)
        return cls(
            mask_type=MaskType.RLE,
            rle_counts=counts,
            rle_start=start,
            rle_height=int(np.asarray(mask).shape[0]),
            rle_width=int(np.asarray(mask).shape[1]),
            class_name=class_name,
            confidence=confidence,
            track=track,
            instance_ref=instance_ref,
            is_predicted=is_predicted,
            prompt=prompt,
            mask_id=mask_id,
            artifact_ref=artifact_ref,
            mask_path=mask_path,
        )

    @classmethod
    def from_mask_ref(
        cls,
        path: str | Path,
        *,
        class_name: str = "",
        confidence: float = float("nan"),
        track: Track | None = None,
        instance_ref: int = -1,
        is_predicted: bool = False,
        prompt: SegmentationPrompt | None = None,
        mask_id: str = "",
        artifact_ref: str = "",
    ) -> SegmentationMask:
        """Create a metadata-only mask reference to an external mask image."""

        return cls(
            mask_type=MaskType.MASK_REF,
            class_name=class_name,
            confidence=confidence,
            track=track,
            instance_ref=instance_ref,
            is_predicted=is_predicted,
            prompt=prompt,
            mask_id=mask_id,
            artifact_ref=artifact_ref,
            mask_path=Path(path).as_posix(),
        )


@dataclass(frozen=True, slots=True)
class SegmentationFrame:
    """Segmentation masks and ROIs associated with one source frame."""

    frame_index: int
    masks: tuple[SegmentationMask, ...] = ()
    rois: tuple[ROI, ...] = ()
    video_id: str = ""
    video_label: str = ""
    video_path: str = ""


def rasterize_polygon(
    vertices: list[np.ndarray] | np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterize polygon rings into a binary ``uint8`` mask.

    The first ring is filled as foreground. Subsequent rings are treated as
    holes, matching the existing polygon-mask documentation.
    """

    if height < 0 or width < 0:
        raise ValueError("height and width must be non-negative.")
    rings = _coerce_polygon_vertices(vertices)
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    if not rings:
        return mask
    exterior = np.round(rings[0]).astype(np.int32)
    cv2.fillPoly(mask, [exterior], color=1)
    for hole in rings[1:]:
        cv2.fillPoly(mask, [np.round(hole).astype(np.int32)], color=0)
    return mask


def _coerce_polygon_vertices(vertices: list[np.ndarray] | np.ndarray | Any) -> list[np.ndarray]:
    if isinstance(vertices, np.ndarray):
        raw_vertices = [vertices]
    else:
        raw_vertices = list(vertices)
    coerced: list[np.ndarray] = []
    for raw in raw_vertices:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                "Polygon vertices must be arrays with shape (N, 2); "
                f"got {arr.shape}"
            )
        if arr.shape[0] < 3:
            raise ValueError("Polygon rings must contain at least three vertices.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Polygon vertices must be finite.")
        coerced.append(arr)
    return coerced


def _prompt_to_dict(prompt: SegmentationPrompt) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": prompt.prompt_type.name.lower()}
    if prompt.box is not None:
        payload["box"] = list(prompt.box)
    if prompt.points is not None:
        payload["points"] = [list(point) for point in prompt.points]
    if prompt.text:
        payload["text"] = prompt.text
    if prompt.model_id:
        payload["model_id"] = prompt.model_id
    if prompt.backend:
        payload["backend"] = prompt.backend
    return payload


def _prompt_from_dict(payload: dict[str, Any]) -> SegmentationPrompt:
    prompt_type = PromptType[str(payload.get("type", "none")).upper()]
    box = None
    if "box" in payload and payload["box"] is not None:
        raw_box = payload["box"]
        box = (float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3]))
    points = None
    if "points" in payload and payload["points"] is not None:
        points = [
            (float(point[0]), float(point[1]), int(point[2]))
            for point in payload["points"]
        ]
    return SegmentationPrompt(
        prompt_type=prompt_type,
        box=box,
        points=points,
        text=str(payload.get("text", "")),
        model_id=str(payload.get("model_id", "")),
        backend=str(payload.get("backend", "")),
    )


__all__ = [
    "MaskType",
    "PromptType",
    "ROI",
    "SegmentationFrame",
    "SegmentationMask",
    "SegmentationPrompt",
    "rasterize_polygon",
]
