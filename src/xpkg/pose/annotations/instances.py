"""Instance-level annotation structures (skeleton + points + track metadata)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import InitVar, asdict, dataclass, field
from typing import TYPE_CHECKING, Any, TypeGuard, cast, overload

import numpy as np

from xpkg.pose.annotations.points import (
    Point,
    PointArray,
    PointCtor,
    PredictedPoint,
    PredictedPointArray,
)
from xpkg.pose.skeleton import Keypoint, Skeleton

if TYPE_CHECKING:
    from xpkg.media.video import Video
    from xpkg.pose.annotations.frames import LabeledFrame
else:
    from typing import Any as LabeledFrame
    from typing import Any as Video

_KeypointIndex = str | Keypoint | int
_KeypointIndexSequence = list[_KeypointIndex] | tuple[_KeypointIndex, ...]
_PointValueSequence = list[Point] | tuple[Point, ...] | np.ndarray


def _is_keypoint_index_sequence(value: object) -> TypeGuard[_KeypointIndexSequence]:
    return isinstance(value, list | tuple)


def _is_point_value_sequence(value: object) -> TypeGuard[_PointValueSequence]:
    return isinstance(value, list | tuple | np.ndarray)


def _ndarray_to_list(value: np.ndarray) -> list[Any]:
    raw = cast(Any, value).tolist()
    return raw if isinstance(raw, list) else [raw]


@dataclass(eq=False)
class Track:
    """Descriptor representing track metadata (spawn time + name)."""

    spawned_on: int = 0
    name: str = ""

    def matches(self, other: Track):
        """Return True when the contents of two tracks are identical."""
        return asdict(self) == asdict(other)

    @property
    def id(self) -> int:
        """Return the stable identifier for this track (mirrors spawned_on)."""
        return int(self.spawned_on)


@dataclass
class Instance:
    """Labeled instance data linking skeletons, points, and predictions."""

    skeleton: Skeleton
    track: Track | None = None
    from_predicted: PredictedInstance | None = None
    frame: LabeledFrame | None = None
    tracking_score: float = 0.0
    init_points: InitVar[PointArray | Mapping[str | Keypoint, Point] | None] = None

    _points: PointArray = field(init=False, repr=False)
    _keypoints: list[Keypoint] = field(init=False, repr=False)
    _point_array_type = PointArray

    def __post_init__(
        self, init_points: PointArray | Mapping[str | Keypoint, Point] | None
    ) -> None:
        if self.skeleton is None:
            raise ValueError("No skeleton set for Instance")

        if self.from_predicted is not None and not isinstance(
            self.from_predicted, PredictedInstance
        ):
            raise TypeError(
                "Instance.from_predicted type must be PredictedInstance "
                f"(not {type(self.from_predicted)})"
            )

        self._points = self._coerce_init_points(init_points)

        self._keypoints = list(self.skeleton.keypoints)

    def _coerce_init_points(
        self, init_points: PointArray | Mapping[str | Keypoint, Point] | None
    ) -> PointArray:
        if init_points is None:
            return self._point_array_type.make_default(len(self.skeleton.keypoints))
        if isinstance(init_points, PointArray):
            return init_points
        if isinstance(init_points, Mapping):
            return self._coerce_mapping_init_points(init_points)
        raise TypeError("Instance points must be PointArray, dict, or None.")

    def _coerce_mapping_init_points(
        self, init_points: Mapping[str | Keypoint, Point]
    ) -> PointArray:
        parray = self._point_array_type.make_default(len(self.skeleton.keypoints))
        normalized = Instance._normalize_points_dict(dict(init_points))
        Instance._points_dict_to_array(normalized, parray, self.skeleton)
        return parray

    @staticmethod
    def _normalize_key(key: str | Keypoint | int | np.integer[Any]) -> str | Keypoint | int:
        if isinstance(key, np.integer):
            return int(key)
        if isinstance(key, str | Keypoint | int):
            return key
        raise TypeError("Keypoint key must be a str, Keypoint, or int.")

    @staticmethod
    def _normalize_key_seq(keys: Iterable[Any]) -> list[str | Keypoint | int]:
        return [Instance._normalize_key(key) for key in keys]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Instance):
            return NotImplemented
        if type(self) is not type(other):
            return NotImplemented
        return (
            self.skeleton == other.skeleton
            and self.track == other.track
            and self.from_predicted == other.from_predicted
            and self.frame == other.frame
            and self.tracking_score == other.tracking_score
            and self._keypoints == other._keypoints
            and np.array_equal(self._points, other._points)
        )

    @staticmethod
    def _normalize_points_dict(
        points: Mapping[str | Keypoint, Point],
    ) -> dict[str | Keypoint, Point]:
        normalized: dict[str | Keypoint, Point] = {}
        for key, value in points.items():
            if not isinstance(key, str | Keypoint):
                raise TypeError("Instance point keys must be strings or Keypoints.")
            if not isinstance(value, Point):
                raise TypeError("Instance point values must be Point-like records.")
            normalized[key] = value
        return normalized

    @staticmethod
    def _coerce_coordinate(value: object, *, axis: str) -> float:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool | int | float):
            return float(value)
        raise ValueError(f"Instance point {axis} coordinate must be numeric.")

    @classmethod
    def _coerce_point_value(cls, value: object) -> Point:
        if isinstance(value, Point):
            return value
        if isinstance(value, np.ndarray):
            if value.shape != (2,):
                raise ValueError("Instance point values must be (x, y) coordinates.")
            coords = tuple(cast(Iterable[Any], value))
            if len(coords) != 2:
                raise ValueError("Instance point values must be (x, y) coordinates.")
            x = cls._coerce_coordinate(coords[0], axis="x")
            y = cls._coerce_coordinate(coords[1], axis="y")
            return PointCtor(x=x, y=y)
        if isinstance(value, Sequence) and not isinstance(value, bytes | str):
            coords = tuple(cast(Sequence[Any], value))
            if len(coords) != 2:
                raise ValueError("Instance point values must be (x, y) coordinates.")
            x = cls._coerce_coordinate(coords[0], axis="x")
            y = cls._coerce_coordinate(coords[1], axis="y")
            return PointCtor(x=x, y=y)
        raise ValueError("Instance point values must be (x, y) coordinates.")

    @staticmethod
    def _points_dict_to_array(
        points: dict[str | Keypoint, Point], parray: PointArray, skeleton: Skeleton
    ):
        if not points:
            return
        is_string_dict = set(map(type, points)) == {str}
        is_kp_dict = set(map(type, points)) == {Keypoint}
        if is_string_dict:
            mapped: dict[str | Keypoint, Point] = {}
            for name, point in points.items():
                kp = skeleton.get_keypoint_by_name(str(name))
                if kp is None:
                    raise KeyError(f"Unknown keypoint: {name}")
                mapped[kp] = point
            points = mapped
        if not is_string_dict and not is_kp_dict:
            raise ValueError(
                "points dictionary must be keyed by either strings "
                + "(keypoint names) or Keypoints."
            )
        for keypoint, point in points.items():
            idx = skeleton.keypoint_to_index(keypoint)
            dst = parray[idx]
            dst["x"] = float(point["x"])
            dst["y"] = float(point["y"])
            dst["visible"] = bool(point["visible"])
            dst["complete"] = bool(point["complete"])
            dst["flags"] = int(point["flags"]) & 0xFF
            # Copy score when destination supports it and source has it
            if "score" in dst.dtype.names and "score" in point.dtype.names:
                dst["score"] = float(point["score"])

    def _keypoint_to_index(self, keypoint: str | Keypoint) -> int:
        return self.skeleton.keypoint_to_index(keypoint)

    def _assert_points_synced(self) -> None:
        if not isinstance(self._points, PointArray):
            raise TypeError("Instance points container must be a PointArray.")
        expected_keypoints = list(self.skeleton.keypoints)
        if len(self._points) != len(expected_keypoints):
            raise ValueError(
                "Instance points length does not match skeleton keypoints; "
                "call realign_points() to resync."
            )
        if self._keypoints != expected_keypoints:
            raise ValueError(
                "Instance keypoints are out of sync with skeleton ordering; "
                "call realign_points() to resync."
            )

    def _ensure_point_records(self) -> PointArray:
        self._assert_points_synced()
        return self._points

    def realign_points(self) -> None:
        """Realign the internal points array to match the current skeleton ordering.

        Call this after a skeleton has been mutated (e.g. keypoints added,
        removed, or reordered) so that each point stays associated with the
        correct keypoint. Keypoints that no longer exist in the skeleton are
        dropped; new skeleton keypoints receive default (empty) point values.

        This is also the recovery path for the ``ValueError`` raised by
        :meth:`_assert_points_synced` when the instance is out of sync.
        """
        skeleton_kps = list(self.skeleton.keypoints)
        if not isinstance(self._points, PointArray):
            raise TypeError("Instance points container must be a PointArray.")
        points = self._points
        cls = type(points)
        new_len = len(skeleton_kps)
        new_array = cls.make_default(new_len)
        for i, kp in enumerate(self._keypoints):
            if i >= len(points):
                break
            if kp not in skeleton_kps:
                continue
            new_index = skeleton_kps.index(kp)
            new_array[new_index] = points[i]
        self._points = new_array
        self._keypoints = skeleton_kps

    @overload
    def __getitem__(self, keypoint: str | Keypoint | int | np.integer[Any]) -> Point: ...

    @overload
    def __getitem__(
        self, keypoint: list[str | Keypoint | int] | tuple[str | Keypoint | int, ...]
    ) -> list[Point]: ...

    @overload
    def __getitem__(self, keypoint: np.ndarray) -> np.ndarray: ...

    def __getitem__(
        self,
        keypoint: str
        | Keypoint
        | int
        | np.integer[Any]
        | list[str | Keypoint | int]
        | tuple[str | Keypoint | int, ...]
        | np.ndarray,
    ) -> Point | list[Point] | np.ndarray:
        self._assert_points_synced()
        if isinstance(keypoint, np.ndarray):
            kp_raw = _ndarray_to_list(keypoint)
            kp_list = Instance._normalize_key_seq(cast(Iterable[Any], kp_raw))
            pts = [self._get_point_at(kp) for kp in kp_list]
            return np.array([[pt.x, pt.y] for pt in pts])
        if _is_keypoint_index_sequence(keypoint):
            kp_list = Instance._normalize_key_seq(cast(Iterable[Any], keypoint))
            pts = [self._get_point_at(kp) for kp in kp_list]
            return pts
        normalized_key = Instance._normalize_key(
            cast(str | Keypoint | int | np.integer[Any], keypoint)
        )
        return self._get_point_at(normalized_key)

    def __contains__(self, keypoint: str | Keypoint | int | np.integer[Any]) -> bool:
        normalized_key = Instance._normalize_key(keypoint)
        if isinstance(normalized_key, Keypoint):
            keypoint_name = normalized_key.name
        elif isinstance(normalized_key, str):
            keypoint_name = normalized_key
        else:
            keypoint_name = None

        if keypoint_name is not None:
            if keypoint_name not in self.skeleton:
                return False
            normalized_key = self._keypoint_to_index(keypoint_name)
        points = self._ensure_point_records()
        if (
            not isinstance(normalized_key, int)
            or normalized_key < 0
            or normalized_key >= len(points)
        ):
            return False
        return not points[normalized_key].isnan()

    @overload
    def __setitem__(
        self, keypoint: str | Keypoint | int | np.integer[Any], value: Point
    ) -> None: ...

    @overload
    def __setitem__(
        self,
        keypoint: list[str | Keypoint | int] | tuple[str | Keypoint | int, ...],
        value: list[Point] | tuple[Point, ...],
    ) -> None: ...

    @overload
    def __setitem__(self, keypoint: np.ndarray, value: np.ndarray) -> None: ...

    def __setitem__(
        self,
        keypoint: str
        | Keypoint
        | int
        | np.integer[Any]
        | list[str | Keypoint | int]
        | tuple[str | Keypoint | int, ...]
        | np.ndarray,
        value: Point | list[Point] | tuple[Point, ...] | np.ndarray,
    ) -> None:
        self._assert_points_synced()
        if isinstance(keypoint, np.ndarray):
            keypoint_list = Instance._normalize_key_seq(
                cast(Iterable[Any], _ndarray_to_list(keypoint))
            )
        elif _is_keypoint_index_sequence(keypoint):
            keypoint_list = Instance._normalize_key_seq(cast(Iterable[Any], keypoint))
        else:
            normalized_key = Instance._normalize_key(
                cast(str | Keypoint | int | np.integer[Any], keypoint)
            )
            self._set_point_at(normalized_key, value)
            return

        if not _is_point_value_sequence(value):
            raise IndexError("Keypoint list for indexing must be same length and value list.")

        if isinstance(value, np.ndarray):
            val_list = [v for v in cast(Iterable[Any], value)]
        else:
            val_list = list(cast(list[Point] | tuple[Point, ...], value))
        if len(val_list) != len(keypoint_list):
            raise IndexError("Keypoint list for indexing must be same length and value list.")
        for n, v in zip(keypoint_list, val_list, strict=False):
            self._set_point_at(n, v)

    def _get_point_at(self, keypoint: str | Keypoint | int | np.integer[Any]) -> Point:
        points = self._ensure_point_records()
        normalized_key = Instance._normalize_key(keypoint)
        if isinstance(normalized_key, Keypoint | str):
            keypoint_name = (
                normalized_key.name if isinstance(normalized_key, Keypoint) else normalized_key
            )
            if not self.skeleton.has_keypoint(keypoint_name):
                return PointCtor()
            normalized_key = self._keypoint_to_index(keypoint_name)
        return points[int(normalized_key)]

    def _set_point_at(
        self,
        keypoint: str | Keypoint | int | np.integer[Any],
        value: list[Point] | Point | np.ndarray | Any,
    ) -> None:
        points = self._ensure_point_records()
        normalized_key = Instance._normalize_key(keypoint)
        if isinstance(normalized_key, Keypoint | str):
            kp_name = (
                normalized_key.name if isinstance(normalized_key, Keypoint) else normalized_key
            )
            keypoint_idx = self._keypoint_to_index(kp_name)
        else:
            keypoint_idx = normalized_key
        value = self._coerce_point_value(value)
        if (
            isinstance(points, PredictedPointArray)
            and isinstance(value, Point)
            and not isinstance(value, PredictedPoint)
        ):
            value = PredictedPoint.from_point(value)
        points[keypoint_idx] = value

    def __delitem__(self, keypoint: str | Keypoint):
        self._assert_points_synced()
        keypoint_idx = self._keypoint_to_index(keypoint)
        self._points[keypoint_idx].x = math.nan
        self._points[keypoint_idx].y = math.nan

    def __repr__(self) -> str:
        pts = ", ".join(f"{kp.name}: ({pt.x:.1f}, {pt.y:.1f})" for kp, pt in self.keypoints_points)
        return (
            "Instance("
            f"video={self.video}, "
            f"frame_idx={self.frame_idx}, "
            f"points=[{pts}], "
            f"track={self.track}, "
            f"tracking_score={self.tracking_score:.2f}"
            ")"
        )

    def matches(self, other: Instance) -> bool:
        """Return True when this Instance mirrors another (points, track, frame)."""
        same_track = (self.track is None and other.track is None) or (
            self.track is not None and other.track is not None and self.track.matches(other.track)
        )
        return (
            isinstance(other, self.__class__)
            and list(self.points) == list(other.points)
            and self.skeleton.matches(other.skeleton)
            and same_track
            and self.frame_idx == other.frame_idx
        )

    @property
    def keypoints(self) -> tuple[Keypoint, ...]:
        """Return the tuple of labeled keypoints that are present."""
        points = self._ensure_point_records()
        skeleton_kps = self.skeleton.keypoints
        result = []
        for i in range(len(points)):
            pt = points[i]

            if np.isnan(pt["x"]) or np.isnan(pt["y"]):
                continue
            kp = self._keypoints[i]
            if kp in skeleton_kps:
                result.append(kp)
        return tuple(result)

    @property
    def keypoints_points(self) -> list[tuple[Keypoint, Point]]:
        """Return (keypoint, point) pairs for existing points.

        Returns:
            list[tuple[Keypoint, Point]]: List of keypoint-point pairs.
        """
        points = self._ensure_point_records()

        skeleton_kps = self.skeleton.keypoints
        result = []
        for i in range(len(points)):
            pt = points[i]
            if np.isnan(pt["x"]) or np.isnan(pt["y"]):
                continue
            kp = self._keypoints[i]
            if kp in skeleton_kps:
                result.append((kp, pt))
        return result

    @property
    def points(self) -> tuple[Point, ...]:
        """Return the tuple of visible points (non-NaN).

        Returns:
            tuple[Point, ...]: Tuple of visible points.
        """
        points = self._ensure_point_records()

        result = []
        for i in range(len(points)):
            pt = points[i]
            if not (np.isnan(pt["x"]) or np.isnan(pt["y"])):
                result.append(pt)
        return tuple(result)

    def point_records(self, *, copy: bool = True) -> np.ndarray:
        """Return the structured point records for storage-level callers.

        Use :meth:`xy_array` or :meth:`xy_score_array` for numeric coordinate
        arrays. Use this method only when the caller needs record fields such as
        ``visible``, ``complete``, ``flags``, or native prediction ``score``.
        """
        records = self._ensure_point_records().view(np.ndarray)
        return records.copy() if copy else records

    def xy_array(self, *, invisible_as_nan: bool = True) -> np.ndarray:
        """Return point coordinates as a plain ``(keypoints, 2)`` float array.

        xpkg stores missing coordinates as NaN. When ``invisible_as_nan`` is
        true, invisible points are also masked to NaN in the returned coordinate
        array. The returned array is detached from the instance's structured
        record storage so consumers can safely mutate it.
        """
        points = self._ensure_point_records().view(np.ndarray)
        xy = np.empty((len(points), 2), dtype=float)
        xy[:, 0] = np.asarray(points["x"], dtype=float)
        xy[:, 1] = np.asarray(points["y"], dtype=float)
        if invisible_as_nan:
            xy[~np.asarray(points["visible"], dtype=bool)] = math.nan
        return xy

    def xy_score_array(
        self,
        *,
        invisible_as_nan: bool = True,
        missing_score: float = math.nan,
    ) -> np.ndarray:
        """Return coordinates plus score as a plain ``(keypoints, 3)`` array.

        Predicted instances carry native point scores. User-labeled instances do
        not, so their score column is filled with ``missing_score``.
        """
        points = self._ensure_point_records().view(np.ndarray)
        xys = np.empty((len(points), 3), dtype=float)
        xys[:, :2] = self.xy_array(invisible_as_nan=invisible_as_nan)
        if "score" in (points.dtype.names or ()):
            xys[:, 2] = np.asarray(points["score"], dtype=float)
        else:
            xys[:, 2] = float(missing_score)
        if invisible_as_nan:
            xys[~np.asarray(points["visible"], dtype=bool), 2] = math.nan
        return xys

    def fill_missing(self, max_x: float | None = None, max_y: float | None = None):
        """Impute missing points to stay within the optional bounds."""
        self._assert_points_synced()
        x1, y1, x2, y2 = self.bounding_box_xyxy
        x1 = float(np.nanmax([x1, 0.0]))
        y1 = float(np.nanmax([y1, 0.0]))
        if max_x is not None:
            x2 = float(np.nanmin([x2, float(max_x)]))
        if max_y is not None:
            y2 = float(np.nanmin([y2, float(max_y)]))
        w = x2 - x1
        h = y2 - y1
        if not (math.isfinite(w) and math.isfinite(h) and w > 0 and h > 0):
            return
        for kp in self.skeleton.keypoints:
            if kp not in self.keypoints or self[kp].isnan():
                off = np.array([w, h]) * np.random.rand(2)
                x, y = off + np.array([x1, y1])
                x = max(x, 0.0)
                y = max(y, 0.0)
                if max_x is not None:
                    x = min(x, max_x)
                if max_y is not None:
                    y = min(y, max_y)
                self[kp] = PointCtor(x=x, y=y, visible=False)

    def transform_points(self, transformation_matrix):
        """Apply the provided matrix to transform all points."""
        points = self.xy_array(invisible_as_nan=False)
        if transformation_matrix.shape[1] == 3:
            rotation = transformation_matrix[:, :2]
            translation = transformation_matrix[:, 2]
            transformed = points @ rotation.T + translation
        else:
            transformed = points @ transformation_matrix.T
        self._points["x"] = transformed[:, 0]
        self._points["y"] = transformed[:, 1]

    @property
    def centroid(self) -> np.ndarray:
        """Return the median coordinate (x,y) of visible points."""
        points = self.xy_array()

        valid_mask = ~np.isnan(points).any(axis=1)
        if not valid_mask.any():
            return np.array([np.nan, np.nan])
        valid_points = np.asarray(points[valid_mask], dtype=float, order="C")

        sorted_points = np.sort(valid_points, axis=0)
        mid = len(sorted_points) // 2
        if len(sorted_points) % 2 == 1:
            return sorted_points[mid]
        return (sorted_points[mid - 1] + sorted_points[mid]) / 2.0

    @property
    def bounding_box_xyxy(self) -> np.ndarray:
        """Bounding box in XYXY order (x1, y1, x2, y2)."""
        points = self.xy_array()
        if np.isnan(points).all():
            return np.array([np.nan, np.nan, np.nan, np.nan])
        mins = np.nanmin(points, axis=0)
        maxs = np.nanmax(points, axis=0)
        x1, y1 = mins[0], mins[1]
        x2, y2 = maxs[0], maxs[1]
        return np.array([x1, y1, x2, y2])

    @property
    def midpoint(self) -> np.ndarray:
        """Return the geometrical center (x,y) of the instance bounding box."""
        x1, y1, x2, y2 = self.bounding_box_xyxy
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

    @property
    def visible_point_count(self) -> int:
        """Return the number of visible (non-NaN) points."""
        return sum(1 for p in self.points if p.visible)

    def __len__(self) -> int:
        """Return the number of visible points (for len())."""
        return self.visible_point_count

    @property
    def video(self) -> Video | None:
        """Return the associated video object if the frame is available."""
        if self.frame is None:
            return None
        else:
            return self.frame.video

    @property
    def frame_idx(self) -> int | None:
        """Return the frame index for this instance, if a frame is attached."""
        if self.frame is None:
            return None
        else:
            return self.frame.frame_idx

    @classmethod
    def from_pointsarray(
        cls, points: np.ndarray, skeleton: Skeleton, track: Track | None = None
    ) -> Instance:
        """Construct an Instance from raw numpy (x,y) coordinate arrays."""
        predicted_points = dict()
        for point, keypoint_name in zip(points, skeleton.keypoint_names, strict=False):
            if np.isnan(point).any():
                continue
            predicted_points[keypoint_name] = PointCtor(x=point[0], y=point[1])
        return cls(init_points=predicted_points, skeleton=skeleton, track=track)

    @classmethod
    def from_numpy(
        cls, points: np.ndarray, skeleton: Skeleton, track: Track | None = None
    ) -> Instance:
        """Convert numpy point arrays into an Instance via pointsarray."""
        return cls.from_pointsarray(points, skeleton, track=track)

    def _merge_keypoints_data(self, base_keypoint: str, merge_keypoint: str):
        """Copy coordinates from one keypoint into another when merging."""
        base_pt = self[base_keypoint]
        merge_pt = self[merge_keypoint]
        if merge_pt.isnan():
            return
        if base_pt.isnan() or not base_pt.visible:
            base_pt.x = merge_pt.x
            base_pt.y = merge_pt.y
            base_pt.visible = merge_pt.visible
            base_pt.complete = merge_pt.complete
            if isinstance(base_pt, PredictedPoint) and isinstance(merge_pt, PredictedPoint):
                base_pt.score = merge_pt.score


@dataclass(eq=False)
class PredictedInstance(Instance):
    """Instance subclass that also tracks prediction scores and confidences."""

    score: float = 0.0
    _point_array_type = PredictedPointArray

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PredictedInstance):
            return NotImplemented
        if not super().__eq__(other):
            return False
        return self.score == other.score

    def __post_init__(self, init_points):
        super().__post_init__(init_points)
        if self.from_predicted is not None:
            raise ValueError("PredictedInstance should not have from_predicted.")

    def __repr__(self) -> str:
        pts = []
        for kp, pt in self.keypoints_points:
            pts.append(f"{kp.name}: ({pt.x:.1f}, {pt.y:.1f}, {pt.score:.2f})")
        pts = ", ".join(pts)
        return (
            "PredictedInstance("
            f"video={self.video}, "
            f"frame_idx={self.frame_idx}, "
            f"points=[{pts}], "
            f"score={self.score:.2f}, "
            f"track={self.track}, "
            f"tracking_score={self.tracking_score:.2f}"
            ")"
        )

    @property
    def scores(self) -> np.ndarray:
        """Return the per-point confidence scores."""
        return self.xy_score_array()[:, 2]

    @classmethod
    def from_instance(
        cls, instance: Instance, score: float, tracking_score: float = 0.0
    ) -> PredictedInstance:
        """Upgrade a general Instance into a PredictedInstance with score metadata.

        Args:
            instance: The source Instance to upgrade.
            score: The prediction confidence score.
            tracking_score: Optional tracking-specific score.

        Returns:
            PredictedInstance: The upgraded predicted instance.
        """
        return cls(
            skeleton=instance.skeleton,
            track=instance.track,
            from_predicted=None,
            init_points=PredictedPointArray.from_array(instance._points),
            frame=instance.frame,
            tracking_score=tracking_score,
            score=score,
        )

    @classmethod
    def from_arrays(
        cls,
        points: np.ndarray,
        point_confidences: np.ndarray,
        instance_score: float,
        skeleton: Skeleton,
        track: Track | None = None,
        tracking_score: float = 0.0,
    ) -> PredictedInstance:
        """Build a PredictedInstance from raw point/confidence arrays.

        Args:
            points: Array of (x, y) coordinates.
            point_confidences: Array of per-point confidence scores.
            instance_score: Overall instance confidence score.
            skeleton: The skeleton associated with the instance.
            track: Optional track assignment.
            tracking_score: Optional tracking-specific score.

        Returns:
            PredictedInstance: The initialized predicted instance.
        """
        predicted_points = dict()
        for point, confidence, keypoint_name in zip(
            points, point_confidences, skeleton.keypoint_names, strict=False
        ):
            if np.isnan(point).any():
                continue
            predicted_points[keypoint_name] = PredictedPoint(
                x=point[0], y=point[1], score=confidence
            )
        return cls(
            init_points=predicted_points,
            skeleton=skeleton,
            score=instance_score,
            track=track,
            tracking_score=tracking_score,
        )


InstanceLike = Instance | PredictedInstance


def is_predicted_instance(instance: InstanceLike) -> TypeGuard[PredictedInstance]:
    """Type guard for distinguishing predicted instances."""
    return isinstance(instance, PredictedInstance)


__all__ = [
    "Instance",
    "InstanceLike",
    "PredictedInstance",
    "Track",
    "is_predicted_instance",
]
