"""Parquet-backed instance-mask tables using the canonical xpkg RLE codec."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from xpkg._core.json_utils import dump_json, parse_json
from xpkg.segmentation.model import MaskType, SegmentationMask
from xpkg.segmentation.rle import XPKG_RLE_ENCODING, XPKG_RLE_ORDER

MASK_TABLE_KIND = "xpkg.segmentation.mask_table"
MASK_TABLE_SCHEMA_VERSION = "1"
DEFAULT_ROW_GROUP_SIZE = 4096
MASK_TABLE_COLUMNS = (
    "frame_index",
    "instance_index",
    "instance_id",
    "class_name",
    "confidence",
    "mask_id",
    "rle_counts",
    "rle_start",
    "height",
    "width",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "is_empty",
    "is_keyframe",
    "is_predicted",
    "status",
    "source",
)

_RESERVED_METADATA_KEYS = {
    "xpkg.kind",
    "schema_version",
    "encoding",
    "order",
    "frame_height",
    "frame_width",
    "instance_roster",
}


@dataclass(frozen=True, slots=True)
class MaskTableInstance:
    """One instance identity in a mask table roster."""

    instance_index: int
    instance_id: str
    class_name: str = ""


@dataclass(frozen=True, slots=True)
class MaskTableInfo:
    """Self-describing metadata stored in the Parquet footer."""

    schema_version: str
    encoding: str
    order: str
    frame_height: int | None
    frame_width: int | None
    instance_roster: tuple[MaskTableInstance, ...] = ()
    custom_metadata: dict[str, str] = field(default_factory=dict)

    @property
    def instance_ids(self) -> tuple[str, ...]:
        """Return instance IDs in roster order."""

        return tuple(item.instance_id for item in self.instance_roster)


@dataclass(frozen=True, slots=True)
class MaskTableRecord:
    """One time-indexed instance mask row."""

    frame_index: int
    instance_index: int
    instance_id: str
    mask: SegmentationMask
    is_keyframe: bool = False
    status: str = ""
    source: str = ""
    bbox_xyxy: tuple[int, int, int, int] | None = None
    is_empty: bool | None = None


def _import_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Parquet mask tables require pyarrow. Install exp-pkg with its "
            "declared dependencies, then rerun this operation."
        ) from exc
    return pa, pq


def _metadata_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return dump_json(value, sort_keys=True, compact=True)


def _metadata_bytes(
    *,
    frame_height: int | None,
    frame_width: int | None,
    instance_roster: Sequence[MaskTableInstance],
    metadata: Mapping[str, object] | None,
) -> dict[bytes, bytes]:
    custom = dict(metadata or {})
    reserved = _RESERVED_METADATA_KEYS.intersection(custom)
    if reserved:
        names = ", ".join(sorted(reserved))
        raise ValueError(f"Mask table metadata keys are reserved: {names}")

    payload: dict[str, object] = {
        "xpkg.kind": MASK_TABLE_KIND,
        "schema_version": MASK_TABLE_SCHEMA_VERSION,
        "encoding": XPKG_RLE_ENCODING,
        "order": XPKG_RLE_ORDER,
        "instance_roster": [
            {
                "instance_index": int(item.instance_index),
                "instance_id": item.instance_id,
                "class_name": item.class_name,
            }
            for item in instance_roster
        ],
    }
    if frame_height is not None:
        payload["frame_height"] = int(frame_height)
    if frame_width is not None:
        payload["frame_width"] = int(frame_width)
    payload.update(custom)
    return {
        key.encode("utf-8"): _metadata_value(value).encode("utf-8")
        for key, value in payload.items()
    }


def _schema(pa: Any, metadata: dict[bytes, bytes] | None = None) -> Any:
    return pa.schema(
        [
            ("frame_index", pa.int64()),
            ("instance_index", pa.int32()),
            ("instance_id", pa.string()),
            ("class_name", pa.string()),
            ("confidence", pa.float64()),
            ("mask_id", pa.string()),
            ("rle_counts", pa.binary()),
            ("rle_start", pa.uint8()),
            ("height", pa.int32()),
            ("width", pa.int32()),
            ("bbox_x1", pa.int32()),
            ("bbox_y1", pa.int32()),
            ("bbox_x2", pa.int32()),
            ("bbox_y2", pa.int32()),
            ("is_empty", pa.bool_()),
            ("is_keyframe", pa.bool_()),
            ("is_predicted", pa.bool_()),
            ("status", pa.string()),
            ("source", pa.string()),
        ],
        metadata=metadata,
    )


def _empty_rows() -> dict[str, list[Any]]:
    return {column: [] for column in MASK_TABLE_COLUMNS}


def _counts_to_bytes(counts: np.ndarray) -> bytes:
    return np.asarray(counts, dtype="<u4").tobytes(order="C")


def _counts_from_bytes(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype="<u4").astype(np.uint32, copy=True)


def _is_empty_rle(counts: np.ndarray, start: int) -> bool:
    if counts.size == 0:
        return True
    if start == 0:
        return int(counts[1::2].sum()) == 0
    return int(counts[0::2].sum()) == 0


def _ensure_rle_mask(
    mask: SegmentationMask,
    *,
    frame_height: int | None,
    frame_width: int | None,
) -> SegmentationMask:
    if mask.mask_type == MaskType.RLE:
        if mask.rle_counts is None:
            raise ValueError("RLE mask is missing run-length counts.")
        return mask
    if mask.mask_type == MaskType.POLYGON:
        if frame_height is None or frame_width is None:
            raise ValueError("Polygon mask-table writes require frame_height and frame_width.")
        dense = mask.to_binary_mask(height=frame_height, width=frame_width)
        return SegmentationMask.from_binary_mask(
            dense,
            class_name=mask.class_name,
            confidence=mask.confidence,
            track=mask.track,
            instance_ref=mask.instance_ref,
            is_predicted=mask.is_predicted,
            prompt=mask.prompt,
            mask_id=mask.mask_id,
            artifact_ref=mask.artifact_ref,
            mask_path=mask.mask_path,
        )
    raise ValueError("Mask references cannot be embedded in a Parquet mask table.")


def _bbox_from_mask(mask: SegmentationMask, *, is_empty: bool) -> tuple[int, int, int, int]:
    if is_empty:
        return (-1, -1, -1, -1)
    bbox = np.rint(mask.bounding_box).astype(np.int64)
    return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))


def _row_from_record(
    record: MaskTableRecord,
    *,
    frame_height: int | None,
    frame_width: int | None,
) -> dict[str, Any]:
    mask = _ensure_rle_mask(record.mask, frame_height=frame_height, frame_width=frame_width)
    if mask.rle_counts is None:
        raise ValueError("RLE mask is missing run-length counts.")
    counts = np.asarray(mask.rle_counts, dtype=np.uint32)
    is_empty = _is_empty_rle(counts, mask.rle_start) if record.is_empty is None else record.is_empty
    bbox = record.bbox_xyxy or _bbox_from_mask(mask, is_empty=bool(is_empty))
    status = record.status or ("empty" if is_empty else "valid")
    confidence = float(mask.confidence)
    if math.isnan(confidence):
        confidence = float("nan")
    return {
        "frame_index": int(record.frame_index),
        "instance_index": int(record.instance_index),
        "instance_id": str(record.instance_id),
        "class_name": str(mask.class_name),
        "confidence": confidence,
        "mask_id": str(mask.mask_id),
        "rle_counts": _counts_to_bytes(counts),
        "rle_start": int(mask.rle_start),
        "height": int(mask.rle_height),
        "width": int(mask.rle_width),
        "bbox_x1": int(bbox[0]),
        "bbox_y1": int(bbox[1]),
        "bbox_x2": int(bbox[2]),
        "bbox_y2": int(bbox[3]),
        "is_empty": bool(is_empty),
        "is_keyframe": bool(record.is_keyframe),
        "is_predicted": bool(mask.is_predicted),
        "status": status,
        "source": str(record.source),
    }


def _append_row(rows: dict[str, list[Any]], row: Mapping[str, Any]) -> None:
    for column in MASK_TABLE_COLUMNS:
        rows[column].append(row[column])


def _table_from_rows(pa: Any, rows: dict[str, list[Any]], schema: Any) -> Any:
    return pa.Table.from_pydict(rows, schema=schema)


def _parse_int_metadata(raw: Mapping[str, str], key: str) -> int | None:
    value = raw.get(key)
    if value is None or value == "":
        return None
    return int(value)


def _parse_instance_roster(raw: Mapping[str, str]) -> tuple[MaskTableInstance, ...]:
    payload = raw.get("instance_roster", "[]")
    values = parse_json(payload)
    if not isinstance(values, list):
        raise ValueError("Mask table metadata field 'instance_roster' must be a list.")
    roster: list[MaskTableInstance] = []
    for item in values:
        if not isinstance(item, Mapping):
            raise ValueError("Mask table roster entries must be objects.")
        entry = cast(Mapping[str, object], item)
        roster.append(
            MaskTableInstance(
                instance_index=int(str(entry["instance_index"])),
                instance_id=str(entry["instance_id"]),
                class_name=str(entry.get("class_name", "")),
            )
        )
    return tuple(roster)


def _info_from_metadata(metadata: Mapping[bytes, bytes] | None) -> MaskTableInfo:
    raw = {
        key.decode("utf-8"): value.decode("utf-8")
        for key, value in (metadata or {}).items()
    }
    kind = raw.get("xpkg.kind", MASK_TABLE_KIND)
    if kind != MASK_TABLE_KIND:
        raise ValueError(f"Not an xpkg mask table: {kind!r}")
    encoding = raw.get("encoding")
    if encoding != XPKG_RLE_ENCODING:
        raise ValueError(f"Unsupported mask table encoding {encoding!r}.")
    order = raw.get("order")
    if order != XPKG_RLE_ORDER:
        raise ValueError(f"Unsupported mask table RLE order {order!r}.")
    custom = {
        key: value
        for key, value in raw.items()
        if key not in _RESERVED_METADATA_KEYS and not key.startswith("ARROW:")
    }
    return MaskTableInfo(
        schema_version=str(raw.get("schema_version", "")),
        encoding=encoding,
        order=order,
        frame_height=_parse_int_metadata(raw, "frame_height"),
        frame_width=_parse_int_metadata(raw, "frame_width"),
        instance_roster=_parse_instance_roster(raw),
        custom_metadata=custom,
    )


def _row_value(row: Mapping[str, Any], key: str) -> Any:
    value = row[key]
    if hasattr(value, "as_py"):
        return value.as_py()
    return value


def _record_from_row(row: Mapping[str, Any]) -> MaskTableRecord:
    counts = _counts_from_bytes(_row_value(row, "rle_counts"))
    height = int(_row_value(row, "height"))
    width = int(_row_value(row, "width"))
    if int(counts.sum()) != height * width:
        raise ValueError("Mask table row has RLE counts that do not match height x width.")
    mask = SegmentationMask(
        mask_type=MaskType.RLE,
        rle_counts=counts,
        rle_start=int(_row_value(row, "rle_start")),
        rle_height=height,
        rle_width=width,
        class_name=str(_row_value(row, "class_name") or ""),
        confidence=float(_row_value(row, "confidence")),
        instance_ref=int(_row_value(row, "instance_index")),
        is_predicted=bool(_row_value(row, "is_predicted")),
        mask_id=str(_row_value(row, "mask_id") or ""),
    )
    return MaskTableRecord(
        frame_index=int(_row_value(row, "frame_index")),
        instance_index=int(_row_value(row, "instance_index")),
        instance_id=str(_row_value(row, "instance_id")),
        mask=mask,
        is_keyframe=bool(_row_value(row, "is_keyframe")),
        status=str(_row_value(row, "status") or ""),
        source=str(_row_value(row, "source") or ""),
        bbox_xyxy=(
            int(_row_value(row, "bbox_x1")),
            int(_row_value(row, "bbox_y1")),
            int(_row_value(row, "bbox_x2")),
            int(_row_value(row, "bbox_y2")),
        ),
        is_empty=bool(_row_value(row, "is_empty")),
    )


def _records_from_table(table: Any) -> tuple[MaskTableRecord, ...]:
    records = [_record_from_row(row) for row in table.to_pylist()]
    records.sort(key=lambda item: (item.frame_index, item.instance_index, item.instance_id))
    return tuple(records)


def _infer_instance_roster(records: Sequence[MaskTableRecord]) -> tuple[MaskTableInstance, ...]:
    roster: dict[tuple[int, str], MaskTableInstance] = {}
    for record in records:
        key = (int(record.instance_index), str(record.instance_id))
        roster.setdefault(
            key,
            MaskTableInstance(
                instance_index=int(record.instance_index),
                instance_id=str(record.instance_id),
                class_name=str(record.mask.class_name),
            ),
        )
    return tuple(roster[key] for key in sorted(roster))


def _infer_constant_size(records: Sequence[MaskTableRecord]) -> tuple[int | None, int | None]:
    sizes = {
        (int(record.mask.rle_height), int(record.mask.rle_width))
        for record in records
        if record.mask.mask_type == MaskType.RLE
    }
    if len(sizes) == 1:
        return next(iter(sizes))
    return None, None


class MaskTableWriter:
    """Write sorted instance masks to a Parquet table."""

    def __init__(
        self,
        path: str | Path,
        *,
        frame_height: int | None = None,
        frame_width: int | None = None,
        instance_roster: Sequence[MaskTableInstance] = (),
        metadata: Mapping[str, object] | None = None,
        row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
        compression: str = "zstd",
        enforce_sorted: bool = True,
    ) -> None:
        if row_group_size <= 0:
            raise ValueError("row_group_size must be positive.")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.row_group_size = int(row_group_size)
        self.compression = compression
        self.enforce_sorted = enforce_sorted
        self._pa, self._pq = _import_pyarrow()
        schema_metadata = _metadata_bytes(
            frame_height=frame_height,
            frame_width=frame_width,
            instance_roster=instance_roster,
            metadata=metadata,
        )
        self._schema = _schema(self._pa, schema_metadata)
        self._rows = _empty_rows()
        self._writer: Any | None = None
        self._last_key: tuple[int, int, str] | None = None
        self._closed = False

    def __enter__(self) -> MaskTableWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def write_mask(
        self,
        *,
        frame_index: int,
        instance_index: int,
        instance_id: str,
        mask: SegmentationMask,
        is_keyframe: bool = False,
        status: str = "",
        source: str = "",
        bbox_xyxy: tuple[int, int, int, int] | None = None,
        is_empty: bool | None = None,
    ) -> None:
        """Write one mask row."""

        self.write_record(
            MaskTableRecord(
                frame_index=frame_index,
                instance_index=instance_index,
                instance_id=instance_id,
                mask=mask,
                is_keyframe=is_keyframe,
                status=status,
                source=source,
                bbox_xyxy=bbox_xyxy,
                is_empty=is_empty,
            )
        )

    def write_records(self, records: Iterable[MaskTableRecord]) -> None:
        """Write multiple records in order."""

        for record in records:
            self.write_record(record)

    def write_record(self, record: MaskTableRecord) -> None:
        """Write one prebuilt mask-table record."""

        if self._closed:
            raise ValueError("Cannot write to a closed MaskTableWriter.")
        key = (int(record.frame_index), int(record.instance_index), str(record.instance_id))
        if self.enforce_sorted and self._last_key is not None and key < self._last_key:
            raise ValueError("Mask table records must be written in frame/instance order.")
        self._last_key = key
        row = _row_from_record(record, frame_height=self.frame_height, frame_width=self.frame_width)
        _append_row(self._rows, row)
        if len(self._rows["frame_index"]) >= self.row_group_size:
            self._flush()

    def close(self) -> None:
        """Flush rows and close the Parquet writer."""

        if self._closed:
            return
        if self._writer is None and not self._rows["frame_index"]:
            table = _table_from_rows(self._pa, self._rows, self._schema)
            self._pq.write_table(table, self.path, compression=self.compression)
            self._closed = True
            return
        self._flush()
        if self._writer is not None:
            self._writer.close()
        self._closed = True

    def _flush(self) -> None:
        if not self._rows["frame_index"]:
            return
        table = _table_from_rows(self._pa, self._rows, self._schema)
        if self._writer is None:
            self._writer = self._pq.ParquetWriter(
                self.path,
                self._schema,
                compression=self.compression,
            )
        writer = self._writer
        if writer is None:
            raise RuntimeError("Parquet writer was not initialized")
        writer.write_table(table, row_group_size=len(self._rows["frame_index"]))
        self._rows = _empty_rows()


class MaskTableReader:
    """Read an xpkg Parquet instance-mask table."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Mask table does not exist: {self.path}")
        self._pa, self._pq = _import_pyarrow()
        parquet_file = self._pq.ParquetFile(self.path)
        self.info = _info_from_metadata(parquet_file.metadata.metadata)

    def read_records(
        self,
        *,
        frame_index: int | None = None,
        start: int | None = None,
        stop: int | None = None,
        instance_ids: Sequence[str] | None = None,
    ) -> tuple[MaskTableRecord, ...]:
        """Read records, optionally filtered by frame/window and instance IDs."""

        filters: list[tuple[str, str, object]] = []
        if frame_index is not None:
            if start is not None or stop is not None:
                raise ValueError("frame_index cannot be combined with start/stop.")
            filters.append(("frame_index", "=", int(frame_index)))
        if start is not None:
            filters.append(("frame_index", ">=", int(start)))
        if stop is not None:
            filters.append(("frame_index", "<", int(stop)))
        if instance_ids is not None:
            filters.append(("instance_id", "in", [str(item) for item in instance_ids]))
        table = self._pq.read_table(self.path, filters=filters or None)
        return _records_from_table(table)

    def read_frame(
        self,
        frame_index: int,
        *,
        instance_ids: Sequence[str] | None = None,
    ) -> tuple[MaskTableRecord, ...]:
        """Read all masks for one frame."""

        return self.read_records(frame_index=frame_index, instance_ids=instance_ids)

    def read_window(
        self,
        start: int,
        stop: int,
        *,
        instance_ids: Sequence[str] | None = None,
    ) -> tuple[MaskTableRecord, ...]:
        """Read masks for ``start <= frame_index < stop``."""

        if stop < start:
            raise ValueError("stop must be greater than or equal to start.")
        return self.read_records(start=start, stop=stop, instance_ids=instance_ids)

    def decode_dense(
        self,
        start: int,
        stop: int,
        *,
        instance_ids: Sequence[str] | None = None,
        dtype: Any = np.bool_,
    ) -> np.ndarray:
        """Decode a frame window as ``frames x instances x height x width``."""

        records = self.read_window(start, stop, instance_ids=instance_ids)
        ids = self._dense_instance_ids(records, instance_ids=instance_ids)
        height, width = self._dense_size(records)
        dense = np.zeros((stop - start, len(ids), height, width), dtype=dtype)
        id_to_column = {instance_id: index for index, instance_id in enumerate(ids)}
        for record in records:
            if record.is_empty:
                continue
            column = id_to_column.get(record.instance_id)
            if column is None:
                continue
            mask = record.mask.to_binary_mask()
            if mask.shape != (height, width):
                raise ValueError("Mask table window contains mixed mask sizes.")
            dense[record.frame_index - start, column] = mask.astype(dtype, copy=False)
        return dense

    def _dense_instance_ids(
        self,
        records: Sequence[MaskTableRecord],
        *,
        instance_ids: Sequence[str] | None,
    ) -> tuple[str, ...]:
        if instance_ids is not None:
            return tuple(str(item) for item in instance_ids)
        if self.info.instance_roster:
            return self.info.instance_ids
        return tuple(sorted({record.instance_id for record in records}))

    def _dense_size(self, records: Sequence[MaskTableRecord]) -> tuple[int, int]:
        if self.info.frame_height is not None and self.info.frame_width is not None:
            return self.info.frame_height, self.info.frame_width
        sizes = {(record.mask.rle_height, record.mask.rle_width) for record in records}
        if len(sizes) == 1:
            return next(iter(sizes))
        raise ValueError("Dense decode requires fixed frame_height and frame_width metadata.")


def write_mask_table(
    path: str | Path,
    records: Iterable[MaskTableRecord],
    *,
    frame_height: int | None = None,
    frame_width: int | None = None,
    instance_roster: Sequence[MaskTableInstance] | None = None,
    metadata: Mapping[str, object] | None = None,
    row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
    compression: str = "zstd",
    sort: bool = True,
) -> Path:
    """Write records to a complete Parquet mask table."""

    record_list = list(records)
    if sort:
        record_list.sort(key=lambda item: (item.frame_index, item.instance_index, item.instance_id))
    if instance_roster is None:
        instance_roster = _infer_instance_roster(record_list)
    inferred_height, inferred_width = _infer_constant_size(record_list)
    if frame_height is None:
        frame_height = inferred_height
    if frame_width is None:
        frame_width = inferred_width
    with MaskTableWriter(
        path,
        frame_height=frame_height,
        frame_width=frame_width,
        instance_roster=instance_roster,
        metadata=metadata,
        row_group_size=row_group_size,
        compression=compression,
        enforce_sorted=sort,
    ) as writer:
        writer.write_records(record_list)
    return Path(path)


__all__ = [
    "DEFAULT_ROW_GROUP_SIZE",
    "MASK_TABLE_KIND",
    "MASK_TABLE_SCHEMA_VERSION",
    "MaskTableInfo",
    "MaskTableInstance",
    "MaskTableReader",
    "MaskTableRecord",
    "MaskTableWriter",
    "write_mask_table",
]
