from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xpkg.segmentation import (
    MASK_TABLE_KIND,
    XPKG_RLE_ENCODING,
    MaskTableInstance,
    MaskTableReader,
    MaskTableRecord,
    MaskTableWriter,
    SegmentationMask,
    write_mask_table,
)


def _binary_mask(*, shift: int = 0) -> np.ndarray:
    mask = np.zeros((6, 7), dtype=np.uint8)
    mask[1 + shift : 4 + shift, 2:5] = 1
    return mask


def _records() -> list[MaskTableRecord]:
    return [
        MaskTableRecord(
            frame_index=1,
            instance_index=0,
            instance_id="lfpaw",
            mask=SegmentationMask.from_binary_mask(
                _binary_mask(shift=1),
                class_name="paw",
                mask_id="f1-lf",
            ),
            source="sam2",
        ),
        MaskTableRecord(
            frame_index=0,
            instance_index=1,
            instance_id="rfpaw",
            mask=SegmentationMask.from_binary_mask(
                np.zeros((6, 7), dtype=np.uint8),
                class_name="paw",
                mask_id="f0-rf",
            ),
            is_keyframe=True,
            source="sam2",
        ),
        MaskTableRecord(
            frame_index=0,
            instance_index=0,
            instance_id="lfpaw",
            mask=SegmentationMask.from_binary_mask(
                _binary_mask(),
                class_name="paw",
                confidence=0.91,
                mask_id="f0-lf",
            ),
            is_keyframe=True,
            source="sam2",
        ),
    ]


def test_mask_table_round_trips_rle_records_and_metadata(tmp_path: Path) -> None:
    table_path = tmp_path / "masks.parquet"
    roster = (
        MaskTableInstance(instance_index=0, instance_id="lfpaw", class_name="paw"),
        MaskTableInstance(instance_index=1, instance_id="rfpaw", class_name="paw"),
    )

    write_mask_table(
        table_path,
        _records(),
        instance_roster=roster,
        metadata={
            "producer": "phrase",
            "qc_thresholds": {"forepaw_max": 600, "hindpaw_max": 1100},
        },
        row_group_size=2,
    )

    reader = MaskTableReader(table_path)

    assert reader.info.schema_version == "1"
    assert reader.info.encoding == XPKG_RLE_ENCODING
    assert reader.info.frame_height == 6
    assert reader.info.frame_width == 7
    assert reader.info.instance_ids == ("lfpaw", "rfpaw")
    assert reader.info.custom_metadata["producer"] == "phrase"
    assert reader.info.custom_metadata["qc_thresholds"] == (
        '{"forepaw_max":600,"hindpaw_max":1100}'
    )

    frame_0 = reader.read_frame(0)
    assert [record.instance_id for record in frame_0] == ["lfpaw", "rfpaw"]
    assert frame_0[0].bbox_xyxy == (2, 1, 5, 4)
    assert frame_0[0].status == "valid"
    assert frame_0[0].is_keyframe
    assert frame_0[0].source == "sam2"
    np.testing.assert_array_equal(frame_0[0].mask.to_binary_mask(), _binary_mask())

    assert frame_0[1].bbox_xyxy == (-1, -1, -1, -1)
    assert frame_0[1].is_empty
    assert frame_0[1].status == "empty"
    assert int(frame_0[1].mask.to_binary_mask().sum()) == 0

    frame_1_lf = reader.read_window(1, 2, instance_ids=["lfpaw"])
    assert len(frame_1_lf) == 1
    assert frame_1_lf[0].frame_index == 1
    np.testing.assert_array_equal(frame_1_lf[0].mask.to_binary_mask(), _binary_mask(shift=1))


def test_mask_table_decodes_dense_windows_in_roster_order(tmp_path: Path) -> None:
    table_path = tmp_path / "masks.parquet"
    write_mask_table(
        table_path,
        _records(),
        instance_roster=(
            MaskTableInstance(instance_index=0, instance_id="lfpaw", class_name="paw"),
            MaskTableInstance(instance_index=1, instance_id="rfpaw", class_name="paw"),
        ),
    )

    dense = MaskTableReader(table_path).decode_dense(0, 2)

    assert dense.shape == (2, 2, 6, 7)
    assert dense.dtype == np.dtype(bool)
    np.testing.assert_array_equal(dense[0, 0], _binary_mask().astype(bool))
    assert not dense[0, 1].any()
    np.testing.assert_array_equal(dense[1, 0], _binary_mask(shift=1).astype(bool))
    assert not dense[1, 1].any()


def test_mask_table_writer_rejects_unsorted_streams(tmp_path: Path) -> None:
    table_path = tmp_path / "masks.parquet"
    writer = MaskTableWriter(table_path, frame_height=6, frame_width=7)
    writer.write_mask(
        frame_index=2,
        instance_index=0,
        instance_id="lfpaw",
        mask=SegmentationMask.from_binary_mask(_binary_mask(), class_name="paw"),
    )

    with pytest.raises(ValueError, match="frame/instance order"):
        writer.write_mask(
            frame_index=1,
            instance_index=0,
            instance_id="lfpaw",
            mask=SegmentationMask.from_binary_mask(_binary_mask(), class_name="paw"),
        )

    writer.close()


def test_mask_table_parquet_footer_identifies_xpkg_contract(tmp_path: Path) -> None:
    table_path = tmp_path / "masks.parquet"
    write_mask_table(table_path, [])

    reader = MaskTableReader(table_path)

    assert reader.info.encoding == XPKG_RLE_ENCODING
    assert reader.info.custom_metadata == {}

    import pyarrow.parquet as pq

    footer = {
        key.decode("utf-8"): value.decode("utf-8")
        for key, value in pq.ParquetFile(table_path).metadata.metadata.items()
    }
    assert footer["xpkg.kind"] == MASK_TABLE_KIND
    assert footer["schema_version"] == "1"
    assert footer["encoding"] == XPKG_RLE_ENCODING
