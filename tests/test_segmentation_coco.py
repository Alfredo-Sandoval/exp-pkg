"""Contract tests for the COCO segmentation conversion boundary.

COCO RLE is column-major and always alternates starting from background
counts; these tests pin that convention with hand-derived payloads.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from xpkg._core.json_utils import write_json
from xpkg.segmentation import coco as coco_module
from xpkg.segmentation.coco import (
    annotation_to_masks,
    annotations_to_masks,
    coco_rle_to_mask,
    mask_to_coco_annotation,
    mask_to_coco_polygons,
    mask_to_coco_rle,
    read_coco_annotations,
    segmentation_to_masks,
)
from xpkg.segmentation.model import MaskType, SegmentationMask


def test_coco_rle_decode_is_column_major() -> None:
    # Column-major traversal of [[0, 1], [1, 0]] is 0, 1, 1, 0 -> counts 1, 2, 1.
    mask = coco_rle_to_mask({"size": [2, 2], "counts": [1, 2, 1]})

    np.testing.assert_array_equal(mask, np.array([[0, 1], [1, 0]], dtype=np.uint8))


def test_coco_rle_encode_is_column_major_starting_from_background() -> None:
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    rle = mask_to_coco_rle(mask)

    assert rle == {"size": [2, 2], "counts": [1, 2, 1]}


def test_coco_rle_encode_all_ones_starts_with_zero_count() -> None:
    rle = mask_to_coco_rle(np.ones((2, 3), dtype=np.uint8))

    assert rle == {"size": [2, 3], "counts": [0, 6]}


def test_coco_rle_encode_all_zeros_is_single_count() -> None:
    rle = mask_to_coco_rle(np.zeros((2, 3), dtype=np.uint8))

    assert rle == {"size": [2, 3], "counts": [6]}


def test_coco_rle_encode_rejects_non_2d_mask() -> None:
    with pytest.raises(ValueError, match="Expected 2D binary mask"):
        mask_to_coco_rle(np.zeros((2, 2, 3), dtype=np.uint8))


def test_coco_rle_decode_rejects_bad_size_field() -> None:
    with pytest.raises(ValueError, match=r"'size' must be \[height, width\]"):
        coco_rle_to_mask({"size": [2], "counts": [4]})


def test_coco_rle_decode_rejects_non_sequence_counts() -> None:
    with pytest.raises(ValueError, match="'counts' must be a sequence"):
        coco_rle_to_mask({"size": [2, 2], "counts": 4})


def test_coco_rle_decode_rejects_count_total_mismatch() -> None:
    with pytest.raises(ValueError, match="decode size mismatch: expected 4, got 3"):
        coco_rle_to_mask({"size": [2, 2], "counts": [1, 2]})


def test_compressed_coco_rle_requires_pycocotools(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing(_name: str) -> object:
        raise ImportError("No module named 'pycocotools'")

    monkeypatch.setattr(
        coco_module,
        "importlib",
        types.SimpleNamespace(import_module=_missing),
    )

    with pytest.raises(ImportError, match="Compressed COCO RLE requires pycocotools"):
        coco_rle_to_mask({"size": [2, 2], "counts": "012"})


def test_segmentation_to_masks_converts_polygon_lists() -> None:
    masks = segmentation_to_masks(
        [[1.0, 2.0, 5.0, 2.0, 5.0, 6.0], [0.0, 0.0, 2.0, 0.0, 0.0, 2.0]],
        class_name="paw",
    )

    assert len(masks) == 2
    assert all(mask.mask_type == MaskType.POLYGON for mask in masks)
    assert masks[0].class_name == "paw"
    assert masks[0].polygon_vertices is not None
    np.testing.assert_array_equal(
        masks[0].polygon_vertices[0],
        np.array([[1.0, 2.0], [5.0, 2.0], [5.0, 6.0]], dtype=np.float32),
    )


def test_segmentation_to_masks_rejects_odd_polygon_coordinates() -> None:
    with pytest.raises(ValueError, match="must contain x/y pairs"):
        segmentation_to_masks([[1.0, 2.0, 3.0]])


def test_segmentation_to_masks_decodes_rle_mapping() -> None:
    masks = segmentation_to_masks({"size": [2, 2], "counts": [1, 2, 1]}, class_name="body")

    assert len(masks) == 1
    np.testing.assert_array_equal(
        masks[0].to_binary_mask(),
        np.array([[0, 1], [1, 0]], dtype=np.uint8),
    )


def test_segmentation_to_masks_rejects_unsupported_payload() -> None:
    with pytest.raises(ValueError, match="Unsupported COCO segmentation payload"):
        segmentation_to_masks(42)


def test_segmentation_to_masks_returns_empty_for_unsupported_with_image_size() -> None:
    assert segmentation_to_masks(42, image_size=(4, 4)) == []


def test_annotation_to_masks_resolves_category_names_and_score() -> None:
    annotation = {
        "id": 7,
        "category_id": 3,
        "score": 0.75,
        "segmentation": [[0.0, 0.0, 4.0, 0.0, 0.0, 4.0]],
    }

    masks = annotation_to_masks(annotation, category_names={3: "mouse"})

    assert len(masks) == 1
    assert masks[0].class_name == "mouse"
    assert masks[0].confidence == pytest.approx(0.75)
    assert masks[0].instance_ref == 7
    assert not masks[0].is_predicted


def test_annotation_to_masks_falls_back_to_numeric_category_name() -> None:
    annotation = {"category_id": 9, "segmentation": [[0.0, 0.0, 4.0, 0.0, 0.0, 4.0]]}

    masks = annotation_to_masks(annotation, category_names={3: "mouse"})

    assert masks[0].class_name == "9"


def test_annotation_to_masks_uses_empty_name_without_category_id() -> None:
    annotation = {"segmentation": [[0.0, 0.0, 4.0, 0.0, 0.0, 4.0]]}

    masks = annotation_to_masks(annotation)

    assert masks[0].class_name == ""
    assert masks[0].instance_ref == -1


def test_annotations_to_masks_applies_categories_and_predicted_flag() -> None:
    masks = annotations_to_masks(
        [
            {"id": 1, "category_id": 1, "segmentation": [[0.0, 0.0, 2.0, 0.0, 0.0, 2.0]]},
            {"id": 2, "category_id": 2, "segmentation": [[1.0, 1.0, 3.0, 1.0, 1.0, 3.0]]},
        ],
        categories=[{"id": 1, "name": "paw"}, {"id": 2, "name": "tail"}],
        is_predicted=True,
    )

    assert [mask.class_name for mask in masks] == ["paw", "tail"]
    assert [mask.instance_ref for mask in masks] == [1, 2]
    assert all(mask.is_predicted for mask in masks)


def test_mask_to_coco_annotation_polygon_payload_keeps_exact_vertices() -> None:
    mask = SegmentationMask.from_polygon(
        np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]], dtype=np.float32),
        class_name="paw",
        confidence=0.5,
    )

    annotation = mask_to_coco_annotation(
        mask,
        image_id=3,
        category_id=2,
        annotation_id=11,
        use_rle=False,
    )

    assert annotation["image_id"] == 3
    assert annotation["category_id"] == 2
    assert annotation["id"] == 11
    assert annotation["iscrowd"] == 0
    assert annotation["score"] == pytest.approx(0.5)
    assert annotation["segmentation"] == [[0.0, 0.0, 4.0, 0.0, 0.0, 4.0]]
    # Right triangle with legs of length 4 has area 8.
    assert annotation["area"] == pytest.approx(8.0)
    assert annotation["bbox"] == [0.0, 0.0, 4.0, 4.0]


def test_mask_to_coco_annotation_omits_score_for_nan_confidence() -> None:
    binary = np.zeros((4, 4), dtype=np.uint8)
    binary[1:3, 1:3] = 1
    mask = SegmentationMask.from_binary_mask(binary)

    annotation = mask_to_coco_annotation(mask, image_id=1, category_id=1)

    assert "score" not in annotation
    assert "id" not in annotation
    assert annotation["iscrowd"] == 1
    assert annotation["area"] == 4
    np.testing.assert_array_equal(
        coco_rle_to_mask(annotation["segmentation"]),
        binary,
    )


def test_mask_to_coco_annotation_rasterizes_rle_mask_without_polygons() -> None:
    binary = np.zeros((6, 6), dtype=np.uint8)
    binary[1:4, 2:5] = 1
    mask = SegmentationMask.from_binary_mask(binary, class_name="body")

    annotation = mask_to_coco_annotation(mask, image_id=1, category_id=1, use_rle=False)

    assert annotation["iscrowd"] == 0
    assert annotation["area"] == 9
    assert len(annotation["segmentation"]) == 1
    assert len(annotation["segmentation"][0]) >= 6


def test_mask_to_coco_polygons_skips_degenerate_contours() -> None:
    single_pixel = np.zeros((4, 4), dtype=np.uint8)
    single_pixel[2, 2] = 1

    assert mask_to_coco_polygons(single_pixel) == []


def test_read_coco_annotations_loads_masks_from_file(tmp_path: Path) -> None:
    payload = {
        "annotations": [
            {
                "id": 4,
                "category_id": 1,
                "score": 0.9,
                "segmentation": {"size": [2, 2], "counts": [1, 2, 1]},
            }
        ],
        "categories": [{"id": 1, "name": "paw"}],
    }
    path = tmp_path / "coco.json"
    write_json(path, payload)

    masks = read_coco_annotations(path, is_predicted=True)

    assert len(masks) == 1
    assert masks[0].class_name == "paw"
    assert masks[0].instance_ref == 4
    assert masks[0].is_predicted
    np.testing.assert_array_equal(
        masks[0].to_binary_mask(),
        np.array([[0, 1], [1, 0]], dtype=np.uint8),
    )


def test_read_coco_annotations_rejects_non_list_annotations(tmp_path: Path) -> None:
    path = tmp_path / "coco.json"
    write_json(path, {"annotations": {"id": 1}, "categories": []})

    with pytest.raises(ValueError, match="'annotations' must be a list"):
        read_coco_annotations(path)


def test_read_coco_annotations_rejects_non_list_categories(tmp_path: Path) -> None:
    path = tmp_path / "coco.json"
    write_json(path, {"annotations": [], "categories": {"id": 1}})

    with pytest.raises(ValueError, match="'categories' must be a list"):
        read_coco_annotations(path)
