from __future__ import annotations

import json
from collections.abc import Callable

import cv2
import numpy as np
import pytest

import xpkg
from xpkg.segmentation import (
    PromptType,
    SegmentationMask,
    SegmentationPrompt,
    SupervisionOverlayError,
    best_mask_index,
    coco_rle_to_mask,
    decode_mask_rle,
    encode_mask_rle,
    format_supervision_overlay_labels,
    mask_to_coco_annotation,
    mask_to_coco_rle,
    masks_from_fiesta_summary,
    masks_from_label_image,
    masks_from_sam_arrays,
    rasterize_polygon,
    read_binary_mask,
    read_label_image,
    read_normalized_polygon_dataset_yaml,
    read_normalized_polygon_labels,
    read_normalized_polygon_rows,
    render_supervision_overlay,
    select_masks_for_save,
    write_binary_mask,
    write_binary_masks,
    write_label_image,
    write_mask_overlay,
    write_normalized_polygon_dataset_yaml,
    write_normalized_polygon_labels,
    write_supervision_overlay,
)
from xpkg.segmentation import images as image_helpers


def test_segmentation_module_is_public_from_root() -> None:
    assert xpkg.segmentation is not None
    assert xpkg.segmentation.SegmentationMask is SegmentationMask


def test_exact_rle_payload_round_trips_binary_mask() -> None:
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[1:4, 2:6] = 1

    payload = encode_mask_rle(mask)
    restored = decode_mask_rle(payload)

    assert payload["encoding"] == "xpkg.rle.v1"
    assert payload["size"] == [5, 7]
    assert payload["order"] == "C"
    np.testing.assert_array_equal(restored, mask)


def test_segmentation_mask_constructors_accept_prompt_and_predicted_flag() -> None:
    prompt = SegmentationPrompt(
        prompt_type=PromptType.TEXT,
        text="front paw",
        model_id="sam3",
        backend="fiesta",
    )
    mask = SegmentationMask.from_binary_mask(
        np.eye(4, dtype=np.uint8),
        class_name="paw",
        prompt=prompt,
        is_predicted=True,
        mask_id="frame-0-mask-0",
    )
    payload = mask.to_dict()
    restored = SegmentationMask.from_dict(json.loads(json.dumps(payload)))

    assert restored.is_predicted
    assert restored.mask_id == "frame-0-mask-0"
    assert restored.prompt is not None
    assert restored.prompt.text == "front paw"
    np.testing.assert_array_equal(restored.to_binary_mask(), np.eye(4, dtype=np.uint8))


def test_polygon_rasterization_is_explicit() -> None:
    vertices = np.array(
        [[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]],
        dtype=np.float32,
    )
    mask = rasterize_polygon(vertices, height=6, width=6)

    assert mask.dtype == np.uint8
    assert mask[2, 2] == 1
    assert mask[0, 0] == 0


def test_binary_and_label_mask_images_round_trip(tmp_path) -> None:
    binary = np.zeros((8, 9), dtype=np.uint8)
    binary[2:5, 3:7] = 1
    binary_path = tmp_path / "mask.png"
    write_binary_mask(binary_path, binary)
    np.testing.assert_array_equal(read_binary_mask(binary_path), binary)

    labels = np.zeros((8, 9), dtype=np.uint16)
    labels[1:3, 1:4] = 3
    labels[4:7, 5:8] = 7
    label_path = tmp_path / "labels.tiff"
    write_label_image(label_path, labels)
    loaded = read_label_image(label_path)
    np.testing.assert_array_equal(loaded, labels)

    split = masks_from_label_image(loaded)
    assert sorted(split) == [3, 7]
    assert int(split[3].sum()) == 6
    assert int(split[7].sum()) == 9


def test_write_binary_masks_preserves_ordered_names(tmp_path) -> None:
    first = np.eye(3, dtype=np.uint8)
    second = np.fliplr(first)

    paths = write_binary_masks(
        tmp_path,
        [first, second],
        png_compression=0,
    )

    assert [path.name for path in paths] == ["mask_000.png", "mask_001.png"]
    np.testing.assert_array_equal(read_binary_mask(paths[0]), first)
    np.testing.assert_array_equal(read_binary_mask(paths[1]), second)


def test_best_mask_index_returns_argmax_and_rejects_non_finite() -> None:
    assert best_mask_index([0.1, 0.9, 0.2]) == 1
    assert best_mask_index([]) == 0
    with pytest.raises(ValueError, match="scores must contain only finite numeric values"):
        best_mask_index([0.5, np.nan])


def test_select_masks_for_save_obeys_policy() -> None:
    masks = [
        np.zeros((2, 2), dtype=np.uint8),
        np.ones((2, 2), dtype=np.uint8),
    ]

    assert select_masks_for_save(masks, save_masks="none") == []

    top1_masks = select_masks_for_save(masks, save_masks="top1")
    assert len(top1_masks) == 1
    np.testing.assert_array_equal(top1_masks[0], masks[0])

    all_masks = select_masks_for_save(masks, save_masks="all")
    assert len(all_masks) == 2
    for saved, original in zip(all_masks, masks, strict=True):
        np.testing.assert_array_equal(saved, original)


def test_write_mask_overlay_tints_masked_pixels(tmp_path) -> None:
    image = np.full((6, 6, 3), 100, dtype=np.uint8)
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[2:4, 2:4] = 1
    path = tmp_path / "overlay.png"

    write_mask_overlay(
        path,
        image,
        mask,
        tint_rgb=(200, 0, 0),
        opacity=0.5,
        png_compression=0,
    )

    loaded = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
    assert loaded is not None
    rgb = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(rgb[0, 0], np.array([100, 100, 100], dtype=np.uint8))
    np.testing.assert_array_equal(rgb[2, 2], np.array([150, 50, 50], dtype=np.uint8))


def test_write_mask_overlay_can_resize_mask_and_draw_box(tmp_path) -> None:
    image = np.full((8, 8, 3), 20, dtype=np.uint8)
    mask = np.ones((4, 4), dtype=np.uint8)
    path = tmp_path / "boxed_overlay.png"

    write_mask_overlay(
        path,
        image,
        mask,
        tint_rgb=(20, 20, 120),
        opacity=0.25,
        box=(1, 1, 6, 6),
        box_outline_rgb=(0, 255, 0),
        resize_mask_to_image=True,
        png_compression=0,
    )

    loaded = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
    assert loaded is not None
    rgb = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(rgb[1, 1], np.array([0, 255, 0], dtype=np.uint8))


def test_format_supervision_overlay_labels_uses_tracker_prefixes() -> None:
    labels = format_supervision_overlay_labels(
        labels=["mouse", "tail"],
        confidences=[0.8123, 0.634],
        tracker_ids=[40, None],
        confidence_digits=1,
    )

    assert labels == ["#40 mouse 0.8", "#pending tail 0.6"]


def test_render_supervision_overlay_draws_masks_boxes_and_labels() -> None:
    image = np.full((12, 18, 3), 24, dtype=np.uint8)
    mask_a = np.zeros((12, 18), dtype=np.uint8)
    mask_a[3:8, 4:10] = 1
    mask_b = np.zeros((12, 18), dtype=np.uint8)
    mask_b[5:10, 11:16] = 1

    overlay = render_supervision_overlay(
        image,
        masks=[mask_a, mask_b],
        boxes_xyxy=[(4.0, 3.0, 10.0, 8.0), (11.0, 5.0, 16.0, 10.0)],
        labels=["mouse", "tail"],
        confidences=[0.82, 0.63],
        tracker_ids=[40, None],
        opacity=0.35,
        palette_rgb=[(46, 204, 113), (52, 152, 219)],
    )

    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
    assert not np.array_equal(overlay, image)
    assert np.any(overlay[mask_a.astype(bool)] != image[mask_a.astype(bool)])


def test_write_supervision_overlay_persists_rgb_png(tmp_path) -> None:
    image = np.full((8, 8, 3), 20, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    path = tmp_path / "supervision_overlay.png"

    write_supervision_overlay(
        path,
        image,
        masks=[mask],
        labels=["mouse"],
        confidences=[0.9],
        png_compression=0,
    )

    loaded = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
    assert loaded is not None
    rgb = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
    assert rgb.shape == image.shape
    assert not np.array_equal(rgb, image)


def test_supervision_overlay_reports_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def _raise_missing_supervision(_name: str) -> Callable[..., object]:
        raise ModuleNotFoundError(name="supervision")

    monkeypatch.setattr(image_helpers, "import_module", _raise_missing_supervision)

    with pytest.raises(SupervisionOverlayError, match="Roboflow supervision is required"):
        write_supervision_overlay(
            tmp_path / "overlay.png",
            np.zeros((4, 4, 3), dtype=np.uint8),
            boxes_xyxy=[(0.0, 0.0, 2.0, 2.0)],
        )


def test_coco_uncompressed_rle_round_trips_with_column_major_contract() -> None:
    mask = np.zeros((6, 5), dtype=np.uint8)
    mask[1:5, 2:4] = 1

    coco_rle = mask_to_coco_rle(mask)
    restored = coco_rle_to_mask(coco_rle)
    annotation = mask_to_coco_annotation(
        SegmentationMask.from_binary_mask(mask, class_name="body"),
        image_id=12,
        category_id=4,
        annotation_id=99,
    )

    np.testing.assert_array_equal(restored, mask)
    assert annotation["id"] == 99
    assert annotation["segmentation"]["size"] == [6, 5]
    assert annotation["area"] == int(mask.sum())


def test_sam_arrays_and_fiesta_rle_summary_convert_to_masks_and_rois() -> None:
    first = np.zeros((6, 6), dtype=np.uint8)
    first[1:4, 2:5] = 1
    second = np.eye(6, dtype=np.uint8)

    result = masks_from_sam_arrays(
        [first, second],
        boxes=[[2, 1, 5, 4], [0, 0, 6, 6]],
        scores=[0.91, None],
        prompt_text="mouse paw",
        backend="sam3",
        model_id="sam3-large",
        class_names=["paw", "body"],
    )

    assert len(result.masks) == 2
    assert len(result.rois) == 2
    assert result.masks[0].class_name == "paw"
    assert result.masks[0].is_predicted
    assert result.masks[0].prompt is not None
    assert result.masks[0].prompt.text == "mouse paw"

    fiesta = masks_from_fiesta_summary(
        {
            "prompt": "mouse paw",
            "backend": "sam3",
            "model_id": "sam3-large",
            "class_names": ["paw"],
            "scores": [0.91],
            "boxes": [[2, 1, 5, 4]],
            "mask_rles": [encode_mask_rle(first)],
        }
    )

    assert len(fiesta.masks) == 1
    assert len(fiesta.rois) == 1
    assert fiesta.masks[0].class_name == "paw"
    np.testing.assert_array_equal(fiesta.masks[0].to_binary_mask(), first)


def test_normalized_polygon_sidecar_labels_round_trip(tmp_path) -> None:
    label_path = tmp_path / "labels" / "frame_001.txt"
    mask = SegmentationMask.from_polygon(
        np.array([[64.0, 48.0], [128.0, 48.0], [128.0, 96.0]], dtype=np.float32),
        class_name="paw",
    )

    write_normalized_polygon_labels(
        label_path,
        [mask],
        image_width=640,
        image_height=480,
        class_name_to_id={"paw": 2},
        precision=4,
    )

    rows = read_normalized_polygon_rows(label_path)
    loaded = read_normalized_polygon_labels(
        label_path,
        image_width=640,
        image_height=480,
        class_names={2: "paw"},
    )

    assert label_path.read_text(encoding="utf-8") == "2 0.1 0.1 0.2 0.1 0.2 0.2\n"
    assert len(rows) == 1
    assert rows[0].class_index == 2
    assert len(loaded) == 1
    assert loaded[0].class_name == "paw"
    assert loaded[0].polygon_vertices is not None
    assert mask.polygon_vertices is not None
    np.testing.assert_allclose(loaded[0].polygon_vertices[0], mask.polygon_vertices[0])


def test_normalized_polygon_dataset_yaml_round_trip(tmp_path) -> None:
    yaml_path = tmp_path / "dataset.yaml"

    write_normalized_polygon_dataset_yaml(
        yaml_path,
        names={0: "body", 1: "paw"},
        train="images/train",
        val="images/val",
        test="images/test",
    )
    payload = read_normalized_polygon_dataset_yaml(yaml_path)

    assert payload["path"] == "."
    assert payload["train"] == "images/train"
    assert payload["val"] == "images/val"
    assert payload["test"] == "images/test"
    assert payload["names"] == {0: "body", 1: "paw"}
