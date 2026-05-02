from __future__ import annotations

import json

import numpy as np

import xpkg
from xpkg.segmentation import (
    PromptType,
    SegmentationMask,
    SegmentationPrompt,
    coco_rle_to_mask,
    decode_mask_rle,
    encode_mask_rle,
    mask_to_coco_annotation,
    mask_to_coco_rle,
    masks_from_fiesta_summary,
    masks_from_label_image,
    masks_from_sam_arrays,
    rasterize_polygon,
    read_binary_mask,
    read_label_image,
    write_binary_mask,
    write_label_image,
)


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
