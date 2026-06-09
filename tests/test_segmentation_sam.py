"""Contract tests for SAM-family and Fiesta segmentation output adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xpkg._core.json_utils import write_json
from xpkg.segmentation.images import write_binary_mask
from xpkg.segmentation.model import PromptType, SegmentationPrompt
from xpkg.segmentation.rle import encode_mask_rle
from xpkg.segmentation.sam import (
    mask_from_sam_array,
    masks_from_fiesta_result_json,
    masks_from_fiesta_summary,
    masks_from_sam_arrays,
)


def _square_mask(size: int = 6, *, lo: int = 1, hi: int = 4) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[lo:hi, lo:hi] = 1
    return mask


def test_mask_from_sam_array_builds_default_prompt_metadata() -> None:
    mask = mask_from_sam_array(
        _square_mask(),
        class_name="paw",
        backend="sam2",
        model_id="sam2-base",
        instance_ref=3,
    )

    assert mask.class_name == "paw"
    assert mask.instance_ref == 3
    assert mask.is_predicted
    assert mask.prompt is not None
    assert mask.prompt.prompt_type == PromptType.NONE
    assert mask.prompt.backend == "sam2"
    assert mask.prompt.model_id == "sam2-base"
    np.testing.assert_array_equal(mask.to_binary_mask(), _square_mask())


def test_mask_from_sam_array_keeps_explicit_prompt() -> None:
    prompt = SegmentationPrompt(prompt_type=PromptType.TEXT, text="front paw")

    mask = mask_from_sam_array(_square_mask(), prompt=prompt, backend="ignored")

    assert mask.prompt is prompt


def test_masks_from_sam_arrays_pads_class_names_and_skips_short_boxes() -> None:
    result = masks_from_sam_arrays(
        [_square_mask(), np.eye(6, dtype=np.uint8)],
        boxes=[[1.0, 1.0, 4.0], [0.0, 0.0, 6.0, 6.0]],
        scores=[float("inf"), 0.25],
        class_names=["paw"],
    )

    assert [mask.class_name for mask in result.masks] == ["paw", "mask_001"]
    # Non-finite scores are normalized to NaN.
    assert np.isnan(result.masks[0].confidence)
    assert result.masks[1].confidence == pytest.approx(0.25)
    # The malformed 3-element box is dropped; only the second box survives.
    assert len(result.rois) == 1
    roi = result.rois[0]
    assert (roi.x1, roi.y1, roi.x2, roi.y2) == (0.0, 0.0, 6.0, 6.0)
    assert roi.instance_ref == 1
    assert roi.class_name == "mask_001"


def test_fiesta_summary_with_results_list_inherits_top_level_metadata() -> None:
    first = _square_mask()
    second = np.eye(6, dtype=np.uint8)
    summary = {
        "prompt": "mouse paw",
        "backend": "sam3",
        "model_id": "sam3-large",
        "results": [
            "not-a-frame",  # ignored entries must not break aggregation
            {"mask_rles": [encode_mask_rle(first)], "scores": [0.9]},
            {"mask_rles": [encode_mask_rle(second)], "backend": "sam2"},
        ],
    }

    result = masks_from_fiesta_summary(summary)

    assert len(result.masks) == 2
    assert result.rois == ()
    np.testing.assert_array_equal(result.masks[0].to_binary_mask(), first)
    np.testing.assert_array_equal(result.masks[1].to_binary_mask(), second)
    assert result.masks[0].confidence == pytest.approx(0.9)
    assert np.isnan(result.masks[1].confidence)
    assert result.masks[0].prompt is not None
    assert result.masks[0].prompt.text == "mouse paw"
    assert result.masks[0].prompt.backend == "sam3"
    # Frame-level metadata overrides the inherited summary value.
    assert result.masks[1].prompt is not None
    assert result.masks[1].prompt.backend == "sam2"


def test_fiesta_result_json_loads_mask_paths_relative_to_json(tmp_path: Path) -> None:
    mask = _square_mask()
    mask_dir = tmp_path / "masks"
    write_binary_mask(mask_dir / "mask_000.png", mask)
    result_path = tmp_path / "result.json"
    write_json(
        result_path,
        {
            "prompt": "paw",
            "mask_paths": ["masks/mask_000.png"],
            "boxes": [[1.0, 1.0, 4.0, 4.0]],
        },
    )

    result = masks_from_fiesta_result_json(result_path)

    assert len(result.masks) == 1
    assert result.masks[0].class_name == "mask_000"
    assert result.masks[0].mask_path == (tmp_path / "masks" / "mask_000.png").as_posix()
    np.testing.assert_array_equal(result.masks[0].to_binary_mask(), mask)
    assert len(result.rois) == 1
    assert (result.rois[0].x1, result.rois[0].y1) == (1.0, 1.0)


def test_fiesta_summary_can_skip_mask_path_loading(tmp_path: Path) -> None:
    result = masks_from_fiesta_summary(
        {"mask_paths": ["masks/missing.png"]},
        result_root=tmp_path,
        load_mask_paths=False,
    )

    assert result.masks == ()
    assert result.rois == ()


def test_fiesta_summary_resolves_absolute_mask_paths(tmp_path: Path) -> None:
    mask = _square_mask()
    mask_path = tmp_path / "external_mask.png"
    write_binary_mask(mask_path, mask)

    result = masks_from_fiesta_summary(
        {"mask_paths": [mask_path.as_posix()]},
        result_root=tmp_path / "elsewhere",
    )

    assert len(result.masks) == 1
    assert result.masks[0].mask_path == mask_path.as_posix()
    np.testing.assert_array_equal(result.masks[0].to_binary_mask(), mask)


def test_fiesta_summary_rejects_non_object_mask_rles() -> None:
    with pytest.raises(ValueError, match="mask_rles entries must be objects"):
        masks_from_fiesta_summary({"mask_rles": ["not-an-object"]})


def test_fiesta_summary_rejects_non_string_mask_paths() -> None:
    with pytest.raises(ValueError, match="mask_paths entries must be strings"):
        masks_from_fiesta_summary({"mask_paths": [123]})


def test_fiesta_summary_class_names_fall_back_to_track_labels() -> None:
    mask = _square_mask()

    result = masks_from_fiesta_summary(
        {
            "track_labels": ["tail"],
            "mask_rles": [encode_mask_rle(mask)],
        }
    )

    assert result.masks[0].class_name == "tail"
