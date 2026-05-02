"""Tests for segmentation mask/ROI schema, RLE, and roundtrip serialization."""

from __future__ import annotations

import json

import numpy as np
import pytest

from xpkg.pose.annotations.regions import (
    ROI,
    MaskType,
    PromptType,
    SegmentationMask,
    SegmentationPrompt,
    rle_decode,
    rle_encode,
)

# ---------------------------------------------------------------------------
# RLE encode / decode
# ---------------------------------------------------------------------------


class TestRLE:
    def test_roundtrip_simple(self):
        mask = np.array(
            [[0, 0, 1, 1, 1], [1, 1, 0, 0, 0], [0, 0, 0, 1, 1]], dtype=np.uint8
        )
        counts, start = rle_encode(mask)
        decoded = rle_decode(counts, start, mask.shape[0], mask.shape[1])
        np.testing.assert_array_equal(decoded, mask)

    def test_roundtrip_all_zeros(self):
        mask = np.zeros((10, 20), dtype=np.uint8)
        counts, start = rle_encode(mask)
        assert start == 0
        assert len(counts) == 1
        assert counts[0] == 200
        decoded = rle_decode(counts, start, 10, 20)
        np.testing.assert_array_equal(decoded, mask)

    def test_roundtrip_all_ones(self):
        mask = np.ones((5, 8), dtype=np.uint8)
        counts, start = rle_encode(mask)
        assert start == 1
        assert len(counts) == 1
        assert counts[0] == 40
        decoded = rle_decode(counts, start, 5, 8)
        np.testing.assert_array_equal(decoded, mask)

    def test_roundtrip_random(self):
        rng = np.random.default_rng(42)
        mask = rng.integers(0, 2, size=(64, 48), dtype=np.uint8)
        counts, start = rle_encode(mask)
        decoded = rle_decode(counts, start, 64, 48)
        np.testing.assert_array_equal(decoded, mask)

    def test_empty_mask(self):
        mask = np.zeros((0, 0), dtype=np.uint8)
        counts, start = rle_encode(mask)
        assert counts.size == 0

    def test_decode_mismatch_raises(self):
        with pytest.raises(ValueError, match="RLE counts sum"):
            rle_decode(np.array([5, 3], dtype=np.uint32), 0, 4, 4)


# ---------------------------------------------------------------------------
# SegmentationMask
# ---------------------------------------------------------------------------


class TestSegmentationMask:
    def test_polygon_construction(self):
        verts = np.array([[10, 20], [30, 20], [30, 50], [10, 50]], dtype=np.float32)
        mask = SegmentationMask.from_polygon(verts, class_name="subject")
        assert mask.is_polygon()
        assert not mask.is_rle()
        assert mask.class_name == "subject"
        assert mask.vertex_count == 4

    def test_rle_construction(self):
        mask = SegmentationMask.from_rle(
            [100, 50, 50], start=0, height=10, width=20, class_name="cell"
        )
        assert mask.is_rle()
        assert mask.rle_height == 10
        assert mask.rle_width == 20

    def test_from_binary_mask(self):
        binary = np.zeros((32, 32), dtype=np.uint8)
        binary[10:20, 5:25] = 1
        mask = SegmentationMask.from_binary_mask(binary, class_name="roi")
        assert mask.is_rle()
        decoded = mask.to_binary_mask()
        np.testing.assert_array_equal(decoded, binary)

    def test_polygon_bounding_box(self):
        verts = np.array(
            [[10.0, 20.0], [30.0, 20.0], [30.0, 50.0], [10.0, 50.0]],
            dtype=np.float32,
        )
        mask = SegmentationMask.from_polygon(verts)
        bb = mask.bounding_box
        np.testing.assert_array_almost_equal(bb, [10.0, 20.0, 30.0, 50.0])

    def test_rle_bounding_box(self):
        binary = np.zeros((10, 10), dtype=np.uint8)
        binary[2:5, 3:8] = 1
        mask = SegmentationMask.from_binary_mask(binary)
        bb = mask.bounding_box
        assert bb[0] == 3  # x1
        assert bb[1] == 2  # y1
        assert bb[2] == 8  # x2
        assert bb[3] == 5  # y2

    def test_dict_roundtrip_polygon(self):
        verts = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32),
            np.array([[2, 2], [8, 2], [8, 8]], dtype=np.float32),
        ]
        mask = SegmentationMask.from_polygon(
            verts, class_name="body", confidence=0.95
        )
        d = mask.to_dict()
        restored = SegmentationMask.from_dict(d)
        assert restored.mask_type == MaskType.POLYGON
        assert restored.class_name == "body"
        assert abs(restored.confidence - 0.95) < 1e-6
        assert restored.polygon_vertices is not None
        assert len(restored.polygon_vertices) == 2
        np.testing.assert_array_almost_equal(
            restored.polygon_vertices[0], verts[0]
        )

    def test_dict_roundtrip_rle(self):
        binary = np.zeros((16, 16), dtype=np.uint8)
        binary[4:12, 4:12] = 1
        mask = SegmentationMask.from_binary_mask(binary, class_name="cell")
        d = mask.to_dict()
        restored = SegmentationMask.from_dict(d)
        assert restored.mask_type == MaskType.RLE
        decoded = restored.to_binary_mask()
        np.testing.assert_array_equal(decoded, binary)

    def test_dict_roundtrip_with_prompt(self):
        prompt = SegmentationPrompt(
            prompt_type=PromptType.BOX,
            box=(10.0, 20.0, 100.0, 200.0),
            model_id="sam2-large",
            backend="torch",
        )
        mask = SegmentationMask.from_polygon(
            np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32),
            class_name="test",
        )
        mask.prompt = prompt
        d = mask.to_dict()
        assert "prompt" in d
        restored = SegmentationMask.from_dict(d)
        assert restored.prompt is not None
        assert restored.prompt.prompt_type == PromptType.BOX
        assert restored.prompt.box == (10.0, 20.0, 100.0, 200.0)
        assert restored.prompt.model_id == "sam2-large"

    def test_invalid_binary_mask_shape_raises(self):
        with pytest.raises(ValueError, match="2D binary mask"):
            SegmentationMask.from_binary_mask(np.zeros((3, 4, 5)))


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------


class TestROI:
    def test_properties(self):
        roi = ROI(x1=10, y1=20, x2=50, y2=80)
        assert roi.width == 40
        assert roi.height == 60
        assert roi.area == 2400
        assert roi.center == (30.0, 50.0)

    def test_as_array(self):
        roi = ROI(x1=1, y1=2, x2=3, y2=4)
        np.testing.assert_array_equal(roi.as_array(), [1, 2, 3, 4])

    def test_dict_roundtrip(self):
        roi = ROI(
            x1=5, y1=10, x2=100, y2=200, class_name="animal", confidence=0.9
        )
        d = roi.to_dict()
        restored = ROI.from_dict(d)
        assert restored.x1 == 5
        assert restored.class_name == "animal"
        assert abs(restored.confidence - 0.9) < 1e-6


# ---------------------------------------------------------------------------
# LabeledFrame integration
# ---------------------------------------------------------------------------


class TestLabeledFrameIntegration:
    def _make_video_stub(self):
        class _VideoStub:
            filename = "test.mp4"
            image_filenames = []
            frames = 100
            height = 480
            width = 640
            channels = 3
            backend = "opencv"
            sha256 = ""
            id = "video_0"
            label = "video-0"

            def close(self):
                pass

            def get_frame(self, idx):
                return np.zeros((self.height, self.width, self.channels), dtype=np.uint8)

        return _VideoStub()

    def test_masks_and_rois_on_frame(self):
        from xpkg.pose.annotations.frames import LabeledFrame

        video = self._make_video_stub()
        mask = SegmentationMask.from_polygon(
            np.array([[0, 0], [10, 0], [10, 10]], dtype=np.float32)
        )
        roi = ROI(x1=0, y1=0, x2=10, y2=10)
        lf = LabeledFrame(video=video, frame_idx=0, masks=[mask], rois=[roi])
        assert lf.has_masks
        assert lf.has_rois
        assert len(lf.masks) == 1
        assert len(lf.rois) == 1

    def test_default_empty(self):
        from xpkg.pose.annotations.frames import LabeledFrame

        video = self._make_video_stub()
        lf = LabeledFrame(video=video, frame_idx=0)
        assert not lf.has_masks
        assert not lf.has_rois
        assert lf.masks == []
        assert lf.rois == []

    def test_copy_preserves_masks(self):
        from xpkg.pose.annotations.frames import LabeledFrame

        video = self._make_video_stub()
        mask = SegmentationMask.from_binary_mask(
            np.eye(8, dtype=np.uint8), class_name="diag"
        )
        roi = ROI(x1=1, y1=2, x2=3, y2=4, class_name="box")
        lf = LabeledFrame(video=video, frame_idx=5, masks=[mask], rois=[roi])
        clone = lf.copy()
        assert len(clone.masks) == 1
        assert clone.masks[0].class_name == "diag"
        assert len(clone.rois) == 1
        assert clone.rois[0].class_name == "box"
        # Ensure deep copy
        assert clone.masks[0] is not mask
        assert clone.rois[0] is not roi

    def test_user_vs_predicted_masks(self):
        from xpkg.pose.annotations.frames import LabeledFrame

        video = self._make_video_stub()
        user_mask = SegmentationMask.from_polygon(
            np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32),
            class_name="user",
        )
        pred_mask = SegmentationMask.from_polygon(
            np.array([[5, 5], [6, 5], [6, 6]], dtype=np.float32),
            class_name="pred",
        )
        pred_mask.is_predicted = True
        lf = LabeledFrame(
            video=video, frame_idx=0, masks=[user_mask, pred_mask]
        )
        assert len(lf.user_masks) == 1
        assert len(lf.predicted_masks) == 1
        assert lf.user_masks[0].class_name == "user"
        assert lf.predicted_masks[0].class_name == "pred"


# ---------------------------------------------------------------------------
# JSON roundtrip
# ---------------------------------------------------------------------------


class TestJSONRoundtrip:
    def test_mask_json_serialization(self):
        prompt = SegmentationPrompt(
            prompt_type=PromptType.TEXT,
            text="a subject",
            model_id="sam3",
            backend="mlx",
        )
        mask = SegmentationMask.from_polygon(
            np.array([[0, 0], [5, 0], [5, 5]], dtype=np.float32),
            class_name="subject",
            confidence=0.92,
        )
        mask.prompt = prompt
        mask.is_predicted = True

        d = mask.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        restored = SegmentationMask.from_dict(parsed)

        assert restored.mask_type == MaskType.POLYGON
        assert restored.class_name == "subject"
        assert restored.is_predicted
        assert restored.prompt is not None
        assert restored.prompt.prompt_type == PromptType.TEXT
        assert restored.prompt.text == "a subject"

    def test_roi_json_serialization(self):
        roi = ROI(x1=10, y1=20, x2=100, y2=200, class_name="region")
        d = roi.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        restored = ROI.from_dict(parsed)
        assert restored.x1 == 10
        assert restored.class_name == "region"

    def test_rle_mask_json_roundtrip(self):
        binary = np.zeros((24, 32), dtype=np.uint8)
        binary[8:16, 8:24] = 1
        mask = SegmentationMask.from_binary_mask(binary, class_name="blob")
        d = mask.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        restored = SegmentationMask.from_dict(parsed)
        decoded = restored.to_binary_mask()
        np.testing.assert_array_equal(decoded, binary)


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


class TestPublicAPISurface:
    def test_model_exports(self):
        from xpkg.model import (
            ROI,
            MaskType,
            PromptType,
            SegmentationMask,
            SegmentationPrompt,
            rle_decode,
            rle_encode,
        )

        assert callable(rle_encode)
        assert callable(rle_decode)
        assert MaskType.POLYGON == 0
        assert MaskType.RLE == 1
        assert PromptType.BOX == 1
        assert ROI.__name__ == "ROI"
        assert SegmentationMask.__name__ == "SegmentationMask"
        assert SegmentationPrompt.__name__ == "SegmentationPrompt"

    def test_annotations_exports(self):
        from xpkg.pose.annotations import (
            ROI,
            MaskType,
            PromptType,
            SegmentationMask,
            SegmentationPrompt,
            rle_decode,
            rle_encode,
        )

        assert callable(rle_encode)
        assert callable(rle_decode)
        assert MaskType.POLYGON == 0
        assert PromptType.BOX == 1
        assert ROI.__name__ == "ROI"
        assert SegmentationMask.__name__ == "SegmentationMask"
        assert SegmentationPrompt.__name__ == "SegmentationPrompt"
