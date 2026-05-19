# ruff: noqa: E402, I001

from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

primitives = pytest.importorskip("primitives")
PrimitivesSession = primitives.PrimitivesSession

from xpkg.adapters import labels_to_primitives_session, project_to_primitives_session
from xpkg.io.labels.model import Labels
from xpkg.media.video import Video
from xpkg.pose.annotations import (
    Instance,
    LabeledFrame,
    Point,
    PredictedInstance,
    PredictedPoint,
    Track,
)
from xpkg.pose.skeleton import Keypoint, Skeleton


def _video(
    *,
    label: str = "top-camera",
    video_id: str = "video-top",
    filename: str = "top-camera.avi",
    frames: int = 4,
    fps: float = 120.0,
) -> Video:
    video = Video.__new__(Video)
    video.label = label
    video.id = video_id
    video.filename = filename
    video.frames = frames
    video.fps = fps
    video.width = 640
    video.height = 480
    video.channels = 3
    video.backend = "test"
    video.sha256 = ""
    return video


def _skeleton() -> Skeleton:
    return Skeleton(
        name="mouse",
        keypoints=[
            Keypoint(id=0, name="nose"),
            Keypoint(id=1, name="tail"),
        ],
        links_ids=[(0, 1)],
        aliases={"snout": "nose"},
    )


def _labels(video: Video, skeleton: Skeleton, frames: list[LabeledFrame]) -> Labels:
    labels = Labels(
        labeled_frames=frames,
        videos=[video],
        skeletons=[skeleton],
        provenance={"converter": "unit-test"},
        preferences={"theme": "quiet"},
        session={"animal_id": "m1"},
    )
    labels.validate()
    return labels


def test_labels_to_primitives_session_converts_user_labels_to_dense_session(
    tmp_path: Path,
) -> None:
    video = _video()
    skeleton = _skeleton()
    track = Track(spawned_on=7, name="mouse-a")
    labels = _labels(
        video,
        skeleton,
        [
            LabeledFrame(
                video=video,
                frame_idx=0,
                instances=[
                    Instance(
                        skeleton=skeleton,
                        track=track,
                        init_points={
                            "nose": Point(1.0, 2.0, visible=True),
                            "tail": Point(3.0, 4.0, visible=True),
                        },
                    )
                ],
            ),
            LabeledFrame(
                video=video,
                frame_idx=2,
                instances=[
                    Instance(
                        skeleton=skeleton,
                        track=track,
                        init_points={
                            "nose": Point(5.0, 6.0, visible=True),
                            "tail": Point(9.0, 10.0, visible=False),
                        },
                    )
                ],
            ),
        ],
    )
    labels.path = tmp_path / "labels.json"

    session = labels_to_primitives_session(labels, track=7)

    assert isinstance(session, PrimitivesSession)
    assert session.label == "top-camera"
    assert session.modality == "xpkg"
    assert session.root == tmp_path
    assert session.frame_count == 4
    assert session.fps == 120.0
    assert session.bodyparts == ("nose", "tail")
    assert session.skeleton.name == "mouse"
    assert session.skeleton.edges == (("nose", "tail"),)
    assert session.skeleton.aliases == {"snout": "nose"}
    np.testing.assert_allclose(session.coords[0, :, :2], [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(session.coords[2, 0, :2], [5.0, 6.0])
    assert np.isnan(session.coords[1, :, :2]).all()
    assert np.isnan(session.coords[2, 1, :2]).all()
    assert session.likelihoods is not None
    assert np.isnan(session.likelihoods).all()
    assert session.videos[0].path == Path("top-camera.avi")
    assert session.extras["xpkg"]["track_id"] == 7
    assert session.extras["xpkg"]["track_name"] == "mouse-a"
    assert session.extras["xpkg"]["provenance"] == {"converter": "unit-test"}


def test_labels_to_primitives_session_prefers_predictions_but_can_select_user_labels() -> None:
    video = _video(frames=1)
    skeleton = _skeleton()
    track = Track(spawned_on=3, name="mouse-a")
    predicted = PredictedInstance(
        skeleton=skeleton,
        track=track,
        init_points={
            "nose": PredictedPoint(10.0, 20.0, score=0.8),
            "tail": PredictedPoint(30.0, 40.0, score=0.6),
        },
        score=0.9,
    )
    user = Instance(
        skeleton=skeleton,
        from_predicted=predicted,
        init_points={
            "nose": Point(1.0, 2.0, visible=True),
            "tail": Point(3.0, 4.0, visible=True),
        },
    )
    labels = _labels(
        video,
        skeleton,
        [LabeledFrame(video=video, frame_idx=0, instances=[user, predicted])],
    )

    predicted_session = labels_to_primitives_session(labels, track="mouse-a")
    user_session = labels_to_primitives_session(
        labels,
        track="mouse-a",
        use_predicted=False,
    )

    np.testing.assert_allclose(
        predicted_session.coords[0, :, :2],
        [[10.0, 20.0], [30.0, 40.0]],
    )
    assert predicted_session.likelihoods is not None
    np.testing.assert_allclose(predicted_session.likelihoods[0], [0.8, 0.6])
    np.testing.assert_allclose(user_session.coords[0, :, :2], [[1.0, 2.0], [3.0, 4.0]])
    assert user_session.likelihoods is not None
    assert np.isnan(user_session.likelihoods[0]).all()


def test_labels_to_primitives_session_requires_track_when_streams_are_ambiguous() -> None:
    video = _video(frames=1)
    skeleton = _skeleton()
    labels = _labels(
        video,
        skeleton,
        [
            LabeledFrame(
                video=video,
                frame_idx=0,
                instances=[
                    Instance(
                        skeleton=skeleton,
                        track=Track(spawned_on=1, name="mouse-a"),
                        init_points={"nose": Point(1.0, 2.0), "tail": Point(3.0, 4.0)},
                    ),
                    Instance(
                        skeleton=skeleton,
                        track=Track(spawned_on=2, name="mouse-b"),
                        init_points={"nose": Point(5.0, 6.0), "tail": Point(7.0, 8.0)},
                    ),
                ],
            )
        ],
    )

    with pytest.raises(ValueError, match="multiple pose streams"):
        labels_to_primitives_session(labels)

    session = labels_to_primitives_session(labels, track="mouse-b")

    np.testing.assert_allclose(session.coords[0, :, :2], [[5.0, 6.0], [7.0, 8.0]])
    assert session.extras["xpkg"]["track_id"] == 2


def test_project_to_primitives_session_uses_service_load_labels(tmp_path: Path) -> None:
    video = _video(frames=1)
    skeleton = _skeleton()
    labels = _labels(
        video,
        skeleton,
        [
            LabeledFrame(
                video=video,
                frame_idx=0,
                instances=[
                    Instance(
                        skeleton=skeleton,
                        init_points={"nose": Point(1.0, 2.0), "tail": Point(3.0, 4.0)},
                    )
                ],
            )
        ],
    )

    class _Project:
        project_root = tmp_path / "Project"
        calls = 0

        def load_labels(self) -> Labels:
            self.calls += 1
            return labels

    project = _Project()

    session = project_to_primitives_session(project)

    assert project.calls == 1
    assert session.root == project.project_root.resolve()
    np.testing.assert_allclose(session.coords[0, :, :2], [[1.0, 2.0], [3.0, 4.0]])
