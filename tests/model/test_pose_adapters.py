from __future__ import annotations

from xpkg.model import (
    ViconMarkerModel,
    skeleton_to_vicon_marker_model,
    vicon_marker_model_to_skeleton,
)


def test_vicon_marker_model_to_skeleton_preserves_names_edges_and_metadata() -> None:
    model = ViconMarkerModel(
        name="mouse_model",
        display_name="Mouse Model",
        marker_names=("L_Hip", "L_Knee", "TailBase"),
        edges=(("L_Hip", "L_Knee"), ("L_Hip", "TailBase")),
        source="vsk",
    )

    skeleton = vicon_marker_model_to_skeleton(model)

    assert skeleton.name == "mouse_model"
    assert skeleton.description == "Mouse Model"
    assert skeleton.keypoint_names == ["L_Hip", "L_Knee", "TailBase"]
    assert skeleton.links_ids == [(0, 1), (0, 2)]
    assert skeleton.metadata == {"source": "vsk"}


def test_skeleton_to_vicon_marker_model_round_trips_adapter_shape() -> None:
    model = ViconMarkerModel(
        name="mouse_model",
        display_name="Mouse Model",
        marker_names=("L_Hip", "L_Knee"),
        edges=(("L_Hip", "L_Knee"),),
        source="vsk",
    )

    skeleton = vicon_marker_model_to_skeleton(model)
    converted = skeleton_to_vicon_marker_model(skeleton, source="skeleton")

    assert converted.name == "mouse_model"
    assert converted.display_name == "Mouse Model"
    assert converted.marker_names == ("L_Hip", "L_Knee")
    assert converted.edges == (("L_Hip", "L_Knee"),)
    assert converted.source == "skeleton"

