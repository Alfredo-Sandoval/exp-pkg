"""Focused adapters between source-native marker models and skeletons."""

from __future__ import annotations

from xpkg.model.vicon import ViconMarkerModel
from xpkg.pose.naming import normalize_marker_name
from xpkg.pose.skeleton import Keypoint, Skeleton, infer_side


def vicon_marker_model_to_skeleton(model: ViconMarkerModel) -> Skeleton:
    """Adapt Vicon marker metadata to a Skeleton without changing Vicon storage."""
    keypoints = [
        Keypoint(id=index, name=name, side=infer_side(normalize_marker_name(name)))
        for index, name in enumerate(model.marker_names)
    ]
    name_to_id = {name: index for index, name in enumerate(model.marker_names)}
    links_ids = [(name_to_id[parent], name_to_id[child]) for parent, child in model.edges]
    return Skeleton(
        name=model.name,
        keypoints=keypoints,
        links_ids=links_ids,
        description=model.display_name,
        metadata={"source": model.source},
    )


def skeleton_to_vicon_marker_model(
    skeleton: Skeleton,
    *,
    source: str = "skeleton",
) -> ViconMarkerModel:
    """Adapt a Skeleton to a Vicon marker model without implying Vicon recording data."""
    marker_names = tuple(skeleton.keypoint_names)
    marker_edges = tuple(
        (marker_names[parent], marker_names[child])
        for parent, child in skeleton.links_ids
    )
    return ViconMarkerModel(
        name=skeleton.name,
        display_name=skeleton.description or skeleton.name,
        marker_names=marker_names,
        edges=marker_edges,
        source=source,
    )


__all__ = ["skeleton_to_vicon_marker_model", "vicon_marker_model_to_skeleton"]
