"""Serialization helpers for annotation primitives."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import cattrs as cattr
import numpy as np

from xpkg.core.annotations.frames import InstancesList
from xpkg.core.annotations.instances import Instance, PredictedInstance
from xpkg.core.annotations.points import Point, PointArray, PredictedPoint, PredictedPointArray


def make_instance_cattr() -> cattr.Converter:
    """Build a cattrs converter for the point/instance helpers."""
    converter = cattr.Converter()
    converter.register_unstructure_hook(np.bool_, bool)
    converter.register_unstructure_hook(PointArray, lambda x: None)
    converter.register_unstructure_hook(PredictedPointArray, lambda x: None)

    def unstructure_point(pt: Point) -> dict[str, Any]:
        """Unstructure Point to a named dict (not positional tuple)."""
        return {
            "x": float(pt["x"]),
            "y": float(pt["y"]),
            "visible": bool(pt["visible"]),
            "complete": bool(pt["complete"]),
            "flags": int(pt["flags"]),
        }

    def unstructure_predicted_point(pt: PredictedPoint) -> dict[str, Any]:
        """Unstructure PredictedPoint to a named dict with score."""
        return {
            "x": float(pt["x"]),
            "y": float(pt["y"]),
            "visible": bool(pt["visible"]),
            "complete": bool(pt["complete"]),
            "score": float(pt["score"]),
            "flags": int(pt["flags"]),
        }

    converter.register_unstructure_hook(Point, unstructure_point)
    converter.register_unstructure_hook(PredictedPoint, unstructure_predicted_point)
    # Also handle np.record/np.void that may come from array indexing
    converter.register_unstructure_hook(
        np.record,
        lambda pt: (
            unstructure_predicted_point(pt) if "score" in pt.dtype.names else unstructure_point(pt)
        ),
    )
    converter.register_unstructure_hook(
        np.void,
        lambda pt: (
            unstructure_predicted_point(pt) if "score" in pt.dtype.names else unstructure_point(pt)
        ),
    )

    def unstructure_instance(x: Instance):
        d = {}
        for fld in fields(x):
            if fld.name not in ["_points", "_keypoints", "frame", "init_points"] and fld.init:
                d[fld.name] = converter.unstructure(x.__dict__[fld.name])
        # Use keypoint names (strings) as keys for serialization
        d["_points"] = {kp.name: converter.unstructure(pt) for kp, pt in x.keypoints_points}
        return d

    converter.register_unstructure_hook(Instance, unstructure_instance)
    converter.register_unstructure_hook(PredictedInstance, unstructure_instance)
    converter.register_unstructure_hook(
        InstancesList, lambda x: [converter.unstructure(inst) for inst in x]
    )

    def structure_points(x, type):
        if "score" in x.keys():
            return cattr.structure(x, PredictedPoint)
        else:
            return cattr.structure(x, Point)

    converter.register_structure_hook(Point | PredictedPoint, structure_points)

    def structure_instances_list(x, type):
        inst_list = []
        for inst_data in x:
            inst = structure_instance(inst_data, type)
            inst_list.append(inst)
        return inst_list

    def structure_instance(inst_data, type):
        from_predicted = None

        if "score" in inst_data.keys():
            base_type = PredictedInstance
        else:
            base_type = Instance

        # Normalize serialized point payloads to Instance's init_points.
        if "_points" in inst_data:
            inst_data["init_points"] = inst_data.pop("_points")

        if "from_predicted" in inst_data and inst_data["from_predicted"] is not None:
            from_predicted = converter.structure(inst_data["from_predicted"], PredictedInstance)
            inst_data["from_predicted"] = None

        inst = converter.structure(inst_data, base_type)
        if from_predicted is not None:
            inst.from_predicted = from_predicted
        return inst

    converter.register_structure_hook(
        list[Instance] | list[PredictedInstance], structure_instances_list
    )
    converter.register_structure_hook(InstancesList, structure_instances_list)

    def structure_point_array(x, _type):
        # Keys are now strings (keypoint names), not Keypoint objects
        if x:
            point1 = x[next(iter(x.keys()))]
            if "score" in point1.keys():
                return converter.structure(x, dict[str, PredictedPoint])
            else:
                return converter.structure(x, dict[str, Point])
        else:
            return {}

    converter.register_structure_hook(PointArray, structure_point_array)
    converter.register_structure_hook(PredictedPointArray, structure_point_array)

    return converter


__all__ = ["make_instance_cattr"]
