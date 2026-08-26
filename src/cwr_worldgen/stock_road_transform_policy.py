# SPDX-License-Identifier: GPL-3.0-or-later
"""Place native stock curves from their actual model-space connectors.

The curve selector works in centerline connector space. Stock ODOLs, however,
are not centered on that connector chord: the chord is locally headed five
degrees and its midpoint is offset from the object origin. Correct the final WRP
transform so the Memory-LOD begin/end connectors land exactly on the fitted
centerline endpoints.
"""
from __future__ import annotations

from dataclasses import replace

from . import playability as _p
from . import stock_road_curve_policy as _curve
from . import stock_road_model_geometry as _geometry

_ORIGINAL_ROAD_OBJECT_ON_SLOPE = None
_ORIGINAL_MODEL_AXIS = None
_REVERSED_FINAL_KEYS: set[tuple[int, str, float, float, float]] = set()
_INSTALLED = False


def _road_object_on_slope(*args, **kwargs):
    if _ORIGINAL_ROAD_OBJECT_ON_SLOPE is None:
        raise RuntimeError("stock road transform policy is not installed")
    model_path = str(args[1] if len(args) > 1 else kwargs.get("model_path", ""))
    geometry = _geometry.stock_curve_connectors(model_path)
    obj = _ORIGINAL_ROAD_OBJECT_ON_SLOPE(*args, **kwargs)
    if geometry is None:
        return obj

    start = args[2] if len(args) > 2 else kwargs["start"]
    end = args[3] if len(args) > 3 else kwargs["end"]

    # The earlier curve policy records handedness by reversing traversal through
    # the same right-hand P3D. Read that flag from the object it just emitted,
    # then solve a fresh connector transform instead of merely rotating it 180°.
    reverse = _curve._curve_object_key(obj) in _curve._REVERSED_CURVE_KEYS
    local_begin = geometry.end if reverse else geometry.begin
    local_end = geometry.begin if reverse else geometry.end
    origin, heading = _geometry.solve_planar_connector_transform(
        tuple(start), tuple(end), local_begin, local_end
    )
    fixed = replace(
        obj,
        x=origin[0],
        z=origin[1],
        heading_degrees=heading,
    )
    if reverse:
        _REVERSED_FINAL_KEYS.add(_curve._curve_object_key(fixed))
    return fixed


def _model_axis(obj, length: float):
    if _ORIGINAL_MODEL_AXIS is None:
        raise RuntimeError("stock road transform policy is not installed")
    geometry = _geometry.stock_curve_connectors(obj.model_path)
    if geometry is None:
        return _ORIGINAL_MODEL_AXIS(obj, length)

    origin = (float(obj.x), float(obj.z))
    begin = _geometry.transform_local(geometry.begin, origin, obj.heading_degrees)
    end = _geometry.transform_local(geometry.end, origin, obj.heading_degrees)
    if _curve._curve_object_key(obj) in _REVERSED_FINAL_KEYS:
        return end, begin
    return begin, end


def install_stock_road_transform_policy() -> None:
    global _ORIGINAL_ROAD_OBJECT_ON_SLOPE, _ORIGINAL_MODEL_AXIS, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_ROAD_OBJECT_ON_SLOPE = _p._road_object_on_slope
    _ORIGINAL_MODEL_AXIS = _p._model_axis
    _p._road_object_on_slope = _road_object_on_slope
    _p._model_axis = _model_axis
    _INSTALLED = True
