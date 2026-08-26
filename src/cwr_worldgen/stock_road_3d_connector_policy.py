# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep stock-road connectors closed when road pieces are pitched on terrain.

A rigid P3D keeps its three-dimensional connector distance when pitched. Fitting
that model to the same horizontal chord length as a flat road therefore creates
a small plan-view gap: the horizontal projection shrinks by ``cos(pitch)``. On
gentle grades this is only centimetres, but that is enough for terrain to show
through the seam.

This policy makes known rigid stock/generated-gravel road families solve each
connector chord in 3D using the already-graded terrain. Unknown custom roads keep
the generic fitter's configured planar-length semantics; measured CWA geometry
must not silently leak into arbitrary third-party assets.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import playability as _p
from . import road_quality_policy as _quality
from . import stock_road_curve_policy as _curve
from . import stock_road_geometry_policy as _geometry
from . import stock_road_model_geometry as _model_geometry
from . import stock_road_transform_policy as _transform

_ORIGINAL_CHAIN = None
_ORIGINAL_ROAD_OBJECT_ON_SLOPE = None
_ORIGINAL_MODEL_AXIS = None
_ORIGINAL_CHAIN_IS_SEAM_SAFE = None
_INSTALLED = False


class _TerrainMeasure:
    """Proxy a polyline while making chord lengths mean 3D connector lengths."""

    def __init__(self, measure, context):
        self._measure = measure
        self._context = context

    def __getattr__(self, name):
        return getattr(self._measure, name)

    def _height(self, x: float, z: float) -> float:
        spec = self._context.spec
        return _p._sample_elevation(
            self._context.elevations,
            spec.cells,
            spec.cell_size,
            x,
            z,
        )

    def chord_endpoint(self, start_distance, chord_length, maximum_distance):
        if chord_length <= 0.0:
            return None
        start_x, start_z, _ = self._measure.point(start_distance)
        start_height = self._height(start_x, start_z)
        probe_distance = min(
            maximum_distance,
            start_distance + max(0.05, float(chord_length)),
        )
        probe_x, probe_z, _ = self._measure.point(probe_distance)
        probe_horizontal = math.dist((start_x, start_z), (probe_x, probe_z))
        probe_height = self._height(probe_x, probe_z)
        grade = (
            (probe_height - start_height) / probe_horizontal
            if probe_horizontal > 1.0e-6
            else 0.0
        )
        horizontal = float(chord_length) / math.sqrt(1.0 + grade * grade)
        horizontal = max(0.02, min(float(chord_length), horizontal))

        endpoint = None
        for _ in range(8):
            endpoint = self._measure.chord_endpoint(
                start_distance, horizontal, maximum_distance
            )
            if endpoint is None:
                horizontal *= 0.97
                if horizontal <= 0.02:
                    return None
                continue
            _distance, end_x, end_z, _heading = endpoint
            delta_height = self._height(end_x, end_z) - start_height
            if abs(delta_height) >= chord_length * 0.98:
                return self._measure.chord_endpoint(
                    start_distance, chord_length, maximum_distance
                )
            desired_horizontal = math.sqrt(
                max(0.0, chord_length * chord_length - delta_height * delta_height)
            )
            if desired_horizontal <= 0.02:
                return endpoint
            if abs(desired_horizontal - horizontal) <= 1.0e-5:
                return endpoint
            horizontal = desired_horizontal
        return endpoint


def _three_dimensional_length(context, start, end) -> float:
    spec = context.spec
    start_height = _p._sample_elevation(
        context.elevations, spec.cells, spec.cell_size, start[0], start[1]
    )
    end_height = _p._sample_elevation(
        context.elevations, spec.cells, spec.cell_size, end[0], end[1]
    )
    horizontal = math.dist(start, end)
    return math.hypot(horizontal, end_height - start_height)


def _chain_is_seam_safe(measure, fitted) -> bool:
    context = _quality._CONTEXT.get()
    if context is None:
        assert _ORIGINAL_CHAIN_IS_SEAM_SAFE is not None
        return _ORIGINAL_CHAIN_IS_SEAM_SAFE(measure, fitted)

    previous_end = None
    for piece, start, end in fitted:
        if previous_end is not None and math.dist(previous_end, start) > 1.0e-4:
            return False
        previous_end = end
        if _curve._curve_match(piece.model_path) is None:
            continue
        if not math.isclose(
            _three_dimensional_length(context, start, end),
            float(piece.length_metres),
            rel_tol=0.0,
            abs_tol=2.0e-3,
        ):
            return False
        if (
            _geometry._curve_turn_error_degrees(measure.points, start, end)
            > _geometry._MAXIMUM_TANGENT_TURN_ERROR_DEGREES
        ):
            return False
    return True


def _uses_measured_rigid_connectors(pieces) -> bool:
    """Limit 3D chord semantics to road families whose connectors we know."""

    return any(
        _model_geometry.stock_straight_match(str(piece.model_path)) is not None
        or _p.is_generated_gravel_road_model(str(piece.model_path))
        for piece in pieces
    )


def _stock_piece_chain(measure, pieces, **kwargs):
    if _ORIGINAL_CHAIN is None:
        raise RuntimeError("3D stock-road connector policy is not installed")
    context = _quality._CONTEXT.get()
    if context is None or not _uses_measured_rigid_connectors(pieces):
        return _ORIGINAL_CHAIN(measure, pieces, **kwargs)
    return _ORIGINAL_CHAIN(_TerrainMeasure(measure, context), pieces, **kwargs)


def _curve_world_point(local, origin, heading_degrees, pitch_degrees):
    x, z = local
    heading = math.radians(float(heading_degrees))
    pitch = math.radians(float(pitch_degrees))
    cosine_heading = math.cos(heading)
    sine_heading = math.sin(heading)
    cosine_pitch = math.cos(pitch)
    sine_pitch = math.sin(pitch)
    return (
        origin[0] + x * cosine_heading + z * sine_heading * cosine_pitch,
        origin[1] + z * sine_pitch,
        origin[2] - x * sine_heading + z * cosine_heading * cosine_pitch,
    )


def _solve_curve_transform(local_begin, local_end, world_begin, world_end):
    local_dx = local_end[0] - local_begin[0]
    local_dz = local_end[1] - local_begin[1]
    world_dx = world_end[0] - world_begin[0]
    world_dy = world_end[1] - world_begin[1]
    world_dz = world_end[2] - world_begin[2]

    local_length = math.hypot(local_dx, local_dz)
    world_length = math.sqrt(world_dx * world_dx + world_dy * world_dy + world_dz * world_dz)
    if local_length <= 1.0e-9 or world_length <= 1.0e-9:
        return None
    if not math.isclose(local_length, world_length, rel_tol=0.0, abs_tol=3.0e-3):
        return None
    if abs(local_dz) <= 1.0e-6:
        return None

    sine_pitch = world_dy / local_dz
    if not -0.999 <= sine_pitch <= 0.999:
        return None
    pitch = math.asin(sine_pitch)
    cosine_pitch = math.cos(pitch)

    local_horizontal_heading = math.atan2(local_dx, local_dz * cosine_pitch)
    world_horizontal_heading = math.atan2(world_dx, world_dz)
    heading = world_horizontal_heading - local_horizontal_heading

    cosine_heading = math.cos(heading)
    sine_heading = math.sin(heading)
    x, z = local_begin
    offset = (
        x * cosine_heading + z * sine_heading * cosine_pitch,
        z * sine_pitch,
        -x * sine_heading + z * cosine_heading * cosine_pitch,
    )
    origin = (
        world_begin[0] - offset[0],
        world_begin[1] - offset[1],
        world_begin[2] - offset[2],
    )
    return origin, math.degrees(heading) % 360.0, math.degrees(pitch)


def _road_object_on_slope(*args, **kwargs):
    if _ORIGINAL_ROAD_OBJECT_ON_SLOPE is None:
        raise RuntimeError("3D stock-road connector policy is not installed")
    obj = _ORIGINAL_ROAD_OBJECT_ON_SLOPE(*args, **kwargs)
    model_path = str(args[1] if len(args) > 1 else kwargs.get("model_path", ""))
    geometry = _model_geometry.stock_curve_connectors(model_path)
    if geometry is None:
        return obj

    start = tuple(args[2] if len(args) > 2 else kwargs["start"])
    end = tuple(args[3] if len(args) > 3 else kwargs["end"])
    elevations = args[4] if len(args) > 4 else kwargs["elevations"]
    spec = args[5] if len(args) > 5 else kwargs["spec"]
    vertical_offset = float(kwargs.get("vertical_offset", 0.0))

    reverse = _curve._curve_object_key(obj) in _transform._REVERSED_FINAL_KEYS
    local_begin = geometry.end if reverse else geometry.begin
    local_end = geometry.begin if reverse else geometry.end
    start_height = _p._sample_elevation(
        elevations, spec.cells, spec.cell_size, start[0], start[1]
    ) + vertical_offset
    end_height = _p._sample_elevation(
        elevations, spec.cells, spec.cell_size, end[0], end[1]
    ) + vertical_offset
    solved = _solve_curve_transform(
        local_begin,
        local_end,
        (start[0], start_height, start[1]),
        (end[0], end_height, end[1]),
    )
    if solved is None:
        return obj
    origin, heading, pitch = solved
    fixed = replace(
        obj,
        x=origin[0],
        y=origin[1],
        z=origin[2],
        heading_degrees=heading,
        pitch_degrees=pitch,
    )
    if reverse:
        _transform._REVERSED_FINAL_KEYS.add(_curve._curve_object_key(fixed))
    return fixed


def _model_axis(obj, length: float):
    if _ORIGINAL_MODEL_AXIS is None:
        raise RuntimeError("3D stock-road connector policy is not installed")
    geometry = _model_geometry.stock_curve_connectors(obj.model_path)
    if geometry is None:
        return _ORIGINAL_MODEL_AXIS(obj, length)

    origin = (float(obj.x), float(obj.y), float(obj.z))
    begin = _curve_world_point(
        geometry.begin, origin, obj.heading_degrees, obj.pitch_degrees
    )
    end = _curve_world_point(
        geometry.end, origin, obj.heading_degrees, obj.pitch_degrees
    )
    begin_xz = (begin[0], begin[2])
    end_xz = (end[0], end[2])
    if _curve._curve_object_key(obj) in _transform._REVERSED_FINAL_KEYS:
        return end_xz, begin_xz
    return begin_xz, end_xz


def install_stock_road_3d_connector_policy() -> None:
    global _ORIGINAL_CHAIN, _ORIGINAL_ROAD_OBJECT_ON_SLOPE, _ORIGINAL_MODEL_AXIS
    global _ORIGINAL_CHAIN_IS_SEAM_SAFE, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_CHAIN = _p._stock_piece_chain
    _ORIGINAL_ROAD_OBJECT_ON_SLOPE = _p._road_object_on_slope
    _ORIGINAL_MODEL_AXIS = _p._model_axis
    _ORIGINAL_CHAIN_IS_SEAM_SAFE = _geometry._chain_is_seam_safe
    _geometry._chain_is_seam_safe = _chain_is_seam_safe
    _p._stock_piece_chain = _stock_piece_chain
    _p._road_object_on_slope = _road_object_on_slope
    _p._model_axis = _model_axis
    _INSTALLED = True
