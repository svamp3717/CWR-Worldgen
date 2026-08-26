# SPDX-License-Identifier: GPL-3.0-or-later
"""Layer generated gravel underneath an uninterrupted paved road at mixed T nodes.

Generated gravel is classified as the stock ``ces`` family only as an internal
connector surrogate. It must never make the visible paved road turn into a dirt
junction. At a mixed paved/gravel T, keep the paved main road visually ordinary:
place a normal 12.5 m stock straight across the node and let the generated gravel
chain continue underneath it.

The straight overlay also provides a rectangular paved footprint over small
heading disagreement between the two paved approaches. That is deliberately more
forgiving than a stock T mesh with fixed connector cut-outs and avoids exposing
terrain wedges when the source through-road bends slightly at the junction.
"""
from __future__ import annotations

import math

from . import gravel_junction_policy as _gravel_junction
from . import playability as _p
from . import road_quality_policy as _quality
from . import stock_road_junction_policy as _junction
from . import stock_road_measured_junction_policy as _measured
from .stock_road_model_geometry import STOCK_JUNCTION_CONNECTOR_RADIUS_METRES

MAXIMUM_LAYERED_MAIN_HEADING_ERROR_DEGREES = 30.0
_GRAVEL_END_TOLERANCE_METRES = 0.25

_ORIGINAL_NATIVE_T = None
_ORIGINAL_QUALITY_WINDOW = None
_INSTALLED = False


def _layered_mixed_t_components(incidents):
    if len(incidents) != 3:
        return None
    gravel = [
        index
        for index, incident in enumerate(incidents)
        if _p.is_generated_gravel_road_model(incident.model_path)
    ]
    if len(gravel) != 1:
        return None
    paved = [index for index in range(3) if index != gravel[0]]
    first_family = incidents[paved[0]].family
    second_family = incidents[paved[1]].family
    if first_family != second_family or first_family not in {"sil", "asf", "kos"}:
        return None
    return first_family, paved[0], paved[1], gravel[0]


def _layered_main_rotation(incidents, first: int, second: int):
    first_heading = _junction._heading(incidents[first].direction)
    second_heading = _junction._heading(incidents[second].direction)
    fits = []
    for actual_zero, actual_180 in (
        (first_heading, second_heading),
        (second_heading, first_heading),
    ):
        rotation, maximum_error = _junction._best_rotation(
            ((0.0, actual_zero), (180.0, actual_180))
        )
        fits.append((maximum_error, rotation))
    return min(fits)


def _native_t_junction(incidents):
    if _ORIGINAL_NATIVE_T is None:
        raise RuntimeError("gravel/paved transition policy is not installed")

    layered = _layered_mixed_t_components(incidents)
    if layered is None:
        return _ORIGINAL_NATIVE_T(incidents)

    family, first, second, _gravel = layered
    maximum_error, rotation = _layered_main_rotation(incidents, first, second)
    if maximum_error > MAXIMUM_LAYERED_MAIN_HEADING_ERROR_DEGREES:
        return _ORIGINAL_NATIVE_T(incidents)

    return _junction._NativeJunction(
        rf"o\road\{family}12.p3d",
        rotation,
        maximum_error,
        family,
    )


def _is_layered_stock_junction(junction) -> bool:
    if junction is None or _gravel_junction._is_gravel_junction(junction):
        return False
    return (
        math.isclose(
            float(junction.half_length),
            STOCK_JUNCTION_CONNECTOR_RADIUS_METRES,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
        and math.isclose(
            float(junction.half_width),
            STOCK_JUNCTION_CONNECTOR_RADIUS_METRES,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
    )


def _pieces_are_generated_gravel(pieces) -> bool:
    return bool(pieces) and all(
        _p.is_generated_gravel_road_model(piece.model_path) for piece in pieces
    )


def _quality_window(
    measure,
    pieces,
    start_distance,
    preferred_end,
    minimum_end,
    maximum_end,
    context,
):
    if _ORIGINAL_QUALITY_WINDOW is None:
        raise RuntimeError("gravel/paved transition policy is not installed")
    start_distance, preferred_end, minimum_end, maximum_end = _ORIGINAL_QUALITY_WINDOW(
        measure,
        pieces,
        start_distance,
        preferred_end,
        minimum_end,
        maximum_end,
        context,
    )
    if not _pieces_are_generated_gravel(pieces):
        return start_distance, preferred_end, minimum_end, maximum_end

    start_junction = context.junctions.get(_p._road_node_key(measure.points[0]))
    end_junction = context.junctions.get(_p._road_node_key(measure.points[-1]))
    shortest = min(float(piece.length_metres) for piece in pieces)

    if _is_layered_stock_junction(start_junction):
        start_distance = 0.0
    if _is_layered_stock_junction(end_junction):
        preferred_end = max(start_distance, measure.total)
        minimum_end = max(
            start_distance,
            measure.total - _GRAVEL_END_TOLERANCE_METRES,
        )
        maximum_end = max(maximum_end, measure.total + shortest * 0.5)
    return start_distance, preferred_end, minimum_end, maximum_end


def install_gravel_asphalt_transition_policy() -> None:
    global _ORIGINAL_NATIVE_T, _ORIGINAL_QUALITY_WINDOW, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_NATIVE_T = _measured._native_t_junction
    _ORIGINAL_QUALITY_WINDOW = _quality._quality_window
    _measured._native_t_junction = _native_t_junction
    _junction._native_t_junction = _native_t_junction
    _quality._quality_window = _quality_window
    _INSTALLED = True
