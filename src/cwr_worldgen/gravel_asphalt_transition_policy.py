# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep paved main roads visually continuous through mixed T intersections.

Generated gravel and stock ``ces`` side roads may both meet a paved ``sil``,
``asf`` or ``kos`` main road.  The Resistance mixed T meshes are useful when the
source geometry is almost exactly their fixed connector template, but a few
degrees of skew produces a conspicuous texture/edge mismatch in game even though
the logical centrelines still meet.

For generated gravel, keep the existing rule: place one normal 6.25 m stock
paved straight over the immediate node and let the gravel approach continue
underneath it.  For stock ``ces`` side roads, retain the purpose-built mixed T
only when its measured connector fit is very close.  Otherwise use the same
stock paved straight overlay instead of forcing a visibly crooked mixed T mesh.

The overlay is only a central surface cover.  Paved and unpaved approaches are
allowed to continue underneath it, so no generated intersection geometry is
needed and the visible main road stays a normal stock road surface.
"""
from __future__ import annotations

import math

from . import gravel_junction_policy as _gravel_junction
from . import playability as _p
from . import road_quality_policy as _quality
from . import stock_road_junction_policy as _junction
from . import stock_road_measured_junction_policy as _measured
from . import stock_road_skew_policy as _skew
from . import stock_road_model_geometry as _model_geometry
from .stock_road_model_geometry import STOCK_JUNCTION_CONNECTOR_RADIUS_METRES

MAXIMUM_LAYERED_MAIN_HEADING_ERROR_DEGREES = 30.0
# 1.5 degrees at the measured 6.25 m connector radius is about 0.16 m of
# lateral displacement.  Larger errors are much easier to see at the painted
# edge than the small source-line deviation introduced by the straight overlay.
MAXIMUM_STOCK_MIXED_NATIVE_HEADING_ERROR_DEGREES = 1.5
_UNPAVED_END_TOLERANCE_METRES = 0.25

_ORIGINAL_NATIVE_T = None
_ORIGINAL_QUALITY_WINDOW = None
_ORIGINAL_RELAXATION_ELIGIBILITY = None
_INSTALLED = False


def _is_generated_gravel_incident(incident) -> bool:
    return _p.is_generated_gravel_road_model(incident.model_path)


def _is_unpaved_incident(incident) -> bool:
    return _is_generated_gravel_incident(incident) or incident.family == "ces"


def _layered_mixed_t_components(incidents):
    """Return paved-main indices and the one unpaved branch for a mixed T."""

    if len(incidents) != 3:
        return None
    unpaved = [
        index
        for index, incident in enumerate(incidents)
        if _is_unpaved_incident(incident)
    ]
    if len(unpaved) != 1:
        return None
    paved = [index for index in range(3) if index != unpaved[0]]
    first_family = incidents[paved[0]].family
    second_family = incidents[paved[1]].family
    if first_family != second_family or first_family not in {"sil", "asf", "kos"}:
        return None
    return (
        first_family,
        paved[0],
        paved[1],
        unpaved[0],
        _is_generated_gravel_incident(incidents[unpaved[0]]),
    )


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
        raise RuntimeError("mixed unpaved/paved transition policy is not installed")

    original = _ORIGINAL_NATIVE_T(incidents)
    layered = _layered_mixed_t_components(incidents)
    if layered is None:
        return original

    family, first, second, _unpaved, generated_gravel = layered

    # A correctly aligned stock mixed T is still the best-looking transition.
    # Generated gravel has no matching stock texture, so it always uses the
    # paved-main overlay.  Stock ces keeps the mixed T only while its measured
    # connector error stays below the visibly harmless threshold above.
    if (
        not generated_gravel
        and original is not None
        and float(original.maximum_heading_error_degrees)
        <= MAXIMUM_STOCK_MIXED_NATIVE_HEADING_ERROR_DEGREES
    ):
        return original

    maximum_error, rotation = _layered_main_rotation(incidents, first, second)
    if maximum_error > MAXIMUM_LAYERED_MAIN_HEADING_ERROR_DEGREES:
        return original

    return _junction._NativeJunction(
        rf"o\road\{family}6.p3d",
        rotation,
        maximum_error,
        family,
    )


def _relaxation_eligible(incidents) -> bool:
    """Keep connector snapping for real T meshes, never for a straight overlay."""

    if _ORIGINAL_RELAXATION_ELIGIBILITY is None:
        raise RuntimeError("mixed unpaved/paved transition policy is not installed")
    if not _ORIGINAL_RELAXATION_ELIGIBILITY(incidents):
        return False
    native = _native_t_junction(incidents)
    if native is None:
        return True
    return _model_geometry.stock_straight_match(native.model_path) is None


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


def _pieces_are_unpaved_branch(pieces) -> bool:
    if not pieces:
        return False
    for piece in pieces:
        if _p.is_generated_gravel_road_model(piece.model_path):
            continue
        if _junction._family(piece.model_path) == "ces":
            continue
        return False
    return True


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
        raise RuntimeError("mixed unpaved/paved transition policy is not installed")
    start_distance, preferred_end, minimum_end, maximum_end = _ORIGINAL_QUALITY_WINDOW(
        measure,
        pieces,
        start_distance,
        preferred_end,
        minimum_end,
        maximum_end,
        context,
    )
    if not _pieces_are_unpaved_branch(pieces):
        return start_distance, preferred_end, minimum_end, maximum_end

    start_junction = context.junctions.get(_p._road_node_key(measure.points[0]))
    end_junction = context.junctions.get(_p._road_node_key(measure.points[-1]))
    shortest = min(float(piece.length_metres) for piece in pieces)

    # The straight stock overlay owns only the visible top surface.  Let the
    # unpaved branch physically reach the node underneath it instead of trimming
    # to the 6.25 m native-junction connector radius and exposing a gap.
    if _is_layered_stock_junction(start_junction):
        start_distance = 0.0
    if _is_layered_stock_junction(end_junction):
        preferred_end = max(start_distance, measure.total)
        minimum_end = max(
            start_distance,
            measure.total - _UNPAVED_END_TOLERANCE_METRES,
        )
        maximum_end = max(maximum_end, measure.total + shortest * 0.5)
    return start_distance, preferred_end, minimum_end, maximum_end


def install_gravel_asphalt_transition_policy() -> None:
    global _ORIGINAL_NATIVE_T, _ORIGINAL_QUALITY_WINDOW
    global _ORIGINAL_RELAXATION_ELIGIBILITY, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_NATIVE_T = _measured._native_t_junction
    _ORIGINAL_QUALITY_WINDOW = _quality._quality_window
    _ORIGINAL_RELAXATION_ELIGIBILITY = _skew._eligible_relaxed_mixed_t
    _measured._native_t_junction = _native_t_junction
    _junction._native_t_junction = _native_t_junction
    _quality._quality_window = _quality_window
    _skew._eligible_relaxed_mixed_t = _relaxation_eligible
    _INSTALLED = True
