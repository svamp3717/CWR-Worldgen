# SPDX-License-Identifier: GPL-3.0-or-later
"""Fit stock bends with connector-locked curves before faceted straights.

The old policy first asked the ordinary greedy stock fitter to build a road,
then tried to promote a bad short-straight result into curves. That made short
straight facets the architecture and native curves the repair. Reference WRPs do
the opposite: they choose stock-compatible curvature first and only fall back to
facets when no safe stock curve chain can represent the source.

For the verified Resistance ``sil/asf/kos/ces`` families this wrapper therefore
attempts the exact stock-curve beam *before* calling the inherited
straight-oriented fitter. The candidate may smooth hard OSM vertices inside the
bounded stock-road corridor, but every sample is also checked against the
source-backed obstacle index when that context is active. Generated gravel and
custom road families remain untouched.
"""
from __future__ import annotations

import math

from . import playability as _p
from . import stock_road_model_geometry as _geometry
from . import stock_road_relaxation_policy as _relax
from . import stock_road_sharp_turn_policy as _sharp

_MAXIMUM_PROMOTION_RUN_METRES = 180.0
_MINIMUM_TOTAL_TURN_DEGREES = 15.0
_MAXIMUM_TOTAL_TURN_DEGREES = 70.0
_MINIMUM_PROMOTED_CURVES = 1
_MINIMUM_ENDPOINT_COVER_METRES = 0.40
_MAXIMUM_UNCOVERED_EXIT_ERROR_DEGREES = 1.50
_END_PROGRESS_TOLERANCE_METRES = 0.20
_MINIMUM_SIGNIFICANT_VERTEX_TURN_DEGREES = 0.45
_MAXIMUM_LOCAL_VERTEX_TURN_DEGREES = 35.0
_MAXIMUM_REVERSE_NOISE_DEGREES = 1.50

_ORIGINAL_CHAIN = None
_INSTALLED = False


def _dominant_bend(points) -> tuple[int, float] | None:
    """Return one coherent bend sign and accumulated turn for a stock run."""

    sign = 0
    count = 0
    total = 0.0
    for previous, point, following in zip(points, points[1:], points[2:]):
        turn = _sharp._signed_turn(previous, point, following)
        magnitude = abs(turn)
        if magnitude < _MINIMUM_SIGNIFICANT_VERTEX_TURN_DEGREES:
            continue
        if magnitude > _MAXIMUM_LOCAL_VERTEX_TURN_DEGREES:
            return None
        current_sign = 1 if turn > 0.0 else -1
        if sign and current_sign != sign:
            if magnitude <= _MAXIMUM_REVERSE_NOISE_DEGREES:
                continue
            return None
        if not sign:
            sign = current_sign
        total += turn
        count += 1

    magnitude = abs(total)
    if (
        sign == 0
        or count < 1
        or magnitude < _MINIMUM_TOTAL_TURN_DEGREES
        or magnitude > _MAXIMUM_TOTAL_TURN_DEGREES
    ):
        return None
    return sign, magnitude


def _piece_tangents(item, turn_sign: int) -> tuple[float, float]:
    piece, start, end = item
    chord = _sharp._heading(start, end)
    if _geometry.stock_curve_match(str(piece.model_path)) is None:
        return chord, chord
    half_turn = _geometry.STOCK_CURVE_ANGLE_DEGREES * 0.5
    if turn_sign > 0:
        return (chord - half_turn) % 360.0, (chord + half_turn) % 360.0
    return (chord + half_turn) % 360.0, (chord - half_turn) % 360.0


def _maximum_internal_tangent_error(fitted, turn_sign: int) -> float:
    maximum = 0.0
    for previous, current in zip(fitted, fitted[1:]):
        previous_end = _piece_tangents(previous, turn_sign)[1]
        current_start = _piece_tangents(current, turn_sign)[0]
        maximum = max(maximum, _p._heading_difference(previous_end, current_start))
    return maximum


def _fallback_chain(
    measure,
    pieces,
    *,
    start_distance,
    preferred_end_distance,
    minimum_end_distance,
    maximum_end_distance,
):
    return _ORIGINAL_CHAIN(
        measure,
        pieces,
        start_distance=start_distance,
        preferred_end_distance=preferred_end_distance,
        minimum_end_distance=minimum_end_distance,
        maximum_end_distance=maximum_end_distance,
    )


def _path_is_obstacle_safe(path) -> bool:
    """Check the smoothed stock alignment against source-backed obstacles."""

    context = _relax._CONTEXT.get()
    if context is None:
        return True
    return all(
        _relax._shortcut_clear(context.obstacles, first, second)
        for first, second in zip(path, path[1:])
    )


def _curve_promotion_chain(
    measure,
    pieces,
    *,
    start_distance,
    preferred_end_distance,
    minimum_end_distance,
    maximum_end_distance,
):
    """Return an exact native-curve chain first, then fall back to old fitting."""

    if _ORIGINAL_CHAIN is None:
        raise RuntimeError("stock road curve-usage policy is not installed")

    fallback_args = dict(
        start_distance=start_distance,
        preferred_end_distance=preferred_end_distance,
        minimum_end_distance=minimum_end_distance,
        maximum_end_distance=maximum_end_distance,
    )
    if _sharp._curveable_family(pieces) is None:
        return _fallback_chain(measure, pieces, **fallback_args)
    if measure.total > _MAXIMUM_PROMOTION_RUN_METRES:
        return _fallback_chain(measure, pieces, **fallback_args)

    bend = _dominant_bend(measure.points)
    if bend is None:
        return _fallback_chain(measure, pieces, **fallback_args)
    turn_sign, _total_turn = bend

    start = max(0.0, min(float(measure.total), float(start_distance)))
    end = max(start, min(float(measure.total), float(preferred_end_distance)))
    if end <= start + 1.0:
        return _fallback_chain(measure, pieces, **fallback_args)

    source_points, entry_heading, source_exit_heading = _sharp._measure_slice(
        measure, start, end
    )
    stock_exit_heading = _sharp._quantised_stock_exit_heading(
        entry_heading,
        source_exit_heading,
        turn_sign,
    )

    # At an exposed feature end, do not trade an interior miter for a visibly
    # rotated final connector. Junction cover may absorb the normal ten-degree
    # stock quantisation at trimmed boundaries.
    end_cover = float(measure.total) - float(preferred_end_distance)
    if (
        end_cover < _MINIMUM_ENDPOINT_COVER_METRES
        and _p._heading_difference(stock_exit_heading, source_exit_heading)
        > _MAXIMUM_UNCOVERED_EXIT_ERROR_DEGREES
    ):
        return _fallback_chain(measure, pieces, **fallback_args)

    locked_path = _sharp._beam_stock_path(
        source_points,
        turn_sign,
        entry_heading,
        stock_exit_heading,
        pieces,
    )
    if locked_path is None or not _path_is_obstacle_safe(locked_path):
        return _fallback_chain(measure, pieces, **fallback_args)

    exact = _sharp._recover_exact_actions(locked_path, pieces, turn_sign)
    if exact is None or _sharp._curve_count(exact) < _MINIMUM_PROMOTED_CURVES:
        return _fallback_chain(measure, pieces, **fallback_args)

    # Connector continuity, not comparison with an already-bad baseline, is the
    # acceptance criterion. This is the architectural inversion the reference
    # WRPs pointed to.
    if _maximum_internal_tangent_error(exact, turn_sign) > 1.0e-4:
        return _fallback_chain(measure, pieces, **fallback_args)

    end_projection = _sharp._nearest_forward(
        measure,
        locked_path[-1],
        start,
        float(maximum_end_distance),
    )
    if end_projection is None:
        return _fallback_chain(measure, pieces, **fallback_args)
    if end_projection[0] > _sharp._MAXIMUM_LOCKED_CORRIDOR_METRES + 1.0e-9:
        return _fallback_chain(measure, pieces, **fallback_args)
    if end_projection[1] < float(minimum_end_distance) - _END_PROGRESS_TOLERANCE_METRES:
        return _fallback_chain(measure, pieces, **fallback_args)

    start_point = measure.point(start)[:2]
    if math.dist(locked_path[0], start_point) > 1.0e-6:
        return _fallback_chain(measure, pieces, **fallback_args)
    return exact


def install_stock_road_curve_usage_policy() -> None:
    """Install exact curve-first stock fitting after the narrow bend policies."""

    global _ORIGINAL_CHAIN, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_CHAIN = _p._stock_piece_chain
    _p._stock_piece_chain = _curve_promotion_chain
    _INSTALLED = True
