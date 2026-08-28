# SPDX-License-Identifier: GPL-3.0-or-later
"""Promote more paved bends from short facets to exact stock curve chains.

The sharp-turn policies intentionally started with a very narrow production
case.  Real worlds still contain many junction-to-junction and feature-end runs
where a coherent 15-70 degree bend is rendered mostly from ``sil6``/``sil12``
rectangles even though a connector-locked sequence of stock ten-degree curves
fits the conditioned centreline.  Those rectangular mitres are the source of
triangular grass wedges on the outside of a turn.

This late road-only policy gives those runs a second chance.  It never moves
terrain or roadside objects, never applies to dirt/gravel, and never loosens the
physical curve connectors.  It simply accepts the existing sharp-turn beam's
exact stock-piece sequence on a broader class of paved runs when that sequence
uses more native curves, stays in the same 0.60 m source corridor, and keeps
internal tangents continuous.
"""
from __future__ import annotations

import math

from . import playability as _p
from . import stock_road_model_geometry as _geometry
from . import stock_road_sharp_exact_policy as _exact
from . import stock_road_sharp_turn_policy as _sharp

_MAXIMUM_PROMOTION_RUN_METRES = 180.0
_MINIMUM_BASELINE_SHORT_STRAIGHTS = 3
_MINIMUM_TOTAL_TURN_DEGREES = 15.0
_MAXIMUM_TOTAL_TURN_DEGREES = 70.0
_MINIMUM_PROMOTED_CURVES = 2
_MAXIMUM_EXTRA_PIECES = 2
_MINIMUM_ENDPOINT_COVER_METRES = 0.40
_MAXIMUM_UNCOVERED_EXIT_ERROR_DEGREES = 1.50
_END_PROGRESS_TOLERANCE_METRES = 0.20
_MINIMUM_SIGNIFICANT_VERTEX_TURN_DEGREES = 0.45
_MAXIMUM_LOCAL_VERTEX_TURN_DEGREES = 24.0
_MAXIMUM_REVERSE_NOISE_DEGREES = 1.50

_ORIGINAL_CHAIN = None
_INSTALLED = False


def _dominant_bend(points) -> tuple[int, float] | None:
    """Return one coherent bend sign and accumulated turn for a paved run."""

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
        or count < 2
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


def _curve_promotion_chain(
    measure,
    pieces,
    *,
    start_distance,
    preferred_end_distance,
    minimum_end_distance,
    maximum_end_distance,
):
    if _ORIGINAL_CHAIN is None:
        raise RuntimeError("stock road curve-usage policy is not installed")

    baseline = _ORIGINAL_CHAIN(
        measure,
        pieces,
        start_distance=start_distance,
        preferred_end_distance=preferred_end_distance,
        minimum_end_distance=minimum_end_distance,
        maximum_end_distance=maximum_end_distance,
    )
    if _sharp._paved_family(pieces) is None:
        return baseline
    if measure.total > _MAXIMUM_PROMOTION_RUN_METRES:
        return baseline
    if _exact._baseline_short_straights(baseline) < _MINIMUM_BASELINE_SHORT_STRAIGHTS:
        return baseline

    bend = _dominant_bend(measure.points)
    if bend is None:
        return baseline
    turn_sign, _total_turn = bend

    start = max(0.0, min(float(measure.total), float(start_distance)))
    end = max(start, min(float(measure.total), float(preferred_end_distance)))
    if end <= start + 1.0:
        return baseline

    source_points, entry_heading, source_exit_heading = _exact._measure_slice(measure, start, end)
    stock_exit_heading = _exact._quantised_stock_exit_heading(
        entry_heading,
        source_exit_heading,
        turn_sign,
    )

    # A junction cap can hide the small quantisation error at a run boundary.
    # Without such cover, accept only an almost exact source tangent so this
    # policy cannot trade an interior grass wedge for a new exposed end seam.
    end_cover = float(measure.total) - float(preferred_end_distance)
    if (
        end_cover < _MINIMUM_ENDPOINT_COVER_METRES
        and _p._heading_difference(stock_exit_heading, source_exit_heading)
        > _MAXIMUM_UNCOVERED_EXIT_ERROR_DEGREES
    ):
        return baseline

    locked_path = _sharp._beam_stock_path(
        source_points,
        turn_sign,
        entry_heading,
        stock_exit_heading,
        pieces,
    )
    if locked_path is None:
        return baseline
    exact = _exact._recover_exact_actions(locked_path, pieces, turn_sign)
    if exact is None:
        return baseline

    exact_curves = _exact._curve_count(exact)
    baseline_curves = _exact._curve_count(baseline)
    if exact_curves < _MINIMUM_PROMOTED_CURVES or exact_curves <= baseline_curves:
        return baseline
    if len(exact) > len(baseline) + _MAXIMUM_EXTRA_PIECES:
        return baseline

    # The entire point of this promotion is to remove rectangular mitres.  Do
    # not accept a recovered sequence unless the physical curve/straight
    # tangents agree at every exposed internal connector.
    if _maximum_internal_tangent_error(exact, turn_sign) > 1.0e-4:
        return baseline

    end_projection = _sharp._nearest_forward(
        measure,
        locked_path[-1],
        start,
        float(maximum_end_distance),
    )
    if end_projection is None:
        return baseline
    if end_projection[0] > _sharp._MAXIMUM_LOCKED_CORRIDOR_METRES + 1.0e-9:
        return baseline
    if end_projection[1] < float(minimum_end_distance) - _END_PROGRESS_TOLERANCE_METRES:
        return baseline

    start_point = measure.point(start)[:2]
    if math.dist(locked_path[0], start_point) > 1.0e-6:
        return baseline
    return exact


def install_stock_road_curve_usage_policy() -> None:
    """Install broader exact curve promotion after the narrow sharp-turn pass."""

    global _ORIGINAL_CHAIN, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_CHAIN = _p._stock_piece_chain
    _p._stock_piece_chain = _curve_promotion_chain
    _INSTALLED = True
