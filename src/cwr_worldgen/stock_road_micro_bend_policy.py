# SPDX-License-Identifier: GPL-3.0-or-later
"""Use native stock curves for gentle bends that still facet into straights.

The first sharp-turn beam deliberately requires at least two native ten-degree
curves. That leaves a common real-world case untreated: a gentle 8-15 degree
bend spread across several short stock pieces. Each individual miter is small,
but the outside road edge still opens enough for terrain to show through.

Keep the existing sharp-turn search as first refusal. If it cannot finish, run
the same connector-locked search with a one-curve minimum and a slightly wider
boundary-tangent allowance. The source corridor remains the same 0.60 m and
all internal stock connectors remain exact.

For short junction-to-junction stock runs with a coherent 7.5-15 degree bend,
retain the recovered stock-piece actions directly rather than feeding the sampled
centreline back through the greedy fitter. Generated gravel remains outside this
stock-asset path.
"""
from __future__ import annotations

import math

from . import playability as _p
from . import stock_road_sharp_turn_policy as _sharp

MINIMUM_MICRO_BEND_TOTAL_TURN_DEGREES = 7.5
MAXIMUM_MICRO_BEND_TOTAL_TURN_DEGREES = 15.0
MAXIMUM_MICRO_BEND_BOUNDARY_TANGENT_ERROR_DEGREES = 4.5

MAXIMUM_MICRO_EXACT_RUN_METRES = 120.0
MINIMUM_MICRO_EXACT_ENDPOINT_COVER_METRES = 0.40
MINIMUM_MICRO_EXACT_SHORT_STRAIGHTS = 2
MINIMUM_MICRO_EXACT_CURVES = 1
MAXIMUM_MICRO_EXACT_EXTRA_PIECES = 2
MAXIMUM_MICRO_VERTEX_TURN_DEGREES = 18.0
MINIMUM_MICRO_VERTEX_TURN_DEGREES = 0.45
MAXIMUM_MICRO_REVERSE_NOISE_DEGREES = 1.0
MICRO_EXACT_END_PROGRESS_TOLERANCE_METRES = 0.20

_ORIGINAL_BEAM = None
_ORIGINAL_CHAIN = None
_INSTALLED = False


def _micro_beam_stock_path(
    source_points,
    turn_sign: int,
    entry_heading: float,
    exit_heading: float,
    pieces,
):
    """Try the strict shared beam first, then allow one native curve section."""

    if _ORIGINAL_BEAM is None:
        raise RuntimeError("stock road micro-bend policy is not installed")

    result = _ORIGINAL_BEAM(
        source_points,
        turn_sign,
        entry_heading,
        exit_heading,
        pieces,
    )
    if result is not None:
        return result
    return _ORIGINAL_BEAM(
        source_points,
        turn_sign,
        entry_heading,
        exit_heading,
        pieces,
        minimum_curve_count=1,
        maximum_boundary_tangent_error_degrees=(
            MAXIMUM_MICRO_BEND_BOUNDARY_TANGENT_ERROR_DEGREES
        ),
    )


def _dominant_micro_bend(points):
    """Return one coherent gentle bend sign and accumulated source turn."""

    sign = 0
    count = 0
    total = 0.0
    for previous, point, following in zip(points, points[1:], points[2:]):
        turn = float(_sharp._signed_turn(previous, point, following))
        magnitude = abs(turn)
        if magnitude < MINIMUM_MICRO_VERTEX_TURN_DEGREES:
            continue
        if magnitude > MAXIMUM_MICRO_VERTEX_TURN_DEGREES:
            return None
        current_sign = 1 if turn > 0.0 else -1
        if sign and current_sign != sign:
            if magnitude <= MAXIMUM_MICRO_REVERSE_NOISE_DEGREES:
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
        or magnitude < MINIMUM_MICRO_BEND_TOTAL_TURN_DEGREES
        or magnitude > MAXIMUM_MICRO_BEND_TOTAL_TURN_DEGREES
    ):
        return None
    return sign, magnitude


def _micro_exact_chain(
    measure,
    pieces,
    *,
    start_distance,
    preferred_end_distance,
    minimum_end_distance,
    maximum_end_distance,
):
    """Retain a one-curve beam result instead of greedily re-faceting it."""

    if _ORIGINAL_CHAIN is None:
        raise RuntimeError("stock road micro-bend exact policy is not installed")

    baseline = _ORIGINAL_CHAIN(
        measure,
        pieces,
        start_distance=start_distance,
        preferred_end_distance=preferred_end_distance,
        minimum_end_distance=minimum_end_distance,
        maximum_end_distance=maximum_end_distance,
    )
    if _sharp._curveable_family(pieces) is None:
        return baseline
    if measure.total > MAXIMUM_MICRO_EXACT_RUN_METRES:
        return baseline
    if _sharp._baseline_short_straights(baseline) < MINIMUM_MICRO_EXACT_SHORT_STRAIGHTS:
        return baseline

    bend = _dominant_micro_bend(measure.points)
    if bend is None:
        return baseline
    turn_sign, _total_turn = bend

    start_cover = float(start_distance)
    end_cover = float(measure.total) - float(preferred_end_distance)
    if (
        start_cover < MINIMUM_MICRO_EXACT_ENDPOINT_COVER_METRES
        or end_cover < MINIMUM_MICRO_EXACT_ENDPOINT_COVER_METRES
    ):
        return baseline

    start = max(0.0, min(float(measure.total), float(start_distance)))
    end = max(start, min(float(measure.total), float(preferred_end_distance)))
    if end <= start + 1.0:
        return baseline

    source_points, entry_heading, source_exit_heading = _sharp._measure_slice(
        measure, start, end
    )
    stock_exit_heading = _sharp._quantised_stock_exit_heading(
        entry_heading,
        source_exit_heading,
        turn_sign,
    )
    locked_path = _micro_beam_stock_path(
        source_points,
        turn_sign,
        entry_heading,
        stock_exit_heading,
        pieces,
    )
    if locked_path is None:
        return baseline

    exact = _sharp._recover_exact_actions(locked_path, pieces, turn_sign)
    if exact is None:
        return baseline
    exact_curves = _sharp._curve_count(exact)
    baseline_curves = _sharp._curve_count(baseline)
    if exact_curves < MINIMUM_MICRO_EXACT_CURVES or exact_curves <= baseline_curves:
        return baseline
    if len(exact) > len(baseline) + MAXIMUM_MICRO_EXACT_EXTRA_PIECES:
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
    if (
        end_projection[1]
        < float(minimum_end_distance) - MICRO_EXACT_END_PROGRESS_TOLERANCE_METRES
    ):
        return baseline

    start_point = measure.point(start)[:2]
    if math.dist(locked_path[0], start_point) > 1.0e-6:
        return baseline

    # Beam actions are propagated connector-to-connector, so retaining the
    # recovered sequence preserves exact internal positions and tangents.
    for previous, current in zip(exact, exact[1:]):
        if math.dist(previous[2], current[1]) > 1.0e-4:
            return baseline
    return exact


def install_stock_road_micro_bend_policy() -> None:
    """Allow and preserve one-curve stock bend repairs."""

    global _ORIGINAL_BEAM, _ORIGINAL_CHAIN, _INSTALLED
    if _INSTALLED:
        return

    # A stock curve turns ten degrees, so a source bend just below that angle is
    # an important case rather than noise. The 0.60 m corridor remains the
    # geometric safety gate before any replacement can be accepted.
    _sharp._MINIMUM_SUSTAINED_TOTAL_TURN_DEGREES = (
        MINIMUM_MICRO_BEND_TOTAL_TURN_DEGREES
    )
    _ORIGINAL_BEAM = _sharp._beam_stock_path
    _sharp._beam_stock_path = _micro_beam_stock_path

    # Preserve exact one-curve actions after the earlier sharp/exact/S-bend
    # wrappers have had first refusal. Later curve-usage policies can still
    # promote larger bends around this result.
    _ORIGINAL_CHAIN = _p._stock_piece_chain
    _p._stock_piece_chain = _micro_exact_chain
    _INSTALLED = True
