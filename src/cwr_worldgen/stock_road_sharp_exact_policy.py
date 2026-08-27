# SPDX-License-Identifier: GPL-3.0-or-later
"""Use the sharp-turn beam result directly for short faceted paved bends.

The general sharp-turn policy can find a better stock-compatible centreline,
but it then feeds that sampled line back through the ordinary greedy fitter.
That second fit is allowed to rotate short straight pieces independently.  At a
sharp bend their centreline endpoints can still touch while the road edges do
not, leaving the familiar triangular grass wedge on one half of the carriageway.

This policy is deliberately narrow.  It only replaces short paved runs whose
bend fills a junction-to-junction span and whose existing result is dominated
by short straight facets.  The replacement is the beam search's stock-piece
sequence itself, so every internal connector has one common position *and*
tangent.  The small residual heading quantisation at each end remains under the
existing junction cover rather than being exposed as an interior road seam.
"""
from __future__ import annotations

import math

from . import playability as _p
from . import stock_road_model_geometry as _model_geometry
from . import stock_road_sharp_turn_policy as _sharp

_MAXIMUM_EXACT_RUN_METRES = 110.0
_MINIMUM_ENDPOINT_TRIM_METRES = 0.40
_MINIMUM_BASELINE_SHORT_STRAIGHTS = 6
_MINIMUM_TOTAL_TURN_DEGREES = 20.0
_MAXIMUM_TOTAL_TURN_DEGREES = 50.0
_MINIMUM_EXACT_CURVES = 3
_END_PROGRESS_TOLERANCE_METRES = 0.20
_ACTION_LENGTH_TOLERANCE_METRES = 1.0e-4

_ORIGINAL_CHAIN = None
_INSTALLED = False


def _measure_slice(measure, start_distance: float, end_distance: float):
    start_x, start_z, start_heading = measure.point(start_distance)
    end_x, end_z, end_heading = measure.point(end_distance)
    points = [(start_x, start_z)]
    for distance, point in zip(measure.cumulative, measure.points):
        if start_distance + 1.0e-7 < distance < end_distance - 1.0e-7:
            points.append(point)
    if math.dist(points[-1], (end_x, end_z)) > 0.05:
        points.append((end_x, end_z))
    return tuple(points), start_heading, end_heading


def _recover_exact_actions(path, pieces, turn_sign: int):
    """Recover one stock action from each beam sampling group."""

    family = _sharp._paved_family(pieces)
    samples_per_action = _sharp._CURVE_SAMPLE_COUNT
    if (
        family is None
        or len(path) < 2
        or (len(path) - 1) % samples_per_action != 0
    ):
        return None
    actions = _sharp._actions(pieces, *family, turn_sign)
    if not actions:
        return None

    recovered = []
    action_count = (len(path) - 1) // samples_per_action
    for index in range(action_count):
        start = path[index * samples_per_action]
        end = path[(index + 1) * samples_per_action]
        chord = math.dist(start, end)
        action = min(
            actions,
            key=lambda candidate: abs(float(candidate.piece.length_metres) - chord),
        )
        if abs(float(action.piece.length_metres) - chord) > _ACTION_LENGTH_TOLERANCE_METRES:
            return None
        recovered.append((action.piece, start, end))
    return tuple(recovered)


def _baseline_short_straights(fitted) -> int:
    return sum(
        1
        for piece, _start, _end in fitted
        if piece.nominal_length in {6, 12}
        and _model_geometry.stock_curve_match(str(piece.model_path)) is None
    )


def _curve_count(fitted) -> int:
    return sum(
        _model_geometry.stock_curve_match(str(piece.model_path)) is not None
        for piece, _start, _end in fitted
    )


def _quantised_stock_exit_heading(
    entry_heading: float,
    source_exit_heading: float,
    turn_sign: int,
) -> float:
    """Return the nearest heading reachable by whole 10-degree stock curves."""

    total_turn = _p._heading_difference(entry_heading, source_exit_heading)
    curve_steps = max(1, int(round(total_turn / 10.0)))
    return (float(entry_heading) + float(turn_sign) * curve_steps * 10.0) % 360.0


def _exact_sharp_turn_chain(
    measure,
    pieces,
    *,
    start_distance,
    preferred_end_distance,
    minimum_end_distance,
    maximum_end_distance,
):
    if _ORIGINAL_CHAIN is None:
        raise RuntimeError("stock road exact sharp-turn policy is not installed")

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
    if measure.total > _MAXIMUM_EXACT_RUN_METRES:
        return baseline

    # This exact replacement is allowed to quantise the two boundary tangents
    # by at most five degrees, so require the normal junction trims at both ends
    # to hide those boundaries. Interior seams remain exactly tangent-matched.
    start_trim = float(start_distance)
    end_trim = float(measure.total) - float(preferred_end_distance)
    if (
        start_trim < _MINIMUM_ENDPOINT_TRIM_METRES
        or end_trim < _MINIMUM_ENDPOINT_TRIM_METRES
    ):
        return baseline
    if _baseline_short_straights(baseline) < _MINIMUM_BASELINE_SHORT_STRAIGHTS:
        return baseline

    for start_index, end_index, turn_sign in _sharp._sharp_turn_spans(measure.points):
        # The whole bend must occupy this short split run.  Do not rewrite an
        # unrelated interior bend plus its surrounding straight road.
        if start_index != 0 or end_index + 1 != len(measure.points) - 1:
            continue

        start = max(float(start_distance), float(measure.cumulative[start_index]))
        end = min(
            float(preferred_end_distance),
            float(measure.cumulative[end_index + 1]),
        )
        if end <= start + 1.0:
            continue

        source_points, entry_heading, source_exit_heading = _measure_slice(
            measure, start, end
        )
        total_turn = _p._heading_difference(entry_heading, source_exit_heading)
        if not (_MINIMUM_TOTAL_TURN_DEGREES <= total_turn <= _MAXIMUM_TOTAL_TURN_DEGREES):
            continue

        # Stock curve ODOLs turn exactly ten degrees.  The nearest reachable
        # exit heading differs from the source by at most five degrees and is
        # hidden beneath the endpoint junction cap.  More importantly, this
        # gives the beam a physically reachable final tangent instead of making
        # it reintroduce an exposed rotated-straight miter to chase a fraction
        # of a degree in the source polyline.
        stock_exit_heading = _quantised_stock_exit_heading(
            entry_heading,
            source_exit_heading,
            turn_sign,
        )
        locked_path = _sharp._beam_stock_path(
            source_points,
            turn_sign,
            entry_heading,
            stock_exit_heading,
            pieces,
        )
        if locked_path is None:
            continue
        exact = _recover_exact_actions(locked_path, pieces, turn_sign)
        if exact is None or _curve_count(exact) < _MINIMUM_EXACT_CURVES:
            continue
        if _curve_count(exact) <= _curve_count(baseline):
            continue
        if len(exact) > len(baseline):
            continue

        end_projection = _sharp._nearest_forward(
            measure,
            locked_path[-1],
            start,
            float(maximum_end_distance),
        )
        if end_projection is None:
            continue
        if end_projection[0] > _sharp._MAXIMUM_LOCKED_CORRIDOR_METRES + 1.0e-9:
            continue
        if (
            end_projection[1]
            < float(minimum_end_distance) - _END_PROGRESS_TOLERANCE_METRES
        ):
            continue
        start_point = measure.point(start)[:2]
        if math.dist(locked_path[0], start_point) > 1.0e-6:
            continue
        return exact

    return baseline


def install_stock_road_sharp_exact_policy() -> None:
    global _ORIGINAL_CHAIN, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_CHAIN = _p._stock_piece_chain
    _p._stock_piece_chain = _exact_sharp_turn_chain
    _INSTALLED = True
