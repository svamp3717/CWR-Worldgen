# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep connector-locked stock actions for endpoint-covered paved S-bends.

The S-bend beam already searches real stock straights and 10-degree curves in
connector space. Historically it returned only a sampled centreline, after
which the ordinary greedy fitter was free to turn that solution back into
independently rotated short straights. That recreates the grass wedges the beam
was meant to remove.

For paved runs whose two boundary errors are hidden by normal junction trims,
retain the beam's recovered stock-piece actions directly. The ordinary S-bend
pass keeps its shorter search cap; this exact pass may search a longer covered
run because there is no exposed endpoint seam to protect. The beam still keeps
every sample inside the existing 0.60 m source corridor. No extra or overlapping
repair road objects are emitted.
"""
from __future__ import annotations

import math
import threading

from . import playability as _p
from . import stock_road_curve_policy as _curve
from . import stock_road_model_geometry as _geometry
from . import stock_road_s_bend_policy as _s_bend
from . import stock_road_sharp_exact_policy as _exact
from . import stock_road_sharp_turn_policy as _sharp

MAXIMUM_EXACT_S_BEND_RUN_METRES = 360.0
MINIMUM_EXACT_S_BEND_ENDPOINT_COVER_METRES = 0.40
MINIMUM_EXACT_S_BEND_SHORT_STRAIGHTS = 4
MINIMUM_EXACT_S_BEND_CURVES = 3
MAXIMUM_EXACT_S_BEND_EXTRA_PIECES = 3
MINIMUM_SIGNIFICANT_REVERSAL_DEGREES = 0.45
MINIMUM_TANGENT_IMPROVEMENT_DEGREES = 0.25
MAXIMUM_EXACT_INTERNAL_TANGENT_ERROR_DEGREES = 1.0e-3
END_PROGRESS_TOLERANCE_METRES = 0.20
_ACTION_LENGTH_TOLERANCE_METRES = 1.0e-4

_ORIGINAL_CHAIN = None
_INSTALLED = False
_BEAM_LIMIT_LOCK = threading.Lock()


def _has_direction_reversal(points) -> bool:
    positive = False
    negative = False
    for previous, point, following in zip(points, points[1:], points[2:]):
        turn = float(_sharp._signed_turn(previous, point, following))
        if turn >= MINIMUM_SIGNIFICANT_REVERSAL_DEGREES:
            positive = True
        elif turn <= -MINIMUM_SIGNIFICANT_REVERSAL_DEGREES:
            negative = True
        if positive and negative:
            return True
    return False


def _recover_exact_actions(path, pieces):
    """Recover one stock piece from each S-bend beam sampling group."""

    samples_per_action = int(_sharp._CURVE_SAMPLE_COUNT)
    if (
        len(path) < 2
        or samples_per_action <= 0
        or (len(path) - 1) % samples_per_action != 0
    ):
        return None

    unique = {}
    for action in _s_bend._s_bend_actions(pieces):
        key = (str(action.piece.model_path).casefold(), float(action.piece.length_metres))
        unique.setdefault(key, action.piece)
    candidates = tuple(unique.values())
    if not candidates:
        return None

    recovered = []
    action_count = (len(path) - 1) // samples_per_action
    for index in range(action_count):
        start = path[index * samples_per_action]
        end = path[(index + 1) * samples_per_action]
        chord = math.dist(start, end)
        piece = min(
            candidates,
            key=lambda candidate: abs(float(candidate.length_metres) - chord),
        )
        if abs(float(piece.length_metres) - chord) > _ACTION_LENGTH_TOLERANCE_METRES:
            return None
        recovered.append((piece, start, end))
    return tuple(recovered)


def _curve_count(fitted) -> int:
    return sum(
        _geometry.stock_curve_match(str(piece.model_path)) is not None
        for piece, _start, _end in fitted
    )


def _piece_tangents(item, source_points):
    piece, start, end = item
    chord = _sharp._heading(start, end)
    if _geometry.stock_curve_match(str(piece.model_path)) is None:
        return chord, chord
    reverse = _curve._curve_reverse_for_run(source_points, start, end)
    if reverse:
        return (chord + 5.0) % 360.0, (chord - 5.0) % 360.0
    return (chord - 5.0) % 360.0, (chord + 5.0) % 360.0


def _maximum_internal_tangent_error(fitted, source_points) -> float:
    maximum = 0.0
    for previous, current in zip(fitted, fitted[1:]):
        previous_end = _piece_tangents(previous, source_points)[1]
        current_start = _piece_tangents(current, source_points)[0]
        maximum = max(maximum, _p._heading_difference(previous_end, current_start))
    return maximum


def _quantised_exit_heading(entry_heading: float, source_exit_heading: float) -> float:
    signed = _p._signed_heading_delta(entry_heading, source_exit_heading)
    steps = int(round(signed / 10.0))
    return (float(entry_heading) + float(steps) * 10.0) % 360.0


def _long_exact_s_bend_path(source_points, entry_heading, exit_heading, pieces):
    """Run the existing beam with a larger cap only for this covered exact pass."""

    with _BEAM_LIMIT_LOCK:
        previous_limit = float(_s_bend._MAXIMUM_S_BEND_SPAN_METRES)
        _s_bend._MAXIMUM_S_BEND_SPAN_METRES = max(
            previous_limit,
            MAXIMUM_EXACT_S_BEND_RUN_METRES,
        )
        try:
            return _s_bend._beam_s_bend_path(
                source_points,
                entry_heading,
                exit_heading,
                pieces,
            )
        finally:
            _s_bend._MAXIMUM_S_BEND_SPAN_METRES = previous_limit


def _exact_s_bend_chain(
    measure,
    pieces,
    *,
    start_distance,
    preferred_end_distance,
    minimum_end_distance,
    maximum_end_distance,
):
    if _ORIGINAL_CHAIN is None:
        raise RuntimeError("stock road exact S-bend policy is not installed")

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
    if float(measure.total) > MAXIMUM_EXACT_S_BEND_RUN_METRES:
        return baseline
    if _exact._baseline_short_straights(baseline) < MINIMUM_EXACT_S_BEND_SHORT_STRAIGHTS:
        return baseline
    if not _has_direction_reversal(measure.points):
        return baseline

    start_cover = float(start_distance)
    end_cover = float(measure.total) - float(preferred_end_distance)
    if (
        start_cover < MINIMUM_EXACT_S_BEND_ENDPOINT_COVER_METRES
        or end_cover < MINIMUM_EXACT_S_BEND_ENDPOINT_COVER_METRES
    ):
        return baseline

    start = max(0.0, min(float(measure.total), float(start_distance)))
    end = max(start, min(float(measure.total), float(preferred_end_distance)))
    if end <= start + 1.0:
        return baseline

    source_points, entry_heading, source_exit_heading = _exact._measure_slice(
        measure, start, end
    )
    stock_exit_heading = _quantised_exit_heading(entry_heading, source_exit_heading)
    locked_path = _long_exact_s_bend_path(
        source_points,
        entry_heading,
        stock_exit_heading,
        pieces,
    )
    if locked_path is None:
        return baseline

    exact = _recover_exact_actions(locked_path, pieces)
    if exact is None:
        return baseline
    exact_curves = _curve_count(exact)
    baseline_curves = _curve_count(baseline)
    if exact_curves < MINIMUM_EXACT_S_BEND_CURVES or exact_curves < baseline_curves:
        return baseline
    if len(exact) > len(baseline) + MAXIMUM_EXACT_S_BEND_EXTRA_PIECES:
        return baseline

    exact_tangent_error = _maximum_internal_tangent_error(exact, source_points)
    baseline_tangent_error = _maximum_internal_tangent_error(baseline, measure.points)
    if exact_tangent_error > MAXIMUM_EXACT_INTERNAL_TANGENT_ERROR_DEGREES:
        return baseline
    if (
        exact_curves == baseline_curves
        and baseline_tangent_error - exact_tangent_error
        < MINIMUM_TANGENT_IMPROVEMENT_DEGREES
    ):
        return baseline

    start_point = measure.point(start)[:2]
    if math.dist(locked_path[0], start_point) > 1.0e-6:
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
    if end_projection[1] < float(minimum_end_distance) - END_PROGRESS_TOLERANCE_METRES:
        return baseline

    for previous, current in zip(exact, exact[1:]):
        if math.dist(previous[2], current[1]) > 1.0e-4:
            return baseline
    return exact


def install_stock_road_s_bend_exact_policy() -> None:
    """Preserve exact S-bend beam actions after the micro-bend wrapper."""

    global _ORIGINAL_CHAIN, _INSTALLED
    if _INSTALLED:
        return
    if not _s_bend._INSTALLED:
        raise RuntimeError("stock road S-bend policy must install first")

    _ORIGINAL_CHAIN = _p._stock_piece_chain
    _p._stock_piece_chain = _exact_s_bend_chain
    _INSTALLED = True
