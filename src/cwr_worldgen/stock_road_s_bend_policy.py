# SPDX-License-Identifier: GPL-3.0-or-later
"""Fit local paved S-bends with connector-locked stock curves.

Lundby20 exposed a residual case that the one-direction sharp-turn fitter cannot
repair: a long paved road can turn one way and immediately reverse.  The two
same-direction spans share the inflection vertex, so treating them separately
either skips the second span or leaves a rotated-straight miter at the join.

This road-only policy detects adjacent opposite-sign bend spans and fits the
combined local window with both left- and right-turning stock 10-degree curves.
The beam still stays inside the existing 0.60 m source corridor.  It prefers a
stable radius within each same-direction curve run and permits the one direction
change required by the source S-bend.  The resulting sampled centreline is then
fed through the existing stock-road chain fitter; terrain and non-road objects
are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from . import playability as _p
from . import stock_road_sharp_turn_policy as _sharp

_MAXIMUM_S_BEND_SPAN_METRES = 230.0
_MAXIMUM_S_BEND_END_ERROR_METRES = 0.75
_MAXIMUM_S_BEND_BOUNDARY_TANGENT_ERROR_DEGREES = 3.5
_MINIMUM_S_BEND_CURVES = 3
_RADIUS_CHANGE_PENALTY = 0.45
_DIRECTION_SWITCH_PENALTY = 0.35
_WRONG_DIRECTION_PENALTY = 4.0

_ORIGINAL_LOCKED_MEASURE = None
_INSTALLED = False


@dataclass(frozen=True, slots=True)
class _State:
    score: float
    x: float
    z: float
    heading_degrees: float
    progress: float
    steps: tuple
    curve_count: int
    last_curve_sign: int
    last_curve_radius: float | None


def _s_bend_actions(pieces):
    family = _sharp._paved_family(pieces)
    if family is None:
        return ()
    prefix, family_name = family
    result = []
    seen_straights = set()
    for sign in (1, -1):
        for action in _sharp._actions(pieces, prefix, family_name, sign):
            if action.turn_sign == 0:
                key = str(action.piece.model_path).casefold()
                if key in seen_straights:
                    continue
                seen_straights.add(key)
            result.append(action)
    return tuple(result)


def _beam_s_bend_path(source_points, entry_heading: float, exit_heading: float, pieces):
    measure = _p._PolylineMeasure.create(source_points)
    if measure.total <= 1.0 or measure.total > _MAXIMUM_S_BEND_SPAN_METRES:
        return None
    actions = _s_bend_actions(pieces)
    if not actions:
        return None

    start = source_points[0]
    beam = (
        _State(
            0.0,
            float(start[0]),
            float(start[1]),
            float(entry_heading),
            0.0,
            (),
            0,
            0,
            None,
        ),
    )
    best = None
    shortest = min(float(action.piece.length_metres) for action in actions)
    maximum_steps = max(3, int(math.ceil(measure.total / shortest)) + 6)

    for _step_index in range(maximum_steps):
        candidates = []
        for state in beam:
            for action in actions:
                end, end_heading, samples = _sharp._advance(state, action)
                progress = state.progress
                maximum_deviation = 0.0
                lookahead = max(18.0, float(action.piece.length_metres) * 2.0 + 10.0)
                valid = True
                for sample in samples:
                    nearest = _sharp._nearest_forward(
                        measure,
                        sample,
                        progress,
                        min(measure.total, state.progress + lookahead),
                    )
                    if (
                        nearest is None
                        or nearest[0] > _sharp._MAXIMUM_LOCKED_CORRIDOR_METRES
                    ):
                        valid = False
                        break
                    maximum_deviation = max(maximum_deviation, nearest[0])
                    progress = max(progress, nearest[1])
                if not valid or progress <= state.progress + 0.40:
                    continue

                source_heading = measure.point(progress)[2]
                tangent_error = _p._heading_difference(end_heading, source_heading)
                if tangent_error > _sharp._MAXIMUM_CANDIDATE_TANGENT_ERROR_DEGREES:
                    continue
                endpoint_nearest = _sharp._nearest_forward(
                    measure,
                    end,
                    state.progress,
                    min(measure.total, state.progress + lookahead),
                )
                endpoint_error = (
                    endpoint_nearest[0] if endpoint_nearest is not None else math.inf
                )
                if endpoint_error > _sharp._MAXIMUM_LOCKED_CORRIDOR_METRES:
                    continue

                signed_source_turn = _p._signed_heading_delta(
                    state.heading_degrees,
                    source_heading,
                )
                signed_action_turn = 10.0 * int(action.turn_sign)
                turn_mismatch = abs(signed_source_turn - signed_action_turn)
                penalty = 0.0
                last_sign = state.last_curve_sign
                last_radius = state.last_curve_radius
                if action.turn_sign:
                    radius = float(action.radius_metres)
                    if signed_source_turn * int(action.turn_sign) < -1.0:
                        penalty += _WRONG_DIRECTION_PENALTY
                    if last_sign and last_sign != int(action.turn_sign):
                        penalty += _DIRECTION_SWITCH_PENALTY
                    if (
                        last_sign == int(action.turn_sign)
                        and last_radius is not None
                        and abs(radius - last_radius) > 1.0e-9
                    ):
                        penalty += (
                            _RADIUS_CHANGE_PENALTY
                            * abs(radius - last_radius)
                            / 25.0
                        )
                    last_sign = int(action.turn_sign)
                    last_radius = radius
                else:
                    # A straight separates curve runs, so radius continuity does
                    # not need to carry across it.
                    last_sign = 0
                    last_radius = None

                score = (
                    state.score
                    + maximum_deviation * maximum_deviation * 7.0
                    + endpoint_error * endpoint_error * 3.0
                    + (tangent_error / 5.0) ** 2
                    + (turn_mismatch / 5.0) ** 2
                    + penalty
                    + 0.03
                )
                step = _sharp._Step(
                    action.piece,
                    (state.x, state.z),
                    end,
                    samples,
                )
                candidate = _State(
                    score,
                    end[0],
                    end[1],
                    end_heading,
                    progress,
                    (*state.steps, step),
                    state.curve_count + int(action.turn_sign != 0),
                    last_sign,
                    last_radius,
                )

                remaining = measure.total - progress
                end_error = math.dist(end, source_points[-1])
                boundary_error = _p._heading_difference(end_heading, exit_heading)
                if (
                    remaining <= _MAXIMUM_S_BEND_END_ERROR_METRES
                    and end_error <= _MAXIMUM_S_BEND_END_ERROR_METRES
                    and boundary_error
                    <= _MAXIMUM_S_BEND_BOUNDARY_TANGENT_ERROR_DEGREES
                    and candidate.curve_count >= _MINIMUM_S_BEND_CURVES
                ):
                    final_score = (
                        score
                        + end_error * end_error * 3.0
                        + (boundary_error / 3.0) ** 2
                    )
                    if best is None or final_score < best[0]:
                        best = (final_score, candidate)

                if progress < measure.total - 0.05:
                    candidates.append(candidate)

        if best is not None:
            break
        if not candidates:
            break

        candidates.sort(
            key=lambda item: (
                item.score + (measure.total - item.progress) * 0.01,
                item.score,
                -item.progress,
            )
        )
        beam_list = []
        seen = {}
        for candidate in candidates:
            radius_bucket = int(round((candidate.last_curve_radius or 0.0) / 25.0))
            key = (
                int(candidate.progress / 2.5),
                int(round(candidate.heading_degrees / 2.5)) % 144,
                candidate.last_curve_sign,
                radius_bucket,
            )
            if seen.get(key, 0) >= 2:
                continue
            seen[key] = seen.get(key, 0) + 1
            beam_list.append(candidate)
            if len(beam_list) >= _sharp._BEAM_WIDTH:
                break
        beam = tuple(beam_list)

    if best is None:
        return None
    state = best[1]
    result = [source_points[0]]
    for step in state.steps:
        for point in step.samples:
            if math.dist(result[-1], point) > 0.05:
                result.append(point)
    return tuple(result)


def _s_bend_replacement(measure, pieces):
    spans = _sharp._sharp_turn_spans(measure.points)
    if len(spans) < 2:
        return None

    replacements = []
    occupied_until = -1
    index = 0
    while index + 1 < len(spans):
        first = spans[index]
        second = spans[index + 1]
        start_a, end_a, sign_a = first
        start_b, end_b, sign_b = second
        if (
            sign_a == sign_b
            or start_a <= occupied_until
            or start_b > end_a + 1
        ):
            index += 1
            continue

        start_index = start_a
        end_index = max(end_a, end_b)
        span_length = float(measure.cumulative[end_index] - measure.cumulative[start_index])
        if span_length > _MAXIMUM_S_BEND_SPAN_METRES:
            index += 1
            continue

        entry_heading = (
            _sharp._heading(measure.points[start_index - 1], measure.points[start_index])
            if start_index > 0
            else _sharp._heading(measure.points[start_index], measure.points[start_index + 1])
        )
        exit_heading = (
            _sharp._heading(measure.points[end_index], measure.points[end_index + 1])
            if end_index + 1 < len(measure.points)
            else _sharp._heading(measure.points[end_index - 1], measure.points[end_index])
        )
        source_points = measure.points[start_index : end_index + 1]
        locked = _beam_s_bend_path(source_points, entry_heading, exit_heading, pieces)
        if locked is None:
            index += 1
            continue

        replacements.append((start_index, end_index, locked))
        occupied_until = end_index
        index += 2

    if not replacements:
        return None

    result = []
    cursor = 0
    for start_index, end_index, locked in replacements:
        for point in measure.points[cursor:start_index]:
            if not result or math.dist(result[-1], point) > 0.05:
                result.append(point)
        for point in locked:
            if not result or math.dist(result[-1], point) > 0.05:
                result.append(point)
        cursor = end_index + 1
    for point in measure.points[cursor:]:
        if not result or math.dist(result[-1], point) > 0.05:
            result.append(point)
    if len(result) < 2:
        return None
    return _p._PolylineMeasure.create(tuple(result))


def _s_bend_locked_measure(measure, pieces, baseline):
    if _ORIGINAL_LOCKED_MEASURE is None:
        raise RuntimeError("stock road S-bend policy is not installed")
    if _sharp._paved_family(pieces) is None:
        return _ORIGINAL_LOCKED_MEASURE(measure, pieces, baseline)

    locked = _s_bend_replacement(measure, pieces)
    if locked is not None:
        return locked
    return _ORIGINAL_LOCKED_MEASURE(measure, pieces, baseline)


def install_stock_road_s_bend_policy() -> None:
    """Install opposite-direction local bend locking for paved stock roads."""

    global _ORIGINAL_LOCKED_MEASURE, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_LOCKED_MEASURE = _sharp._locked_measure
    _sharp._locked_measure = _s_bend_locked_measure
    _INSTALLED = True
