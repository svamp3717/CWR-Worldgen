# SPDX-License-Identifier: GPL-3.0-or-later
"""Own connector-locked stock fitting for paved S-bends.

Lundby20 exposed a residual case that the one-direction sharp-turn fitter cannot
repair: a long paved road can turn one way and immediately reverse. The two
same-direction spans share the inflection vertex, so treating them separately
either skips the second span or leaves a rotated-straight miter at the join.

The first phase detects adjacent opposite-sign bend spans and fits the combined
local window with both left- and right-turning stock 10-degree curves while
remaining inside the existing 0.60 m source corridor. A later exact phase runs
after the micro-bend policy and retains those connector-locked stock actions
rather than feeding the sampled centreline back through the greedy fitter.

The two phases intentionally remain separate installation moments because the
micro-bend policy sits between them. They live in one module because they are
one S-bend algorithm and share the same beam, action catalogue and geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading

from . import playability as _p
from . import stock_road_curve_policy as _curve
from . import stock_road_model_geometry as _geometry
from . import stock_road_sharp_turn_policy as _sharp

_MAXIMUM_S_BEND_SPAN_METRES = 230.0
_MAXIMUM_S_BEND_END_ERROR_METRES = 0.75
_MAXIMUM_S_BEND_BOUNDARY_TANGENT_ERROR_DEGREES = 3.5
_MINIMUM_S_BEND_CURVES = 3
_RADIUS_CHANGE_PENALTY = 0.45
_DIRECTION_SWITCH_PENALTY = 0.35
_WRONG_DIRECTION_PENALTY = 4.0

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
_STEP_TURN_EPSILON_DEGREES = 1.0

_ORIGINAL_LOCKED_MEASURE = None
_ORIGINAL_CHAIN = None
_ORIGINAL_CURVED_MODEL_FOR_RUN = None
_ORIGINAL_FIT_STOCK_ROADS = None
_INSTALLED = False
_EXACT_INSTALLED = False
_BEAM_LIMIT_LOCK = threading.Lock()
_EXACT_CURVE_REVERSE: dict[
    tuple[str, float, float, float, float], bool
] = {}


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


def _curve_key(model_path, start, end):
    return (
        str(model_path).casefold(),
        round(float(start[0]), 6),
        round(float(start[1]), 6),
        round(float(end[0]), 6),
        round(float(end[1]), 6),
    )


def _recover_exact_steps(path, pieces):
    """Recover stock pieces plus the beam's original turn sign."""

    samples_per_action = int(_sharp._CURVE_SAMPLE_COUNT)
    if (
        len(path) < 2
        or samples_per_action <= 0
        or (len(path) - 1) % samples_per_action != 0
    ):
        return None

    unique = {}
    for action in _s_bend_actions(pieces):
        key = (str(action.piece.model_path).casefold(), float(action.piece.length_metres))
        unique.setdefault(key, action.piece)
    candidates = tuple(unique.values())
    if not candidates:
        return None

    recovered = []
    action_count = (len(path) - 1) // samples_per_action
    for index in range(action_count):
        offset = index * samples_per_action
        start = path[offset]
        end = path[offset + samples_per_action]
        chord = math.dist(start, end)
        piece = min(
            candidates,
            key=lambda candidate: abs(float(candidate.length_metres) - chord),
        )
        if abs(float(piece.length_metres) - chord) > _ACTION_LENGTH_TOLERANCE_METRES:
            return None

        first_sample = path[offset + 1]
        penultimate = path[offset + samples_per_action - 1]
        first_heading = _sharp._heading(start, first_sample)
        last_heading = _sharp._heading(penultimate, end)
        sampled_turn = _p._signed_heading_delta(first_heading, last_heading)
        if sampled_turn > _STEP_TURN_EPSILON_DEGREES:
            turn_sign = 1
        elif sampled_turn < -_STEP_TURN_EPSILON_DEGREES:
            turn_sign = -1
        else:
            turn_sign = 0

        is_curve = _geometry.stock_curve_match(str(piece.model_path)) is not None
        if is_curve != bool(turn_sign):
            return None
        recovered.append((piece, start, end, turn_sign))
    return tuple(recovered)


def _recover_exact_actions(path, pieces):
    steps = _recover_exact_steps(path, pieces)
    if steps is None:
        return None
    return tuple((piece, start, end) for piece, start, end, _sign in steps)


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


def _step_tangents(step):
    piece, start, end, turn_sign = step
    chord = _sharp._heading(start, end)
    if _geometry.stock_curve_match(str(piece.model_path)) is None:
        return chord, chord
    if turn_sign < 0:
        return (chord + 5.0) % 360.0, (chord - 5.0) % 360.0
    return (chord - 5.0) % 360.0, (chord + 5.0) % 360.0


def _maximum_internal_tangent_error(fitted, source_points) -> float:
    maximum = 0.0
    for previous, current in zip(fitted, fitted[1:]):
        previous_end = _piece_tangents(previous, source_points)[1]
        current_start = _piece_tangents(current, source_points)[0]
        maximum = max(maximum, _p._heading_difference(previous_end, current_start))
    return maximum


def _maximum_step_tangent_error(steps) -> float:
    maximum = 0.0
    for previous, current in zip(steps, steps[1:]):
        previous_end = _step_tangents(previous)[1]
        current_start = _step_tangents(current)[0]
        maximum = max(maximum, _p._heading_difference(previous_end, current_start))
    return maximum


def _quantised_exit_heading(entry_heading: float, source_exit_heading: float) -> float:
    signed = _p._signed_heading_delta(entry_heading, source_exit_heading)
    steps = int(round(signed / 10.0))
    return (float(entry_heading) + float(steps) * 10.0) % 360.0


def _long_exact_s_bend_path(source_points, entry_heading, exit_heading, pieces):
    """Run the S-bend beam with a larger cap only for the covered exact pass."""

    global _MAXIMUM_S_BEND_SPAN_METRES
    with _BEAM_LIMIT_LOCK:
        previous_limit = float(_MAXIMUM_S_BEND_SPAN_METRES)
        _MAXIMUM_S_BEND_SPAN_METRES = max(
            previous_limit,
            MAXIMUM_EXACT_S_BEND_RUN_METRES,
        )
        try:
            return _beam_s_bend_path(
                source_points,
                entry_heading,
                exit_heading,
                pieces,
            )
        finally:
            _MAXIMUM_S_BEND_SPAN_METRES = previous_limit


def _curved_model_for_run(model_path, run, start, end):
    if _ORIGINAL_CURVED_MODEL_FOR_RUN is None:
        raise RuntimeError("stock road exact S-bend curve placement is not installed")

    match = _geometry.stock_curve_match(str(model_path))
    if match is None:
        return _ORIGINAL_CURVED_MODEL_FOR_RUN(model_path, run, start, end)

    reverse = _EXACT_CURVE_REVERSE.pop(_curve_key(model_path, start, end), None)
    if reverse is None:
        return _ORIGINAL_CURVED_MODEL_FOR_RUN(model_path, run, start, end)

    _curve._CURVE_REVERSE.set(bool(reverse))
    return model_path


def _fit_stock_roads(*args, **kwargs):
    if _ORIGINAL_FIT_STOCK_ROADS is None:
        raise RuntimeError("stock road exact S-bend fit wrapper is not installed")
    _EXACT_CURVE_REVERSE.clear()
    try:
        return _ORIGINAL_FIT_STOCK_ROADS(*args, **kwargs)
    finally:
        _EXACT_CURVE_REVERSE.clear()


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
    if _sharp._baseline_short_straights(baseline) < MINIMUM_EXACT_S_BEND_SHORT_STRAIGHTS:
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

    source_points, entry_heading, source_exit_heading = _sharp._measure_slice(
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

    exact_steps = _recover_exact_steps(locked_path, pieces)
    if exact_steps is None:
        return baseline
    exact = tuple(
        (piece, step_start, step_end)
        for piece, step_start, step_end, _sign in exact_steps
    )
    exact_curves = _curve_count(exact)
    baseline_curves = _curve_count(baseline)
    if exact_curves < MINIMUM_EXACT_S_BEND_CURVES or exact_curves < baseline_curves:
        return baseline
    if len(exact) > len(baseline) + MAXIMUM_EXACT_S_BEND_EXTRA_PIECES:
        return baseline

    exact_tangent_error = _maximum_step_tangent_error(exact_steps)
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

    for piece, step_start, step_end, turn_sign in exact_steps:
        if turn_sign:
            _EXACT_CURVE_REVERSE[
                _curve_key(piece.model_path, step_start, step_end)
            ] = turn_sign < 0
    return exact


def install_stock_road_s_bend_policy() -> None:
    """Install opposite-direction local bend locking for paved stock roads."""

    global _ORIGINAL_LOCKED_MEASURE, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_LOCKED_MEASURE = _sharp._locked_measure
    _sharp._locked_measure = _s_bend_locked_measure
    _INSTALLED = True


def install_stock_road_s_bend_exact_policy() -> None:
    """Preserve exact S-bend actions after the intervening micro-bend stage."""

    global _ORIGINAL_CHAIN, _ORIGINAL_CURVED_MODEL_FOR_RUN
    global _ORIGINAL_FIT_STOCK_ROADS, _EXACT_INSTALLED
    if _EXACT_INSTALLED:
        return
    if not _INSTALLED:
        raise RuntimeError("stock road S-bend policy must install first")

    _ORIGINAL_CHAIN = _p._stock_piece_chain
    _ORIGINAL_CURVED_MODEL_FOR_RUN = _p._curved_gravel_model_for_run
    _ORIGINAL_FIT_STOCK_ROADS = _p._fit_stock_piece_road_objects
    _p._stock_piece_chain = _exact_s_bend_chain
    _p._curved_gravel_model_for_run = _curved_model_for_run
    _p._fit_stock_piece_road_objects = _fit_stock_roads
    _EXACT_INSTALLED = True
