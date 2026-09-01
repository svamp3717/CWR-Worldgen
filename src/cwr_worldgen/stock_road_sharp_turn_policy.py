# SPDX-License-Identifier: GPL-3.0-or-later
"""Own connector-locked stock fitting for difficult road bends.

A difficult real-world bend can be too irregular for the single-radius curve
regularizer while still being perfectly representable by a short sequence of
stock straights and ten-degree curves. Falling all the way back to rotated short
rectangles makes every heading change a visible mitre: the road surface clips
and the borders no longer meet.

This owner handles both sustained same-direction bends and isolated stock-road
corners surrounded by quiet source geometry. A small beam search propagates the
actual connector pose from piece to piece, so every accepted internal seam has
one common position and tangent. A second, separately installed exact phase
retains those beam actions directly for short junction-covered runs instead of
feeding them back through the greedy fitter. Resistance ``ces`` uses the same
verified ten-degree stock-curve geometry as the paved families; generated gravel,
junction selection and terrain remain untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
import bisect
import math
import re

from . import playability as _p
from . import stock_road_model_geometry as _model_geometry

_MINIMUM_SIGNIFICANT_TURN_DEGREES = 0.70
_MAXIMUM_LOCAL_SUSTAINED_TURN_DEGREES = 18.0
_MINIMUM_SUSTAINED_TOTAL_TURN_DEGREES = 7.5
_MAXIMUM_QUIET_DISTANCE_METRES = 35.0
_MINIMUM_SINGLE_VERTEX_TURN_DEGREES = 7.5
_MAXIMUM_SINGLE_VERTEX_TURN_DEGREES = 35.0
_MAXIMUM_ADJACENT_SIGNIFICANT_TURN_DEGREES = 0.70
_MAXIMUM_LOCKED_CORRIDOR_METRES = 0.60
_MAXIMUM_LOCKED_END_ERROR_METRES = 1.25
_MAXIMUM_LOCKED_BOUNDARY_TANGENT_ERROR_DEGREES = 3.0
_MAXIMUM_CANDIDATE_TANGENT_ERROR_DEGREES = 9.0
_BEAM_WIDTH = 256
_MAXIMUM_SPAN_METRES = 180.0
_CURVE_SAMPLE_COUNT = 4

_MAXIMUM_EXACT_RUN_METRES = 110.0
_MINIMUM_ENDPOINT_TRIM_METRES = 0.40
_MINIMUM_BASELINE_SHORT_STRAIGHTS = 6
_MINIMUM_TOTAL_TURN_DEGREES = 20.0
_MAXIMUM_TOTAL_TURN_DEGREES = 50.0
_MINIMUM_EXACT_CURVES = 3
_END_PROGRESS_TOLERANCE_METRES = 0.20
_ACTION_LENGTH_TOLERANCE_METRES = 1.0e-4

_STOCK_PAVED_STRAIGHT = re.compile(
    r"^(?P<prefix>.*[\\/])(?P<family>sil|asf|kos)(?P<length>25|12|6)\.p3d$",
    re.IGNORECASE,
)
_STOCK_CURVEABLE_STRAIGHT = re.compile(
    r"^(?P<prefix>.*[\\/])(?P<family>sil|asf|kos|ces)(?P<length>25|12|6)\.p3d$",
    re.IGNORECASE,
)

_ORIGINAL_CHAIN = None
_ORIGINAL_EXACT_CHAIN = None
_INSTALLED = False
_EXACT_INSTALLED = False


@dataclass(frozen=True, slots=True)
class _Action:
    piece: object
    turn_sign: int
    radius_metres: float | None


@dataclass(frozen=True, slots=True)
class _Step:
    piece: object
    start: tuple[float, float]
    end: tuple[float, float]
    samples: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class _State:
    score: float
    x: float
    z: float
    heading_degrees: float
    progress: float
    steps: tuple[_Step, ...]
    curve_count: int


def _heading(start, end) -> float:
    return math.degrees(
        math.atan2(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
    ) % 360.0


def _signed_turn(previous, point, following) -> float:
    return _p._signed_heading_delta(
        _heading(previous, point),
        _heading(point, following),
    )


def _coherent_bend(
    points,
    *,
    minimum_vertex_turn_degrees: float,
    maximum_vertex_turn_degrees: float,
    maximum_reverse_noise_degrees: float,
    minimum_total_turn_degrees: float,
    maximum_total_turn_degrees: float,
    minimum_significant_vertices: int = 1,
) -> tuple[int, float] | None:
    """Return one coherent bend sign using caller-owned acceptance thresholds."""

    sign = 0
    count = 0
    total = 0.0
    for previous, point, following in zip(points, points[1:], points[2:]):
        turn = float(_signed_turn(previous, point, following))
        magnitude = abs(turn)
        if magnitude < float(minimum_vertex_turn_degrees):
            continue
        if magnitude > float(maximum_vertex_turn_degrees):
            return None
        current_sign = 1 if turn > 0.0 else -1
        if sign and current_sign != sign:
            if magnitude <= float(maximum_reverse_noise_degrees):
                continue
            return None
        if not sign:
            sign = current_sign
        total += turn
        count += 1

    magnitude = abs(total)
    if (
        sign == 0
        or count < int(minimum_significant_vertices)
        or magnitude < float(minimum_total_turn_degrees)
        or magnitude > float(maximum_total_turn_degrees)
    ):
        return None
    return sign, magnitude


def _sustained_sharp_turn_spans(points):
    """Return sustained same-direction bend spans, splitting long quiet gaps."""

    cleaned = tuple(points)
    if len(cleaned) < 4:
        return ()
    turns = [0.0] * len(cleaned)
    for index in range(1, len(cleaned) - 1):
        turns[index] = _signed_turn(cleaned[index - 1], cleaned[index], cleaned[index + 1])

    spans = []
    sign = 0
    first_significant = None
    last_significant = None
    significant_count = 0
    accumulated_turn = 0.0
    quiet_distance = 0.0

    def finish() -> None:
        nonlocal sign, first_significant, last_significant
        nonlocal significant_count, accumulated_turn, quiet_distance
        if (
            first_significant is not None
            and last_significant is not None
            and significant_count >= 2
            and abs(accumulated_turn) >= _MINIMUM_SUSTAINED_TOTAL_TURN_DEGREES
        ):
            start = max(0, first_significant - 1)
            end = last_significant
            length = sum(
                math.dist(a, b)
                for a, b in zip(cleaned[start:end], cleaned[start + 1 : end + 1])
            )
            if length <= _MAXIMUM_SPAN_METRES:
                spans.append((start, end, sign))
        sign = 0
        first_significant = None
        last_significant = None
        significant_count = 0
        accumulated_turn = 0.0
        quiet_distance = 0.0

    for index in range(1, len(cleaned) - 1):
        turn = turns[index]
        magnitude = abs(turn)
        if magnitude > _MAXIMUM_LOCAL_SUSTAINED_TURN_DEGREES:
            finish()
            continue
        if magnitude >= _MINIMUM_SIGNIFICANT_TURN_DEGREES:
            current_sign = 1 if turn > 0.0 else -1
            if sign and current_sign != sign:
                finish()
            if not sign:
                sign = current_sign
                first_significant = index
            last_significant = index
            significant_count += 1
            accumulated_turn += turn
            quiet_distance = 0.0
            continue
        if sign:
            accumulated_turn += turn
            quiet_distance += math.dist(cleaned[index - 1], cleaned[index])
            if quiet_distance > _MAXIMUM_QUIET_DISTANCE_METRES:
                finish()

    finish()
    return tuple(spans)


def _isolated_single_vertex_spans(points, existing=()):
    """Return curve-beam spans for isolated stock corners not already covered."""

    cleaned = tuple(points)
    if len(cleaned) < 5:
        return ()

    covered = tuple((int(start), int(end)) for start, end, _sign in existing)
    turns = [0.0] * len(cleaned)
    for index in range(1, len(cleaned) - 1):
        turns[index] = _signed_turn(
            cleaned[index - 1], cleaned[index], cleaned[index + 1]
        )

    result = []
    # Two quiet boundary segments let _locked_measure derive stable entry and
    # exit tangents around the three-point corner span. Corners near a run edge
    # remain the responsibility of endpoint/junction fitting.
    for index in range(2, len(cleaned) - 2):
        turn = float(turns[index])
        magnitude = abs(turn)
        if not (
            _MINIMUM_SINGLE_VERTEX_TURN_DEGREES
            <= magnitude
            <= _MAXIMUM_SINGLE_VERTEX_TURN_DEGREES
        ):
            continue
        if any(start <= index <= end + 1 for start, end in covered):
            continue
        if (
            abs(float(turns[index - 1]))
            >= _MAXIMUM_ADJACENT_SIGNIFICANT_TURN_DEGREES
            or abs(float(turns[index + 1]))
            >= _MAXIMUM_ADJACENT_SIGNIFICANT_TURN_DEGREES
        ):
            continue

        start = index - 1
        end = index + 1
        length = sum(
            math.dist(a, b)
            for a, b in zip(cleaned[start:end], cleaned[start + 1 : end + 1])
        )
        if length <= 1.0 or length > _MAXIMUM_SPAN_METRES:
            continue
        result.append((start, end, 1 if turn > 0.0 else -1))
    return tuple(result)


def _sharp_turn_spans(points):
    """Return all stock bend spans owned by the connector-locked curve beam."""

    existing = _sustained_sharp_turn_spans(points)
    additions = _isolated_single_vertex_spans(points, existing)
    if not additions:
        return existing
    return tuple(sorted((*existing, *additions), key=lambda item: (item[0], item[1])))


def _family_for_pattern(pieces, pattern) -> tuple[str, str] | None:
    family = None
    prefix = None
    found = False
    for piece in pieces:
        match = pattern.fullmatch(str(piece.model_path).replace("/", "\\"))
        if match is None:
            continue
        current_family = match.group("family").casefold()
        current_prefix = match.group("prefix")
        if family is None:
            family = current_family
            prefix = current_prefix
        elif current_family != family or current_prefix.casefold() != str(prefix).casefold():
            return None
        found = True
    if not found or family is None or prefix is None:
        return None
    return prefix, family


def _paved_family(pieces) -> tuple[str, str] | None:
    """Return only paved stock families for callers that intentionally exclude ces."""

    return _family_for_pattern(pieces, _STOCK_PAVED_STRAIGHT)


def _curveable_family(pieces) -> tuple[str, str] | None:
    """Return any stock family with verified Resistance ten-degree curve assets."""

    return _family_for_pattern(pieces, _STOCK_CURVEABLE_STRAIGHT)


def _actions(pieces, prefix: str, family: str, turn_sign: int) -> tuple[_Action, ...]:
    result: list[_Action] = []
    straights = []
    for piece in pieces:
        match = _STOCK_CURVEABLE_STRAIGHT.fullmatch(
            str(piece.model_path).replace("/", "\\")
        )
        if match is None or match.group("family").casefold() != family:
            continue
        straights.append(piece)
    for piece in sorted(
        straights, key=lambda item: (-float(item.length_metres), str(item.model_path).casefold())
    ):
        result.append(_Action(piece, 0, None))

    for radius in (100, 75, 50, 25):
        model_path = f"{prefix}{family}10 {radius}.p3d"
        geometry = _model_geometry.stock_curve_connectors(model_path)
        if geometry is None:
            continue
        result.append(
            _Action(
                _p._RoadPiece(model_path, geometry.chord_length_metres, 10),
                turn_sign,
                float(radius),
            )
        )
    return tuple(result)


def _nearest_forward(measure, point, minimum_distance: float, maximum_distance: float):
    """Project a point onto a bounded forward interval of one polyline measure."""

    minimum = max(0.0, min(float(measure.total), float(minimum_distance)))
    maximum = max(minimum, min(float(measure.total), float(maximum_distance)))
    first = max(0, bisect.bisect_right(measure.cumulative, minimum) - 2)
    last = min(
        len(measure.points) - 2,
        bisect.bisect_right(measure.cumulative, maximum),
    )
    best = None
    for index in range(first, last + 1):
        segment_start = float(measure.cumulative[index])
        segment_end = float(measure.cumulative[index + 1])
        low = max(minimum, segment_start)
        high = min(maximum, segment_end)
        if high < low - 1.0e-9:
            continue
        start = measure.points[index]
        end = measure.points[index + 1]
        dx = float(end[0]) - float(start[0])
        dz = float(end[1]) - float(start[1])
        length = max(1.0e-9, segment_end - segment_start)
        denominator = dx * dx + dz * dz
        if denominator <= 1.0e-12:
            continue
        low_t = max(0.0, min(1.0, (low - segment_start) / length))
        high_t = max(low_t, min(1.0, (high - segment_start) / length))
        t = (
            (float(point[0]) - float(start[0])) * dx
            + (float(point[1]) - float(start[1])) * dz
        ) / denominator
        t = max(low_t, min(high_t, t))
        projected = (float(start[0]) + dx * t, float(start[1]) + dz * t)
        distance = math.dist((float(point[0]), float(point[1])), projected)
        along = segment_start + length * t
        candidate = (distance, along)
        if best is None or candidate < best:
            best = candidate
    return best


def _advance(state: _State, action: _Action):
    if action.turn_sign == 0:
        length = float(action.piece.length_metres)
        angle = math.radians(state.heading_degrees)
        end = (
            state.x + math.sin(angle) * length,
            state.z + math.cos(angle) * length,
        )
        samples = tuple(
            (
                state.x + (end[0] - state.x) * (sample / _CURVE_SAMPLE_COUNT),
                state.z + (end[1] - state.z) * (sample / _CURVE_SAMPLE_COUNT),
            )
            for sample in range(1, _CURVE_SAMPLE_COUNT + 1)
        )
        return end, state.heading_degrees, samples

    radius = float(action.radius_metres)
    heading = math.radians(state.heading_degrees)
    right = (math.cos(heading), -math.sin(heading))
    centre = (
        state.x + right[0] * radius * action.turn_sign,
        state.z + right[1] * radius * action.turn_sign,
    )
    vector = (state.x - centre[0], state.z - centre[1])
    turn = 10.0 * action.turn_sign
    samples = []
    for sample in range(1, _CURVE_SAMPLE_COUNT + 1):
        angle = -math.radians(turn * (sample / _CURVE_SAMPLE_COUNT))
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotated = (
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
        )
        samples.append((centre[0] + rotated[0], centre[1] + rotated[1]))
    return samples[-1], (state.heading_degrees + turn) % 360.0, tuple(samples)


def _beam_stock_path(
    source_points,
    turn_sign: int,
    entry_heading: float,
    exit_heading: float,
    pieces,
    *,
    minimum_curve_count: int = 2,
    maximum_boundary_tangent_error_degrees: float = _MAXIMUM_LOCKED_BOUNDARY_TANGENT_ERROR_DEGREES,
):
    """Fit one exact-pose stock sequence through a difficult bend."""

    measure = _p._PolylineMeasure.create(source_points)
    if measure.total <= 1.0:
        return None
    family = _curveable_family(pieces)
    if family is None:
        return None
    prefix, family_name = family
    actions = _actions(pieces, prefix, family_name, turn_sign)
    if not actions:
        return None

    start = source_points[0]
    beam = (
        _State(0.0, float(start[0]), float(start[1]), entry_heading, 0.0, (), 0),
    )
    best = None
    shortest = min(float(action.piece.length_metres) for action in actions)
    maximum_steps = max(3, int(math.ceil(measure.total / shortest)) + 5)

    for _step_index in range(maximum_steps):
        candidates: list[_State] = []
        for state in beam:
            for action in actions:
                end, end_heading, samples = _advance(state, action)
                progress = state.progress
                maximum_deviation = 0.0
                lookahead = max(12.0, float(action.piece.length_metres) * 2.0 + 8.0)
                valid = True
                for sample in samples:
                    nearest = _nearest_forward(
                        measure,
                        sample,
                        progress,
                        min(measure.total, state.progress + lookahead),
                    )
                    if nearest is None or nearest[0] > _MAXIMUM_LOCKED_CORRIDOR_METRES:
                        valid = False
                        break
                    maximum_deviation = max(maximum_deviation, nearest[0])
                    progress = max(progress, nearest[1])
                if not valid or progress <= state.progress + 0.40:
                    continue

                source_heading = measure.point(progress)[2]
                tangent_error = _p._heading_difference(end_heading, source_heading)
                if tangent_error > _MAXIMUM_CANDIDATE_TANGENT_ERROR_DEGREES:
                    continue
                source_turn = _p._heading_difference(state.heading_degrees, source_heading)
                action_turn = 10.0 if action.turn_sign else 0.0
                turn_mismatch = abs(source_turn - action_turn)
                endpoint_nearest = _nearest_forward(
                    measure,
                    end,
                    state.progress,
                    min(measure.total, state.progress + lookahead),
                )
                endpoint_error = endpoint_nearest[0] if endpoint_nearest is not None else math.inf
                if endpoint_error > _MAXIMUM_LOCKED_CORRIDOR_METRES:
                    continue

                score = (
                    state.score
                    + maximum_deviation * maximum_deviation * 7.0
                    + endpoint_error * endpoint_error * 3.0
                    + (tangent_error / 5.0) ** 2
                    + (turn_mismatch / 5.0) ** 2
                    + 0.03
                )
                step = _Step(action.piece, (state.x, state.z), end, samples)
                candidate = _State(
                    score,
                    end[0],
                    end[1],
                    end_heading,
                    progress,
                    (*state.steps, step),
                    state.curve_count + int(action.turn_sign != 0),
                )

                remaining = measure.total - progress
                end_error = math.dist(end, source_points[-1])
                boundary_error = _p._heading_difference(end_heading, exit_heading)
                if (
                    remaining <= _MAXIMUM_LOCKED_END_ERROR_METRES
                    and end_error <= _MAXIMUM_LOCKED_END_ERROR_METRES
                    and boundary_error <= maximum_boundary_tangent_error_degrees
                    and candidate.curve_count >= int(minimum_curve_count)
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
        seen: dict[tuple[int, int], int] = {}
        for candidate in candidates:
            key = (
                int(candidate.progress / 2.5),
                int(round(candidate.heading_degrees / 2.5)) % 144,
            )
            if seen.get(key, 0) >= 2:
                continue
            seen[key] = seen.get(key, 0) + 1
            beam_list.append(candidate)
            if len(beam_list) >= _BEAM_WIDTH:
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


def _piece_midpoint_distance(measure, start, end) -> float | None:
    midpoint = ((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5)
    nearest = _nearest_forward(measure, midpoint, 0.0, measure.total)
    return nearest[1] if nearest is not None else None


def _has_coherent_native_curve_run(
    fitted,
    measure,
    span_start: float,
    span_end: float,
    *,
    minimum_count: int = 2,
) -> bool:
    """True only for adjacent same-radius native curves inside one bend span."""

    run_key = None
    run_count = 0
    for piece, start, end in fitted:
        distance = _piece_midpoint_distance(measure, start, end)
        match = _model_geometry.stock_curve_match(str(piece.model_path))
        if (
            distance is None
            or distance < span_start - 0.5
            or distance > span_end + 0.5
            or match is None
        ):
            run_key = None
            run_count = 0
            continue
        key = (match.group("family").casefold(), int(match.group("radius")))
        if key == run_key:
            run_count += 1
        else:
            run_key = key
            run_count = 1
        if run_count >= minimum_count:
            return True
    return False


def _locked_measure(measure, pieces, baseline):
    spans = _sharp_turn_spans(measure.points)
    if not spans:
        return None
    replacements = []
    occupied_until = -1
    for start_index, end_index, turn_sign in spans:
        if start_index <= occupied_until or end_index + 1 >= len(measure.points):
            continue
        span_start = float(measure.cumulative[start_index])
        span_end = float(measure.cumulative[end_index])
        if _has_coherent_native_curve_run(baseline, measure, span_start, span_end):
            continue

        entry_heading = (
            _heading(measure.points[start_index - 1], measure.points[start_index])
            if start_index > 0
            else _heading(measure.points[start_index], measure.points[start_index + 1])
        )
        exit_heading = _heading(measure.points[end_index], measure.points[end_index + 1])
        source_span = measure.points[start_index : end_index + 1]
        locked = _beam_stock_path(source_span, turn_sign, entry_heading, exit_heading, pieces)
        if locked is None:
            continue
        replacements.append((start_index, end_index, locked))
        occupied_until = end_index

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


def _sharp_turn_chain(
    measure,
    pieces,
    *,
    start_distance,
    preferred_end_distance,
    minimum_end_distance,
    maximum_end_distance,
):
    if _ORIGINAL_CHAIN is None:
        raise RuntimeError("stock road sharp-turn policy is not installed")

    baseline = _ORIGINAL_CHAIN(
        measure,
        pieces,
        start_distance=start_distance,
        preferred_end_distance=preferred_end_distance,
        minimum_end_distance=minimum_end_distance,
        maximum_end_distance=maximum_end_distance,
    )
    if _curveable_family(pieces) is None:
        return baseline

    locked = _locked_measure(measure, pieces, baseline)
    if locked is None:
        return baseline

    tail_preferred = float(measure.total) - float(preferred_end_distance)
    tail_minimum = float(measure.total) - float(minimum_end_distance)
    tail_maximum = float(maximum_end_distance) - float(measure.total)
    new_start = min(float(locked.total), float(start_distance))
    new_preferred = max(new_start, float(locked.total) - tail_preferred)
    new_minimum = max(new_start, float(locked.total) - tail_minimum)
    new_maximum = max(new_preferred, float(locked.total) + tail_maximum)

    fitted = _ORIGINAL_CHAIN(
        locked,
        pieces,
        start_distance=new_start,
        preferred_end_distance=new_preferred,
        minimum_end_distance=new_minimum,
        maximum_end_distance=new_maximum,
    )
    if not fitted:
        return baseline
    locked_curves = sum(
        _model_geometry.stock_curve_match(str(piece.model_path)) is not None
        for piece, _start, _end in fitted
    )
    baseline_curves = sum(
        _model_geometry.stock_curve_match(str(piece.model_path)) is not None
        for piece, _start, _end in baseline
    )
    return fitted if locked_curves > baseline_curves else baseline


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

    family = _curveable_family(pieces)
    samples_per_action = _CURVE_SAMPLE_COUNT
    if (
        family is None
        or len(path) < 2
        or (len(path) - 1) % samples_per_action != 0
    ):
        return None
    actions = _actions(pieces, *family, turn_sign)
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
    if _ORIGINAL_EXACT_CHAIN is None:
        raise RuntimeError("stock road exact sharp-turn policy is not installed")

    baseline = _ORIGINAL_EXACT_CHAIN(
        measure,
        pieces,
        start_distance=start_distance,
        preferred_end_distance=preferred_end_distance,
        minimum_end_distance=minimum_end_distance,
        maximum_end_distance=maximum_end_distance,
    )
    if _curveable_family(pieces) is None:
        return baseline
    if measure.total > _MAXIMUM_EXACT_RUN_METRES:
        return baseline

    start_trim = float(start_distance)
    end_trim = float(measure.total) - float(preferred_end_distance)
    if (
        start_trim < _MINIMUM_ENDPOINT_TRIM_METRES
        or end_trim < _MINIMUM_ENDPOINT_TRIM_METRES
    ):
        return baseline
    if _baseline_short_straights(baseline) < _MINIMUM_BASELINE_SHORT_STRAIGHTS:
        return baseline

    for start_index, end_index, turn_sign in _sharp_turn_spans(measure.points):
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

        stock_exit_heading = _quantised_stock_exit_heading(
            entry_heading,
            source_exit_heading,
            turn_sign,
        )
        locked_path = _beam_stock_path(
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

        end_projection = _nearest_forward(
            measure,
            locked_path[-1],
            start,
            float(maximum_end_distance),
        )
        if end_projection is None:
            continue
        if end_projection[0] > _MAXIMUM_LOCKED_CORRIDOR_METRES + 1.0e-9:
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


def install_stock_road_sharp_turn_policy() -> None:
    global _ORIGINAL_CHAIN, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_CHAIN = _p._stock_piece_chain
    _p._stock_piece_chain = _sharp_turn_chain
    _INSTALLED = True


def install_stock_road_sharp_exact_policy() -> None:
    global _ORIGINAL_EXACT_CHAIN, _EXACT_INSTALLED
    if _EXACT_INSTALLED:
        return
    if not _INSTALLED:
        raise RuntimeError("stock road sharp-turn policy must install first")
    _ORIGINAL_EXACT_CHAIN = _p._stock_piece_chain
    _p._stock_piece_chain = _exact_sharp_turn_chain
    _EXACT_INSTALLED = True
