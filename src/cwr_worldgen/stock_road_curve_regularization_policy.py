# SPDX-License-Identifier: GPL-3.0-or-later
"""Preserve coherent stock-compatible arcs before local corner filleting.

The normalizer can quantize a smooth sampled curve enough that consecutive local
turns alternate between roughly ten degrees and nearly zero.  The ordinary
corner fillet then treats only the larger samples as separate bends and changes
the source into several unrelated radii.  Native CWA curve P3Ds still connect at
their centreline slots, but their painted borders visibly clip at those radius
changes.

Before the local fillet stage, detect only sustained same-direction curvature and
ask whether the affected source samples can lie on one verified stock radius
(25/50/75/100 m) without moving more than the existing 0.45 m fillet corridor.
The candidate must also keep source order on the arc and match both entry and
exit tangents.  When the observed turn is already close to a whole number of
native ten-degree sections, distribute the normalizer error across both arc
boundaries and snap to that exact stock angle.  This lets adjacent curve P3Ds
share both radius and tangent instead of leaving the final few degrees to short
faceted straights.

Ordinary corners, S-bends and source wiggles continue through the existing
rounding implementation unchanged.
"""
from __future__ import annotations

import math

from . import playability as _p
from . import stock_road_geometry_policy as _geometry
from . import stock_road_relaxation_policy as _relax

MINIMUM_SIGNIFICANT_CURVE_TURN_DEGREES = 0.70
MAXIMUM_SUSTAINED_CURVE_VERTEX_TURN_DEGREES = 18.0
MAXIMUM_CURVE_NOISE_GAP_VERTICES = 2
MINIMUM_SUSTAINED_CURVE_TOTAL_TURN_DEGREES = 15.0
MAXIMUM_REGULARIZED_ARC_DEGREES = 135.0
MAXIMUM_REGULARIZED_TANGENT_ERROR_DEGREES = 3.0
MAXIMUM_REGULARIZED_TURN_ERROR_DEGREES = 3.0
MAXIMUM_STOCK_ANGLE_SNAP_ERROR_DEGREES = 1.5
REGULARIZED_ARC_SAMPLE_DEGREES = 2.5
_FIXED_RADIUS_FIT_ITERATIONS = 8
_FIXED_RADIUS_MAXIMUM_STEP_METRES = 2.0

_ORIGINAL_ROUNDED = None
_INSTALLED = False


def _heading(start, end) -> float:
    return math.degrees(
        math.atan2(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
    ) % 360.0


def _signed_turn(previous, point, following) -> float:
    return _p._signed_heading_delta(
        _heading(previous, point),
        _heading(point, following),
    )


def _sustained_curve_spans(points):
    """Return candidate point-index spans and their source turn sign.

    A span is started only by significant gentle turns and may bridge a couple
    of almost-straight quantized samples.  A significant opposite turn or one
    hard local corner ends it immediately.  Circle/tangent validation later is
    deliberately stricter than this inexpensive signal detector.
    """

    if len(points) < 4:
        return ()
    turns = [0.0] * len(points)
    for index in range(1, len(points) - 1):
        turns[index] = _signed_turn(points[index - 1], points[index], points[index + 1])

    spans = []
    sign = 0
    first_significant = None
    last_significant = None
    significant_count = 0
    accumulated_turn = 0.0
    noise_gap = 0

    def finish():
        nonlocal sign, first_significant, last_significant
        nonlocal significant_count, accumulated_turn, noise_gap
        if (
            first_significant is not None
            and last_significant is not None
            and significant_count >= 2
            and abs(accumulated_turn) >= MINIMUM_SUSTAINED_CURVE_TOTAL_TURN_DEGREES
        ):
            spans.append((max(0, first_significant - 1), last_significant, sign))
        sign = 0
        first_significant = None
        last_significant = None
        significant_count = 0
        accumulated_turn = 0.0
        noise_gap = 0

    for index in range(1, len(points) - 1):
        turn = turns[index]
        magnitude = abs(turn)
        if magnitude > MAXIMUM_SUSTAINED_CURVE_VERTEX_TURN_DEGREES:
            finish()
            continue
        if magnitude >= MINIMUM_SIGNIFICANT_CURVE_TURN_DEGREES:
            current_sign = 1 if turn > 0.0 else -1
            if sign and current_sign != sign:
                finish()
            if not sign:
                sign = current_sign
                first_significant = index
            last_significant = index
            significant_count += 1
            accumulated_turn += turn
            noise_gap = 0
            continue
        if sign:
            accumulated_turn += turn
            noise_gap += 1
            if noise_gap > MAXIMUM_CURVE_NOISE_GAP_VERTICES:
                finish()

    finish()
    return tuple(spans)


def _circle_centres(start, end, radius: float):
    dx = float(end[0]) - float(start[0])
    dz = float(end[1]) - float(start[1])
    chord = math.hypot(dx, dz)
    if chord <= 1.0e-9 or chord > radius * 2.0 + 1.0e-9:
        return ()
    midpoint = (
        (float(start[0]) + float(end[0])) * 0.5,
        (float(start[1]) + float(end[1])) * 0.5,
    )
    height = math.sqrt(max(0.0, radius * radius - (chord * 0.5) ** 2))
    normal = (-dz / chord, dx / chord)
    return (
        (midpoint[0] + normal[0] * height, midpoint[1] + normal[1] * height),
        (midpoint[0] - normal[0] * height, midpoint[1] - normal[1] * height),
    )


def _directed_arc_degrees(point, centre, origin_angle: float, turn_sign: int) -> float:
    angle = math.atan2(float(point[1]) - centre[1], float(point[0]) - centre[0])
    if turn_sign > 0:
        delta = (origin_angle - angle) % (2.0 * math.pi)
    else:
        delta = (angle - origin_angle) % (2.0 * math.pi)
    return math.degrees(delta)


def _circle_tangent_heading(point, centre, turn_sign: int) -> float:
    radial_x = float(point[0]) - centre[0]
    radial_z = float(point[1]) - centre[1]
    # Positive source heading turn is clockwise in the world X/Z plane.
    angular_sign = -1.0 if turn_sign > 0 else 1.0
    tangent_x = angular_sign * -radial_z
    tangent_z = angular_sign * radial_x
    return math.degrees(math.atan2(tangent_x, tangent_z)) % 360.0


def _fit_fixed_radius_centre(points, radius: float, initial_centre):
    """Gauss-Newton fit of one known stock radius to source samples."""

    cx, cz = float(initial_centre[0]), float(initial_centre[1])
    for _ in range(_FIXED_RADIUS_FIT_ITERATIONS):
        hxx = hxz = hzz = 0.0
        gx = gz = 0.0
        for point in points:
            dx = cx - float(point[0])
            dz = cz - float(point[1])
            distance = math.hypot(dx, dz)
            if distance <= 1.0e-9:
                continue
            jx = dx / distance
            jz = dz / distance
            residual = distance - radius
            hxx += jx * jx
            hxz += jx * jz
            hzz += jz * jz
            gx += jx * residual
            gz += jz * residual
        determinant = hxx * hzz - hxz * hxz
        if abs(determinant) <= 1.0e-12:
            break
        move_x = (-gx * hzz + hxz * gz) / determinant
        move_z = (hxz * gx - hxx * gz) / determinant
        movement = math.hypot(move_x, move_z)
        if movement > _FIXED_RADIUS_MAXIMUM_STEP_METRES:
            scale = _FIXED_RADIUS_MAXIMUM_STEP_METRES / movement
            move_x *= scale
            move_z *= scale
        cx += move_x
        cz += move_z
        if math.hypot(move_x, move_z) <= 1.0e-8:
            break
    return cx, cz


def _uniform_arc(centre, radius: float, origin_angle: float, turn_sign: int, arc_degrees: float):
    sections = max(2, int(round(arc_degrees / REGULARIZED_ARC_SAMPLE_DEGREES)))
    angular_sign = -1.0 if turn_sign > 0 else 1.0
    arc_radians = math.radians(arc_degrees)
    result = []
    for sample in range(sections + 1):
        angle = origin_angle + angular_sign * arc_radians * (sample / sections)
        result.append(
            (
                centre[0] + math.cos(angle) * radius,
                centre[1] + math.sin(angle) * radius,
            )
        )
    return tuple(result)


def _snapped_stock_arc(
    source_span,
    radius: float,
    initial_centre,
    turn_sign: int,
    entry_heading: float,
    exit_heading: float,
    observed_turn: float,
    maximum_deviation: float,
):
    native_sections = max(1, int(round(observed_turn / _geometry.STOCK_CURVE_ANGLE_DEGREES)))
    snapped_turn = native_sections * _geometry.STOCK_CURVE_ANGLE_DEGREES
    if not (
        MINIMUM_SUSTAINED_CURVE_TOTAL_TURN_DEGREES
        <= snapped_turn
        <= MAXIMUM_REGULARIZED_ARC_DEGREES
    ):
        return None
    if abs(snapped_turn - observed_turn) > MAXIMUM_STOCK_ANGLE_SNAP_ERROR_DEGREES:
        return None

    centre = _fit_fixed_radius_centre(source_span, radius, initial_centre)
    radial_deviations = [abs(math.dist(point, centre) - radius) for point in source_span]
    if max(radial_deviations, default=0.0) > maximum_deviation + 1.0e-9:
        return None

    first_angle = math.atan2(
        float(source_span[0][1]) - centre[1], float(source_span[0][0]) - centre[0]
    )
    source_arc = _directed_arc_degrees(source_span[-1], centre, first_angle, turn_sign)
    slack = snapped_turn - source_arc
    # A slightly longer stock arc can distribute quantization error across both
    # boundaries.  Do not trim a meaningfully longer source arc to force a fit.
    if slack < -0.05:
        return None
    angular_sign = -1.0 if turn_sign > 0 else 1.0
    origin_angle = first_angle - angular_sign * math.radians(slack * 0.5)
    arc = _uniform_arc(centre, radius, origin_angle, turn_sign, snapped_turn)

    if math.dist(source_span[0], arc[0]) > maximum_deviation + 1.0e-9:
        return None
    if math.dist(source_span[-1], arc[-1]) > maximum_deviation + 1.0e-9:
        return None

    entry_error = _p._heading_difference(
        entry_heading, _circle_tangent_heading(arc[0], centre, turn_sign)
    )
    exit_error = _p._heading_difference(
        exit_heading, _circle_tangent_heading(arc[-1], centre, turn_sign)
    )
    if max(entry_error, exit_error) > MAXIMUM_REGULARIZED_TANGENT_ERROR_DEGREES:
        return None

    progress = [
        _directed_arc_degrees(point, centre, origin_angle, turn_sign)
        for point in source_span
    ]
    if any(following + 0.25 < previous for previous, following in zip(progress, progress[1:])):
        return None
    if any(value > snapped_turn + 0.25 for value in progress):
        return None
    return arc


def _regularized_stock_arc(points, start_index: int, end_index: int, turn_sign: int):
    if end_index <= start_index + 1:
        return None
    start = points[start_index]
    end = points[end_index]
    entry_heading = (
        _heading(points[start_index - 1], start)
        if start_index > 0
        else _heading(start, points[start_index + 1])
    )
    exit_heading = (
        _heading(end, points[end_index + 1])
        if end_index + 1 < len(points)
        else _heading(points[end_index - 1], end)
    )
    observed_turn = abs(_p._signed_heading_delta(entry_heading, exit_heading))
    if observed_turn < MINIMUM_SUSTAINED_CURVE_TOTAL_TURN_DEGREES:
        return None

    best = None
    source_span = points[start_index : end_index + 1]
    maximum_deviation = float(_geometry._MAXIMUM_STOCK_FILLET_DEVIATION_METRES)

    for radius in _geometry._STOCK_RADII_METRES:
        for centre in _circle_centres(start, end, float(radius)):
            origin_angle = math.atan2(
                float(start[1]) - centre[1], float(start[0]) - centre[0]
            )
            arc_degrees = _directed_arc_degrees(end, centre, origin_angle, turn_sign)
            if not (
                MINIMUM_SUSTAINED_CURVE_TOTAL_TURN_DEGREES
                <= arc_degrees
                <= MAXIMUM_REGULARIZED_ARC_DEGREES
            ):
                continue

            entry_tangent = _circle_tangent_heading(start, centre, turn_sign)
            exit_tangent = _circle_tangent_heading(end, centre, turn_sign)
            entry_error = _p._heading_difference(entry_heading, entry_tangent)
            exit_error = _p._heading_difference(exit_heading, exit_tangent)
            tangent_error = max(entry_error, exit_error)
            turn_error = abs(arc_degrees - observed_turn)
            if tangent_error > MAXIMUM_REGULARIZED_TANGENT_ERROR_DEGREES:
                continue
            if turn_error > MAXIMUM_REGULARIZED_TURN_ERROR_DEGREES:
                continue

            deviations = [
                abs(math.dist(point, centre) - float(radius)) for point in source_span
            ]
            if max(deviations, default=0.0) > maximum_deviation + 1.0e-9:
                continue

            progress = [
                _directed_arc_degrees(point, centre, origin_angle, turn_sign)
                for point in source_span
            ]
            if any(
                following + 0.25 < previous
                for previous, following in zip(progress, progress[1:])
            ):
                continue
            if any(value > arc_degrees + 0.25 for value in progress):
                continue

            rms = math.sqrt(
                sum(value * value for value in deviations) / max(1, len(deviations))
            )
            score = (
                max(deviations, default=0.0),
                rms,
                tangent_error,
                turn_error,
                -float(radius),
            )
            if best is None or score < best[0]:
                best = (score, centre, float(radius), arc_degrees)

    if best is None:
        return None

    _score, centre, radius, arc_degrees = best
    snapped = _snapped_stock_arc(
        source_span,
        radius,
        centre,
        turn_sign,
        entry_heading,
        exit_heading,
        observed_turn,
        maximum_deviation,
    )
    if snapped is not None:
        return snapped

    origin_angle = math.atan2(float(start[1]) - centre[1], float(start[0]) - centre[0])
    return _uniform_arc(centre, radius, origin_angle, turn_sign, arc_degrees)


def _append_unique(target, values) -> None:
    for point in values:
        point = (float(point[0]), float(point[1]))
        if not target or math.dist(target[-1], point) > 0.05:
            target.append(point)


def _curve_regularized_rounded_run(points, **kwargs):
    if _ORIGINAL_ROUNDED is None:
        raise RuntimeError("stock road curve regularization policy is not installed")
    cleaned = tuple(_p._clean_road_points(points))
    if len(cleaned) < 4:
        return _ORIGINAL_ROUNDED(points, **kwargs)

    accepted = []
    occupied_until = -1
    for start_index, end_index, turn_sign in _sustained_curve_spans(cleaned):
        if start_index <= occupied_until:
            continue
        arc = _regularized_stock_arc(cleaned, start_index, end_index, turn_sign)
        if arc is None:
            continue
        accepted.append((start_index, end_index, arc))
        occupied_until = end_index
    if not accepted:
        return _ORIGINAL_ROUNDED(points, **kwargs)

    result = []
    cursor = 0
    for start_index, end_index, arc in accepted:
        if start_index > cursor:
            chunk = list(cleaned[cursor : start_index + 1])
            if result:
                chunk[0] = result[-1]
            chunk[-1] = arc[0]
            _append_unique(result, _ORIGINAL_ROUNDED(tuple(chunk), **kwargs))
        elif not result:
            _append_unique(result, (arc[0],))
        _append_unique(result, arc)
        cursor = end_index

    if cursor < len(cleaned) - 1:
        chunk = [result[-1], *cleaned[cursor + 1 :]]
        _append_unique(result, _ORIGINAL_ROUNDED(tuple(chunk), **kwargs))
    return tuple(result)


def install_stock_road_curve_regularization_policy() -> None:
    global _ORIGINAL_ROUNDED, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_ROUNDED = _relax._ORIGINAL_ROUNDED
    if _ORIGINAL_ROUNDED is None:
        raise RuntimeError("stock road relaxation policy must be installed first")
    _relax._ORIGINAL_ROUNDED = _curve_regularized_rounded_run
    _INSTALLED = True
