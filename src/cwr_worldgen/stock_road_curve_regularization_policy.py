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
exit tangents.  Accepted spans are resampled uniformly on that exact circle;
ordinary corners, S-bends and source wiggles continue through the existing
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
REGULARIZED_ARC_SAMPLE_DEGREES = 2.5

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
    sections = max(2, int(math.ceil(arc_degrees / REGULARIZED_ARC_SAMPLE_DEGREES)))
    origin_angle = math.atan2(float(start[1]) - centre[1], float(start[0]) - centre[0])
    angular_sign = -1.0 if turn_sign > 0 else 1.0
    arc_radians = math.radians(arc_degrees)
    result = [start]
    for sample in range(1, sections):
        angle = origin_angle + angular_sign * arc_radians * (sample / sections)
        result.append(
            (
                centre[0] + math.cos(angle) * radius,
                centre[1] + math.sin(angle) * radius,
            )
        )
    result.append(end)
    return tuple(result)


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
            _append_unique(
                result,
                _ORIGINAL_ROUNDED(cleaned[cursor : start_index + 1], **kwargs),
            )
        elif not result:
            _append_unique(result, (cleaned[start_index],))
        _append_unique(result, arc)
        cursor = end_index

    if cursor < len(cleaned) - 1:
        _append_unique(result, _ORIGINAL_ROUNDED(cleaned[cursor:], **kwargs))
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
