# SPDX-License-Identifier: GPL-3.0-or-later
"""Preserve sustained source curvature while removing stock-road line noise.

CWA stock road curves are rigid ten-degree P3Ds. A correctly sampled shallow
arc can lie only a few decimetres from its endpoint chord, which makes a generic
sub-metre simplifier liable to misclassify the arc as source noise. Once that
happens the curve selector never sees the curvature and has no choice but to
facet the run with short straight slabs.

Keep samples that participate in repeated same-direction turning. A one-off
three-point dog-leg is still ordinary simplification material; a sequence of
same-sign turns is a real curve signal and remains dense enough for native CWA
curve selection. Existing obstacle checks remain authoritative.
"""
from __future__ import annotations

from functools import lru_cache
import math

from . import playability as _p
from . import stock_road_geometry_policy as _geometry
from . import stock_road_path_conditioning_policy as _path
from . import stock_road_relaxation_policy as _relax

MINIMUM_CURVE_SIGNAL_TANGENT_CHANGE_DEGREES = 4.0
MINIMUM_CURVE_SIGNAL_DEVIATION_METRES = 0.12
MINIMUM_LOCAL_SUSTAINED_TURN_DEGREES = 0.75
CURVE_SIDE_EPSILON_METRES = 0.03

_ORIGINAL_RELAXABLE = None
_ORIGINAL_PATH_SHORTCUT_SAFE = None
_ORIGINAL_SIMPLIFY_MICRO_BENDS = None
_INSTALLED = False


def _heading(start, end) -> float:
    return math.degrees(math.atan2(end[0] - start[0], end[1] - start[1])) % 360.0


def _signed_turn(previous, point, following) -> float:
    return _p._signed_heading_delta(
        _heading(previous, point),
        _heading(point, following),
    )


def _signed_chord_offset(point, start, end) -> float:
    dx, dz = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dz)
    if length <= 1.0e-9:
        return 0.0
    return (dx * (point[1] - start[1]) - dz * (point[0] - start[0])) / length


def _candidate_is_sustained_curve(points, first: int, last: int) -> bool:
    """Return whether a shortcut would erase coherent repeated curvature."""

    if last <= first + 2:
        # One interior vertex is a dog-leg/corner, not a sustained curve run.
        return False
    start, end = points[first], points[last]
    if math.dist(start, end) <= 0.10:
        return False

    entry = _heading(points[first], points[first + 1])
    exit_heading = _heading(points[last - 1], points[last])
    tangent_change = _p._heading_difference(entry, exit_heading)
    if tangent_change < MINIMUM_CURVE_SIGNAL_TANGENT_CHANGE_DEGREES:
        return False

    positive = False
    negative = False
    maximum_offset = 0.0
    for index in range(first + 1, last):
        offset = _signed_chord_offset(points[index], start, end)
        maximum_offset = max(maximum_offset, abs(offset))
        if offset > CURVE_SIDE_EPSILON_METRES:
            positive = True
        elif offset < -CURVE_SIDE_EPSILON_METRES:
            negative = True
        if positive and negative:
            return False
    if maximum_offset < MINIMUM_CURVE_SIGNAL_DEVIATION_METRES:
        return False

    turn_sign = 0
    significant_turns = 0
    for index in range(first + 1, last):
        turn = _signed_turn(points[index - 1], points[index], points[index + 1])
        if abs(turn) < MINIMUM_LOCAL_SUSTAINED_TURN_DEGREES:
            continue
        sign = 1 if turn > 0.0 else -1
        if turn_sign and sign != turn_sign:
            return False
        turn_sign = sign
        significant_turns += 1
    return significant_turns >= 2


@lru_cache(maxsize=2048)
def _curve_anchor_points_cached(
    cleaned: tuple[tuple[float, float], ...],
) -> frozenset[tuple[float, float]]:
    """Return repeated-curvature samples once per immutable source polyline."""

    if len(cleaned) < 4:
        return frozenset()
    turns = [0.0] * len(cleaned)
    for index in range(1, len(cleaned) - 1):
        turns[index] = _signed_turn(cleaned[index - 1], cleaned[index], cleaned[index + 1])

    protected: set[tuple[float, float]] = set()
    for index in range(1, len(cleaned) - 1):
        current = turns[index]
        if abs(current) < MINIMUM_LOCAL_SUSTAINED_TURN_DEGREES:
            continue
        sign = 1 if current > 0.0 else -1
        neighbours = []
        if index > 1:
            neighbours.append(turns[index - 1])
        if index + 1 < len(cleaned) - 1:
            neighbours.append(turns[index + 1])
        if any(
            abs(value) >= MINIMUM_LOCAL_SUSTAINED_TURN_DEGREES
            and (1 if value > 0.0 else -1) == sign
            for value in neighbours
        ):
            protected.add(cleaned[index])
    return frozenset(protected)


def _curve_anchor_points(points) -> frozenset[tuple[float, float]]:
    """Protect samples participating in consecutive same-direction turns."""

    cleaned = tuple(_p._clean_road_points(points))
    return _curve_anchor_points_cached(cleaned)


def _candidate_contains_curve_anchor(points, first: int, last: int) -> bool:
    if last <= first + 1:
        return False
    anchors = _curve_anchor_points(points)
    return any(points[index] in anchors for index in range(first + 1, last))


def _candidate_is_relaxable(points, first: int, last: int, obstacles) -> bool:
    if _ORIGINAL_RELAXABLE is None:
        raise RuntimeError("stock road curve preservation policy is not installed")
    if (
        _candidate_is_sustained_curve(points, first, last)
        or _candidate_contains_curve_anchor(points, first, last)
    ):
        return False
    return _ORIGINAL_RELAXABLE(points, first, last, obstacles)


def _shortcut_is_safe(points, first, last, protected, obstacles) -> bool:
    if _ORIGINAL_PATH_SHORTCUT_SAFE is None:
        raise RuntimeError("stock road curve preservation policy is not installed")
    if (
        _candidate_is_sustained_curve(points, first, last)
        or _candidate_contains_curve_anchor(points, first, last)
    ):
        return False
    return _ORIGINAL_PATH_SHORTCUT_SAFE(points, first, last, protected, obstacles)


def _simplify_micro_bends(points):
    """Run the existing micro-bend cleanup without collapsing smooth curve runs."""

    if _ORIGINAL_SIMPLIFY_MICRO_BENDS is None:
        raise RuntimeError("stock road curve preservation policy is not installed")
    protected = _curve_anchor_points(points)
    if not protected:
        return _ORIGINAL_SIMPLIFY_MICRO_BENDS(points)

    result = list(_p._clean_road_points(points))
    changed = True
    while changed and len(result) >= 3:
        changed = False
        simplified = [result[0]]
        for index in range(1, len(result) - 1):
            previous, point, following = result[index - 1], result[index], result[index + 1]
            if point in protected:
                simplified.append(point)
                continue
            turn = _p._turn_degrees(previous, point, following)
            deviation = _geometry._point_segment_distance(point, previous, following)
            if (
                turn <= _geometry._MAXIMUM_MICRO_BEND_DEGREES
                and deviation <= _geometry._MAXIMUM_MICRO_BEND_DEVIATION_METRES
            ):
                changed = True
                continue
            simplified.append(point)
        simplified.append(result[-1])
        result = simplified
    return tuple(result)


def install_stock_road_curve_preservation_policy() -> None:
    global _ORIGINAL_RELAXABLE, _ORIGINAL_PATH_SHORTCUT_SAFE
    global _ORIGINAL_SIMPLIFY_MICRO_BENDS, _INSTALLED
    if _INSTALLED:
        return

    _ORIGINAL_RELAXABLE = _relax._candidate_is_relaxable
    _ORIGINAL_PATH_SHORTCUT_SAFE = _path._shortcut_is_safe
    _ORIGINAL_SIMPLIFY_MICRO_BENDS = _geometry._simplify_micro_bends

    _relax._candidate_is_relaxable = _candidate_is_relaxable
    _path._shortcut_is_safe = _shortcut_is_safe
    _geometry._simplify_micro_bends = _simplify_micro_bends
    _INSTALLED = True
