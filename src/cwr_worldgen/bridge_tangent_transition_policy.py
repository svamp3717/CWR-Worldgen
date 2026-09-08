# SPDX-License-Identifier: GPL-3.0-or-later
"""Match stock bridge abutments to the tangent of the fitted paved road.

The bridge deck can be at exactly the right height while still exposing a
wall-like concrete corner when the adjoining road reaches the abutment at a
different heading. A straight 6 m filler closes the centreline gap but cannot
remove that angular mismatch.

Use the stock 10-degree paved-road curve models for the final dry-land handoff
when one turn can closely match the fitted road tangent. The curve starts
exactly at the bridge centreline endpoint with the bridge tangent, then overlaps
the existing fitted road where their tangents nearly agree. Small residual
heading differences still use the established straight filler.

This policy also corrects the stock straight connector lengths used by the
abutment geometry: sil/asf/kos ``6`` and ``12`` models are 6.25 m and 12.5 m,
not 6.0 m and 12.0 m.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import bridge_final_alignment_policy as _final
from . import paved_junction_policy as _paved
from .model import WorldObject

_INSTALLED = False
_ORIGINAL_ADD_BRIDGE_APPROACH_FILLERS = None
_ORIGINAL_STRAIGHT_ROAD_NOMINAL_LENGTH = None

_STRAIGHT_MODEL_LENGTHS = {
    6: 6.25,
    12: 12.5,
    25: 25.0,
}
_CURVE_RADII_METRES = (25, 50, 75, 100)
_CURVE_MINIMUM_HEADING_DELTA_DEGREES = 6.0
_CURVE_MAXIMUM_HEADING_DELTA_DEGREES = 14.0
_CURVE_MAXIMUM_RESIDUAL_DEGREES = 3.0
_CURVE_MAXIMUM_JOIN_DISTANCE_METRES = 1.25
_STRAIGHT_MAXIMUM_HEADING_DELTA_DEGREES = 6.0


def _signed_heading_delta(first: float, second: float) -> float:
    """Return signed shortest turn from ``first`` heading to ``second``."""
    return (float(second) - float(first) + 180.0) % 360.0 - 180.0


def _heading(direction: tuple[float, float]) -> float:
    return math.degrees(math.atan2(float(direction[0]), float(direction[1]))) % 360.0


def _actual_straight_road_length(model_path: str) -> float | None:
    filename = str(model_path).replace("/", "\\").rsplit("\\", 1)[-1]
    match = _final._STRAIGHT_ROAD_LENGTH_RE.search(filename)
    if match is None:
        return None
    nominal = int(match.group("length"))
    return _STRAIGHT_MODEL_LENGTHS.get(nominal)


def _road_near_far(candidate, bridge_point):
    endpoints = _final._road_endpoints(candidate)
    if endpoints is None:
        return None
    first, second = endpoints
    first_distance = math.dist(
        (float(first[0]), float(first[2])),
        (float(bridge_point[0]), float(bridge_point[1])),
    )
    second_distance = math.dist(
        (float(second[0]), float(second[2])),
        (float(bridge_point[0]), float(bridge_point[1])),
    )
    return (first, second) if first_distance <= second_distance else (second, first)


def _point_segment_measure(point, start, end):
    px, pz = float(point[0]), float(point[1])
    ax, az = float(start[0]), float(start[2])
    bx, bz = float(end[0]), float(end[2])
    dx, dz = bx - ax, bz - az
    length2 = dx * dx + dz * dz
    if length2 <= 1.0e-12:
        return math.hypot(px - ax, pz - az), 0.0
    fraction = ((px - ax) * dx + (pz - az) * dz) / length2
    fraction = max(0.0, min(1.0, fraction))
    qx = ax + dx * fraction
    qz = az + dz * fraction
    return math.hypot(px - qx, pz - qz), fraction


def _curve_object_from_abutment(
    object_id: int,
    family: str,
    radius: int,
    bridge_point: tuple[float, float],
    outward_heading: float,
    turn_sign: int,
    deck_y: float,
):
    """Place one stock 10-degree curve with its first connector at the bridge."""
    begin, end = _paved._curve_points(family, float(radius))
    if turn_sign > 0:
        local_start, local_end = begin, end
        yaw = float(outward_heading)
        next_heading = float(outward_heading) + float(_paved._TURN_DEGREES)
    else:
        local_start, local_end = end, begin
        yaw = float(outward_heading) - (180.0 + float(_paved._TURN_DEGREES))
        next_heading = float(outward_heading) - float(_paved._TURN_DEGREES)

    sx, sz = _paved._rotate(local_start, yaw)
    origin_x = float(bridge_point[0]) - sx
    origin_z = float(bridge_point[1]) - sz
    ex, ez = _paved._rotate(local_end, yaw)
    finish = (origin_x + ex, origin_z + ez)

    obj = WorldObject(
        int(object_id),
        rf"o\road\{family}10 {int(radius)}.p3d",
        origin_x,
        float(deck_y),
        origin_z,
        yaw % 360.0,
        0.0,
    )
    return obj, finish, next_heading % 360.0


def _tangent_curve_filler(
    object_id: int,
    bridge_point: tuple[float, float],
    outward: tuple[float, float],
    candidate,
    bridge_deck_y: float,
):
    """Return the best one-curve tangent transition, or ``None``."""
    family = _paved._family(candidate.model_path)
    if family not in _paved._WIDTH:
        return None

    near_far = _road_near_far(candidate, bridge_point)
    if near_far is None:
        return None
    near, far = near_far

    outward_heading = _heading(outward)
    road_heading = math.degrees(
        math.atan2(
            float(far[0]) - float(near[0]),
            float(far[2]) - float(near[2]),
        )
    ) % 360.0
    delta = _signed_heading_delta(outward_heading, road_heading)
    magnitude = abs(delta)
    if not (
        _CURVE_MINIMUM_HEADING_DELTA_DEGREES
        <= magnitude
        <= _CURVE_MAXIMUM_HEADING_DELTA_DEGREES
    ):
        return None

    turn_sign = 1 if delta > 0.0 else -1
    expected_heading = (
        outward_heading + turn_sign * float(_paved._TURN_DEGREES)
    ) % 360.0
    residual = abs(_signed_heading_delta(expected_heading, road_heading))
    if residual > _CURVE_MAXIMUM_RESIDUAL_DEGREES:
        return None

    if (
        abs(float(bridge_deck_y) - float(near[1]))
        > _final._APPROACH_MAX_VERTICAL_MISMATCH_METRES
    ):
        return None

    best = None
    for radius in _CURVE_RADII_METRES:
        obj, finish, final_heading = _curve_object_from_abutment(
            object_id,
            family,
            radius,
            bridge_point,
            outward_heading,
            turn_sign,
            bridge_deck_y,
        )
        join_distance, fraction = _point_segment_measure(finish, near, far)
        score = (join_distance, abs(float(radius)), radius)
        if best is None or score < best[0]:
            best = (
                score,
                obj,
                finish,
                final_heading,
                join_distance,
                fraction,
                road_heading,
            )

    if best is None or best[4] > _CURVE_MAXIMUM_JOIN_DISTANCE_METRES:
        return None
    return best[1:]


def _small_heading_straight_filler(
    object_id: int,
    bridge_point: tuple[float, float],
    outward: tuple[float, float],
    bridge_heading: float,
    candidate,
    candidate_near,
    gap: float,
    bridge_deck_y: float,
):
    outward_heading = _heading(outward)
    near_far = _road_near_far(candidate, bridge_point)
    if near_far is None:
        return None
    near, far = near_far
    road_heading = math.degrees(
        math.atan2(
            float(far[0]) - float(near[0]),
            float(far[2]) - float(near[2]),
        )
    ) % 360.0
    if (
        abs(_signed_heading_delta(outward_heading, road_heading))
        > _STRAIGHT_MAXIMUM_HEADING_DELTA_DEGREES
    ):
        return None
    return _final._approach_filler(
        object_id,
        bridge_point,
        outward,
        bridge_heading,
        candidate,
        candidate_near,
        gap,
        bridge_deck_y,
    )


def _add_bridge_approach_fillers(report, spans, elevations, spec):
    """Join each abutment with a tangent curve when the fitted road turns."""
    if not spans or not getattr(report, "objects", ()):
        return report, 0

    objects = list(report.objects)
    road_objects = tuple(objects)
    next_id = max((int(obj.object_id) for obj in objects), default=0) + 1
    added = 0

    for span in spans:
        if len(span.points) < 2:
            continue
        start = (float(span.points[0][0]), float(span.points[0][1]))
        end = (float(span.points[-1][0]), float(span.points[-1][1]))
        dx = end[0] - start[0]
        dz = end[1] - start[1]
        length = math.hypot(dx, dz)
        if length <= 0.1:
            continue

        axis = (dx / length, dz / length)
        bridge_heading = _heading(axis)

        for bridge_point, outward in (
            (start, (-axis[0], -axis[1])),
            (end, axis),
        ):
            selected = _final._approach_candidate(
                bridge_point,
                bridge_heading,
                road_objects,
            )
            if selected is None:
                continue
            candidate, candidate_near, gap = selected
            deck_y = _final._bridge_endpoint_deck_height(
                bridge_point,
                outward,
                float(candidate_near[1]),
                elevations,
                spec,
            )

            curved = _tangent_curve_filler(
                next_id,
                bridge_point,
                outward,
                candidate,
                deck_y,
            )
            if curved is not None:
                filler = curved[0]
            else:
                filler = _small_heading_straight_filler(
                    next_id,
                    bridge_point,
                    outward,
                    bridge_heading,
                    candidate,
                    candidate_near,
                    gap,
                    deck_y,
                )

            if filler is None:
                continue
            objects.append(filler)
            next_id += 1
            added += 1

    if not added:
        return report, 0

    first_id = min(int(obj.object_id) for obj in objects)
    renumbered = tuple(
        replace(obj, object_id=first_id + index)
        for index, obj in enumerate(objects)
    )
    return replace(report, objects=renumbered), added


def install_bridge_tangent_transition_policy() -> None:
    """Replace straight-only bridge fillers with tangent-matched stock curves."""
    global _INSTALLED
    global _ORIGINAL_ADD_BRIDGE_APPROACH_FILLERS
    global _ORIGINAL_STRAIGHT_ROAD_NOMINAL_LENGTH

    if _INSTALLED:
        return

    _ORIGINAL_ADD_BRIDGE_APPROACH_FILLERS = _final._add_bridge_approach_fillers
    _ORIGINAL_STRAIGHT_ROAD_NOMINAL_LENGTH = _final._straight_road_nominal_length

    _final._straight_road_nominal_length = _actual_straight_road_length
    _final._APPROACH_FILL_NOMINAL_LENGTH_METRES = _STRAIGHT_MODEL_LENGTHS[6]
    _final._add_bridge_approach_fillers = _add_bridge_approach_fillers
    _INSTALLED = True
