# SPDX-License-Identifier: GPL-3.0-or-later
"""Improve source-intersection matching in the read-only Road Inspector.

Real fitted approaches often stop a few metres short of the logical OSM node
because a stock T/X mesh or a low intersection cap owns the centre. Requiring an
emitted endpoint to lie within less than a metre of the source node therefore
manufactures large heading errors on otherwise-correct intersections.

For diagnostics only, identify both ordinary stock endpoints clearly aimed at a
nearby normalized junction and straight stock pieces whose physical axis actually
crosses a low-cap junction. The latter matters when a fitted piece spans the
logical node with both connectors several metres away. The selected central cap
is explicitly excluded so its own straight axis cannot masquerade as a missing
approach. No WRP object, source feature, fitter, or generator state is modified.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import road_inspector as _core


MAXIMUM_APPROACH_INSET_METRES = 6.50
MAXIMUM_APPROACH_ALIGNMENT_ERROR_DEGREES = 30.0
MINIMUM_APPROACH_CENTER_DISTANCE_METRES = 1.50
MINIMUM_CENTER_BEYOND_ENDPOINT_METRES = 0.25
MAXIMUM_STRAIGHT_AXIS_JUNCTION_DISTANCE_METRES = 0.50
MINIMUM_CROSSING_ENDPOINT_DISTANCE_METRES = 0.90

_ORIGINAL_SOURCE_INTERSECTION_ISSUES = None
_INSTALLED = False


def _heading(start: tuple[float, float], end: tuple[float, float]) -> float:
    return math.degrees(
        math.atan2(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
    ) % 360.0


def _nearest_source_junction(point, junctions):
    best = None
    for junction in junctions:
        distance = math.dist(point, junction.point)
        if distance > MAXIMUM_APPROACH_INSET_METRES:
            continue
        candidate = (distance, junction.point[0], junction.point[1], junction)
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    return None if best is None else (best[0], best[3])


def _snapped_endpoint(road, endpoint, junctions):
    """Return a diagnostic endpoint at its logical node when geometry proves it."""

    nearest = _nearest_source_junction(endpoint.point, junctions)
    if nearest is None:
        return endpoint
    endpoint_distance, junction = nearest
    if endpoint_distance <= _core.DEFAULT_JUNCTION_MATCH_TOLERANCE_METRES:
        return endpoint

    center_distance = math.dist(road.logical_center, junction.point)
    if center_distance < MINIMUM_APPROACH_CENTER_DISTANCE_METRES:
        return endpoint
    if center_distance < endpoint_distance + MINIMUM_CENTER_BEYOND_ENDPOINT_METRES:
        return endpoint

    # Raw RoadEndpoint.outward_heading_degrees points away from the piece. At an
    # inner approach endpoint that direction points toward the intersection, so
    # node -> piece is its opposite. The runtime layer performs that +180
    # correction before comparing with source headings; use the same direction
    # here only to prove this endpoint really belongs to this source node.
    direction_into_piece = (float(endpoint.outward_heading_degrees) + 180.0) % 360.0
    node_to_endpoint = _heading(junction.point, endpoint.point)
    if (
        _core._angular_distance(direction_into_piece, node_to_endpoint)
        > MAXIMUM_APPROACH_ALIGNMENT_ERROR_DEGREES
    ):
        return endpoint

    return replace(endpoint, point=(float(junction.point[0]), float(junction.point[1])))


def _selected_cap(roads, junction, match_tolerance: float):
    """Mirror the core Inspector's cap choice for one normalized junction."""

    candidates = [
        road
        for road in roads
        if (
            road.kind in {"junction_t", "junction_x", "paved_fill"}
            or (
                road.kind == "straight"
                and float(road.nominal_length_metres) <= 6.26
            )
        )
        and math.dist(road.logical_center, junction.point) <= match_tolerance
    ]
    return min(
        candidates,
        key=lambda road: math.dist(road.logical_center, junction.point),
        default=None,
    )


def _point_segment_projection(point, start, end):
    dx = float(end[0]) - float(start[0])
    dz = float(end[1]) - float(start[1])
    denominator = dx * dx + dz * dz
    if denominator <= 1.0e-12:
        return math.dist(point, start), 0.0
    fraction = (
        (float(point[0]) - float(start[0])) * dx
        + (float(point[1]) - float(start[1])) * dz
    ) / denominator
    projected = (
        float(start[0]) + dx * fraction,
        float(start[1]) + dz * fraction,
    )
    return math.dist(point, projected), fraction


def _crossing_junction(road, junctions, selected_caps, match_tolerance: float):
    """Return a low-cap junction physically crossed inside one stock straight."""

    if road.kind != "straight" or len(road.endpoints) != 2:
        return None
    first, second = road.endpoints
    if (
        math.dist(first.point, second.point) <= 1.0e-6
        or not math.isfinite(float(road.nominal_length_metres))
    ):
        return None

    limit = min(
        float(match_tolerance),
        MAXIMUM_STRAIGHT_AXIS_JUNCTION_DISTANCE_METRES,
    )
    best = None
    for junction in junctions:
        cap = selected_caps.get(id(junction))
        # Only generic central-fill nodes need this diagnostic completion. Native
        # T/X connectors are measured and must remain strict.
        if cap is None or cap.kind not in {"straight", "paved_fill"}:
            continue
        if int(cap.object_id) == int(road.object_id):
            continue
        distance, fraction = _point_segment_projection(
            junction.point,
            first.point,
            second.point,
        )
        if not 0.0 < fraction < 1.0 or distance > limit:
            continue
        if min(
            math.dist(first.point, junction.point),
            math.dist(second.point, junction.point),
        ) < MINIMUM_CROSSING_ENDPOINT_DISTANCE_METRES:
            # Ordinary near-node endpoints are already handled by the inset
            # matcher. Keep this path specifically for an interior crossing.
            continue
        candidate = (
            distance,
            abs(0.5 - fraction),
            junction.point[0],
            junction.point[1],
            junction,
        )
        if best is None or candidate[:4] < best[:4]:
            best = candidate
    return None if best is None else best[4]


def _crossing_axis_copy(road, junction):
    """Expose both directions of one physically crossing straight at the node."""

    point = (float(junction.point[0]), float(junction.point[1]))
    endpoints = tuple(replace(endpoint, point=point) for endpoint in road.endpoints)
    return replace(road, endpoints=endpoints)


def _source_intersection_issues(roads, junctions, *, match_tolerance):
    if _ORIGINAL_SOURCE_INTERSECTION_ISSUES is None:
        raise RuntimeError("Road Inspector intersection context is not installed")
    if not junctions:
        return _ORIGINAL_SOURCE_INTERSECTION_ISSUES(
            roads,
            junctions,
            match_tolerance=match_tolerance,
        )

    selected_caps = {
        id(junction): _selected_cap(roads, junction, match_tolerance)
        for junction in junctions
    }
    adjusted = []
    crossing_copies = []
    for road in roads:
        # Native junction connectors are already measured at their physical
        # Memory-LOD positions. Moving them would hide the very mismatch this
        # inspector is supposed to diagnose.
        if road.kind in {"junction_t", "junction_x"}:
            adjusted.append(road)
            continue
        endpoints = tuple(
            _snapped_endpoint(road, endpoint, junctions)
            for endpoint in road.endpoints
        )
        adjusted.append(replace(road, endpoints=endpoints))

        crossing = _crossing_junction(
            road,
            junctions,
            selected_caps,
            match_tolerance,
        )
        if crossing is not None:
            crossing_copies.append(_crossing_axis_copy(road, crossing))

    return _ORIGINAL_SOURCE_INTERSECTION_ISSUES(
        tuple((*adjusted, *crossing_copies)),
        junctions,
        match_tolerance=match_tolerance,
    )


def install() -> None:
    """Install only into Road Inspector diagnostics, never normal generation."""

    global _ORIGINAL_SOURCE_INTERSECTION_ISSUES, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_SOURCE_INTERSECTION_ISSUES = _core._source_intersection_issues
    _core._source_intersection_issues = _source_intersection_issues
    _INSTALLED = True
