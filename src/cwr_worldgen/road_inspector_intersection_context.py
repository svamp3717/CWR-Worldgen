# SPDX-License-Identifier: GPL-3.0-or-later
"""Improve source-intersection matching in the read-only Road Inspector.

Real fitted approaches often stop a few metres short of the logical OSM node
because a stock T/X mesh or a low intersection cap owns the centre.  Requiring
an emitted endpoint to lie within less than a metre of the source node therefore
manufactures large heading errors on otherwise-correct intersections.

For diagnostics only, identify the inner endpoint of an ordinary stock road that
is clearly aimed at a nearby normalized junction and snap that endpoint to the
logical node before the existing source/intersection audit runs.  No WRP object,
source feature, fitter, or generator state is modified.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import road_inspector as _core


MAXIMUM_APPROACH_INSET_METRES = 6.50
MAXIMUM_APPROACH_ALIGNMENT_ERROR_DEGREES = 30.0
MINIMUM_APPROACH_CENTER_DISTANCE_METRES = 1.50
MINIMUM_CENTER_BEYOND_ENDPOINT_METRES = 0.25

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

    # Raw RoadEndpoint.outward_heading_degrees points away from the piece.  At
    # an inner approach endpoint that direction points toward the intersection,
    # so node -> piece is its opposite.  The runtime layer performs that +180
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


def _source_intersection_issues(roads, junctions, *, match_tolerance):
    if _ORIGINAL_SOURCE_INTERSECTION_ISSUES is None:
        raise RuntimeError("Road Inspector intersection context is not installed")
    if not junctions:
        return _ORIGINAL_SOURCE_INTERSECTION_ISSUES(
            roads,
            junctions,
            match_tolerance=match_tolerance,
        )

    adjusted = []
    for road in roads:
        # Native junction connectors are already measured at their physical
        # Memory-LOD positions.  Moving them would hide the very mismatch this
        # inspector is supposed to diagnose.
        if road.kind in {"junction_t", "junction_x"}:
            adjusted.append(road)
            continue
        endpoints = tuple(
            _snapped_endpoint(road, endpoint, junctions)
            for endpoint in road.endpoints
        )
        adjusted.append(replace(road, endpoints=endpoints))

    return _ORIGINAL_SOURCE_INTERSECTION_ISSUES(
        tuple(adjusted),
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
