# SPDX-License-Identifier: GPL-3.0-or-later
"""Suppress paved seam alarms caused by intentional overlap/repair geometry.

Road Inspector audits raw stock connectors, but the final generator may place
low same-family six-metre underlays beneath a visible paved seam. Lundby25 first
exposed nearly coincident duplicate pieces; Lundby26 then exposed two further
read-only diagnostic cases:

* adjacent dual-underlay fans can be paired with each other even though both are
  below the visible road surface; and
* a third aligned paved straight can already bridge both reported connectors.

Keep these filters deliberately geometric and paved-only. Dirt/mixed-family
findings remain visible, and a nearby road is never enough by itself: bridge
suppression requires one same-family straight to contain both reported connector
points and the finding point while sharing the involved road axis.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

from . import road_inspector as _core

_OVERLAP_CATEGORIES = frozenset({"straight_miter", "connector_gap", "curve_transition"})
_PAVED_FAMILIES = frozenset({"sil", "asf", "kos"})
_MAXIMUM_CENTRE_DISTANCE_METRES = 0.50
_MAXIMUM_AXIS_ERROR_DEGREES = 2.0
_MAXIMUM_VERTICAL_DISTANCE_METRES = 0.15
_DUAL_UNDERLAY_CENTRE_TOLERANCE_METRES = 0.03
_DUAL_UNDERLAY_MINIMUM_AXIS_ERROR_DEGREES = 2.0
_DUAL_UNDERLAY_MAXIMUM_AXIS_ERROR_DEGREES = 35.0
_DUAL_UNDERLAY_VERTICAL_TOLERANCE_METRES = 0.03
_BRIDGE_AXIS_ERROR_DEGREES = 5.0
_BRIDGE_VERTICAL_TOLERANCE_METRES = 0.25
_BRIDGE_SURFACE_MARGIN_METRES = 0.10

_ORIGINAL_INSPECT = None
_INSTALLED = False


def _axis_error(first: float, second: float) -> float:
    difference = abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)
    return min(difference, abs(180.0 - difference))


def _paved_straight(road) -> bool:
    return (
        road is not None
        and road.family in _PAVED_FAMILIES
        and road.kind == "straight"
    )


def _issue_roads(issue, roads):
    if issue.category not in _OVERLAP_CATEGORIES or len(issue.object_ids) != 2:
        return None
    road_by_id = {int(road.object_id): road for road in roads}
    first = road_by_id.get(int(issue.object_ids[0]))
    second = road_by_id.get(int(issue.object_ids[1]))
    if not _paved_straight(first) or not _paved_straight(second):
        return None
    if first.family != second.family:
        return None
    return first, second


def _overlapping_paved_pair(issue, roads) -> bool:
    involved = _issue_roads(issue, roads)
    if involved is None:
        return False
    first, second = involved
    if (
        math.dist(
            (float(first.x), float(first.z)),
            (float(second.x), float(second.z)),
        )
        > _MAXIMUM_CENTRE_DISTANCE_METRES
    ):
        return False
    if _axis_error(first.heading_degrees, second.heading_degrees) > _MAXIMUM_AXIS_ERROR_DEGREES:
        return False
    return abs(float(first.y) - float(second.y)) <= _MAXIMUM_VERTICAL_DISTANCE_METRES


def _dual_underlay_member(road, roads) -> bool:
    """Return whether a short paved straight has a co-centred tangent sibling."""

    if not _paved_straight(road):
        return False
    # Final repair fans are ordinary six-metre stock straights. Longer visible
    # road pieces must never acquire this diagnostic-only classification.
    if float(getattr(road, "nominal_length_metres", 0.0)) > 6.30:
        return False
    for other in roads:
        if int(other.object_id) == int(road.object_id):
            continue
        if not _paved_straight(other) or other.family != road.family:
            continue
        if float(getattr(other, "nominal_length_metres", 0.0)) > 6.30:
            continue
        if (
            math.dist(
                (float(road.x), float(road.z)),
                (float(other.x), float(other.z)),
            )
            > _DUAL_UNDERLAY_CENTRE_TOLERANCE_METRES
        ):
            continue
        axis_error = _axis_error(road.heading_degrees, other.heading_degrees)
        if not (
            _DUAL_UNDERLAY_MINIMUM_AXIS_ERROR_DEGREES
            <= axis_error
            <= _DUAL_UNDERLAY_MAXIMUM_AXIS_ERROR_DEGREES
        ):
            continue
        if (
            abs(float(road.y) - float(other.y))
            > _DUAL_UNDERLAY_VERTICAL_TOLERANCE_METRES
        ):
            continue
        return True
    return False


def _dual_underlay_seam(issue, roads) -> bool:
    involved = _issue_roads(issue, roads)
    if involved is None:
        return False
    first, second = involved
    return _dual_underlay_member(first, roads) and _dual_underlay_member(second, roads)


def _nearest_endpoint(road, point: tuple[float, float]):
    if len(road.endpoints) != 2:
        return None
    return min(road.endpoints, key=lambda endpoint: math.dist(endpoint.point, point))


def _straight_contains(road, point: tuple[float, float]) -> bool:
    if not _paved_straight(road) or len(road.endpoints) != 2:
        return False
    start = road.endpoints[0].point
    end = road.endpoints[1].point
    dx = float(end[0]) - float(start[0])
    dz = float(end[1]) - float(start[1])
    length = math.hypot(dx, dz)
    if length <= 1.0e-9:
        return False
    ux, uz = dx / length, dz / length
    px = float(point[0]) - float(start[0])
    pz = float(point[1]) - float(start[1])
    along = px * ux + pz * uz
    lateral = abs(px * -uz + pz * ux)
    half_width = float(road.endpoints[0].half_width_metres)
    return (
        -_BRIDGE_SURFACE_MARGIN_METRES
        <= along
        <= length + _BRIDGE_SURFACE_MARGIN_METRES
        and lateral <= half_width + _BRIDGE_SURFACE_MARGIN_METRES
    )


def _bridged_by_aligned_paved_straight(issue, roads) -> bool:
    """Suppress a raw connector pair only when one third road bridges it."""

    involved = _issue_roads(issue, roads)
    if involved is None:
        return False
    first, second = involved
    if (
        _axis_error(first.heading_degrees, second.heading_degrees)
        > _BRIDGE_AXIS_ERROR_DEGREES * 2.0
    ):
        return False

    issue_point = (float(issue.x), float(issue.z))
    first_endpoint = _nearest_endpoint(first, issue_point)
    second_endpoint = _nearest_endpoint(second, issue_point)
    if first_endpoint is None or second_endpoint is None:
        return False
    required_points = (first_endpoint.point, second_endpoint.point, issue_point)

    involved_ids = {int(first.object_id), int(second.object_id)}
    for candidate in roads:
        if int(candidate.object_id) in involved_ids:
            continue
        if not _paved_straight(candidate) or candidate.family != first.family:
            continue
        if (
            _axis_error(candidate.heading_degrees, first.heading_degrees)
            > _BRIDGE_AXIS_ERROR_DEGREES
            or _axis_error(candidate.heading_degrees, second.heading_degrees)
            > _BRIDGE_AXIS_ERROR_DEGREES
        ):
            continue
        if (
            abs(float(candidate.y) - float(first.y))
            > _BRIDGE_VERTICAL_TOLERANCE_METRES
            or abs(float(candidate.y) - float(second.y))
            > _BRIDGE_VERTICAL_TOLERANCE_METRES
        ):
            continue
        if all(_straight_contains(candidate, point) for point in required_points):
            return True
    return False


def _covered_repair_geometry(issue, roads) -> bool:
    return (
        _overlapping_paved_pair(issue, roads)
        or _dual_underlay_seam(issue, roads)
        or _bridged_by_aligned_paved_straight(issue, roads)
    )


def inspect_road_geometry(
    input_path: Path,
    *,
    roads_geojson: Path | None = None,
    endpoint_tolerance: float = _core.DEFAULT_ENDPOINT_TOLERANCE_METRES,
    minimum_edge_gap: float = _core.DEFAULT_MINIMUM_EDGE_GAP_METRES,
    minimum_tangent_error: float = _core.DEFAULT_MINIMUM_TANGENT_ERROR_DEGREES,
    junction_match_tolerance: float = _core.DEFAULT_JUNCTION_MATCH_TOLERANCE_METRES,
):
    if _ORIGINAL_INSPECT is None:
        raise RuntimeError("road inspector overlap filter is not installed")
    result = _ORIGINAL_INSPECT(
        input_path,
        roads_geojson=roads_geojson,
        endpoint_tolerance=endpoint_tolerance,
        minimum_edge_gap=minimum_edge_gap,
        minimum_tangent_error=minimum_tangent_error,
        junction_match_tolerance=junction_match_tolerance,
    )
    remaining = tuple(
        issue
        for issue in result.issues
        if not _covered_repair_geometry(issue, result.road_objects)
    )
    if len(remaining) == len(result.issues):
        return result
    return replace(result, issues=_core._number_issues(remaining))


def install() -> None:
    """Install after the paved surface-coverage filter."""

    global _ORIGINAL_INSPECT, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_INSPECT = _core.inspect_road_geometry
    _core.inspect_road_geometry = inspect_road_geometry
    _INSTALLED = True
