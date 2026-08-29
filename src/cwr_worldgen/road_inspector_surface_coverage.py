# SPDX-License-Identifier: GPL-3.0-or-later
"""Suppress Road Inspector seam alarms that are already covered by asphalt.

The inspector intentionally audits raw stock connectors, but the generator also
uses low same-family straight pieces as visual seam underlays. A second road can
likewise cross the same point and cover the entire open wedge. Reporting the raw
pair as a visible defect in either case sends debugging toward geometry that CWA
never exposes.

Filter only ordinary paved seam categories and only when another same-family
straight surface covers sampled points across both road-edge gaps and the
centreline gap. Dirt and mixed-family diagnostics are untouched. The test is
conservative: a nearby road centre is not enough, and vertically distant bridge
surfaces cannot hide a ground-level seam.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

from . import road_inspector as _core

_COVERABLE_CATEGORIES = frozenset({"straight_miter", "curve_transition", "connector_gap"})
_PAVED_FAMILIES = frozenset({"sil", "asf", "kos"})
_SURFACE_MARGIN_METRES = 0.08
_VERTICAL_MARGIN_METRES = 0.15
_SAMPLE_FRACTIONS = (0.20, 0.40, 0.60, 0.80)

_ORIGINAL_INSPECT = None
_INSTALLED = False


def _nearest_endpoint(road, point: tuple[float, float]):
    return min(road.endpoints, key=lambda endpoint: math.dist(endpoint.point, point))


def _matched_edge_pairs(first, second):
    first_edges = _core._cross_section_edges(first)
    second_edges = _core._cross_section_edges(second)
    direct = (
        (first_edges[0], second_edges[0]),
        (first_edges[1], second_edges[1]),
    )
    crossed = (
        (first_edges[0], second_edges[1]),
        (first_edges[1], second_edges[0]),
    )
    direct_length = sum(math.dist(start, end) for start, end in direct)
    crossed_length = sum(math.dist(start, end) for start, end in crossed)
    return direct if direct_length <= crossed_length else crossed


def _segment_samples(start, end):
    return tuple(
        (
            float(start[0]) + (float(end[0]) - float(start[0])) * fraction,
            float(start[1]) + (float(end[1]) - float(start[1])) * fraction,
        )
        for fraction in _SAMPLE_FRACTIONS
    )


def _gap_samples(first, second):
    samples = []
    for start, end in _matched_edge_pairs(first, second):
        samples.extend(_segment_samples(start, end))
    samples.extend(_segment_samples(first.point, second.point))
    return tuple(samples)


def _straight_contains(road, point: tuple[float, float]) -> bool:
    if road.kind != "straight" or len(road.endpoints) != 2:
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
        -_SURFACE_MARGIN_METRES <= along <= length + _SURFACE_MARGIN_METRES
        and lateral <= half_width + _SURFACE_MARGIN_METRES
    )


def _covered_by_other_paved_surface(issue, roads) -> bool:
    if issue.category not in _COVERABLE_CATEGORIES or len(issue.object_ids) != 2:
        return False

    road_by_id = {int(road.object_id): road for road in roads}
    involved = [road_by_id.get(int(object_id)) for object_id in issue.object_ids]
    if any(road is None for road in involved):
        return False
    first_road, second_road = involved
    if first_road.family != second_road.family or first_road.family not in _PAVED_FAMILIES:
        return False

    point = (float(issue.x), float(issue.z))
    first = _nearest_endpoint(first_road, point)
    second = _nearest_endpoint(second_road, point)
    samples = _gap_samples(first, second)
    if not samples:
        return False

    minimum_y = min(float(first_road.y), float(second_road.y)) - _VERTICAL_MARGIN_METRES
    maximum_y = max(float(first_road.y), float(second_road.y)) + _VERTICAL_MARGIN_METRES
    candidates = tuple(
        road
        for road in roads
        if (
            int(road.object_id) not in issue.object_ids
            and road.family == first_road.family
            and road.kind == "straight"
            and minimum_y <= float(road.y) <= maximum_y
        )
    )
    if not candidates:
        return False

    # Allow the covered area to be the union of several same-family pieces. This
    # matches what the renderer sees while still requiring every sampled point
    # across both open road edges to have actual asphalt underneath it.
    return all(any(_straight_contains(road, sample) for road in candidates) for sample in samples)


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
        raise RuntimeError("road inspector surface-coverage policy is not installed")
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
        if not _covered_by_other_paved_surface(issue, result.road_objects)
    )
    if len(remaining) == len(result.issues):
        return result
    return replace(result, issues=_core._number_issues(remaining))


def install() -> None:
    """Install the final read-only visual-coverage filter."""

    global _ORIGINAL_INSPECT, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_INSPECT = _core.inspect_road_geometry
    _core.inspect_road_geometry = inspect_road_geometry
    _INSTALLED = True
