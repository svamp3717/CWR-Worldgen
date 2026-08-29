# SPDX-License-Identifier: GPL-3.0-or-later
"""Suppress seam alarms between nearly coincident overlapping paved pieces.

Lundby25 exposed a detector edge case: two same-family ``sil6`` objects can sit
almost on top of each other as cap/approach or repair geometry. Their rendered
rectangles overlap heavily, but nearest-endpoint diagnostics can pair their
corresponding ends and describe the small placement offset as an end-to-end
connector gap.

This read-only filter removes only straight paved pairs whose object centres are
within half a metre, whose undirected axes agree within two degrees, and whose
vertical origins are close. Real adjacent road pieces are roughly one stock
piece length apart and therefore never satisfy this overlap test.
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

_ORIGINAL_INSPECT = None
_INSTALLED = False


def _axis_error(first: float, second: float) -> float:
    difference = abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)
    return min(difference, abs(180.0 - difference))


def _overlapping_paved_pair(issue, roads) -> bool:
    if issue.category not in _OVERLAP_CATEGORIES or len(issue.object_ids) != 2:
        return False
    road_by_id = {int(road.object_id): road for road in roads}
    involved = [road_by_id.get(int(object_id)) for object_id in issue.object_ids]
    if any(road is None for road in involved):
        return False
    first, second = involved
    if (
        first.family != second.family
        or first.family not in _PAVED_FAMILIES
        or first.kind != "straight"
        or second.kind != "straight"
    ):
        return False
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
        if not _overlapping_paved_pair(issue, result.road_objects)
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
