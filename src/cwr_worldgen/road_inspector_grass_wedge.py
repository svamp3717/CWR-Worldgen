# SPDX-License-Identifier: GPL-3.0-or-later
"""Classify actual exposed paved-turn wedges in Road Inspector.

The base inspector already measures centre, tangent, and road-edge discontinuity
at stock-road seams. A heading discontinuity is only a proxy for the visible
failure users care about, though: on a faceted paved turn the two *outer* road
edges can terminate at different points and leave a triangular patch of terrain
between them.

This layer turns that proxy into an explicit geometric diagnostic for paved
stock-road families only (``sil``, ``asf``, and ``kos``). Gravel/dirt families
are deliberately outside this detector.

For an ordinary same-family paved seam, extend both physical outer edge rays
beyond their emitted connectors. When the rays meet in front of both road pieces,
the triangle between the two connector-edge points and that miter intersection
is the potential exposed terrain wedge. Road Inspector reports it as
``grass_wedge`` with area, depth, opening, and miter coordinates.

The layer is read-only. It does not add cover pieces or change generated road
geometry. Existing surface/overlap filters run before this classifier, but a
surviving paved seam is always evaluated geometrically here so a real exposed
asphalt wedge cannot disappear merely because the base seam category changed.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

from . import road_inspector as _core


_PAVED_FAMILIES = frozenset({"sil", "asf", "kos"})
# Keep the explicit legacy categories, but do not depend on them exclusively:
# runtime/source-context layers may refine a paved seam's category before this
# final classifier runs. Eligibility is therefore determined from the two road
# objects and their nearest endpoints as well.
_SEAM_CATEGORIES = frozenset({"straight_miter", "curve_transition"})
MINIMUM_GRASS_WEDGE_AREA_SQUARE_METRES = 0.001
MINIMUM_GRASS_WEDGE_DEPTH_METRES = 0.005
MINIMUM_GRASS_WEDGE_TURN_DEGREES = 0.75
MAXIMUM_GRASS_WEDGE_CENTER_GAP_METRES = 0.35
MAXIMUM_GRASS_WEDGE_MITER_EXTENSION_METRES = 10.0
SOURCE_JUNCTION_EXCLUSION_METRES = 1.25

_ORIGINAL_INSPECT = None
_INSTALLED = False


def _cross(first: tuple[float, float], second: tuple[float, float]) -> float:
    return float(first[0]) * float(second[1]) - float(first[1]) * float(second[0])


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


def _forward_ray_intersection(
    first_point: tuple[float, float],
    first_heading: float,
    second_point: tuple[float, float],
    second_heading: float,
):
    first_direction = _core._heading_unit(first_heading)
    second_direction = _core._heading_unit(second_heading)
    denominator = _cross(first_direction, second_direction)
    if abs(denominator) <= 1.0e-8:
        return None

    delta = (
        float(second_point[0]) - float(first_point[0]),
        float(second_point[1]) - float(first_point[1]),
    )
    first_distance = _cross(delta, second_direction) / denominator
    second_distance = _cross(delta, first_direction) / denominator
    if first_distance <= 1.0e-5 or second_distance <= 1.0e-5:
        return None
    if (
        first_distance > MAXIMUM_GRASS_WEDGE_MITER_EXTENSION_METRES
        or second_distance > MAXIMUM_GRASS_WEDGE_MITER_EXTENSION_METRES
    ):
        return None

    intersection = (
        float(first_point[0]) + first_direction[0] * first_distance,
        float(first_point[1]) + first_direction[1] * first_distance,
    )
    return intersection, first_distance, second_distance


def _triangle_depth(
    first: tuple[float, float],
    second: tuple[float, float],
    apex: tuple[float, float],
) -> tuple[float, float, float]:
    base = math.dist(first, second)
    if base <= 1.0e-8:
        return 0.0, 0.0, 0.0
    vector = (
        float(second[0]) - float(first[0]),
        float(second[1]) - float(first[1]),
    )
    apex_vector = (
        float(apex[0]) - float(first[0]),
        float(apex[1]) - float(first[1]),
    )
    area = abs(_cross(vector, apex_vector)) * 0.5
    depth = 2.0 * area / base
    return area, depth, base


def _grass_wedge_geometry(first, second):
    """Return the largest forward outer-edge miter triangle for one seam."""

    center_gap = math.dist(first.point, second.point)
    turn = _core._axis_heading_difference(
        first.tangent_axis_degrees,
        second.tangent_axis_degrees,
    )
    if center_gap > MAXIMUM_GRASS_WEDGE_CENTER_GAP_METRES:
        return None
    if turn < MINIMUM_GRASS_WEDGE_TURN_DEGREES:
        return None

    candidates = []
    for first_edge, second_edge in _matched_edge_pairs(first, second):
        intersection = _forward_ray_intersection(
            first_edge,
            first.outward_heading_degrees,
            second_edge,
            second.outward_heading_degrees,
        )
        if intersection is None:
            continue
        apex, first_extension, second_extension = intersection
        area, depth, opening = _triangle_depth(first_edge, second_edge, apex)
        if (
            area < MINIMUM_GRASS_WEDGE_AREA_SQUARE_METRES
            or depth < MINIMUM_GRASS_WEDGE_DEPTH_METRES
        ):
            continue
        centroid = (
            (float(first_edge[0]) + float(second_edge[0]) + float(apex[0])) / 3.0,
            (float(first_edge[1]) + float(second_edge[1]) + float(apex[1])) / 3.0,
        )
        candidates.append(
            (
                area,
                depth,
                opening,
                apex,
                centroid,
                first_extension,
                second_extension,
                turn,
                center_gap,
            )
        )

    return max(candidates, key=lambda value: (value[0], value[1])) if candidates else None


def _nearest_issue_endpoints(issue, roads):
    if len(issue.object_ids) != 2:
        return None
    road_by_id = {int(road.object_id): road for road in roads}
    first_road = road_by_id.get(int(issue.object_ids[0]))
    second_road = road_by_id.get(int(issue.object_ids[1]))
    if first_road is None or second_road is None:
        return None
    if (
        first_road.family != second_road.family
        or first_road.family not in _PAVED_FAMILIES
        or first_road.kind.startswith("junction_")
        or second_road.kind.startswith("junction_")
    ):
        return None

    issue_point = (float(issue.x), float(issue.z))
    first = min(first_road.endpoints, key=lambda endpoint: math.dist(endpoint.point, issue_point))
    second = min(second_road.endpoints, key=lambda endpoint: math.dist(endpoint.point, issue_point))
    return first_road, second_road, first, second


def _near_source_junction(point, source_junctions, match_tolerance: float) -> bool:
    limit = max(SOURCE_JUNCTION_EXCLUSION_METRES, float(match_tolerance) * 1.5)
    return any(math.dist(point, junction.point) <= limit for junction in source_junctions)


def _classify_grass_wedge(issue, roads, source_junctions, match_tolerance: float):
    matched = _nearest_issue_endpoints(issue, roads)
    if matched is None:
        return issue
    _first_road, _second_road, first, second = matched

    # The base inspector normally labels these straight_miter/curve_transition.
    # If a later read-only layer has refined the label, still evaluate the paved
    # two-object seam. Conversely, categories that are explicitly about source
    # intersections remain excluded below by the junction proximity check.
    if issue.category not in _SEAM_CATEGORIES and len(issue.object_ids) != 2:
        return issue

    issue_point = (float(issue.x), float(issue.z))
    if _near_source_junction(issue_point, source_junctions, match_tolerance):
        return issue

    geometry = _grass_wedge_geometry(first, second)
    if geometry is None:
        return issue
    (
        area,
        depth,
        opening,
        apex,
        centroid,
        first_extension,
        second_extension,
        turn,
        center_gap,
    ) = geometry

    wedge_score = min(
        100.0,
        15.0
        + min(40.0, depth * 45.0)
        + min(25.0, area * 20.0)
        + min(30.0, turn * 2.0),
    )
    score = max(float(issue.score), wedge_score)
    return replace(
        issue,
        category="grass_wedge",
        x=round(float(centroid[0]), 4),
        z=round(float(centroid[1]), 4),
        score=round(score, 2),
        severity=_core._severity(score),
        message=(
            f"Potential exposed grass wedge between {first.model_path} and "
            f"{second.model_path}: {turn:.2f}° turn, {area:.3f} m² triangular "
            f"opening, {depth:.3f} m maximum wedge depth."
        ),
        candidate_fix=(
            "Cover the exposed outside paved turn with the bounded borderless "
            "paved wedge overlay, or refit the turn with connector-locked native "
            "stock curves whose physical road edges remain continuous."
        ),
        metrics={
            **issue.metrics,
            "grass_wedge_detector": "forward_edge_ray_miter_triangle",
            "grass_wedge_area_square_metres": round(area, 5),
            "grass_wedge_depth_metres": round(depth, 5),
            "grass_wedge_opening_metres": round(opening, 5),
            "grass_wedge_turn_degrees": round(turn, 5),
            "grass_wedge_center_gap_metres": round(center_gap, 5),
            "grass_wedge_miter_x": round(float(apex[0]), 5),
            "grass_wedge_miter_z": round(float(apex[1]), 5),
            "grass_wedge_first_edge_extension_metres": round(first_extension, 5),
            "grass_wedge_second_edge_extension_metres": round(second_extension, 5),
        },
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
        raise RuntimeError("road inspector grass-wedge detector is not installed")
    result = _ORIGINAL_INSPECT(
        input_path,
        roads_geojson=roads_geojson,
        endpoint_tolerance=endpoint_tolerance,
        minimum_edge_gap=minimum_edge_gap,
        minimum_tangent_error=minimum_tangent_error,
        junction_match_tolerance=junction_match_tolerance,
    )
    source_junctions = _core._source_junctions(roads_geojson) if roads_geojson else ()
    classified = tuple(
        _classify_grass_wedge(
            issue,
            result.road_objects,
            source_junctions,
            junction_match_tolerance,
        )
        for issue in result.issues
    )
    if classified == result.issues:
        return result
    return replace(result, issues=_core._number_issues(classified))


def install() -> None:
    """Install the final read-only paved grass-wedge classifier."""

    global _ORIGINAL_INSPECT, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_INSPECT = _core.inspect_road_geometry
    _core.inspect_road_geometry = inspect_road_geometry
    _INSTALLED = True
