# SPDX-License-Identifier: GPL-3.0-or-later
"""Audit exposed paved outside-miter triangles from final WRP geometry.

This pass is intentionally independent of the ordinary seam thresholds.  It
reconstructs stock connectors with the actual RVW4 pitch projection, samples the
whole outside triangle instead of only its apex/centroid, and accepts coverage
only where a real paved surface contains the sample with a tiny geometric margin.
That prevents the old 8 cm diagnostic fuzz from declaring a visible grass sliver
"covered".  Only sil/asf/kos participate; dirt/gravel remains untouched.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

from . import road_inspector as _core
from . import road_inspector_grass_wedge as _grass
from . import road_inspector_surface_coverage as _coverage
from .paved_wedge_geometry import (
    paved_wedge_local_points as _wide_wedge_points,
    triangle_samples,
)


_PAVED_FAMILIES = frozenset({"sil", "asf", "kos"})
_SCAN_TOLERANCE_METRES = _grass.MAXIMUM_GRASS_WEDGE_CENTER_GAP_METRES
_STRICT_SURFACE_MARGIN_METRES = 0.003
_MINIMUM_VISIBLE_CLEARANCE_METRES = 0.005
_ORIGINAL_INSPECT = None
_ORIGINAL_RECORD_PARSER = None
_INSTALLED = False


def _paved_wedge_local_points(turn_degrees: float):
    return _wide_wedge_points(
        turn_degrees,
        radius_metres=_core.GENERATED_PAVED_FILL_RADIUS_METRES,
        maximum_turn_degrees=35.0,
    )


def _project_local_point(road, point: tuple[float, float]) -> tuple[float, float]:
    local_x, local_z = float(point[0]), float(point[1])
    heading = math.radians(float(road.heading_degrees))
    cosine_heading = math.cos(heading)
    sine_heading = math.sin(heading)
    cosine_pitch = math.cos(math.radians(float(road.pitch_degrees)))
    return (
        float(road.x)
        + local_x * cosine_heading
        + local_z * sine_heading * cosine_pitch,
        float(road.z)
        - local_x * sine_heading
        + local_z * cosine_heading * cosine_pitch,
    )


def _physical_record_parser(values):
    if _ORIGINAL_RECORD_PARSER is None:
        raise RuntimeError("road inspector physical endpoint parser is not installed")
    road = _ORIGINAL_RECORD_PARSER(values)
    if road is None or road.kind not in {"straight", "curve"}:
        return road

    if road.kind == "straight":
        match = _core._geometry.stock_straight_match(road.model_path)
        if match is None:
            return road
        length = float(
            _core._geometry.STOCK_STRAIGHT_LENGTHS_METRES[
                int(match.group("length"))
            ]
        )
        local_points = ((0.0, -length * 0.5), (0.0, length * 0.5))
        tangents = (road.heading_degrees, road.heading_degrees)
    else:
        geometry = _core._geometry.stock_curve_connectors(road.model_path)
        if geometry is None:
            return road
        local_points = (geometry.begin, geometry.end)
        tangents = (
            road.heading_degrees,
            road.heading_degrees + _core._geometry.STOCK_CURVE_ANGLE_DEGREES,
        )

    points = tuple(_project_local_point(road, point) for point in local_points)
    endpoints = []
    for endpoint_index, (point, tangent) in enumerate(zip(points, tangents)):
        other = points[1 - endpoint_index]
        outward_vector = (
            float(point[0]) - float(other[0]),
            float(point[1]) - float(other[1]),
        )
        tangent_unit = _core._heading_unit(tangent)
        outward = (
            tangent
            if outward_vector[0] * tangent_unit[0] + outward_vector[1] * tangent_unit[1]
            >= 0.0
            else tangent + 180.0
        )
        endpoints.append(
            _core._endpoint(
                object_id=int(road.object_id),
                model_path=road.model_path,
                family=road.family,
                kind=road.kind,
                endpoint_index=endpoint_index,
                point=point,
                tangent_axis_degrees=tangent,
                outward_heading_degrees=outward,
            )
        )

    logical_center = (
        (float(points[0][0]) + float(points[1][0])) * 0.5,
        (float(points[0][1]) + float(points[1][1])) * 0.5,
    )
    if road.kind == "straight":
        logical_center = (float(road.x), float(road.z))
    return replace(
        road,
        logical_center=logical_center,
        endpoints=tuple(endpoints),
    )


def _candidate_pairs(roads):
    endpoints = tuple(
        endpoint
        for road in roads
        if road.family in _PAVED_FAMILIES
        and not road.kind.startswith("junction_")
        for endpoint in road.endpoints
    )
    clusters = _core._endpoint_clusters(endpoints, _SCAN_TOLERANCE_METRES)
    pairs = []
    for cluster in clusters:
        unique = {
            (int(endpoint.object_id), int(endpoint.endpoint_index)): endpoint
            for endpoint in cluster
        }
        if len(unique) != 2:
            continue
        first, second = sorted(
            unique.values(),
            key=lambda endpoint: (int(endpoint.object_id), int(endpoint.endpoint_index)),
        )
        if first.object_id == second.object_id or first.family != second.family:
            continue
        pairs.append((first, second))
    return tuple(pairs)


def _provisional_issue(first, second):
    center_gap = math.dist(first.point, second.point)
    tangent_error = _core._axis_heading_difference(
        first.tangent_axis_degrees,
        second.tangent_axis_degrees,
    )
    edge_max, edge_min, edge_mean = _core._edge_discontinuity(first, second)
    score = _core._score_geometry(
        center_gap=center_gap,
        edge_gap=edge_max,
        tangent_error=tangent_error,
    )
    return _core.RoadIssue(
        issue_id="",
        severity=_core._severity(score),
        score=score,
        category="connector_gap",
        x=(float(first.point[0]) + float(second.point[0])) * 0.5,
        z=(float(first.point[1]) + float(second.point[1])) * 0.5,
        object_ids=tuple(sorted((int(first.object_id), int(second.object_id)))),
        models=(first.model_path, second.model_path),
        message="Direct paved outside-miter audit candidate.",
        candidate_fix="Refit the paved seam so its physical outside edges remain continuous.",
        metrics={
            "center_gap_metres": round(center_gap, 5),
            "tangent_error_degrees": round(tangent_error, 5),
            "edge_gap_max_metres": round(edge_max, 5),
            "edge_gap_min_metres": round(edge_min, 5),
            "edge_gap_mean_metres": round(edge_mean, 5),
        },
    )


def _strict_straight_contains(road, point: tuple[float, float]) -> bool:
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
    margin = _STRICT_SURFACE_MARGIN_METRES
    return (
        -margin <= along <= length + margin
        and lateral <= half_width + margin
    )


def _strict_miter_contains(road, point: tuple[float, float]) -> bool:
    turn = _core.paved_miter_angle_degrees(road.model_path)
    if turn is None:
        return False
    dx = float(point[0]) - float(road.logical_center[0])
    dz = float(point[1]) - float(road.logical_center[1])
    heading = math.radians(float(road.heading_degrees))
    local_x = dx * math.cos(heading) - dz * math.sin(heading)
    cosine_pitch = math.cos(math.radians(float(road.pitch_degrees)))
    if abs(cosine_pitch) <= 1.0e-9:
        return False
    local_z = (
        dx * math.sin(heading) + dz * math.cos(heading)
    ) / cosine_pitch
    radius = (
        _core.GENERATED_PAVED_FILL_RADIUS_METRES
        + _core.GENERATED_PAVED_MITER_SAFETY_METRES
    )
    margin = _STRICT_SURFACE_MARGIN_METRES
    if math.hypot(local_x, local_z) <= radius + margin:
        return True
    half_angle = math.radians(float(turn) * 0.5)
    cosine = math.cos(half_angle)
    if cosine <= 1.0e-9:
        return False
    base_x = radius * cosine
    apex_x = radius / cosine
    absolute_x = abs(local_x)
    if absolute_x < base_x - margin or absolute_x > apex_x + margin:
        return False
    depth = apex_x - base_x
    if depth <= 1.0e-9:
        return False
    fraction = max(0.0, min(1.0, (apex_x - absolute_x) / depth))
    return (
        abs(local_z)
        <= radius * math.sin(half_angle) * fraction + margin
    )


def _strict_surface_contains(road, point: tuple[float, float]) -> bool:
    if road.kind == "straight":
        return _strict_straight_contains(road, point)
    if road.kind == "paved_fill":
        return math.dist(road.logical_center, point) <= (
            _core.GENERATED_PAVED_FILL_RADIUS_METRES + _STRICT_SURFACE_MARGIN_METRES
        )
    if road.kind == "paved_miter":
        return _strict_miter_contains(road, point)
    if road.kind == "paved_wedge":
        return _core._paved_wedge_contains(
            road,
            point,
            margin=_STRICT_SURFACE_MARGIN_METRES,
        )
    return False


def _wedge_triangle(first, second, geometry):
    apex = geometry[3]
    best = None
    for first_edge, second_edge in _grass._matched_edge_pairs(first, second):
        intersection = _grass._forward_ray_intersection(
            first_edge,
            first.outward_heading_degrees,
            second_edge,
            second.outward_heading_degrees,
        )
        if intersection is None:
            continue
        candidate_apex = intersection[0]
        score = math.dist(candidate_apex, apex)
        if best is None or score < best[0]:
            best = (score, first_edge, second_edge, candidate_apex)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _strictly_covered_by_other_paved_surface(
    first,
    second,
    geometry,
    roads,
    terrain,
) -> bool:
    triangle = _wedge_triangle(first, second, geometry)
    if triangle is None:
        return False
    samples = triangle_samples(*triangle)
    involved_ids = {int(first.object_id), int(second.object_id)}
    road_by_id = {int(road.object_id): road for road in roads}
    involved = [road_by_id.get(value) for value in involved_ids]
    involved = [road for road in involved if road is not None]
    if len(involved) != 2:
        return False
    minimum_y = min(float(road.y) for road in involved) - 0.20
    maximum_y = max(float(road.y) for road in involved) + 0.55

    candidates = tuple(
        road
        for road in roads
        if int(road.object_id) not in involved_ids
        and minimum_y <= float(road.y) <= maximum_y
        and (
            (road.kind == "straight" and road.family == first.family)
            or road.kind in {"paved_fill", "paved_miter", "paved_wedge"}
        )
    )
    if not candidates:
        return False

    return all(
        any(
            _strict_surface_contains(road, sample)
            and (
                terrain is None
                or (
                    _coverage._surface_height(road, sample)
                    - _coverage._terrain_height(terrain, sample)
                    >= _MINIMUM_VISIBLE_CLEARANCE_METRES
                )
            )
            for road in candidates
        )
        for sample in samples
    )


def _scan_missing_grass_wedges(result, source_junctions, match_tolerance, terrain=None):
    existing_pairs = {
        tuple(sorted(int(value) for value in issue.object_ids))
        for issue in result.issues
        if issue.category == "grass_wedge" and len(issue.object_ids) == 2
    }
    additions = []
    for first, second in _candidate_pairs(result.road_objects):
        pair = tuple(sorted((int(first.object_id), int(second.object_id))))
        if pair in existing_pairs:
            continue
        midpoint = (
            (float(first.point[0]) + float(second.point[0])) * 0.5,
            (float(first.point[1]) + float(second.point[1])) * 0.5,
        )
        if _grass._near_source_junction(midpoint, source_junctions, match_tolerance):
            continue
        geometry = _grass._grass_wedge_geometry(first, second)
        if geometry is None:
            continue
        if _strictly_covered_by_other_paved_surface(
            first,
            second,
            geometry,
            result.road_objects,
            terrain,
        ):
            continue

        provisional = _provisional_issue(first, second)
        classified = _grass._classify_grass_wedge(
            provisional,
            result.road_objects,
            source_junctions,
            match_tolerance,
        )
        if classified.category != "grass_wedge":
            continue
        additions.append(classified)
        existing_pairs.add(pair)

    if not additions:
        return result
    return replace(
        result,
        issues=_core._number_issues(tuple(result.issues) + tuple(additions)),
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
        raise RuntimeError("road inspector paved-wedge audit is not installed")
    result = _ORIGINAL_INSPECT(
        input_path,
        roads_geojson=roads_geojson,
        endpoint_tolerance=endpoint_tolerance,
        minimum_edge_gap=minimum_edge_gap,
        minimum_tangent_error=minimum_tangent_error,
        junction_match_tolerance=junction_match_tolerance,
    )
    source_junctions = _core._source_junctions(roads_geojson) if roads_geojson else ()
    terrain = _coverage._terrain_context(Path(input_path))
    return _scan_missing_grass_wedges(
        result,
        source_junctions,
        junction_match_tolerance,
        terrain,
    )


def install() -> None:
    global _ORIGINAL_INSPECT, _ORIGINAL_RECORD_PARSER, _INSTALLED
    if _INSTALLED:
        return

    # Connector-gap seams are legitimate grass-wedge candidates too.
    _grass._SEAM_CATEGORIES = frozenset(
        set(_grass._SEAM_CATEGORIES) | {"connector_gap"}
    )

    # Make every later inspector layer see the same widened generated helper and
    # the actual pitch-projected stock connectors that the game renders.
    _core.paved_wedge_local_points = _paved_wedge_local_points
    _ORIGINAL_RECORD_PARSER = _core._road_object_from_record
    _core._road_object_from_record = _physical_record_parser

    _ORIGINAL_INSPECT = _core.inspect_road_geometry
    _core.inspect_road_geometry = inspect_road_geometry
    _INSTALLED = True
