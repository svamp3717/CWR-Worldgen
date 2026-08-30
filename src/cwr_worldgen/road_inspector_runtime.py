# SPDX-License-Identifier: GPL-3.0-or-later
"""Runtime accuracy layer for the read-only Road Inspector.

Keep these corrections isolated from the generator.  The inspector reads the
actual WRP/PBO after a build, so its geometry must match the RVW4 transform
rather than the simpler yaw-only fitting helpers used by some road policies.
It also projects normalized WGS84 road GeoJSON back into world metres before
comparing source junctions with emitted P3Ds.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import math
from typing import Iterable, Sequence

from . import road_inspector as _core
from . import stock_road_model_geometry as _geometry


MAXIMUM_NEARBY_CONNECTOR_GAP_METRES = 1.50
MAXIMUM_NEARBY_CONNECTOR_ALIGNMENT_ERROR_DEGREES = 20.0

_ORIGINAL_SOURCE_INTERSECTION_ISSUES = _core._source_intersection_issues
_ORIGINAL_INSPECT = _core.inspect_road_geometry
_INSTALLED = False


def _world_point(
    local: tuple[float, float],
    origin: tuple[float, float],
    heading_degrees: float,
    pitch_degrees: float,
) -> tuple[float, float]:
    """Project a local model X/Z point through the actual RVW4 yaw/pitch matrix."""

    x, z = float(local[0]), float(local[1])
    heading = math.radians(float(heading_degrees))
    pitch = math.radians(float(pitch_degrees))
    cosine_heading = math.cos(heading)
    sine_heading = math.sin(heading)
    cosine_pitch = math.cos(pitch)
    return (
        float(origin[0]) + x * cosine_heading + z * sine_heading * cosine_pitch,
        float(origin[1]) - x * sine_heading + z * cosine_heading * cosine_pitch,
    )


def _world_heading(
    local_heading_degrees: float,
    object_heading_degrees: float,
    pitch_degrees: float,
) -> float:
    """Return the horizontal heading of one local road tangent after RVW4 pitch."""

    local = math.radians(float(local_heading_degrees))
    x = math.sin(local)
    z = math.cos(local)
    heading = math.radians(float(object_heading_degrees))
    pitch = math.radians(float(pitch_degrees))
    cosine_heading = math.cos(heading)
    sine_heading = math.sin(heading)
    cosine_pitch = math.cos(pitch)
    world_x = x * cosine_heading + z * sine_heading * cosine_pitch
    world_z = -x * sine_heading + z * cosine_heading * cosine_pitch
    if math.hypot(world_x, world_z) <= 1.0e-12:
        return float(object_heading_degrees) % 360.0
    return math.degrees(math.atan2(world_x, world_z)) % 360.0


def _road_object_from_record(values):
    """Decode stock road geometry using the same 3D transform written to RVW4."""

    raw_model = values[13].split(b"\0", 1)[0]
    if not raw_model:
        return None
    try:
        model_path = raw_model.decode("ascii")
    except UnicodeDecodeError:
        return None

    normalized = _core._normalise_model(model_path)
    object_id = int(values[12])
    x, y, z = float(values[9]), float(values[10]), float(values[11])
    heading = math.degrees(math.atan2(-float(values[2]), float(values[0]))) % 360.0
    pitch_sine = max(-1.0, min(1.0, float(values[7])))
    pitch = math.degrees(math.asin(pitch_sine))
    origin = (x, z)

    if _core._PAVED_FILL.fullmatch(normalized) is not None:
        return _core.RoadObject(
            object_id,
            model_path,
            x,
            y,
            z,
            heading,
            pitch,
            "sil",
            "paved_fill",
            float(_geometry.STOCK_HALF_WIDTHS_METRES["sil"]) * 2.0,
            origin,
            (),
        )

    if _core._PAVED_MITER.fullmatch(normalized) is not None:
        turn = _core.paved_miter_angle_degrees(normalized)
        if turn is None:
            return None
        half_angle = math.radians(turn * 0.5)
        apex = (
            _core.GENERATED_PAVED_FILL_RADIUS_METRES
            + _core.GENERATED_PAVED_MITER_SAFETY_METRES
        ) / math.cos(half_angle)
        return _core.RoadObject(
            object_id,
            model_path,
            x,
            y,
            z,
            heading,
            pitch,
            "sil",
            "paved_miter",
            apex * 2.0,
            origin,
            (),
        )

    straight = _geometry.stock_straight_match(normalized)
    if straight is not None:
        family = straight.group("family").casefold()
        nominal = int(straight.group("length"))
        length = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[nominal])
        first = _world_point((0.0, -length * 0.5), origin, heading, pitch)
        second = _world_point((0.0, length * 0.5), origin, heading, pitch)
        tangent = _world_heading(0.0, heading, pitch)
        endpoints = (
            _core._endpoint(
                object_id=object_id,
                model_path=model_path,
                family=family,
                kind="straight",
                endpoint_index=0,
                point=first,
                tangent_axis_degrees=tangent,
                outward_heading_degrees=_world_heading(180.0, heading, pitch),
            ),
            _core._endpoint(
                object_id=object_id,
                model_path=model_path,
                family=family,
                kind="straight",
                endpoint_index=1,
                point=second,
                tangent_axis_degrees=tangent,
                outward_heading_degrees=tangent,
            ),
        )
        return _core.RoadObject(
            object_id,
            model_path,
            x,
            y,
            z,
            heading,
            pitch,
            family,
            "straight",
            length,
            origin,
            endpoints,
        )

    curve = _geometry.stock_curve_connectors(normalized)
    if curve is not None:
        family = curve.family
        begin = _world_point(curve.begin, origin, heading, pitch)
        end = _world_point(curve.end, origin, heading, pitch)
        begin_tangent = _world_heading(0.0, heading, pitch)
        end_tangent = _world_heading(
            _geometry.STOCK_CURVE_ANGLE_DEGREES,
            heading,
            pitch,
        )
        endpoints = (
            _core._endpoint(
                object_id=object_id,
                model_path=model_path,
                family=family,
                kind="curve",
                endpoint_index=0,
                point=begin,
                tangent_axis_degrees=begin_tangent,
                outward_heading_degrees=_world_heading(180.0, heading, pitch),
            ),
            _core._endpoint(
                object_id=object_id,
                model_path=model_path,
                family=family,
                kind="curve",
                endpoint_index=1,
                point=end,
                tangent_axis_degrees=end_tangent,
                outward_heading_degrees=end_tangent,
            ),
        )
        return _core.RoadObject(
            object_id,
            model_path,
            x,
            y,
            z,
            heading,
            pitch,
            family,
            "curve",
            float(curve.chord_length_metres),
            ((begin[0] + end[0]) * 0.5, (begin[1] + end[1]) * 0.5),
            endpoints,
        )

    match = _core._T_JUNCTION.fullmatch(normalized)
    if match is not None:
        main = match.group("main").casefold()
        branch = match.group("branch").casefold()
        local_center = _geometry.native_junction_intersection_offset(normalized)
        if local_center is None:
            return None
        center = _world_point(local_center, origin, heading, pitch)
        cx, cz = local_center
        radius = float(_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES)
        local_connectors = (
            ((cx, cz + radius), main, 0.0),
            ((cx, cz - radius), main, 180.0),
            ((cx - radius, cz), branch, 270.0),
        )
        endpoints = tuple(
            _core._endpoint(
                object_id=object_id,
                model_path=model_path,
                family=family,
                kind="junction",
                endpoint_index=index,
                point=_world_point(local, origin, heading, pitch),
                tangent_axis_degrees=_world_heading(local_heading, heading, pitch),
                outward_heading_degrees=_world_heading(local_heading, heading, pitch),
            )
            for index, (local, family, local_heading) in enumerate(local_connectors)
        )
        return _core.RoadObject(
            object_id,
            model_path,
            x,
            y,
            z,
            heading,
            pitch,
            main,
            "junction_t",
            radius,
            center,
            endpoints,
        )

    if _core._X_JUNCTION.fullmatch(normalized) is not None:
        family = "sil"
        radius = float(_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES)
        center = origin
        local_connectors = (
            ((0.0, radius), 0.0),
            ((0.0, -radius), 180.0),
            ((radius, 0.0), 90.0),
            ((-radius, 0.0), 270.0),
        )
        endpoints = tuple(
            _core._endpoint(
                object_id=object_id,
                model_path=model_path,
                family=family,
                kind="junction",
                endpoint_index=index,
                point=_world_point(local, origin, heading, pitch),
                tangent_axis_degrees=_world_heading(local_heading, heading, pitch),
                outward_heading_degrees=_world_heading(local_heading, heading, pitch),
            )
            for index, (local, local_heading) in enumerate(local_connectors)
        )
        return _core.RoadObject(
            object_id,
            model_path,
            x,
            y,
            z,
            heading,
            pitch,
            family,
            "junction_x",
            radius,
            center,
            endpoints,
        )
    return None


def _geojson_projector(payload) -> callable:
    """Return a GeoJSON coordinate -> world-X/Z projection when metadata permits."""

    if not isinstance(payload, dict):
        return lambda coordinate: (float(coordinate[0]), float(coordinate[1]))
    world = payload.get("cwr_world")
    bbox = payload.get("bbox")
    if not isinstance(world, dict) or not isinstance(bbox, list) or len(bbox) < 4:
        return lambda coordinate: (float(coordinate[0]), float(coordinate[1]))
    reference = str(world.get("coordinate_reference", "")).casefold()
    if "wgs84" not in reference and "longitude" not in reference:
        return lambda coordinate: (float(coordinate[0]), float(coordinate[1]))
    try:
        west, south, east, north = map(float, bbox[:4])
        size = float(world.get("world_size_metres", 0.0))
        if size <= 0.0:
            size = float(world.get("grid_cells", 0.0)) * float(
                world.get("cell_size_metres", 0.0)
            )
    except (TypeError, ValueError):
        return lambda coordinate: (float(coordinate[0]), float(coordinate[1]))
    if not (
        math.isfinite(west)
        and math.isfinite(south)
        and math.isfinite(east)
        and math.isfinite(north)
        and math.isfinite(size)
        and east > west
        and north > south
        and size > 0.0
    ):
        return lambda coordinate: (float(coordinate[0]), float(coordinate[1]))

    def project(coordinate):
        longitude = float(coordinate[0])
        latitude = float(coordinate[1])
        return (
            (longitude - west) / (east - west) * size,
            (latitude - south) / (north - south) * size,
        )

    return project


def _source_junctions(path: Path | None):
    """Read normalized roads and compare them in the same metre space as the WRP."""

    if path is None:
        return ()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    features = payload.get("features", []) if isinstance(payload, dict) else []
    project = _geojson_projector(payload)
    nodes: dict[tuple[int, int], tuple[tuple[float, float], list[float]]] = {}
    quantum = 0.05
    for feature in features:
        if not isinstance(feature, dict):
            continue
        for coordinates in _core._line_coordinate_sequences(feature.get("geometry")):
            points: list[tuple[float, float]] = []
            for coordinate in coordinates:
                if not isinstance(coordinate, (list, tuple)) or len(coordinate) < 2:
                    continue
                try:
                    point = project(coordinate)
                except (TypeError, ValueError, ZeroDivisionError):
                    continue
                if math.isfinite(point[0]) and math.isfinite(point[1]):
                    points.append(point)
            for index, point in enumerate(points):
                key = (round(point[0] / quantum), round(point[1] / quantum))
                _stored_point, headings = nodes.setdefault(key, (point, []))
                for neighbour_index in (index - 1, index + 1):
                    if not 0 <= neighbour_index < len(points):
                        continue
                    neighbour = points[neighbour_index]
                    dx, dz = neighbour[0] - point[0], neighbour[1] - point[1]
                    if math.hypot(dx, dz) <= 0.05:
                        continue
                    headings.append(math.degrees(math.atan2(dx, dz)) % 360.0)

    result = []
    for point, headings in nodes.values():
        unique = _core._dedupe_headings(headings)
        if len(unique) >= 3:
            result.append(_core.SourceJunction(point, unique))
    return tuple(result)


def _source_intersection_issues(roads, junctions, *, match_tolerance):
    """Compare source arms with directions from the node into ordinary pieces."""

    corrected = []
    for road in roads:
        if road.kind in {"junction_t", "junction_x"}:
            corrected.append(road)
            continue
        endpoints = tuple(
            replace(
                endpoint,
                outward_heading_degrees=(
                    float(endpoint.outward_heading_degrees) + 180.0
                )
                % 360.0,
            )
            for endpoint in road.endpoints
        )
        corrected.append(replace(road, endpoints=endpoints))
    return _ORIGINAL_SOURCE_INTERSECTION_ISSUES(
        tuple(corrected),
        junctions,
        match_tolerance=match_tolerance,
    )


def _heading_between(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.degrees(
        math.atan2(float(second[0]) - float(first[0]), float(second[1]) - float(first[1]))
    ) % 360.0


def _nearby_unmatched_issues(
    roads,
    *,
    endpoint_tolerance: float,
    minimum_edge_gap: float,
    minimum_tangent_error: float,
):
    """Find physical connector gaps just outside the normal clustering tolerance.

    Only mutually-nearest, directionally-facing endpoints are accepted.  This
    avoids turning ordinary dead-end roads into a festival of false positives.
    """

    endpoints = tuple(endpoint for road in roads for endpoint in road.endpoints)
    if not endpoints:
        return []
    clusters = _core._endpoint_clusters(endpoints, endpoint_tolerance)
    connected = {
        (endpoint.object_id, endpoint.endpoint_index)
        for cluster in clusters
        for endpoint in cluster
    }
    unmatched = [
        endpoint
        for endpoint in endpoints
        if (endpoint.object_id, endpoint.endpoint_index) not in connected
    ]
    if len(unmatched) < 2:
        return []

    bucket_size = MAXIMUM_NEARBY_CONNECTOR_GAP_METRES
    buckets: dict[tuple[str, int, int], list[int]] = {}
    for index, endpoint in enumerate(unmatched):
        bx, bz = _core._bucket(endpoint.point, bucket_size)
        buckets.setdefault((endpoint.family, bx, bz), []).append(index)

    def best_for(index: int):
        endpoint = unmatched[index]
        bx, bz = _core._bucket(endpoint.point, bucket_size)
        best = None
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for candidate_index in buckets.get(
                    (endpoint.family, bx + dx, bz + dz), ()
                ):
                    if candidate_index == index:
                        continue
                    candidate = unmatched[candidate_index]
                    if candidate.object_id == endpoint.object_id:
                        continue
                    distance = math.dist(endpoint.point, candidate.point)
                    if not (
                        endpoint_tolerance < distance <= MAXIMUM_NEARBY_CONNECTOR_GAP_METRES
                    ):
                        continue
                    heading = _heading_between(endpoint.point, candidate.point)
                    first_error = _core._angular_distance(
                        endpoint.outward_heading_degrees, heading
                    )
                    second_error = _core._angular_distance(
                        candidate.outward_heading_degrees, (heading + 180.0) % 360.0
                    )
                    if max(first_error, second_error) > MAXIMUM_NEARBY_CONNECTOR_ALIGNMENT_ERROR_DEGREES:
                        continue
                    candidate_score = (
                        distance,
                        max(first_error, second_error),
                        first_error + second_error,
                        candidate.object_id,
                        candidate.endpoint_index,
                        candidate_index,
                    )
                    if best is None or candidate_score < best:
                        best = candidate_score
        return best

    best = [best_for(index) for index in range(len(unmatched))]
    issues = []
    used: set[tuple[int, int]] = set()
    for index, candidate in enumerate(best):
        if candidate is None:
            continue
        other_index = int(candidate[-1])
        reverse = best[other_index]
        if reverse is None or int(reverse[-1]) != index:
            continue
        pair = tuple(sorted((index, other_index)))
        if pair in used:
            continue
        used.add(pair)
        first, second = unmatched[pair[0]], unmatched[pair[1]]
        first_is_junction = first.object_kind == "junction"
        second_is_junction = second.object_kind == "junction"
        if first_is_junction and second_is_junction:
            continue
        if first_is_junction != second_is_junction:
            connector, approach = (
                (first, second) if first_is_junction else (second, first)
            )
            issue = _core._junction_connector_issue(
                connector,
                approach,
                minimum_edge_gap=minimum_edge_gap,
                minimum_tangent_error=minimum_tangent_error,
            )
        else:
            issue = _core._ordinary_seam_issue(
                first,
                second,
                minimum_edge_gap=minimum_edge_gap,
                minimum_tangent_error=minimum_tangent_error,
            )
        if issue is None:
            continue
        gap_heading = _heading_between(first.point, second.point)
        issue = replace(
            issue,
            metrics={
                **issue.metrics,
                "detector": "nearby_unmatched_connector",
                "gap_alignment_first_degrees": round(
                    _core._angular_distance(first.outward_heading_degrees, gap_heading),
                    5,
                ),
                "gap_alignment_second_degrees": round(
                    _core._angular_distance(
                        second.outward_heading_degrees,
                        (gap_heading + 180.0) % 360.0,
                    ),
                    5,
                ),
            },
        )
        issues.append(issue)
    return issues


def inspect_road_geometry(
    input_path: Path,
    *,
    roads_geojson: Path | None = None,
    endpoint_tolerance: float = _core.DEFAULT_ENDPOINT_TOLERANCE_METRES,
    minimum_edge_gap: float = _core.DEFAULT_MINIMUM_EDGE_GAP_METRES,
    minimum_tangent_error: float = _core.DEFAULT_MINIMUM_TANGENT_ERROR_DEGREES,
    junction_match_tolerance: float = _core.DEFAULT_JUNCTION_MATCH_TOLERANCE_METRES,
):
    result = _ORIGINAL_INSPECT(
        input_path,
        roads_geojson=roads_geojson,
        endpoint_tolerance=endpoint_tolerance,
        minimum_edge_gap=minimum_edge_gap,
        minimum_tangent_error=minimum_tangent_error,
        junction_match_tolerance=junction_match_tolerance,
    )
    extra = _nearby_unmatched_issues(
        result.road_objects,
        endpoint_tolerance=endpoint_tolerance,
        minimum_edge_gap=minimum_edge_gap,
        minimum_tangent_error=minimum_tangent_error,
    )
    if not extra:
        return result
    return replace(
        result,
        issues=_core._number_issues((*result.issues, *extra)),
    )


def install() -> None:
    """Patch inspector diagnostics only; never patch world-generation behavior."""

    global _INSTALLED
    if _INSTALLED:
        return
    _core._road_object_from_record = _road_object_from_record
    _core._source_junctions = _source_junctions
    _core._source_intersection_issues = _source_intersection_issues
    _core.inspect_road_geometry = inspect_road_geometry
    _INSTALLED = True
