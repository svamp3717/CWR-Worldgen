# SPDX-License-Identifier: GPL-3.0-or-later
"""Prefer larger stock-road pieces by relaxing tiny source-line noise safely.

The stock fitter historically follows every small heading change in the source
centreline. Even when a 25 m or 12.5 m slab would stay visually inside the same
road corridor, the quality score can then fall back to many 6.25 m pieces. That
increases visible joins and encourages later seam-repair objects to overlap one
another.

This policy keeps real junction endpoints fixed but may simplify very small bends
inside a bounded 0.75 m corridor. A shortcut is rejected near source-backed
buildings, utility structures, individual trees, fences, walls, hedges,
retaining walls and tree rows. Linear roadside features are indexed segment by
segment so one bent fence does not create a giant rectangular exclusion zone.

The same policy makes junction seam repair aware of complete road axes. If a
larger approach piece already passes underneath a junction connector, do not add
another short repair slab merely because neither endpoint happens to sit on that
connector.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import math
from typing import Mapping

from . import generator as _generator
from . import playability as _p
from . import stock_road_model_geometry as _geometry
from . import stock_road_surface_overlap_policy as _surface

MAXIMUM_LOCAL_RELAXATION_METRES = 0.75
MAXIMUM_RELAXED_HEADING_CHANGE_DEGREES = 7.0
MAXIMUM_RELAXED_CHORD_METRES = 55.0
MAXIMUM_RELAX_LOOKAHEAD_POINTS = 24
ROAD_SURFACE_HALF_WIDTH_METRES = 4.55
ROAD_RELAXATION_OBJECT_MARGIN_METRES = 0.55
_OBSTACLE_BUCKET_METRES = 32.0
_AXIS_BUCKET_METRES = 32.0
_AXIS_ALIGNMENT_COSINE = math.cos(math.radians(35.0))
_BARRIER_HALF_WIDTH_METRES = {
    "hedge": 1.25,
    "wall": 0.60,
    "retaining_wall": 0.75,
    "fence": 0.45,
}
_TREE_ROW_HALF_WIDTH_METRES = 1.75


@dataclass(frozen=True, slots=True)
class _Obstacle:
    min_x: float
    min_z: float
    max_x: float
    max_z: float


@dataclass(frozen=True, slots=True)
class _ObstacleIndex:
    obstacles: tuple[_Obstacle, ...]
    buckets: Mapping[tuple[int, int], tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class _RelaxContext:
    obstacles: _ObstacleIndex


@dataclass(frozen=True, slots=True)
class _AxisRecord:
    start: tuple[float, float]
    end: tuple[float, float]
    family: str


_CONTEXT: ContextVar[_RelaxContext | None] = ContextVar(
    "cwr_stock_road_relaxation", default=None
)
_ORIGINAL_ROUNDED = None
_ORIGINAL_FIT = None
_INSTALLED = False


def _bucket_range(minimum: float, maximum: float, size: float):
    return range(math.floor(minimum / size), math.floor(maximum / size) + 1)


def _add_bbox(obstacles: list[_Obstacle], minimum_x, minimum_z, maximum_x, maximum_z):
    values = (float(minimum_x), float(minimum_z), float(maximum_x), float(maximum_z))
    if not all(math.isfinite(value) for value in values):
        return
    min_x, min_z, max_x, max_z = values
    if max_x < min_x:
        min_x, max_x = max_x, min_x
    if max_z < min_z:
        min_z, max_z = max_z, min_z
    obstacles.append(_Obstacle(min_x, min_z, max_x, max_z))


def _point_bbox(point: tuple[float, float], radius: float) -> tuple[float, float, float, float]:
    x, z = point
    return x - radius, z - radius, x + radius, z + radius


def _line_radius(feature, default: float) -> float:
    kind = str(getattr(feature, "tags", {}).get("barrier", "")).strip().casefold()
    return _BARRIER_HALF_WIDTH_METRES.get(kind, default)


def _line_obstacles(features, projection, *, default_radius: float):
    """Represent mapped roadside lines as segment-sized obstacle boxes."""

    result = []
    for feature in features:
        points = tuple(projection.to_world(point) for point in feature.points)
        if len(points) < 2:
            continue
        radius = max(0.0, _line_radius(feature, default_radius))
        for start, end in zip(points, points[1:]):
            if math.dist(start, end) <= 1.0e-6:
                continue
            result.append(
                _Obstacle(
                    min(start[0], end[0]) - radius,
                    min(start[1], end[1]) - radius,
                    max(start[0], end[0]) + radius,
                    max(start[1], end[1]) + radius,
                )
            )
    return result


def _reindex(obstacles) -> _ObstacleIndex:
    bucket_lists: dict[tuple[int, int], list[int]] = {}
    for index, obstacle in enumerate(obstacles):
        for bx in _bucket_range(obstacle.min_x, obstacle.max_x, _OBSTACLE_BUCKET_METRES):
            for bz in _bucket_range(obstacle.min_z, obstacle.max_z, _OBSTACLE_BUCKET_METRES):
                bucket_lists.setdefault((bx, bz), []).append(index)
    return _ObstacleIndex(
        tuple(obstacles),
        {key: tuple(values) for key, values in bucket_lists.items()},
    )


def _build_obstacle_index(dataset, projection) -> _ObstacleIndex:
    obstacles: list[_Obstacle] = []

    # Source-backed building geometry is authoritative. Treat polygon bounding
    # boxes conservatively; if a road runs close enough that the box matters,
    # retain the source centreline instead of gambling on a shortcut.
    for feature in getattr(dataset, "building_polygons", ()):
        for polygon in getattr(feature, "polygons", ()):
            points = tuple(projection.to_world(point) for point in polygon.outer)
            if not points:
                continue
            _add_bbox(
                obstacles,
                min(point[0] for point in points),
                min(point[1] for point in points),
                max(point[0] for point in points),
                max(point[1] for point in points),
            )

    # Point buildings have no reliable source footprint. Six metres is
    # deliberately conservative for the stock/procedural house families.
    for feature in getattr(dataset, "building_points", ()):
        point = projection.to_world(feature.point)
        _add_bbox(obstacles, *_point_bbox(point, 6.0))

    utility_radius = {
        "power_pole": 1.25,
        "power_tower": 6.0,
        "water_tower": 5.0,
    }
    for feature in getattr(dataset, "utility_points", ()):
        point = projection.to_world(feature.point)
        radius = utility_radius.get(str(feature.tags.get("utility", "")).casefold(), 1.5)
        _add_bbox(obstacles, *_point_bbox(point, radius))

    for feature in getattr(dataset, "individual_trees", ()):
        point = projection.to_world(feature.point)
        _add_bbox(obstacles, *_point_bbox(point, 1.25))

    obstacles.extend(
        _line_obstacles(
            getattr(dataset, "barriers", ()),
            projection,
            default_radius=0.50,
        )
    )
    # Tree rows become repeated stock tree objects. Give their source centreline
    # a modest crown/trunk envelope before _shortcut_clear adds road-width margin.
    obstacles.extend(
        _line_obstacles(
            getattr(dataset, "tree_rows", ()),
            projection,
            default_radius=_TREE_ROW_HALF_WIDTH_METRES,
        )
    )
    return _reindex(obstacles)


def _segment_intersects_box(
    start: tuple[float, float],
    end: tuple[float, float],
    obstacle: _Obstacle,
    margin: float,
) -> bool:
    min_x = obstacle.min_x - margin
    min_z = obstacle.min_z - margin
    max_x = obstacle.max_x + margin
    max_z = obstacle.max_z + margin
    x0, z0 = start
    x1, z1 = end
    dx, dz = x1 - x0, z1 - z0

    # Liang-Barsky clipping against the expanded rectangle.
    lower, upper = 0.0, 1.0
    for p, q in (
        (-dx, x0 - min_x),
        (dx, max_x - x0),
        (-dz, z0 - min_z),
        (dz, max_z - z0),
    ):
        if abs(p) <= 1.0e-12:
            if q < 0.0:
                return False
            continue
        value = q / p
        if p < 0.0:
            if value > upper:
                return False
            lower = max(lower, value)
        else:
            if value < lower:
                return False
            upper = min(upper, value)
    return lower <= upper


def _shortcut_clear(index: _ObstacleIndex, start, end) -> bool:
    margin = ROAD_SURFACE_HALF_WIDTH_METRES + ROAD_RELAXATION_OBJECT_MARGIN_METRES
    min_x = min(start[0], end[0]) - margin
    max_x = max(start[0], end[0]) + margin
    min_z = min(start[1], end[1]) - margin
    max_z = max(start[1], end[1]) + margin
    seen: set[int] = set()
    for bx in _bucket_range(min_x, max_x, _OBSTACLE_BUCKET_METRES):
        for bz in _bucket_range(min_z, max_z, _OBSTACLE_BUCKET_METRES):
            for obstacle_index in index.buckets.get((bx, bz), ()):
                if obstacle_index in seen:
                    continue
                seen.add(obstacle_index)
                obstacle = index.obstacles[obstacle_index]
                if _segment_intersects_box(start, end, obstacle, margin):
                    return False
    return True


def _point_segment_distance(point, start, end) -> float:
    dx, dz = end[0] - start[0], end[1] - start[1]
    length2 = dx * dx + dz * dz
    if length2 <= 1.0e-12:
        return math.dist(point, start)
    fraction = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dz
    ) / length2
    fraction = max(0.0, min(1.0, fraction))
    nearest = (start[0] + dx * fraction, start[1] + dz * fraction)
    return math.dist(point, nearest)


def _heading(start, end) -> float:
    return math.degrees(math.atan2(end[0] - start[0], end[1] - start[1])) % 360.0


def _candidate_is_relaxable(points, first: int, last: int, obstacles: _ObstacleIndex) -> bool:
    if last <= first + 1:
        return True
    start, end = points[first], points[last]
    chord = math.dist(start, end)
    if chord > MAXIMUM_RELAXED_CHORD_METRES + 1.0e-9:
        return False
    if chord <= 0.10:
        return False

    entry_heading = _heading(points[first], points[first + 1])
    exit_heading = _heading(points[last - 1], points[last])
    if (
        _p._heading_difference(entry_heading, exit_heading)
        > MAXIMUM_RELAXED_HEADING_CHANGE_DEGREES
    ):
        return False
    if any(
        _point_segment_distance(points[index], start, end)
        > MAXIMUM_LOCAL_RELAXATION_METRES
        for index in range(first + 1, last)
    ):
        return False
    return _shortcut_clear(obstacles, start, end)


def _simplify_open_run(points, obstacles: _ObstacleIndex):
    points = tuple(_p._clean_road_points(points))
    if len(points) < 3:
        return points
    result = [points[0]]
    first = 0
    while first < len(points) - 1:
        best = first + 1
        limit = min(len(points) - 1, first + MAXIMUM_RELAX_LOOKAHEAD_POINTS)
        for last in range(first + 2, limit + 1):
            if math.dist(points[first], points[last]) > MAXIMUM_RELAXED_CHORD_METRES:
                break
            if _candidate_is_relaxable(points, first, last, obstacles):
                best = last
        result.append(points[best])
        first = best
    return tuple(result)


def _rounded_road_run(points, **kwargs):
    if _ORIGINAL_ROUNDED is None:
        raise RuntimeError("stock road relaxation policy is not installed")
    rounded = _ORIGINAL_ROUNDED(points, **kwargs)
    context = _CONTEXT.get()
    if context is None:
        return rounded
    return _simplify_open_run(rounded, context.obstacles)


def _axis_index(objects):
    buckets: dict[tuple[str, int, int], list[_AxisRecord]] = {}
    for obj in objects:
        match = _geometry.stock_straight_match(obj.model_path)
        if match is None:
            continue
        family = match.group("family").casefold()
        axis = _surface._object_axis(obj)
        if axis is None:
            continue
        start, end = tuple(axis[0]), tuple(axis[1])
        midpoint = ((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5)
        bx = math.floor(midpoint[0] / _AXIS_BUCKET_METRES)
        bz = math.floor(midpoint[1] / _AXIS_BUCKET_METRES)
        buckets.setdefault((family, bx, bz), []).append(_AxisRecord(start, end, family))
    return {key: tuple(values) for key, values in buckets.items()}


def _segment_projection(point, start, end):
    dx, dz = end[0] - start[0], end[1] - start[1]
    length2 = dx * dx + dz * dz
    if length2 <= 1.0e-12:
        return 0.0, math.dist(point, start), (0.0, 1.0)
    fraction = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dz
    ) / length2
    nearest_fraction = max(0.0, min(1.0, fraction))
    nearest = (
        start[0] + dx * nearest_fraction,
        start[1] + dz * nearest_fraction,
    )
    length = math.sqrt(length2)
    return nearest_fraction, math.dist(point, nearest), (dx / length, dz / length)


def _connector_already_covered(axis_buckets, connector) -> bool:
    bx = math.floor(connector.point[0] / _AXIS_BUCKET_METRES)
    bz = math.floor(connector.point[1] / _AXIS_BUCKET_METRES)
    half_width = _geometry.STOCK_HALF_WIDTHS_METRES.get(connector.family, 1.75)
    lateral_limit = max(0.20, half_width - 0.25)
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            for axis in axis_buckets.get((connector.family, bx + dx, bz + dz), ()):
                fraction, distance, direction = _segment_projection(
                    connector.point, axis.start, axis.end
                )
                alignment = abs(
                    direction[0] * connector.outward[0]
                    + direction[1] * connector.outward[1]
                )
                if alignment < _AXIS_ALIGNMENT_COSINE:
                    continue
                # If the connector projects into the interior of an existing
                # larger piece and remains inside that road's visible width,
                # terrain is already covered. Do not add a short repair slab.
                if 1.0e-4 < fraction < 1.0 - 1.0e-4 and distance <= lateral_limit:
                    return True
                if distance < _surface.MINIMUM_CONNECTOR_COVER_GAP_METRES:
                    return True
    return False


def _connector_cover_plans(report):
    cap_count = min(
        int(getattr(report, "junction_cap_objects", 0)), len(report.objects)
    )
    if cap_count <= 0:
        return ()
    caps = report.objects[:cap_count]
    chains = report.objects[cap_count:]
    endpoints = _surface._chain_endpoints(chains)
    if not endpoints:
        return ()
    endpoint_buckets = _surface._endpoint_index(endpoints)
    axis_buckets = _axis_index(chains)

    used_endpoints: set[tuple[int, int]] = set()
    plans = []
    for cap in caps:
        for connector in _surface._native_cap_connectors(cap):
            if connector.family not in _surface._STOCK_FAMILIES:
                continue
            if _connector_already_covered(axis_buckets, connector):
                continue
            nearest = _surface._nearest_endpoint(endpoint_buckets, connector)
            if nearest is None:
                continue
            gap, endpoint = nearest
            endpoint_key = (endpoint.object_id, endpoint.endpoint_index)
            if endpoint_key in used_endpoints:
                continue
            if gap < _surface.MINIMUM_CONNECTOR_COVER_GAP_METRES:
                continue
            if gap > _surface.MAXIMUM_CONNECTOR_COVER_GAP_METRES:
                continue

            centre = (
                (connector.point[0] + endpoint.point[0]) * 0.5,
                (connector.point[1] + endpoint.point[1]) * 0.5,
            )
            vector = (
                endpoint.point[0] - connector.point[0],
                endpoint.point[1] - connector.point[1],
            )
            direction = _surface._normalised(vector)
            plans.append(
                _surface._CoverPlan(
                    _surface._cover_model(connector.family),
                    centre,
                    direction,
                )
            )
            used_endpoints.add(endpoint_key)
    return tuple(plans)


def _fit(dataset, projection, elevations, spec, *, starting_id=1, progress_callback=None):
    if _ORIGINAL_FIT is None:
        raise RuntimeError("stock road relaxation policy is not installed")
    context = _RelaxContext(_build_obstacle_index(dataset, projection))
    token = _CONTEXT.set(context)
    try:
        return _ORIGINAL_FIT(
            dataset,
            projection,
            elevations,
            spec,
            starting_id=starting_id,
            progress_callback=progress_callback,
        )
    finally:
        _CONTEXT.reset(token)


def install_stock_road_relaxation_policy() -> None:
    global _ORIGINAL_ROUNDED, _ORIGINAL_FIT, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_ROUNDED = _p._rounded_road_run
    _ORIGINAL_FIT = _p.fit_road_objects
    _p._rounded_road_run = _rounded_road_run
    _surface._connector_cover_plans = _connector_cover_plans
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
