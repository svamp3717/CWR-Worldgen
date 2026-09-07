# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep explicit bridges over mapped water on coarse CWA terrain grids.

A short OSM ``bridge=yes`` can cross real mapped water even when a coarse WRP
terrain grid has no sampled vertex below nominal sea level.  The historical
terrain-only bridge test then downgraded the crossing to an ordinary road, while
CWA's tide could still cover that road in game.

This policy gives the terrain solver, stock-road fitter, bridge renderer and
underlay cleanup one shared fallback: source-backed mapped water.  Terrain below
sea level remains the primary signal; mapped water only fills the coarse-grid
blind spot.  It also raises ordinary road/causeway fallback above the same
5.5-metre tide-safe floor used by stock bridges.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import math
from typing import Sequence

from . import bridge_render_policy as _bridge
from . import bridge_underlay_cleanup_policy as _cleanup
from . import osm as _osm
from . import playability as _playability
from . import terrain_solver as _terrain

_INSTALLED = False
_ORIGINAL_WATER_TEST = None
_ORIGINAL_STOCK_PLAN = None
_ORIGINAL_STOCK_FIT = None
_ORIGINAL_BRIDGE_SPANS = None
_ORIGINAL_BRIDGE_CORE_GENERATE = None
_ORIGINAL_SOLVE_TERRAIN = None

# Keep this synchronized with bridge_water_deck_clamp_policy: 5 m maximum tide
# plus a 0.5 m visual/wave margin.
_TIDE_SAFE_ROAD_CLEARANCE_METRES = 5.5
_SOURCE_WATER_SAMPLE_STEP_METRES = 0.5
_SOURCE_WATER_REFINEMENT_STEPS = 12

PointXZ = tuple[float, float]


@dataclass(frozen=True, slots=True)
class _ProjectedWaterPolygon:
    outer: tuple[PointXZ, ...]
    holes: tuple[tuple[PointXZ, ...], ...]
    bounds: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class _SourceContext:
    dataset: object
    projection: object
    water: tuple[_ProjectedWaterPolygon, ...]


_CONTEXT: ContextVar[_SourceContext | None] = ContextVar(
    "cwr_bridge_source_water_context", default=None
)


def _ring_bounds(points: Sequence[PointXZ]) -> tuple[float, float, float, float]:
    xs = [float(point[0]) for point in points]
    zs = [float(point[1]) for point in points]
    return min(xs), min(zs), max(xs), max(zs)


def _project_water(dataset, projection) -> tuple[_ProjectedWaterPolygon, ...]:
    result: list[_ProjectedWaterPolygon] = []
    for feature in getattr(dataset, "water", ()):
        for polygon in getattr(feature, "polygons", ()):
            outer = tuple(projection.to_world(point) for point in polygon.outer)
            if len(outer) < 3:
                continue
            holes = tuple(
                tuple(projection.to_world(point) for point in ring)
                for ring in getattr(polygon, "holes", ())
                if len(ring) >= 3
            )
            result.append(
                _ProjectedWaterPolygon(
                    outer=outer,
                    holes=holes,
                    bounds=_ring_bounds(outer),
                )
            )
    return tuple(result)


def _make_context(dataset, projection) -> _SourceContext:
    return _SourceContext(dataset, projection, _project_water(dataset, projection))


def _activate(dataset, projection):
    current = _CONTEXT.get()
    if (
        current is not None
        and current.dataset is dataset
        and current.projection == projection
    ):
        return None
    return _CONTEXT.set(_make_context(dataset, projection))


def _deactivate(token) -> None:
    if token is not None:
        _CONTEXT.reset(token)


def _point_on_segment(point: PointXZ, start: PointXZ, end: PointXZ) -> bool:
    px, pz = point
    ax, az = start
    bx, bz = end
    dx, dz = bx - ax, bz - az
    length2 = dx * dx + dz * dz
    # GeoJSON rings are normally explicitly closed, so the first ray-cast edge
    # can be a zero-length last->first segment.  Treat only the actual coincident
    # point as on that degenerate edge; otherwise every point would appear to be
    # on every closed ring, which is an impressively efficient way to erase lakes.
    if length2 <= 1.0e-14:
        return math.hypot(px - ax, pz - az) <= 1.0e-7
    cross = (px - ax) * dz - (pz - az) * dx
    tolerance = 1.0e-7 * max(1.0, math.hypot(dx, dz))
    if abs(cross) > tolerance:
        return False
    dot = (px - ax) * dx + (pz - az) * dz
    if dot < -tolerance:
        return False
    return dot <= length2 + tolerance


def _point_in_ring(point: PointXZ, ring: Sequence[PointXZ]) -> bool:
    if len(ring) < 3:
        return False
    px, pz = float(point[0]), float(point[1])
    inside = False
    previous = ring[-1]
    for current in ring:
        if _point_on_segment((px, pz), previous, current):
            return True
        x0, z0 = float(previous[0]), float(previous[1])
        x1, z1 = float(current[0]), float(current[1])
        if (z0 > pz) != (z1 > pz):
            x_cross = x0 + (pz - z0) * (x1 - x0) / (z1 - z0)
            if px < x_cross:
                inside = not inside
        previous = current
    return inside


def _point_in_water(point: PointXZ, polygons: Sequence[_ProjectedWaterPolygon]) -> bool:
    x, z = float(point[0]), float(point[1])
    for polygon in polygons:
        min_x, min_z, max_x, max_z = polygon.bounds
        if x < min_x or x > max_x or z < min_z or z > max_z:
            continue
        if not _point_in_ring((x, z), polygon.outer):
            continue
        if any(_point_in_ring((x, z), hole) for hole in polygon.holes):
            continue
        return True
    return False


def _polyline_measure(points: Sequence[PointXZ]):
    cleaned: list[PointXZ] = []
    for point in points:
        value = float(point[0]), float(point[1])
        if not cleaned or math.dist(value, cleaned[-1]) > 0.01:
            cleaned.append(value)
    if len(cleaned) < 2:
        return tuple(cleaned), (0.0,)
    cumulative = [0.0]
    for start, end in zip(cleaned, cleaned[1:]):
        cumulative.append(cumulative[-1] + math.dist(start, end))
    return tuple(cleaned), tuple(cumulative)


def _point_at(points: Sequence[PointXZ], cumulative: Sequence[float], distance: float) -> PointXZ:
    if len(points) < 2:
        return tuple(points[0]) if points else (0.0, 0.0)
    target = max(0.0, min(float(cumulative[-1]), float(distance)))
    for index in range(len(points) - 1):
        start_distance = float(cumulative[index])
        end_distance = float(cumulative[index + 1])
        if target > end_distance and index + 2 < len(points):
            continue
        length = end_distance - start_distance
        if length <= 1.0e-9:
            return float(points[index + 1][0]), float(points[index + 1][1])
        fraction = (target - start_distance) / length
        start, end = points[index], points[index + 1]
        return (
            float(start[0]) + (float(end[0]) - float(start[0])) * fraction,
            float(start[1]) + (float(end[1]) - float(start[1])) * fraction,
        )
    return float(points[-1][0]), float(points[-1][1])


def _source_mapped_water_interval(
    points: Sequence[PointXZ],
    context: _SourceContext | None = None,
) -> tuple[float, float] | None:
    """Return first/last along-distance inside normalized mapped water."""

    context = _CONTEXT.get() if context is None else context
    if context is None or not context.water:
        return None
    cleaned, cumulative = _polyline_measure(points)
    if len(cleaned) < 2 or cumulative[-1] <= 0.01:
        return None

    min_x = min(point[0] for point in cleaned)
    min_z = min(point[1] for point in cleaned)
    max_x = max(point[0] for point in cleaned)
    max_z = max(point[1] for point in cleaned)
    candidates = tuple(
        polygon
        for polygon in context.water
        if not (
            polygon.bounds[2] < min_x
            or polygon.bounds[0] > max_x
            or polygon.bounds[3] < min_z
            or polygon.bounds[1] > max_z
        )
    )
    if not candidates:
        return None

    total = float(cumulative[-1])
    count = max(1, int(math.ceil(total / _SOURCE_WATER_SAMPLE_STEP_METRES)))
    distances = tuple(total * index / count for index in range(count + 1))
    wet = tuple(
        _point_in_water(_point_at(cleaned, cumulative, distance), candidates)
        for distance in distances
    )
    indices = [index for index, value in enumerate(wet) if value]
    if not indices:
        return None

    first = indices[0]
    last = indices[-1]
    start = distances[first]
    end = distances[last]

    if first > 0 and not wet[first - 1]:
        low, high = distances[first - 1], distances[first]
        for _ in range(_SOURCE_WATER_REFINEMENT_STEPS):
            middle = (low + high) * 0.5
            if _point_in_water(_point_at(cleaned, cumulative, middle), candidates):
                high = middle
            else:
                low = middle
        start = high

    if last + 1 < len(distances) and not wet[last + 1]:
        low, high = distances[last], distances[last + 1]
        for _ in range(_SOURCE_WATER_REFINEMENT_STEPS):
            middle = (low + high) * 0.5
            if _point_in_water(_point_at(cleaned, cumulative, middle), candidates):
                low = middle
            else:
                high = middle
        end = low

    return (start, end) if end > start + 1.0e-4 else None


def _source_aware_water_test(
    points,
    elevations,
    *,
    cells: int,
    cell_size: float,
    sea_level: float,
    width: float = 6.0,
) -> bool:
    if _ORIGINAL_WATER_TEST(
        points,
        elevations,
        cells=cells,
        cell_size=cell_size,
        sea_level=sea_level,
        width=width,
    ):
        return True
    return _source_mapped_water_interval(points) is not None


def _mapped_water_stock_plan(points, elevations, spec):
    """Create a minimal fixed-stock span when coarse terrain misses mapped water."""

    plan = _ORIGINAL_STOCK_PLAN(points, elevations, spec)
    if plan is not None:
        return plan
    interval = _source_mapped_water_interval(points)
    if interval is None:
        return None

    cleaned, cumulative = _polyline_measure(points)
    if len(cleaned) < 2:
        return None
    start_distance, end_distance = interval
    wet_start = _point_at(cleaned, cumulative, start_distance)
    wet_end = _point_at(cleaned, cumulative, end_distance)
    dx = wet_end[0] - wet_start[0]
    dz = wet_end[1] - wet_start[1]
    wet_length = math.hypot(dx, dz)
    if wet_length <= 0.05:
        return None
    unit_x, unit_z = dx / wet_length, dz / wet_length

    module_length = float(_bridge._STOCK_MODULE_SPACING_METRES)
    module_count = max(1, int(math.ceil((wet_length - 1.0e-9) / module_length)))
    target_length = module_count * module_length
    half_target = target_length * 0.5
    centre = (
        (wet_start[0] + wet_end[0]) * 0.5,
        (wet_start[1] + wet_end[1]) * 0.5,
    )

    projections = [
        (point[0] - centre[0]) * unit_x + (point[1] - centre[1]) * unit_z
        for point in cleaned
    ]
    source_min = min(projections)
    source_max = max(projections)
    if source_max - source_min >= target_length:
        minimum_shift = source_min + half_target
        maximum_shift = source_max - half_target
        shift = max(minimum_shift, min(0.0, maximum_shift))
        centre = centre[0] + unit_x * shift, centre[1] + unit_z * shift

    start = centre[0] - unit_x * half_target, centre[1] - unit_z * half_target
    end = centre[0] + unit_x * half_target, centre[1] + unit_z * half_target
    world_size = float(getattr(spec, "world_size", 0.0) or 0.0)
    if world_size > 0.0 and not all(
        0.0 <= coordinate < world_size
        for point in (start, end)
        for coordinate in point
    ):
        return None

    return _bridge.StockBridgeSpanPlan(
        points=(start, end),
        module_count=module_count,
        wet_start=wet_start,
        wet_end=wet_end,
        wet_length=wet_length,
    )


def install_bridge_source_water_policy() -> None:
    """Install one source-water fallback across bridge planning and road grading."""

    global _INSTALLED
    global _ORIGINAL_WATER_TEST
    global _ORIGINAL_STOCK_PLAN
    global _ORIGINAL_STOCK_FIT
    global _ORIGINAL_BRIDGE_SPANS
    global _ORIGINAL_BRIDGE_CORE_GENERATE
    global _ORIGINAL_SOLVE_TERRAIN

    if _INSTALLED:
        return

    _ORIGINAL_WATER_TEST = _osm.road_span_has_in_game_water
    _ORIGINAL_STOCK_PLAN = _bridge.stock_bridge_span_plan
    _ORIGINAL_STOCK_FIT = _playability._fit_stock_piece_road_objects
    _ORIGINAL_BRIDGE_SPANS = _cleanup._bridge_spans
    _ORIGINAL_BRIDGE_CORE_GENERATE = _bridge._ORIGINAL_GENERATE_WORLD_OBJECTS
    _ORIGINAL_SOLVE_TERRAIN = _terrain.solve_terrain_constraints

    # These modules imported the historical terrain-only helper by value. Replace
    # all live bindings so each caller sees the same context-aware answer.
    _osm.road_span_has_in_game_water = _source_aware_water_test
    _playability.road_span_has_in_game_water = _source_aware_water_test
    _terrain.road_span_has_in_game_water = _source_aware_water_test
    _bridge.stock_bridge_span_plan = _mapped_water_stock_plan

    @wraps(_ORIGINAL_STOCK_FIT)
    def source_aware_stock_fit(dataset, projection, elevations, spec, *args, **kwargs):
        token = _activate(dataset, projection)
        try:
            return _ORIGINAL_STOCK_FIT(
                dataset, projection, elevations, spec, *args, **kwargs
            )
        finally:
            _deactivate(token)

    @wraps(_ORIGINAL_BRIDGE_SPANS)
    def source_aware_bridge_spans(dataset, projection, elevations, spec):
        token = _activate(dataset, projection)
        try:
            return _ORIGINAL_BRIDGE_SPANS(dataset, projection, elevations, spec)
        finally:
            _deactivate(token)

    @wraps(_ORIGINAL_BRIDGE_CORE_GENERATE)
    def source_aware_bridge_core(
        dataset, projection, raster, elevations, spec, *args, **kwargs
    ):
        token = _activate(dataset, projection)
        try:
            return _ORIGINAL_BRIDGE_CORE_GENERATE(
                dataset, projection, raster, elevations, spec, *args, **kwargs
            )
        finally:
            _deactivate(token)

    @wraps(_ORIGINAL_SOLVE_TERRAIN)
    def source_aware_solve(
        elevations, dataset, projection, raster, spec, *args, **kwargs
    ):
        token = _activate(dataset, projection)
        try:
            return _ORIGINAL_SOLVE_TERRAIN(
                elevations, dataset, projection, raster, spec, *args, **kwargs
            )
        finally:
            _deactivate(token)

    _playability._fit_stock_piece_road_objects = source_aware_stock_fit
    _cleanup._bridge_spans = source_aware_bridge_spans
    _bridge._ORIGINAL_GENERATE_WORLD_OBJECTS = source_aware_bridge_core
    _terrain.solve_terrain_constraints = source_aware_solve

    # If a crossing is intentionally rendered as an ordinary road/causeway, its
    # terrain floor must also clear CWA's tide range.  The previous 0.35 m floor
    # was only safe for a static nominal water plane.
    _terrain.ROAD_WATER_MINIMUM_CLEARANCE_METRES = max(
        float(_terrain.ROAD_WATER_MINIMUM_CLEARANCE_METRES),
        _TIDE_SAFE_ROAD_CLEARANCE_METRES,
    )

    _INSTALLED = True
