# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import math
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from shapely import box as vectorized_box, intersects as vectorized_intersects, prepare as prepare_geometry
from shapely.geometry import LineString, Point, Polygon, box

from .model import ConstraintPlayabilitySpec
from .osm import (
    BuildingPlacementPlan, BboxProjection, GeoPolygon, OsmDataset, OsmLineFeature, OsmRaster,
    conservative_water_interior_mask, renderable_water_mask, road_bridge_crosses_ditch_only,
    road_span_has_in_game_water, road_width_metres,
)


PRIORITY_BOUNDARY = 100
PRIORITY_NATURAL = 200
PRIORITY_WATERCOURSE = 300
PRIORITY_BUILDING = 400
PRIORITY_MINOR_ROAD = 500
PRIORITY_MAJOR_ROAD = 600
PRIORITY_SEMANTIC_BUILDING_CORE = 650
PRIORITY_BRIDGE_TUNNEL = 700
PRIORITY_WATER = 800
PRIORITY_OSM_DRY_LAND = 850
# A side-hill road platform must beat shoreline/dry-land shaping. Otherwise a
# coastal bank can simply reintroduce the transverse slope through the rendered
# road after the road bench has been detected correctly.
PRIORITY_ROAD_SIDEHILL_BENCH = 870
# Ordinary roads occasionally cross mapped water without an OSM bridge tag.
# Give a narrow causeway fill permission to override the global water-bed rule,
# while explicit bridges/tunnels keep their existing higher-priority behavior.
PRIORITY_ROAD_WATER_FILL = 875
PRIORITY_BRIDGE_SUPPORT = 900

_MAJOR_HIGHWAYS = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link",
}

TERRAIN_SMOOTHING_REFERENCE_WORLD_SIZE_METRES = 6_400.0
OSM_DRY_LAND_MINIMUM_CLEARANCE_METRES = 7.0
ROAD_WATER_MINIMUM_CLEARANCE_METRES = 0.35
ROAD_WATER_BANK_WIDTH_CELLS = 0.75
# Side-hill roads need a transverse bench, not a globally flatter longitudinal
# profile. These thresholds intentionally key off the terrain gradient *across*
# the road so a road climbing a steep hill is left alone, while one traversing
# the same hill receives a narrow cut/fill platform beneath the carriageway.
ROAD_SIDEHILL_TRIGGER_SLOPE_PERCENT = 35.0
ROAD_SIDEHILL_FULL_BENCH_SLOPE_PERCENT = 70.0
ROAD_SIDEHILL_BENCH_EXTRA_CELLS = 1.0
ROAD_SIDEHILL_BLEND_CELLS = 1.25
ROAD_SIDEHILL_MINIMUM_PLATFORM_MARGIN_METRES = 3.0


@dataclass(frozen=True, slots=True)
class EffectiveTerrainSmoothing:
    scale: float
    shoreline_transition_cells: int
    lake_shore_smoothing_cells: int
    world_edge_blend_cells: int
    natural_smoothing_strength: float
    solver_iterations: int


def _effective_terrain_smoothing(spec: ConstraintPlayabilitySpec) -> EffectiveTerrainSmoothing:
    """Scale terrain relaxation from the 6.4 km reference-world defaults."""

    scale = max(
        1.0,
        float(spec.world_size) / TERRAIN_SMOOTHING_REFERENCE_WORLD_SIZE_METRES,
    )

    def scaled_cells(value: int) -> int:
        value = max(0, int(value))
        return 0 if value == 0 else min(32, max(1, int(round(value * scale))))

    base_strength = max(0.0, min(1.0, float(spec.natural_smoothing_strength)))
    requested_strength = base_strength * scale * scale
    effective_strength = min(1.0, requested_strength)
    base_iterations = max(0, int(spec.solver_iterations))
    iteration_multiplier = (
        max(1.0, requested_strength / max(1.0e-9, effective_strength))
        if effective_strength > 0.0
        else 1.0
    )
    effective_iterations = min(
        200,
        max(0, int(round(base_iterations * iteration_multiplier))),
    )
    return EffectiveTerrainSmoothing(
        scale=scale,
        shoreline_transition_cells=scaled_cells(spec.shoreline_transition_cells),
        lake_shore_smoothing_cells=scaled_cells(spec.lake_shore_smoothing_cells),
        world_edge_blend_cells=scaled_cells(spec.world_edge_blend_cells),
        natural_smoothing_strength=effective_strength,
        solver_iterations=effective_iterations,
    )


@dataclass(frozen=True, slots=True)
class ConstraintTerrainReport:
    elevations: tuple[float, ...]
    changed_cells: int
    road_seed_cells: int
    building_pad_cells: int
    maximum_cut: float
    maximum_fill: float
    maximum_road_slope_before_percent: float
    maximum_road_slope_after_percent: float
    building_roughness_before: float
    building_roughness_after: float
    solver: str
    smoothing_reference_scale: float
    natural_smoothing_strength: float
    iterations: int
    constrained_cells: int
    hard_constraint_cells: int
    water_cells: int
    deep_water_cells: int
    uncertain_water_cells_preserved: int
    protected_shore_cells: int
    shoreline_transition_cells: int
    coastal_water_components: int
    inland_water_components: int
    lake_shore_cells: int
    osm_land_floor_cells: int
    lake_shore_smoothing_cells: int
    lake_shore_maximum_slope_percent: float
    maximum_lake_shore_slope_before_percent: float
    maximum_lake_shore_slope_after_percent: float
    bridge_segments: int
    tunnel_segments_excluded: int
    embankment_segments: int
    road_water_fill_cells: int
    road_sidehill_segments: int
    road_sidehill_bench_cells: int
    major_road_cells: int
    minor_road_cells: int
    watercourse_cells: int
    downhill_violations_before: int
    downhill_violations_after: int
    downhill_total_violations_after: int
    downhill_protected_crossings: int
    water_roughness_before: float
    water_roughness_after: float
    out_of_bounds_sampling: str
    edge_blend_cells: int
    total_cut_volume_m3: float
    total_fill_volume_m3: float
    category_adjustments: Mapping[str, Mapping[str, float | int]]
    priority_order: tuple[str, ...]


@dataclass(slots=True)
class _ConstraintField:
    priorities: list[int]
    targets: list[float]
    strengths: list[float]
    hard: list[bool]
    categories: list[str]

    @classmethod
    def create(cls, size: int) -> "_ConstraintField":
        return cls(
            priorities=[0] * size,
            targets=[0.0] * size,
            strengths=[0.0] * size,
            hard=[False] * size,
            categories=["natural"] * size,
        )

    def apply(
        self,
        index: int,
        target: float,
        *,
        priority: int,
        strength: float,
        hard: bool,
        category: str,
    ) -> None:
        if not math.isfinite(target) or strength <= 0.0:
            return
        strength = max(0.0, min(1.0, strength))
        current_priority = self.priorities[index]
        if priority > current_priority:
            self.priorities[index] = priority
            self.targets[index] = target
            self.strengths[index] = strength
            self.hard[index] = hard
            self.categories[index] = category
        elif priority == current_priority:
            old_weight = self.strengths[index]
            total = old_weight + strength
            if total > 0:
                self.targets[index] = (self.targets[index] * old_weight + target * strength) / total
            self.strengths[index] = min(1.0, total)
            self.hard[index] = self.hard[index] or hard
            if category < self.categories[index]:
                self.categories[index] = category
        else:
            return


def _sample_elevation(elevations: Sequence[float], cells: int, cell_size: float, x: float, z: float) -> float:
    fx = max(0.0, min(cells - 1.0, x / cell_size))
    fz = max(0.0, min(cells - 1.0, z / cell_size))
    x0 = int(math.floor(fx))
    z0 = int(math.floor(fz))
    x1 = min(cells - 1, x0 + 1)
    z1 = min(cells - 1, z0 + 1)
    tx = fx - x0
    tz = fz - z0
    a = elevations[z0 * cells + x0] * (1.0 - tx) + elevations[z0 * cells + x1] * tx
    b = elevations[z1 * cells + x0] * (1.0 - tx) + elevations[z1 * cells + x1] * tx
    return a * (1.0 - tz) + b * tz

def _cell_center(index: int, cells: int, cell_size: float) -> tuple[float, float]:
    x = index % cells
    z = index // cells
    return x * cell_size, z * cell_size


def _cell_polygon(index: int, cells: int, cell_size: float) -> Polygon:
    x = index % cells
    z = index // cells
    half = cell_size * 0.5
    return box(x * cell_size - half, z * cell_size - half, x * cell_size + half, z * cell_size + half)


def _candidate_cells(bounds: tuple[float, float, float, float], cells: int, cell_size: float) -> Iterable[int]:
    min_x, min_z, max_x, max_z = bounds
    x0 = max(0, min(cells - 1, int(math.ceil(min_x / cell_size - 0.5))))
    z0 = max(0, min(cells - 1, int(math.ceil(min_z / cell_size - 0.5))))
    x1 = max(0, min(cells - 1, int(math.floor(max_x / cell_size + 0.5))))
    z1 = max(0, min(cells - 1, int(math.floor(max_z / cell_size + 0.5))))
    for z in range(z0, z1 + 1):
        for x in range(x0, x1 + 1):
            yield z * cells + x


def _local_polygon(polygon: GeoPolygon, projection: BboxProjection) -> Polygon:
    outer = [projection.to_world(point) for point in polygon.outer]
    holes = [[projection.to_world(point) for point in ring] for ring in polygon.holes]
    geometry = Polygon(outer, holes)
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    return geometry


def _line_geometry(feature: OsmLineFeature, projection: BboxProjection) -> LineString | None:
    points = [projection.to_world(point) for point in feature.points]
    cleaned: list[tuple[float, float]] = []
    for point in points:
        if not cleaned or math.dist(point, cleaned[-1]) > 0.05:
            cleaned.append(point)
    if len(cleaned) < 2:
        return None
    line = LineString(cleaned)
    return line if line.length > 0.05 else None


def _dry_osm_land_floor_cells(
    dataset: OsmDataset,
    projection: BboxProjection,
    raster: OsmRaster,
    spec: ConstraintPlayabilitySpec,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Return sorted terrain-cell indices backed by mapped non-water OSM land.

    This used to build a Python ``set[int]`` and then instantiate one Shapely
    polygon for every candidate terrain cell. On multi-thousand-cell worlds that
    produced millions of Python objects and hash-table insertions. Keep the exact
    cell-intersection semantics, but do the mask combination and polygon tests in
    vectorized C loops instead.
    """

    total_cells = len(raster.water)
    water = np.asarray(raster.water, dtype=np.bool_)
    marked = np.asarray(raster.forest, dtype=np.bool_).copy()
    for layer in (raster.farmland, raster.urban, raster.roads, raster.buildings):
        np.logical_or(marked, np.asarray(layer, dtype=np.bool_), out=marked)
    np.logical_and(marked, np.logical_not(water), out=marked)

    polygon_groups = (
        dataset.forests,
        dataset.farmland,
        dataset.urban,
        dataset.sites,
        dataset.surface_areas,
        dataset.rural_vegetation,
        dataset.aeroway_areas,
    )
    polygon_feature_total = sum(len(group) for group in polygon_groups)
    polygon_feature_index = 0
    polygon_progress_interval = max(1, polygon_feature_total // 16)
    half = spec.cell_size * 0.5
    # Bound temporary Shapely arrays. 65k boxes is large enough to amortize the
    # ufunc call but small enough to avoid ugly memory spikes on 50 km worlds.
    chunk_size = 65_536

    if progress_callback is not None:
        progress_callback(
            f"Collecting dry-land raster cells and {polygon_feature_total:,} mapped polygon features"
        )

    for group in polygon_groups:
        for feature in group:
            polygon_feature_index += 1
            for geo_polygon in feature.polygons:
                geometry = _local_polygon(geo_polygon, projection)
                if geometry.is_empty:
                    continue
                prepare_geometry(geometry)
                min_x, min_z, max_x, max_z = geometry.bounds
                x0 = max(0, min(spec.cells - 1, int(math.ceil(min_x / spec.cell_size - 0.5))))
                z0 = max(0, min(spec.cells - 1, int(math.ceil(min_z / spec.cell_size - 0.5))))
                x1 = max(0, min(spec.cells - 1, int(math.floor(max_x / spec.cell_size + 0.5))))
                z1 = max(0, min(spec.cells - 1, int(math.floor(max_z / spec.cell_size + 0.5))))
                width = x1 - x0 + 1
                height = z1 - z0 + 1
                if width <= 0 or height <= 0:
                    continue
                candidate_count = width * height
                for start in range(0, candidate_count, chunk_size):
                    stop = min(candidate_count, start + chunk_size)
                    offsets = np.arange(start, stop, dtype=np.int64)
                    xs = x0 + offsets % width
                    zs = z0 + offsets // width
                    indices = zs * spec.cells + xs
                    dry = np.logical_not(water[indices])
                    if not np.any(dry):
                        continue
                    if not np.all(dry):
                        xs = xs[dry]
                        zs = zs[dry]
                        indices = indices[dry]
                    x_centres = xs.astype(np.float64) * spec.cell_size
                    z_centres = zs.astype(np.float64) * spec.cell_size
                    cell_boxes = vectorized_box(
                        x_centres - half,
                        z_centres - half,
                        x_centres + half,
                        z_centres + half,
                    )
                    hits = vectorized_intersects(geometry, cell_boxes)
                    if np.any(hits):
                        marked[indices[hits]] = True

            if (
                progress_callback is not None
                and (
                    polygon_feature_index == polygon_feature_total
                    or polygon_feature_index % polygon_progress_interval == 0
                )
            ):
                progress_callback(
                    f"Collecting OSM dry-land polygons {polygon_feature_index:,}/{polygon_feature_total:,}"
                )

    point_groups = (
        dataset.building_points,
        dataset.places,
        dataset.landmarks,
        dataset.individual_trees,
        dataset.utility_points,
    )
    point_total = sum(len(group) for group in point_groups)
    point_index = 0
    point_progress_interval = max(1, point_total // 8)
    point_radius = max(1.0, spec.cell_size * 0.55)
    if progress_callback is not None and point_total:
        progress_callback(f"Collecting {point_total:,} mapped dry-land point features")
    for group in point_groups:
        for feature in group:
            point_index += 1
            x, z = projection.to_world(feature.point)
            bounds = (x - point_radius, z - point_radius, x + point_radius, z + point_radius)
            for index in _candidate_cells(bounds, spec.cells, spec.cell_size):
                if not raster.water[index]:
                    marked[index] = True
            if (
                progress_callback is not None
                and (point_index == point_total or point_index % point_progress_interval == 0)
            ):
                progress_callback(f"Collecting OSM dry-land points {point_index:,}/{point_total:,}")

    if marked.size != total_cells:
        raise ValueError("OSM raster size does not match terrain grid")
    return np.flatnonzero(marked)


def _components(mask: Sequence[bool], cells: int) -> list[list[int]]:
    remaining = {index for index, value in enumerate(mask) if value}
    components: list[list[int]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        queue = deque([seed])
        component: list[int] = []
        while queue:
            index = queue.popleft()
            component.append(index)
            x, z = index % cells, index // cells
            for nx, nz in ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)):
                neighbour = nz * cells + nx
                if 0 <= nx < cells and 0 <= nz < cells and neighbour in remaining:
                    remaining.remove(neighbour)
                    queue.append(neighbour)
        components.append(component)
    return components


def _distance_from_mask(mask: Sequence[bool], cells: int, maximum: int) -> list[int]:
    distances = [10**9] * len(mask)
    queue: deque[int] = deque()
    for index, value in enumerate(mask):
        if value:
            distances[index] = 0
            queue.append(index)
    while queue:
        index = queue.popleft()
        distance = distances[index]
        if distance >= maximum:
            continue
        x, z = index % cells, index // cells
        for nx, nz in ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)):
            if not (0 <= nx < cells and 0 <= nz < cells):
                continue
            neighbour = nz * cells + nx
            if distances[neighbour] > distance + 1:
                distances[neighbour] = distance + 1
                queue.append(neighbour)
    return distances


def _component_touches_world_edge(component: Sequence[int], cells: int) -> bool:
    return any(
        index % cells in {0, cells - 1} or index // cells in {0, cells - 1}
        for index in component
    )


def _mask_from_components(components: Sequence[Sequence[int]], size: int) -> list[bool]:
    mask = [False] * size
    for component in components:
        for index in component:
            mask[index] = True
    return mask


def _euclidean_distance_from_mask(mask: Sequence[bool], cells: int, maximum: int) -> list[float]:
    """Return an eight-neighbour distance field measured in terrain cells."""
    distances = [math.inf] * len(mask)
    queue: list[tuple[float, int]] = []
    for index, value in enumerate(mask):
        if value:
            distances[index] = 0.0
            heapq.heappush(queue, (0.0, index))
    if maximum <= 0:
        return distances
    neighbours = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
    )
    while queue:
        distance, index = heapq.heappop(queue)
        if distance != distances[index] or distance >= maximum:
            continue
        x, z = index % cells, index // cells
        for dx, dz, cost in neighbours:
            nx, nz = x + dx, z + dz
            if not (0 <= nx < cells and 0 <= nz < cells):
                continue
            candidate = distance + cost
            if candidate > maximum or candidate >= distances[nz * cells + nx]:
                continue
            neighbour = nz * cells + nx
            distances[neighbour] = candidate
            heapq.heappush(queue, (candidate, neighbour))
    return distances


def _maximum_lake_bank_slope(
    elevations: Sequence[float],
    lake_water: Sequence[bool],
    lake_bank: Sequence[bool],
    cells: int,
    cell_size: float,
    sea_level: float,
) -> float:
    """Measure the steepest lake-bank edge, using the water surface rather than bed depth."""
    active = [water or bank for water, bank in zip(lake_water, lake_bank)]
    maximum = 0.0
    for index in range(len(elevations)):
        x, z = index % cells, index // cells
        for nx, nz, distance in ((x + 1, z, cell_size), (x, z + 1, cell_size)):
            if nx >= cells or nz >= cells:
                continue
            neighbour = nz * cells + nx
            if not (active[index] or active[neighbour]):
                continue
            left = sea_level if lake_water[index] else elevations[index]
            right = sea_level if lake_water[neighbour] else elevations[neighbour]
            maximum = max(maximum, abs(right - left) / distance * 100.0)
    return maximum


def _profile(line: LineString, original: Sequence[float], spec: ConstraintPlayabilitySpec, maximum_grade: float) -> tuple[list[float], list[float]]:
    spacing = max(2.0, min(10.0, spec.cell_size / 3.0))
    count = max(1, int(math.ceil(line.length / spacing)))
    distances = [line.length * index / count for index in range(count + 1)]
    heights = []
    for distance in distances:
        point = line.interpolate(distance)
        heights.append(_sample_elevation(original, spec.cells, spec.cell_size, point.x, point.y))
    ratio = maximum_grade / 100.0
    for index in range(1, len(heights)):
        limit = max(0.01, distances[index] - distances[index - 1]) * ratio
        heights[index] = min(max(heights[index], heights[index - 1] - limit), heights[index - 1] + limit)
    for index in range(len(heights) - 2, -1, -1):
        limit = max(0.01, distances[index + 1] - distances[index]) * ratio
        heights[index] = min(max(heights[index], heights[index + 1] - limit), heights[index + 1] + limit)
    return distances, heights


def _linear_profile(line: LineString, original: Sequence[float], spec: ConstraintPlayabilitySpec) -> tuple[list[float], list[float]]:
    start = Point(line.coords[0])
    end = Point(line.coords[-1])
    start_h = _sample_elevation(original, spec.cells, spec.cell_size, start.x, start.y)
    end_h = _sample_elevation(original, spec.cells, spec.cell_size, end.x, end.y)
    return [0.0, line.length], [start_h, end_h]


def _profile_height(distance: float, distances: Sequence[float], heights: Sequence[float]) -> float:
    if distance <= distances[0]:
        return heights[0]
    if distance >= distances[-1]:
        return heights[-1]
    low = 0
    high = len(distances) - 1
    while low + 1 < high:
        middle = (low + high) // 2
        if distances[middle] <= distance:
            low = middle
        else:
            high = middle
    span = max(1e-9, distances[high] - distances[low])
    fraction = (distance - distances[low]) / span
    return heights[low] * (1.0 - fraction) + heights[high] * fraction




def _road_cross_slope_profile(
    line: LineString,
    original: Sequence[float],
    spec: ConstraintPlayabilitySpec,
    distances: Sequence[float],
    width: float,
) -> list[float]:
    """Measure source-terrain slope perpendicular to a road centreline.

    Longitudinal grade is deliberately ignored here. A road can climb a steep
    mountain normally; this profile only detects the troublesome case where the
    road runs along the side of that mountain and one edge is buried while the
    opposite edge floats above the slope.

    One cross-section sample is not reliable on coarse DEMs. A road can sit on
    the narrow shoulder of a coastal ridge while both near samples still land on
    that same shoulder, hiding the cliff immediately beyond it. Sample several
    transverse spans and keep the steepest result. This catches the real
    side-hill relief without confusing ordinary longitudinal hill climbing with
    cross-slope.
    """

    if not distances:
        return []
    near_offset = max(
        float(width) * 0.75,
        float(spec.cell_size) * 0.65,
        2.0,
    )
    sample_offsets = (
        near_offset,
        max(near_offset * 1.5, float(width) * 1.25, float(spec.cell_size) * 1.15),
        max(near_offset * 2.5, float(width) * 2.0, float(spec.cell_size) * 2.0),
    )
    tangent_probe = max(2.0, min(12.0, float(spec.cell_size) * 0.5))
    slopes: list[float] = []
    for distance in distances:
        before = line.interpolate(max(0.0, float(distance) - tangent_probe))
        after = line.interpolate(min(line.length, float(distance) + tangent_probe))
        dx = after.x - before.x
        dz = after.y - before.y
        magnitude = math.hypot(dx, dz)
        if magnitude <= 1.0e-6:
            slopes.append(0.0)
            continue
        # Unit normal in world X/Z. Sampling symmetrically cancels most of the
        # longitudinal grade even around gentle curves.
        nx = -dz / magnitude
        nz = dx / magnitude
        centre = line.interpolate(float(distance))
        maximum_cross_slope = 0.0
        for sample_offset in sample_offsets:
            left_h = _sample_elevation(
                original,
                spec.cells,
                spec.cell_size,
                centre.x + nx * sample_offset,
                centre.y + nz * sample_offset,
            )
            right_h = _sample_elevation(
                original,
                spec.cells,
                spec.cell_size,
                centre.x - nx * sample_offset,
                centre.y - nz * sample_offset,
            )
            cross_slope = (
                abs(left_h - right_h)
                / max(0.01, sample_offset * 2.0)
                * 100.0
            )
            maximum_cross_slope = max(maximum_cross_slope, cross_slope)
        slopes.append(maximum_cross_slope)
    return slopes


def _road_corridor_intersects_mask(
    line: LineString,
    mask: Sequence[bool],
    spec: ConstraintPlayabilitySpec,
    width: float,
) -> bool:
    """Return whether a road-width corridor covers any selected terrain cell."""

    radius = max(float(width) * 0.5, float(spec.cell_size) * 0.35)
    corridor = line.buffer(radius, cap_style=2, join_style=2)
    for index in _candidate_cells(corridor.bounds, spec.cells, spec.cell_size):
        if not mask[index]:
            continue
        if corridor.covers(Point(_cell_center(index, spec.cells, spec.cell_size))):
            return True
    return False


def _road_sidehill_factor(cross_slope_percent: float) -> float:
    """Return a smooth 0..1 terrace strength for steep transverse slopes."""

    if cross_slope_percent <= ROAD_SIDEHILL_TRIGGER_SLOPE_PERCENT:
        return 0.0
    span = max(
        0.01,
        ROAD_SIDEHILL_FULL_BENCH_SLOPE_PERCENT - ROAD_SIDEHILL_TRIGGER_SLOPE_PERCENT,
    )
    value = max(
        0.0,
        min(
            1.0,
            (cross_slope_percent - ROAD_SIDEHILL_TRIGGER_SLOPE_PERCENT) / span,
        ),
    )
    return value * value * (3.0 - 2.0 * value)


def _road_width(tags: Mapping[str, str]) -> float:
    raw = tags.get("width", "")
    try:
        value = float(str(raw).replace(",", "."))
    except ValueError:
        value = road_width_metres(tags)
    if not math.isfinite(value) or value <= 0:
        value = road_width_metres(tags)
    return max(1.0, min(40.0, value))


def _road_slope_percent(elevations: Sequence[float], dataset: OsmDataset, projection: BboxProjection, spec: ConstraintPlayabilitySpec) -> float:
    maximum = 0.0
    spacing = max(2.0, spec.cell_size * 0.35)
    for feature in dataset.roads:
        if feature.tags.get("tunnel") not in {None, "", "no"}:
            continue
        line = _line_geometry(feature, projection)
        if line is None:
            continue
        count = max(1, int(math.ceil(line.length / spacing)))
        previous = line.interpolate(0.0)
        previous_h = _sample_elevation(elevations, spec.cells, spec.cell_size, previous.x, previous.y)
        for index in range(1, count + 1):
            point = line.interpolate(line.length * index / count)
            height = _sample_elevation(elevations, spec.cells, spec.cell_size, point.x, point.y)
            distance = max(0.01, point.distance(previous))
            maximum = max(maximum, abs(height - previous_h) / distance * 100.0)
            previous, previous_h = point, height
    return maximum


def _component_roughness(elevations: Sequence[float], components: Sequence[Sequence[int]]) -> float:
    maximum = 0.0
    for component in components:
        if component:
            values = [elevations[index] for index in component]
            maximum = max(maximum, max(values) - min(values))
    return maximum


def _water_roughness(elevations: Sequence[float], components: Sequence[Sequence[int]]) -> float:
    return _component_roughness(elevations, components)


def _point_cell_index(point: Point, spec: ConstraintPlayabilitySpec) -> int:
    x = min(spec.cells - 1, max(0, int(point.x / spec.cell_size)))
    z = min(spec.cells - 1, max(0, int(point.y / spec.cell_size)))
    return z * spec.cells + x




def _dilate_mask(mask: Sequence[bool], cells: int, radius: int = 1) -> list[bool]:
    result = list(mask)
    if radius <= 0:
        return result
    seeds = [index for index, value in enumerate(mask) if value]
    for index in seeds:
        x, z = index % cells, index // cells
        for dz in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, nz = x + dx, z + dz
                if 0 <= nx < cells and 0 <= nz < cells:
                    result[nz * cells + nx] = True
    return result


def _watercourse_violation_stats(
    elevations: Sequence[float],
    dataset: OsmDataset,
    projection: BboxProjection,
    spec: ConstraintPlayabilitySpec,
    protected: Sequence[bool] | None = None,
) -> tuple[int, int, int]:
    resolvable = 0
    total = 0
    protected_crossings = 0
    for feature in dataset.watercourses:
        if feature.tags.get("tunnel") not in {None, "", "no"}:
            continue
        line = _line_geometry(feature, projection)
        if line is None:
            continue
        start = Point(line.coords[0])
        end = Point(line.coords[-1])
        start_h = _sample_elevation(elevations, spec.cells, spec.cell_size, start.x, start.y)
        end_h = _sample_elevation(elevations, spec.cells, spec.cell_size, end.x, end.y)
        reverse = start_h < end_h
        count = max(2, int(math.ceil(line.length / max(5.0, spec.cell_size / 3.0))))
        previous_h: float | None = None
        previous_protected = False
        for index in range(count + 1):
            distance = line.length * (count - index if reverse else index) / count
            point = line.interpolate(distance)
            height = _sample_elevation(elevations, spec.cells, spec.cell_size, point.x, point.y)
            current_protected = bool(
                protected
                and any(protected[cell] for cell in _bilinear_indices(point.x, point.y, spec))
            )
            if previous_h is not None and height > previous_h + 0.02:
                total += 1
                if previous_protected or current_protected:
                    protected_crossings += 1
                else:
                    resolvable += 1
            previous_h = height
            previous_protected = current_protected
    return resolvable, total, protected_crossings




def _bilinear_indices(x: float, z: float, spec: ConstraintPlayabilitySpec) -> tuple[int, ...]:
    fx = max(0.0, min(spec.cells - 1.0, x / spec.cell_size))
    fz = max(0.0, min(spec.cells - 1.0, z / spec.cell_size))
    x0 = int(math.floor(fx))
    z0 = int(math.floor(fz))
    x1 = min(spec.cells - 1, x0 + 1)
    z1 = min(spec.cells - 1, z0 + 1)
    return tuple(dict.fromkeys((z0 * spec.cells + x0, z0 * spec.cells + x1, z1 * spec.cells + x0, z1 * spec.cells + x1)))


def _enforce_downhill_watercourses(
    result: list[float],
    original: Sequence[float],
    dataset: OsmDataset,
    projection: BboxProjection,
    spec: ConstraintPlayabilitySpec,
    field: _ConstraintField,
) -> None:
    # One-cell dilation covers bilinear influence around a road/building/water
    # crossing on coarse CWA terrain grids. Those cells represent a culvert or
    # bridge conflict where the higher-priority surface intentionally wins.
    protected = _dilate_mask(
        [priority > PRIORITY_WATERCOURSE for priority in field.priorities],
        spec.cells,
        2,
    )
    minimum_ratio = spec.watercourse_minimum_gradient_percent / 100.0
    spacing = max(3.0, spec.cell_size * 0.45)
    for _ in range(12):
        changed = 0
        for feature in dataset.watercourses:
            if feature.tags.get("tunnel") not in {None, "", "no"}:
                continue
            line = _line_geometry(feature, projection)
            if line is None:
                continue
            start = Point(line.coords[0])
            end = Point(line.coords[-1])
            start_h = _sample_elevation(original, spec.cells, spec.cell_size, start.x, start.y)
            end_h = _sample_elevation(original, spec.cells, spec.cell_size, end.x, end.y)
            reverse = start_h < end_h
            count = max(1, int(math.ceil(line.length / spacing)))
            previous_index: int | None = None
            previous_height: float | None = None
            previous_distance: float | None = None
            for step in range(count + 1):
                distance = line.length * (count - step if reverse else step) / count
                point = line.interpolate(distance)
                index = _point_cell_index(point, spec)
                if index == previous_index:
                    continue
                if protected[index]:
                    previous_index = None
                    previous_height = None
                    previous_distance = None
                    continue
                current = result[index]
                if previous_height is not None and previous_distance is not None:
                    travel = abs(distance - previous_distance)
                    desired_maximum = previous_height - travel * minimum_ratio
                    if current > desired_maximum:
                        result[index] = max(original[index] - spec.maximum_grade_adjustment, desired_maximum)
                        current = result[index]
                        field.categories[index] = "watercourse"
                        changed += 1
                previous_index = index
                previous_height = current
                previous_distance = distance
        if changed == 0:
            break

    # Bilinear sampling can still see a bump between corrected terrain vertices.
    # Lower the four contributing vertices together, while never touching a
    # higher-priority road/building/water crossing.
    fine_spacing = max(2.0, spec.cell_size / 3.0)
    for _ in range(12):
        changed = 0
        for feature in dataset.watercourses:
            if feature.tags.get("tunnel") not in {None, "", "no"}:
                continue
            line = _line_geometry(feature, projection)
            if line is None:
                continue
            start = Point(line.coords[0])
            end = Point(line.coords[-1])
            reverse = (
                _sample_elevation(original, spec.cells, spec.cell_size, start.x, start.y)
                < _sample_elevation(original, spec.cells, spec.cell_size, end.x, end.y)
            )
            count = max(1, int(math.ceil(line.length / fine_spacing)))
            previous_height: float | None = None
            previous_distance: float | None = None
            for step in range(count + 1):
                distance = line.length * (count - step if reverse else step) / count
                point = line.interpolate(distance)
                indices = _bilinear_indices(point.x, point.y, spec)
                if any(protected[index] for index in indices):
                    previous_height = None
                    previous_distance = None
                    continue
                current = _sample_elevation(result, spec.cells, spec.cell_size, point.x, point.y)
                if previous_height is not None and previous_distance is not None:
                    travel = abs(distance - previous_distance)
                    desired_maximum = previous_height - travel * minimum_ratio
                    if current > desired_maximum:
                        excess = current - desired_maximum
                        for index in indices:
                            result[index] = max(
                                original[index] - spec.maximum_grade_adjustment,
                                result[index] - excess,
                            )
                            field.categories[index] = "watercourse"
                        changed += 1
                        current = _sample_elevation(result, spec.cells, spec.cell_size, point.x, point.y)
                previous_height = current
                previous_distance = distance
        if changed == 0:
            break


def _raw_dem_sampler(path: Path | None):
    if path is None or not path.is_file():
        return None
    try:
        import rasterio
        from rasterio.transform import rowcol
    except Exception:  # noqa: BLE001
        return None
    try:
        dataset = rasterio.open(path)
        array = dataset.read(1, masked=True)
    except Exception:  # noqa: BLE001
        return None

    def sample(latitude: float, longitude: float) -> float | None:
        try:
            row, column = rowcol(dataset.transform, longitude, latitude, op=float)
            row -= 0.5
            column -= 0.5
            r0 = int(math.floor(row))
            c0 = int(math.floor(column))
            r1 = r0 + 1
            c1 = c0 + 1
            if r0 < 0 or c0 < 0 or r1 >= array.shape[0] or c1 >= array.shape[1]:
                return None
            values = (array[r0, c0], array[r0, c1], array[r1, c0], array[r1, c1])
            if any(getattr(value, "mask", False) for value in values):
                return None
            tx = column - c0
            tz = row - r0
            a = float(values[0]) * (1.0 - tx) + float(values[1]) * tx
            b = float(values[2]) * (1.0 - tx) + float(values[3]) * tx
            value = a * (1.0 - tz) + b * tz
            return value if math.isfinite(value) else None
        except Exception:  # noqa: BLE001
            return None

    return sample


def _apply_world_edges(
    field: _ConstraintField,
    original: Sequence[float],
    projection: BboxProjection,
    spec: ConstraintPlayabilitySpec,
    *,
    blend_cells: int | None = None,
    vertical_datum_offset: float = 0.0,
) -> str:
    width = spec.world_edge_blend_cells if blend_cells is None else int(blend_cells)
    if width <= 0:
        return "disabled"
    raw_sampler = _raw_dem_sampler(spec.out_of_bounds_dem_path)
    method = "raw-dem-halo" if raw_sampler is not None else "linear-edge-extrapolation"
    cells = spec.cells
    size = spec.world_size

    def edge_sample(x: float, z: float, fallback: float) -> float:
        if raw_sampler is None:
            return fallback
        latitude, longitude = projection.to_latlon((x, z))
        value = raw_sampler(latitude, longitude)
        if value is None:
            return fallback
        return value - vertical_datum_offset

    for layer in range(width):
        blend = (width - layer) / (width + 1.0)
        for position in range(cells):
            samples = (
                (position, layer, position * spec.cell_size, -spec.cell_size),
                (position, cells - 1 - layer, position * spec.cell_size, size),
                (layer, position, -spec.cell_size, position * spec.cell_size),
                (cells - 1 - layer, position, size, position * spec.cell_size),
            )
            for x_cell, z_cell, outside_x, outside_z in samples:
                index = z_cell * cells + x_cell
                fallback = original[index]
                if raw_sampler is None:
                    inward_x = min(cells - 1, max(0, x_cell + (1 if x_cell < cells / 2 else -1)))
                    inward_z = min(cells - 1, max(0, z_cell + (1 if z_cell < cells / 2 else -1)))
                    inward = original[inward_z * cells + inward_x]
                    fallback = original[index] + (original[index] - inward)
                outside = edge_sample(outside_x, outside_z, fallback)
                target = original[index] * (1.0 - blend) + outside * blend
                field.apply(
                    index,
                    target,
                    priority=PRIORITY_BOUNDARY,
                    strength=0.45 * blend,
                    hard=False,
                    category="world-boundary",
                )
    return method


def solve_terrain_constraints(
    elevations: Sequence[float],
    dataset: OsmDataset,
    projection: BboxProjection,
    raster: OsmRaster,
    spec: ConstraintPlayabilitySpec,
    *,
    building_placement_plans: Sequence[BuildingPlacementPlan] | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> ConstraintTerrainReport:
    def progress(percent: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0, min(100, int(percent))), stage)

    progress(0, f"Preparing terrain constraint field for {spec.cells:,}×{spec.cells:,} cells")
    raw_original = tuple(float(value) for value in elevations)
    if len(raw_original) != spec.cells * spec.cells:
        raise ValueError("constraint solver elevation grid has the wrong size")
    smoothing = _effective_terrain_smoothing(spec)
    renderable_source_water = renderable_water_mask(
        raw_original, raster, sea_level=spec.sea_level, water_depth=spec.water_depth
    )
    conservative_interior = conservative_water_interior_mask(raster)
    water_components = _components(raster.water, spec.cells)
    edge_water_components = [
        component for component in water_components
        if _component_touches_world_edge(component, spec.cells)
    ]
    mapped_inland_water_components = [
        component for component in water_components
        if not _component_touches_world_edge(component, spec.cells)
    ]

    # A cropped map can cut through an inland lake. The old edge test treated
    # every edge-touching water component as ocean, even when the source DEM
    # puts that lake one hundred metres above sea level. CWA/OFP has only one
    # global water plane, so locally excavating such a lake creates enormous
    # beaches/cliffs. On an inland map with no real near-sea edge water, recenter
    # the whole terrain datum toward the lowest large edge lake instead.
    maximum_near_sea_surface = spec.sea_level + spec.water_depth
    edge_surfaces = [
        (component, median(raw_original[index] for index in component))
        for component in edge_water_components
    ]
    near_sea_edge_components = [
        component
        for component, surface in edge_surfaces
        if surface <= maximum_near_sea_surface
    ]
    vertical_datum_offset = 0.0
    minimum_datum_component_cells = max(64, int(round(len(raw_original) * 0.005)))
    datum_candidates = [
        (component, surface)
        for component, surface in edge_surfaces
        if surface > maximum_near_sea_surface
        and len(component) >= minimum_datum_component_cells
    ]
    if datum_candidates and not near_sea_edge_components:
        lowest_edge_lake_surface = min(surface for _component, surface in datum_candidates)

        # Do not blindly lower the map through unrelated dry valleys. Use dry
        # terrain at least three cells away from mapped water as a safety floor,
        # while allowing a tiny low-tail fraction to be outliers.
        water_proximity = _distance_from_mask(raster.water, spec.cells, 3)
        dry_reference = sorted(
            raw_original[index]
            for index, is_water in enumerate(raster.water)
            if not is_water
            and water_proximity[index] >= 3
            and math.isfinite(raw_original[index])
        )
        dry_safe_floor = math.inf
        if dry_reference:
            safe_index = min(
                len(dry_reference) - 1,
                max(0, int(round((len(dry_reference) - 1) * 0.005))),
            )
            dry_safe_floor = dry_reference[safe_index] - max(1.0, spec.height_scale * 2.0)

        requested_offset = max(0.0, lowest_edge_lake_surface - spec.sea_level)
        safe_offset = max(0.0, dry_safe_floor - spec.sea_level)
        candidate_offset = min(requested_offset, safe_offset)
        if candidate_offset >= max(10.0, spec.water_depth * 2.0):
            vertical_datum_offset = candidate_offset

    original = tuple(value - vertical_datum_offset for value in raw_original)
    field = _ConstraintField.create(len(original))
    if vertical_datum_offset > 0.0:
        progress(
            3,
            f"Lowering whole-world terrain datum by {vertical_datum_offset:.1f} m "
            "to fit elevated edge lakes to CWA's global water plane",
        )

    # Reclassify water after the optional whole-world datum shift. Genuine sea
    # components remain coastal. Elevated edge lakes can now use the same broad
    # lake-bank treatment as inland water if the remaining elevation difference
    # can be graded within the solver's 32-cell safety cap.
    lake_rise_per_cell = (
        spec.cell_size * spec.lake_shore_maximum_slope_percent / 100.0
    )
    maximum_edge_lake_surface = (
        spec.sea_level + 32 * lake_rise_per_cell
    )
    coastal_water_components: list[list[int]] = []
    recentered_edge_lake_components: list[list[int]] = []
    elevated_edge_water_components: list[list[int]] = []
    for component in edge_water_components:
        component_surface = median(original[index] for index in component)
        if component_surface <= maximum_near_sea_surface and vertical_datum_offset <= 0.0:
            coastal_water_components.append(component)
        elif (
            vertical_datum_offset > 0.0
            and component_surface <= maximum_edge_lake_surface
        ):
            recentered_edge_lake_components.append(component)
        elif component_surface <= maximum_near_sea_surface:
            coastal_water_components.append(component)
        else:
            elevated_edge_water_components.append(component)

    # Ordinary inland ponds still follow the crater fix: only near-sea water is
    # rendered unless the whole map was recentered enough to bring it close to
    # the global water plane.
    maximum_inland_water_surface = spec.sea_level + spec.water_depth
    inland_water_components: list[list[int]] = []
    elevated_inland_water_components: list[list[int]] = []
    for component in mapped_inland_water_components:
        component_surface = median(original[index] for index in component)
        if component_surface <= maximum_inland_water_surface:
            inland_water_components.append(component)
        else:
            elevated_inland_water_components.append(component)

    selected_coastal_water_mask = _mask_from_components(coastal_water_components, len(original))
    selected_recentered_edge_lake_mask = _mask_from_components(
        recentered_edge_lake_components, len(original)
    )
    selected_inland_water_mask = _mask_from_components(inland_water_components, len(original))
    coastal_water_mask = tuple(
        selected and renderable_source_water[index]
        for index, selected in enumerate(selected_coastal_water_mask)
    )
    recentered_edge_lake_mask = tuple(
        selected and renderable_source_water[index]
        for index, selected in enumerate(selected_recentered_edge_lake_mask)
    )
    inland_water_mask = tuple(
        selected and renderable_source_water[index]
        for index, selected in enumerate(selected_inland_water_mask)
    )
    lake_water_mask = tuple(
        edge_lake or inland
        for edge_lake, inland in zip(recentered_edge_lake_mask, inland_water_mask)
    )
    active_water_mask = tuple(
        coastal or lake
        for coastal, lake in zip(coastal_water_mask, lake_water_mask)
    )
    deep_water_mask = tuple(
        active and conservative_interior[index]
        for index, active in enumerate(active_water_mask)
    )
    uncertain_water_cells_preserved = sum(
        1
        for index, is_water in enumerate(raster.water)
        if is_water and not active_water_mask[index]
    )
    building_pad_groups: list[tuple[int, ...]] = []
    progress(5, (
        f"Found {len(coastal_water_components):,} coastal, "
        f"{len(recentered_edge_lake_components):,} recentered edge-lake, "
        f"{len(inland_water_components):,} renderable inland, "
        f"{len(elevated_edge_water_components) + len(elevated_inland_water_components):,} "
        "elevated water component(s) preserved at DEM height"
    ))

    # 1. Water bodies. Only conservative interior cells get the full depth.
    # Low-confidence shoreline cells that are already near sea level are held
    # just below the water plane, avoiding deep excavation of mixed land/water
    # terrain vertices.
    water_target = spec.sea_level - spec.water_depth
    shallow_water_target = spec.sea_level - min(0.35, max(0.05, spec.water_depth * 0.10))
    progress(8, f"Applying conservative water-bed constraints to {sum(active_water_mask):,} cells")
    for index, is_water in enumerate(active_water_mask):
        if not is_water:
            continue
        target = water_target if deep_water_mask[index] else shallow_water_target
        field.apply(index, target, priority=PRIORITY_WATER, strength=1.0, hard=True, category="water")

    # Build an adaptive coastal ramp. A fixed two-cell beach cannot possibly
    # respect an 8% target when a coarse DEM shoreline vertex is 20 m high.
    coastal_rise_per_cell = max(
        spec.height_scale, spec.cell_size * spec.lake_shore_maximum_slope_percent / 100.0
    )
    first_coastal = _distance_from_mask(coastal_water_mask, spec.cells, 1)
    first_ring_high = max(
        (original[index] for index, distance in enumerate(first_coastal) if distance == 1),
        default=spec.sea_level,
    )
    required_coastal_cells = int(math.ceil(
        max(0.0, first_ring_high - spec.sea_level) / max(1.0e-9, coastal_rise_per_cell)
    )) + 1
    effective_coastal_shore_cells = min(
        32, max(smoothing.shoreline_transition_cells, required_coastal_cells)
    )
    progress(12, f"Building coastal shoreline transition across {effective_coastal_shore_cells:,} cells")
    shore_distances = _distance_from_mask(
        coastal_water_mask, spec.cells, effective_coastal_shore_cells
    )
    protected_shore_cells = 0
    for index, distance in enumerate(shore_distances):
        if distance <= 0 or distance > effective_coastal_shore_cells or active_water_mask[index]:
            continue
        slope_limited = spec.sea_level + distance * coastal_rise_per_cell
        shoreline_cut_budget = max(0.5, float(getattr(spec, "beach_height", 3.0)))
        if original[index] > slope_limited + shoreline_cut_budget:
            continue
        target = max(spec.sea_level, min(original[index], slope_limited))
        clipped_for_grade = original[index] > slope_limited + 1.0e-7
        fraction = distance / max(1.0, float(effective_coastal_shore_cells))
        field.apply(
            index,
            target,
            priority=PRIORITY_WATER,
            strength=1.0 if clipped_for_grade else max(0.35, 0.9 - fraction * 0.55),
            hard=clipped_for_grade or distance == 1,
            category="shoreline",
        )
        protected_shore_cells += 1

    # Inland lakes and recentered edge lakes need a broader treatment than a
    # sea coast. Expand the bank width when the remaining lake level difference
    # requires it, while retaining the configured maximum shoreline slope.
    active_lake_surfaces = [
        median(original[index] for index in component)
        for component in (*recentered_edge_lake_components, *inland_water_components)
    ]
    required_lake_cells = 0
    if active_lake_surfaces and lake_rise_per_cell > 0.0:
        required_lake_cells = int(math.ceil(
            max(0.0, max(active_lake_surfaces) - spec.sea_level) / lake_rise_per_cell
        ))
    effective_lake_shore_cells = min(
        32,
        max(smoothing.lake_shore_smoothing_cells, required_lake_cells + 1),
    )
    progress(
        16,
        f"Building lake-bank distance field across {effective_lake_shore_cells:,} cells",
    )
    lake_bank_mask = [False] * len(original)
    lake_distances = _euclidean_distance_from_mask(
        lake_water_mask,
        spec.cells,
        effective_lake_shore_cells,
    )
    lake_shore_cells = 0
    for index, distance in enumerate(lake_distances):
        if (
            not math.isfinite(distance)
            or distance <= 0.0
            or distance > effective_lake_shore_cells
            or active_water_mask[index]
        ):
            continue
        lake_bank_mask[index] = True
        x, z = index % spec.cells, index // spec.cells
        local_values = [original[index]]
        for dz in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, nz = x + dx, z + dz
                if dx == 0 and dz == 0:
                    continue
                if 0 <= nx < spec.cells and 0 <= nz < spec.cells:
                    neighbour = nz * spec.cells + nx
                    if not lake_water_mask[neighbour]:
                        local_values.append(original[neighbour])
        local_smoothed = sum(local_values) / len(local_values)
        slope_limited = spec.sea_level + distance * lake_rise_per_cell
        shoreline_cut_budget = max(0.5, float(getattr(spec, "beach_height", 3.0)))
        if original[index] > slope_limited + shoreline_cut_budget:
            continue
        target = max(
            spec.sea_level,
            min(original[index], local_smoothed, slope_limited),
        )
        clipped_for_grade = original[index] > slope_limited + 1e-7
        fraction = distance / max(
            1.0, float(effective_lake_shore_cells)
        )
        field.apply(
            index,
            target,
            priority=PRIORITY_WATER,
            strength=1.0 if clipped_for_grade else max(0.35, 0.9 - fraction * 0.55),
            hard=clipped_for_grade or distance <= 1.0,
            category="lake-shoreline",
        )
        protected_shore_cells += 1
        lake_shore_cells += 1

    maximum_lake_shore_slope_before = _maximum_lake_bank_slope(
        original,
        lake_water_mask,
        lake_bank_mask,
        spec.cells,
        spec.cell_size,
        spec.sea_level,
    )

    # OSM dry-land features occasionally sit in DEM depressions that fall below
    # CWA/OFP's global water plane. Lift only cells backed by non-water OSM data,
    # leaving mapped water cells submerged and unchanged.
    dry_land_floor = spec.sea_level + max(
        OSM_DRY_LAND_MINIMUM_CLEARANCE_METRES,
        float(getattr(spec, "osm_land_minimum_clearance", 0.0)),
        spec.height_scale * 3.0,
    )
    osm_land_floor_cells = 0
    progress(20, "Lifting OSM dry-land cells above the global water plane")
    dry_land_indices = _dry_osm_land_floor_cells(
        dataset,
        projection,
        raster,
        spec,
        progress_callback=lambda message: progress(20, message),
    )
    dry_land_total = len(dry_land_indices)
    progress(21, f"Applying dry-land elevation floor to {dry_land_total:,} candidate cells")
    dry_land_interval = max(1, dry_land_total // 16)
    priorities = field.priorities
    targets = field.targets
    strengths = field.strengths
    hard_constraints = field.hard
    categories = field.categories
    for position, raw_index in enumerate(dry_land_indices, start=1):
        index = int(raw_index)
        if original[index] < dry_land_floor:
            # At this point only boundary/water constraints have been applied,
            # so PRIORITY_OSM_DRY_LAND is guaranteed to replace the existing
            # value. Avoid the generic method-call/branch path millions of times.
            priorities[index] = PRIORITY_OSM_DRY_LAND
            targets[index] = dry_land_floor
            strengths[index] = 1.0
            hard_constraints[index] = True
            categories[index] = "osm-land-floor"
            osm_land_floor_cells += 1
        if position == dry_land_total or position % dry_land_interval == 0:
            progress(21, f"Applying dry-land floor {position:,}/{dry_land_total:,} ({osm_land_floor_cells:,} lifted)")

    # 2-4. Road constraints, with explicit bridge/tunnel/embankment handling.
    progress(22, f"Preparing terrain profiles for {len(dataset.roads):,} roads")
    bridge_segments = 0
    tunnel_segments = 0
    embankment_segments = 0
    road_seed_cells: set[int] = set()
    road_water_fill_cells: set[int] = set()
    road_water_floor_cells: set[int] = set()
    road_sidehill_segments = 0
    road_sidehill_bench_cells: set[int] = set()
    road_water_floor = float(spec.sea_level) + max(
        ROAD_WATER_MINIMUM_CLEARANCE_METRES,
        float(spec.height_scale) * 2.0,
    )
    major_cells: set[int] = set()
    minor_cells: set[int] = set()
    road_total = len(dataset.roads)
    road_interval = max(1, road_total // 16)
    for road_index, feature in enumerate(dataset.roads, start=1):
        if road_index == road_total or road_index % road_interval == 0:
            value = 22 + round(22 * road_index / max(1, road_total))
            progress(value, f"Applying road terrain constraints {road_index:,}/{road_total:,}")
        line = _line_geometry(feature, projection)
        if line is None:
            continue
        tags = feature.tags
        if tags.get("tunnel") not in {None, "", "no"}:
            tunnel_segments += 1
            continue
        bridge_value = str(tags.get("bridge", "")).casefold()
        width = _road_width(tags)
        road_surface_width = max(6.0, width)
        is_bridge = (
            bridge_value not in {"", "no", "false", "0"}
            or str(tags.get("man_made", "")).casefold() == "bridge"
        )
        if is_bridge and road_bridge_crosses_ditch_only(feature, dataset, projection):
            # Ditch crossings are intentionally ordinary roads, not bridge decks.
            # Grade them with the road network instead of preserving a bridge gap.
            is_bridge = False

        # A bridge tag is only useful if CWA will actually show water beneath
        # the span. Check both already-submerged source terrain and mapped water
        # that this solver is about to lower below the global water plane. Dry
        # bridge tags become ordinary roads and receive normal grade/bench work.
        bridge_has_water = (
            road_span_has_in_game_water(
                tuple((float(x), float(z)) for x, z in line.coords),
                original,
                cells=spec.cells, cell_size=spec.cell_size,
                sea_level=spec.sea_level, width=road_surface_width,
            )
            or _road_corridor_intersects_mask(
                line, active_water_mask, spec, road_surface_width
            )
        )
        if is_bridge and not bridge_has_water:
            is_bridge = False

        if not is_bridge:
            try:
                positive_layer = float(str(tags.get("layer", "0")).replace(",", ".")) > 0.0
            except ValueError:
                positive_layer = False
            if positive_layer and bridge_has_water:
                is_bridge = True
        is_embankment = tags.get("embankment") not in {None, "", "no"}
        highway = tags.get("highway", "road")
        major = highway in _MAJOR_HIGHWAYS
        priority = PRIORITY_MAJOR_ROAD if major else PRIORITY_MINOR_ROAD
        category = "major-road" if major else "minor-road"
        maximum_grade = spec.major_road_grade_percent if major else spec.maximum_road_grade_percent
        shoulder = 0.0 if is_bridge else spec.road_grade_radius
        if is_bridge:
            bridge_segments += 1
            if bool(getattr(spec, "procedural_bridges", True)):
                # Generated bridges benefit from a modest terrain lift beneath
                # the span so coarse DEM riverbeds and shoreline sinks do not
                # leave the surrounding land unrealistically low. Keep a clear
                # gap below the road profile instead of flattening terrain all
                # the way up to deck level.
                distances, heights = _linear_profile(line, original, spec)
                # A procedural bridge is assembled from rigid modules. Keep the
                # terrain support beneath the complete span at one constant
                # elevation too, rather than following a rising/falling DEM
                # profile under a visually flat deck. The terrain remains below
                # the deck, but is raised enough that the bridge does not appear
                # to hover over an unnecessarily deep artificial trench.
                underfill_gap = 0.9
                minimum_underfill = float(spec.sea_level) + 2.5
                flat_bridge_profile = max(heights) if heights else float(spec.sea_level)
                flat_underfill_target = max(minimum_underfill, flat_bridge_profile - underfill_gap)
                bridge_hard_radius = width * 0.5 + spec.cell_size * math.sqrt(0.35)
                bridge_soft_shoulder = max(1.0, spec.cell_size * 0.55)
                corridor = line.buffer(bridge_hard_radius + bridge_soft_shoulder, cap_style=2, join_style=2)
                for index in _candidate_cells(corridor.bounds, spec.cells, spec.cell_size):
                    centre = Point(_cell_center(index, spec.cells, spec.cell_size))
                    distance_to_line = line.distance(centre)
                    if distance_to_line > bridge_hard_radius + bridge_soft_shoulder:
                        continue
                    # Preserve the actual water opening. The previous underfill
                    # could raise an unmapped below-sea channel above CWA's
                    # global water plane and then leave a perfectly good bridge
                    # standing over dry ground. Only dry approach/support cells
                    # receive the bridge-support terrace.
                    if (
                        active_water_mask[index]
                        or original[index] < float(spec.sea_level) - 0.05
                    ):
                        continue
                    target = flat_underfill_target
                    if distance_to_line <= bridge_hard_radius:
                        strength = 1.0
                        hard = True
                    else:
                        strength = max(0.0, 1.0 - (distance_to_line - bridge_hard_radius) / max(0.01, bridge_soft_shoulder)) ** 2
                        hard = False
                    field.apply(index, target, priority=priority, strength=strength, hard=hard, category="bridge-underfill")
                continue
            # Stock bridge decks are objects above the landscape. Do not grade,
            # fill, flatten, or otherwise raise terrain beneath their span.
            continue
        elif is_embankment:
            embankment_segments += 1
            distances, heights = _linear_profile(line, original, spec)
        else:
            distances, heights = _profile(line, original, spec, maximum_grade)
        cross_slopes = _road_cross_slope_profile(
            line, original, spec, distances, road_surface_width
        )
        maximum_sidehill_factor = max(
            (_road_sidehill_factor(value) for value in cross_slopes),
            default=0.0,
        )
        if maximum_sidehill_factor > 0.0:
            road_sidehill_segments += 1
        # Bilinear terrain sampling uses the four surrounding WRP vertices. A
        # road corridor only as wide as the rendered model can therefore leave
        # one or more contributing cells unconstrained and create a steeper
        # interpolated centreline than either neighbouring cell. Cover the full
        # half-cell diagonal so every terrain sample beneath the road uses the
        # same grade-limited profile.
        hard_radius = road_surface_width * 0.5 + spec.cell_size * math.sqrt(0.5)
        # If an ordinary road crosses active mapped water without a bridge tag,
        # build a narrow terrain causeway instead of letting the higher-priority
        # water-bed constraint sink the road model below the global water plane.
        # The extra bank is intentionally much narrower than road_grade_radius:
        # it only needs enough room to blend the raised carriageway back into the
        # underwater bed without creating a razor-edged embankment.
        water_floor = road_water_floor
        water_bank_width = max(
            road_surface_width,
            float(spec.cell_size) * ROAD_WATER_BANK_WIDTH_CELLS,
        )
        sidehill_platform_margin = max(
            ROAD_SIDEHILL_MINIMUM_PLATFORM_MARGIN_METRES,
            road_surface_width * 0.5,
            float(spec.cell_size) * ROAD_SIDEHILL_BENCH_EXTRA_CELLS,
        )
        maximum_sidehill_extra = (
            0.0
            if maximum_sidehill_factor <= 0.0
            else sidehill_platform_margin * (0.55 + 0.45 * maximum_sidehill_factor)
        )
        maximum_sidehill_blend = (
            0.0
            if maximum_sidehill_factor <= 0.0
            else max(
                road_surface_width,
                float(spec.cell_size) * ROAD_SIDEHILL_BLEND_CELLS,
            ) * (0.65 + 0.35 * maximum_sidehill_factor)
        )
        corridor_radius = hard_radius + max(
            shoulder,
            water_bank_width,
            maximum_sidehill_extra + maximum_sidehill_blend,
        )
        corridor = line.buffer(corridor_radius, cap_style=2, join_style=2)
        for index in _candidate_cells(corridor.bounds, spec.cells, spec.cell_size):
            centre = Point(_cell_center(index, spec.cells, spec.cell_size))
            distance_to_line = line.distance(centre)
            if distance_to_line > corridor_radius:
                continue
            along = line.project(centre)
            profile_target = _profile_height(along, distances, heights)

            # Water itself normally outranks roads. This is the one deliberate
            # exception: an untagged road through active water gets a compact
            # causeway. Explicit bridges already exited above and are never
            # filled here. The bank target blends from road level to the source
            # bed so cells beyond the bank remain ordinary water.
            if (
                active_water_mask[index]
                and distance_to_line <= hard_radius + water_bank_width
            ):
                raised_target = max(profile_target, water_floor)
                if distance_to_line <= hard_radius:
                    target = raised_target
                    hard = True
                else:
                    bank_fraction = max(
                        0.0,
                        1.0 - (distance_to_line - hard_radius) / max(0.01, water_bank_width),
                    )
                    # Smoothstep avoids a visible angular shoulder at the water
                    # edge while still reaching the full road elevation at the
                    # hard corridor boundary.
                    bank_fraction = bank_fraction * bank_fraction * (3.0 - 2.0 * bank_fraction)
                    target = original[index] + (raised_target - original[index]) * bank_fraction
                    hard = False
                field.apply(
                    index,
                    target,
                    priority=PRIORITY_ROAD_WATER_FILL,
                    strength=1.0,
                    hard=hard,
                    category="road-water-fill",
                )
                road_water_fill_cells.add(index)
                if hard:
                    road_water_floor_cells.add(index)
                road_seed_cells.add(index)
                (major_cells if major else minor_cells).add(index)
                continue

            sidehill_factor = _road_sidehill_factor(
                _profile_height(along, distances, cross_slopes)
                if cross_slopes else 0.0
            )
            sidehill_extra = (
                0.0
                if sidehill_factor <= 0.0
                else sidehill_platform_margin * (0.55 + 0.45 * sidehill_factor)
            )
            sidehill_blend = (
                0.0
                if sidehill_factor <= 0.0
                else max(
                    road_surface_width,
                    float(spec.cell_size) * ROAD_SIDEHILL_BLEND_CELLS,
                ) * (0.65 + 0.35 * sidehill_factor)
            )
            bench_radius = hard_radius + sidehill_extra
            active_radius = max(
                hard_radius + shoulder,
                bench_radius + sidehill_blend,
            )
            if distance_to_line > active_radius:
                continue

            hard = distance_to_line <= hard_radius
            strength = 1.0 if hard else 0.0
            if not hard and shoulder > 0.0 and distance_to_line <= hard_radius + shoulder:
                strength = max(
                    strength,
                    max(
                        0.0,
                        1.0 - (distance_to_line - hard_radius) / max(0.01, shoulder),
                    ) ** 2,
                )

            # On a steep transverse hillside, extend the exact road platform
            # beyond the normal interpolation guard. This creates a narrow
            # cut/fill bench that follows the road's longitudinal profile while
            # remaining flat across the carriageway. The outer band blends the
            # bench back into the untouched hillside.
            sidehill_bench = sidehill_factor > 0.0 and distance_to_line <= bench_radius
            if sidehill_bench:
                hard = True
                strength = 1.0
                road_sidehill_bench_cells.add(index)
            elif (
                sidehill_factor > 0.0
                and sidehill_blend > 0.0
                and distance_to_line <= bench_radius + sidehill_blend
            ):
                blend = max(
                    0.0,
                    1.0 - (distance_to_line - bench_radius) / sidehill_blend,
                )
                blend = blend * blend * (3.0 - 2.0 * blend)
                strength = max(strength, blend)
                if blend > 0.0:
                    road_sidehill_bench_cells.add(index)

            normal_target = max(profile_target, water_floor)
            sidehill_owned = sidehill_factor > 0.0 and (
                sidehill_bench
                or distance_to_line <= bench_radius + sidehill_blend
            )
            field.apply(
                index,
                normal_target,
                priority=(
                    PRIORITY_ROAD_SIDEHILL_BENCH if sidehill_owned else priority
                ),
                strength=strength,
                hard=hard,
                category=("sidehill-road-bench" if sidehill_owned else category),
            )
            if profile_target < water_floor:
                road_water_fill_cells.add(index)
                if hard:
                    road_water_floor_cells.add(index)
            road_seed_cells.add(index)
            (major_cells if major else minor_cells).add(index)

    # 5. Building pads use the exact final model footprints whenever a building
    # plan is available. Coarse CWA terrain is sampled bilinearly, so flattening
    # only cells whose polygons touch the model still allows neighbouring cell
    # centres to raise terrain through a large nave or tower. Expand the hard
    # terrace by half a cell diagonal so every contributor beneath the final
    # footprint receives the same target. Churches get a wider terrace and their
    # exact core outranks ordinary road grading; a road can blend around a church,
    # but it cannot push the church floor back into a hillside.
    building_entries: list[tuple[Polygon, str]] = []
    if building_placement_plans is None:
        for feature in dataset.building_polygons:
            for geo_polygon in feature.polygons:
                geometry = _local_polygon(geo_polygon, projection)
                if not geometry.is_empty:
                    building_entries.append((geometry, ""))
    else:
        for plan in building_placement_plans:
            if len(plan.support_polygon) < 3:
                continue
            family = str(getattr(plan, "building_family", "")).casefold()
            if not family:
                family = str(
                    getattr(
                        getattr(plan.procedural_placement, "selected", None),
                        "family",
                        "",
                    )
                ).casefold()
            building_entries.append((Polygon(plan.support_polygon), family))

    building_total = len(building_entries)
    progress(46, f"Preparing terrain pads for {building_total:,} final building footprints")
    building_pad_cells: set[int] = set()
    building_interval = max(1, building_total // 12)
    interpolation_guard = spec.cell_size * math.sqrt(0.5) + 0.5
    for building_index, (geometry, family) in enumerate(building_entries, start=1):
        if building_index == building_total or building_index % building_interval == 0:
            value = 46 + round(13 * building_index / max(1, building_total))
            progress(value, f"Applying final building terrain pads {building_index:,}/{building_total:,}")
        if geometry.is_empty:
            continue
        min_x, min_z, max_x, max_z = geometry.bounds
        span = max(max_x - min_x, max_z - min_z)
        semantic_extra_margin = 0.0
        pad = geometry.buffer(
            spec.building_pad_margin + semantic_extra_margin, join_style=2
        )
        hard_pad = pad.buffer(interpolation_guard, join_style=2)
        source_candidates = [
            index for index in _candidate_cells(pad.bounds, spec.cells, spec.cell_size)
            if _cell_polygon(index, spec.cells, spec.cell_size).intersects(pad)
        ]
        hard_candidates = [
            index for index in _candidate_cells(hard_pad.bounds, spec.cells, spec.cell_size)
            if _cell_polygon(index, spec.cells, spec.cell_size).intersects(hard_pad)
        ]
        if not hard_candidates:
            centroid = geometry.centroid
            cx = min(spec.cells - 1, max(0, int(centroid.x / spec.cell_size)))
            cz = min(spec.cells - 1, max(0, int(centroid.y / spec.cell_size)))
            hard_candidates = [cz * spec.cells + cx]
        if not source_candidates:
            source_candidates = list(hard_candidates)
        building_pad_groups.append(tuple(sorted(set(hard_candidates))))
        source_values = [original[index] for index in source_candidates]
        source_values.extend(
            _sample_elevation(original, spec.cells, spec.cell_size, x, z)
            for x, z in geometry.exterior.coords
        )
        # Use the same median support terrace for churches and houses. The old
        # church-only maximum-height terrace could raise a long church footprint
        # toward one distant hillside sample, which is exactly the opposite of
        # what the model grounding pass needed. Large churches instead receive a
        # deeper hidden foundation skirt after the final quantized terrain is known.
        target = median(source_values)
        transition_radius = spec.building_grade_radius + semantic_extra_margin
        transition = hard_pad.buffer(transition_radius, join_style=2)
        semantic_core = geometry.buffer(
            spec.building_pad_margin + semantic_extra_margin * 0.5, join_style=2
        )
        for index in _candidate_cells(transition.bounds, spec.cells, spec.cell_size):
            cell = _cell_polygon(index, spec.cells, spec.cell_size)
            if not cell.intersects(transition):
                continue
            centre = Point(_cell_center(index, spec.cells, spec.cell_size))
            distance = hard_pad.distance(centre)
            hard = cell.intersects(hard_pad)
            strength = (
                1.0
                if hard
                else max(0.0, 1.0 - distance / max(0.01, transition_radius)) ** 2
            )
            priority = PRIORITY_BUILDING
            field.apply(
                index,
                target,
                priority=priority,
                strength=strength,
                hard=hard,
                category="building",
            )
            if hard:
                building_pad_cells.add(index)

    # 6. Watercourses are directed from the higher endpoint to the lower one,
    # carved slightly, and constrained to remain monotonically downhill.
    watercourse_total = len(dataset.watercourses)
    progress(61, f"Preparing channels for {watercourse_total:,} watercourses")
    watercourse_cells: set[int] = set()
    watercourse_interval = max(1, watercourse_total // 10)
    for watercourse_index, feature in enumerate(dataset.watercourses, start=1):
        if watercourse_index == watercourse_total or watercourse_index % watercourse_interval == 0:
            value = 61 + round(9 * watercourse_index / max(1, watercourse_total))
            progress(value, f"Applying watercourse constraints {watercourse_index:,}/{watercourse_total:,}")
        if feature.tags.get("tunnel") not in {None, "", "no"}:
            continue
        line = _line_geometry(feature, projection)
        if line is None:
            continue
        start = Point(line.coords[0])
        end = Point(line.coords[-1])
        start_h = _sample_elevation(original, spec.cells, spec.cell_size, start.x, start.y)
        end_h = _sample_elevation(original, spec.cells, spec.cell_size, end.x, end.y)
        reverse = start_h < end_h
        kind = feature.tags.get("waterway", "stream")
        depth = spec.river_channel_depth if kind in {"river", "canal"} else spec.stream_channel_depth
        spacing = max(3.0, min(10.0, spec.cell_size / 3.0))
        count = max(1, int(math.ceil(line.length / spacing)))
        ordered_distances = [line.length * (count - index if reverse else index) / count for index in range(count + 1)]
        samples = []
        for distance in ordered_distances:
            point = line.interpolate(distance)
            samples.append(_sample_elevation(original, spec.cells, spec.cell_size, point.x, point.y) - depth)
        minimum_ratio = spec.watercourse_minimum_gradient_percent / 100.0
        for index in range(1, len(samples)):
            travel = abs(ordered_distances[index] - ordered_distances[index - 1])
            samples[index] = min(samples[index], samples[index - 1] - travel * minimum_ratio)
        distances = list(reversed(ordered_distances)) if reverse else ordered_distances
        heights = list(reversed(samples)) if reverse else samples
        if distances and distances[0] > distances[-1]:
            distances.reverse()
            heights.reverse()
        bank = max(spec.cell_size * 0.75, 8.0)
        corridor = line.buffer(bank, cap_style=2, join_style=2)
        for index in _candidate_cells(corridor.bounds, spec.cells, spec.cell_size):
            centre = Point(_cell_center(index, spec.cells, spec.cell_size))
            distance_to_line = line.distance(centre)
            if distance_to_line > bank:
                continue
            target = _profile_height(line.project(centre), distances, heights)
            strength = max(0.0, 1.0 - distance_to_line / bank) ** 2
            field.apply(
                index,
                target,
                priority=PRIORITY_WATERCOURSE,
                strength=max(0.25, strength),
                hard=distance_to_line <= max(2.0, spec.cell_size * 0.2),
                category="watercourse",
            )
            watercourse_cells.add(index)

    # 8. World boundary constraints are deliberately lowest priority.
    progress(72, "Applying world-edge blending constraints")
    out_of_bounds_method = _apply_world_edges(
        field,
        original,
        projection,
        spec,
        blend_cells=smoothing.world_edge_blend_cells,
        vertical_datum_offset=vertical_datum_offset,
    )

    # Unified relaxation. All constraints remain in the same field; higher
    # priorities already won conflicts before this stage.
    progress(75, "Combining terrain constraints by priority")
    result = list(original)
    for index, priority in enumerate(field.priorities):
        if priority <= 0:
            continue
        target = field.targets[index]
        if priority == PRIORITY_WATER:
            result[index] = target
        else:
            adjustment = max(
                -spec.maximum_grade_adjustment,
                min(spec.maximum_grade_adjustment, target - original[index]),
            )
            result[index] = original[index] + adjustment * field.strengths[index]

    iterations = smoothing.solver_iterations
    progress(79, f"Relaxing terrain constraints: 0/{iterations:,} iterations")
    iteration_interval = max(1, iterations // 12)

    # This relaxation used to perform four Python neighbour lookups plus several
    # scalar branches for every cell on every iteration. A 2048² terrain with
    # ~200 large-world iterations turns that into hundreds of millions of Python
    # operations. Keep the same Jacobi update (all cells read the previous
    # iteration), but perform the grid arithmetic in NumPy.
    shape = (spec.cells, spec.cells)
    current = np.asarray(result, dtype=np.float64).reshape(shape).copy()
    original_grid = np.asarray(original, dtype=np.float64).reshape(shape)
    priorities_grid = np.asarray(field.priorities, dtype=np.int32).reshape(shape)
    targets_grid = np.asarray(field.targets, dtype=np.float64).reshape(shape)
    strengths_grid = np.asarray(field.strengths, dtype=np.float64).reshape(shape)
    hard_grid = np.asarray(field.hard, dtype=np.bool_).reshape(shape)
    constrained_grid = priorities_grid > 0
    water_grid = priorities_grid == PRIORITY_WATER
    lower_grid = original_grid - spec.maximum_grade_adjustment
    upper_grid = original_grid + spec.maximum_grade_adjustment
    neighbour_count = np.full(shape, 4.0, dtype=np.float64)
    neighbour_count[0, :] -= 1.0
    neighbour_count[-1, :] -= 1.0
    neighbour_count[:, 0] -= 1.0
    neighbour_count[:, -1] -= 1.0
    neighbour_sum = np.empty(shape, dtype=np.float64)
    natural = np.empty(shape, dtype=np.float64)
    candidate = np.empty(shape, dtype=np.float64)

    for iteration in range(iterations):
        if iteration == iterations - 1 or iteration % iteration_interval == 0:
            value = 79 + round(12 * (iteration + 1) / max(1, iterations))
            progress(value, f"Relaxing terrain constraints: {iteration + 1:,}/{iterations:,} iterations")

        neighbour_sum.fill(0.0)
        # Preserve the historical neighbour order: left, right, up, down.
        neighbour_sum[:, 1:] += current[:, :-1]
        neighbour_sum[:, :-1] += current[:, 1:]
        neighbour_sum[1:, :] += current[:-1, :]
        neighbour_sum[:-1, :] += current[1:, :]
        np.divide(neighbour_sum, neighbour_count, out=natural)
        natural -= current
        natural *= smoothing.natural_smoothing_strength
        natural += current

        # Natural cells relax towards their original DEM; constrained cells
        # blend towards their winning target/strength.
        np.multiply(natural, 0.65, out=candidate)
        candidate += original_grid * 0.35
        if np.any(constrained_grid):
            candidate[constrained_grid] = (
                natural[constrained_grid] * (1.0 - strengths_grid[constrained_grid])
                + targets_grid[constrained_grid] * strengths_grid[constrained_grid]
            )

        non_water = ~water_grid
        np.maximum(candidate, lower_grid, out=candidate, where=non_water)
        np.minimum(candidate, upper_grid, out=candidate, where=non_water)
        candidate[hard_grid] = targets_grid[hard_grid]
        current, candidate = candidate, current

    result = current.reshape(-1).tolist()

    progress(92, "Reasserting water and other hard terrain constraints")
    # Reassert hard constraints after relaxation. Water is uncapped because the
    # global CWA water plane otherwise cannot be reached by elevated lake DEMs.
    for index, is_hard in enumerate(field.hard):
        if is_hard:
            result[index] = field.targets[index]

    # Coarse CWA cells can reintroduce tiny uphill steps through interpolation.
    # Repair all cells not owned by a higher-priority road/building/water rule.
    progress(94, "Repairing downhill drainage after terrain relaxation")
    _enforce_downhill_watercourses(result, original, dataset, projection, spec, field)

    # A constraint profile can be individually grade-limited yet still produce
    # a steeper bilinear centreline where several roads, buildings, or water
    # rules meet. Never keep a road adjustment that makes the measured playable
    # grade materially worse than the source terrain. Attenuate only road-owned
    # cells, retaining water and other higher-priority constraints.
    road_slope_before = _road_slope_percent(original, dataset, projection, spec)
    road_slope_after = _road_slope_percent(result, dataset, projection, spec)
    road_grade_limit = max(spec.maximum_road_grade_percent, road_slope_before + 0.10)
    if road_slope_after > road_grade_limit + 1e-7:
        progress(95, "Stabilizing road grades at intersecting terrain constraints")
        road_owned = [
            index
            for index, category in enumerate(field.categories)
            if category in {"major-road", "minor-road"}
        ]
        best = list(result)
        best_slope = road_slope_after
        for strength in (0.75, 0.50, 0.25, 0.0):
            candidate = list(result)
            for index in road_owned:
                candidate[index] = original[index] + (result[index] - original[index]) * strength
            candidate_slope = _road_slope_percent(candidate, dataset, projection, spec)
            if candidate_slope < best_slope:
                best = candidate
                best_slope = candidate_slope
            if candidate_slope <= road_grade_limit + 1e-7:
                break
        result = best
        road_slope_after = best_slope

    # The grade stabilizer is allowed to attenuate ordinary road terrain toward
    # the source DEM. Never let that safety pass push a road core back below the
    # global water plane. Water-owned cells that did not receive a causeway keep
    # their normal water constraint because they are absent from this set.
    for index in road_water_floor_cells:
        if field.categories[index] in {
            "major-road", "minor-road", "road-water-fill", "osm-land-floor"
        }:
            result[index] = max(result[index], road_water_floor)
    if road_water_floor_cells:
        road_slope_after = _road_slope_percent(result, dataset, projection, spec)

    progress(96, "Calculating terrain cut, fill and slope diagnostics")
    changed = 0
    maximum_cut = 0.0
    maximum_fill = 0.0
    total_cut = 0.0
    total_fill = 0.0
    cell_area = spec.cell_size * spec.cell_size
    category_stats: dict[str, dict[str, float | int]] = {}
    for index, (before, after) in enumerate(zip(original, result)):
        delta = after - before
        if abs(delta) <= 1e-7:
            continue
        changed += 1
        maximum_cut = max(maximum_cut, -delta)
        maximum_fill = max(maximum_fill, delta)
        if delta < 0:
            total_cut += -delta * cell_area
        else:
            total_fill += delta * cell_area
        category = field.categories[index]
        stats = category_stats.setdefault(category, {
            "changed_cells": 0,
            "cut_volume_m3": 0.0,
            "fill_volume_m3": 0.0,
            "maximum_cut_m": 0.0,
            "maximum_fill_m": 0.0,
        })
        stats["changed_cells"] = int(stats["changed_cells"]) + 1
        if delta < 0:
            stats["cut_volume_m3"] = float(stats["cut_volume_m3"]) + (-delta * cell_area)
            stats["maximum_cut_m"] = max(float(stats["maximum_cut_m"]), -delta)
        else:
            stats["fill_volume_m3"] = float(stats["fill_volume_m3"]) + (delta * cell_area)
            stats["maximum_fill_m"] = max(float(stats["maximum_fill_m"]), delta)

    protected_watercourse_crossings = _dilate_mask(
        [priority > PRIORITY_WATERCOURSE for priority in field.priorities],
        spec.cells,
        2,
    )
    downhill_before = _watercourse_violation_stats(
        original, dataset, projection, spec, protected_watercourse_crossings
    )
    downhill_after = _watercourse_violation_stats(
        result, dataset, projection, spec, protected_watercourse_crossings
    )
    maximum_lake_shore_slope_after = _maximum_lake_bank_slope(
        result,
        lake_water_mask,
        lake_bank_mask,
        spec.cells,
        spec.cell_size,
        spec.sea_level,
    )
    effective_building_groups = [
        tuple(
            index
            for index in group
            if field.hard[index] and field.categories[index] == "building"
        )
        for group in building_pad_groups
    ]

    progress(100, "Terrain constraint solution ready")
    return ConstraintTerrainReport(
        elevations=tuple(result),
        changed_cells=changed,
        road_seed_cells=len(road_seed_cells),
        building_pad_cells=len(building_pad_cells),
        maximum_cut=maximum_cut,
        maximum_fill=maximum_fill,
        maximum_road_slope_before_percent=road_slope_before,
        maximum_road_slope_after_percent=road_slope_after,
        building_roughness_before=_component_roughness(original, effective_building_groups),
        building_roughness_after=_component_roughness(result, effective_building_groups),
        solver="unified-priority-constraint-relaxation",
        smoothing_reference_scale=smoothing.scale,
        natural_smoothing_strength=smoothing.natural_smoothing_strength,
        iterations=iterations,
        constrained_cells=sum(priority > 0 for priority in field.priorities),
        hard_constraint_cells=sum(field.hard),
        water_cells=sum(active_water_mask),
        deep_water_cells=sum(deep_water_mask),
        uncertain_water_cells_preserved=uncertain_water_cells_preserved,
        protected_shore_cells=protected_shore_cells,
        shoreline_transition_cells=effective_coastal_shore_cells,
        coastal_water_components=len(coastal_water_components),
        inland_water_components=len(inland_water_components) + len(recentered_edge_lake_components),
        lake_shore_cells=lake_shore_cells,
        osm_land_floor_cells=osm_land_floor_cells,
        lake_shore_smoothing_cells=effective_lake_shore_cells,
        lake_shore_maximum_slope_percent=spec.lake_shore_maximum_slope_percent,
        maximum_lake_shore_slope_before_percent=maximum_lake_shore_slope_before,
        maximum_lake_shore_slope_after_percent=maximum_lake_shore_slope_after,
        bridge_segments=bridge_segments,
        tunnel_segments_excluded=tunnel_segments,
        embankment_segments=embankment_segments,
        road_water_fill_cells=len(road_water_fill_cells),
        road_sidehill_segments=road_sidehill_segments,
        road_sidehill_bench_cells=len(road_sidehill_bench_cells),
        major_road_cells=len(major_cells),
        minor_road_cells=len(minor_cells),
        watercourse_cells=len(watercourse_cells),
        downhill_violations_before=downhill_before[0],
        downhill_violations_after=downhill_after[0],
        downhill_total_violations_after=downhill_after[1],
        downhill_protected_crossings=downhill_after[2],
        water_roughness_before=_water_roughness(original, water_components),
        water_roughness_after=_water_roughness(result, water_components),
        out_of_bounds_sampling=out_of_bounds_method,
        edge_blend_cells=smoothing.world_edge_blend_cells,
        total_cut_volume_m3=total_cut,
        total_fill_volume_m3=total_fill,
        category_adjustments={key: category_stats[key] for key in sorted(category_stats)},
        priority_order=(
            "water-bodies",
            "inland-lake-banks",
            "bridges-and-tunnels",
            "road-water-causeways",
            "osm-dry-land-floor",
            "sidehill-road-benches",
            "major-roads",
            "minor-roads",
            "buildings",
            "watercourses",
            "natural-smoothing",
            "world-boundaries",
        ),
    )
