# SPDX-License-Identifier: GPL-3.0-or-later
"""Accelerate source-mapped water tests used by bridge terrain grading.

The source-water fallback historically sampled long roads every 0.5 metres and
then ran a Python ray cast through every candidate water polygon for every
sample. A 10-20 km explicit bridge-tagged road can therefore perform tens of
thousands of point-in-ring tests and stall the road-constraint stage.

Keep the historical sampling interval and boundary-refinement policy, but index
projected water bounds once per active source context, interpolate all sample
positions in NumPy, and classify sample points with Shapely vector operations.
Outer and hole rings are tested separately so hole boundaries retain the old
"dry" semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
from shapely import covers as vectorized_covers, points as vectorized_points, prepare as prepare_geometry
from shapely.geometry import LineString, Polygon, box
from shapely.strtree import STRtree

from . import bridge_source_water_policy as _source


_DIAGNOSTIC_SAMPLE_THRESHOLD = 10_000
_CACHE_LIMIT = 4
_INSTALLED = False
_ORIGINAL_SOURCE_INTERVAL = _source._source_mapped_water_interval

PointXZ = tuple[float, float]


@dataclass(slots=True)
class _IndexedWaterPolygon:
    source: Any
    outer: Any
    holes: tuple[Any, ...]


@dataclass(slots=True)
class _WaterIndex:
    context: Any
    polygons: tuple[_IndexedWaterPolygon, ...]
    bounds_geometries: tuple[Any, ...]
    tree: Any | None


_INDEX_CACHE: dict[int, _WaterIndex] = {}


def _polygon_geometry(ring: Sequence[PointXZ]) -> Any:
    geometry = Polygon(tuple((float(x), float(z)) for x, z in ring))
    try:
        prepare_geometry(geometry)
    except Exception:
        pass
    return geometry


def _build_index(context: Any) -> _WaterIndex:
    indexed: list[_IndexedWaterPolygon] = []
    bounds_geometries: list[Any] = []
    for source_polygon in context.water:
        outer = _polygon_geometry(source_polygon.outer)
        holes = tuple(_polygon_geometry(ring) for ring in source_polygon.holes)
        indexed.append(_IndexedWaterPolygon(source_polygon, outer, holes))
        bounds_geometries.append(box(*source_polygon.bounds))
    tree = STRtree(bounds_geometries) if bounds_geometries else None
    return _WaterIndex(
        context=context,
        polygons=tuple(indexed),
        bounds_geometries=tuple(bounds_geometries),
        tree=tree,
    )


def _water_index(context: Any) -> _WaterIndex:
    key = id(context)
    cached = _INDEX_CACHE.get(key)
    if cached is not None and cached.context is context:
        return cached
    if len(_INDEX_CACHE) >= _CACHE_LIMIT:
        _INDEX_CACHE.clear()
    result = _build_index(context)
    _INDEX_CACHE[key] = result
    return result


def _fallback_candidates(
    index: _WaterIndex,
    cleaned: Sequence[PointXZ],
) -> tuple[_IndexedWaterPolygon, ...]:
    min_x = min(float(point[0]) for point in cleaned)
    min_z = min(float(point[1]) for point in cleaned)
    max_x = max(float(point[0]) for point in cleaned)
    max_z = max(float(point[1]) for point in cleaned)
    return tuple(
        polygon
        for polygon in index.polygons
        if not (
            polygon.source.bounds[2] < min_x
            or polygon.source.bounds[0] > max_x
            or polygon.source.bounds[3] < min_z
            or polygon.source.bounds[1] > max_z
        )
    )


def _candidate_polygons(
    context: Any,
    cleaned: Sequence[PointXZ],
) -> tuple[_IndexedWaterPolygon, ...]:
    index = _water_index(context)
    if index.tree is None or not index.polygons:
        return ()
    try:
        line = LineString(tuple((float(x), float(z)) for x, z in cleaned))
        hits = np.asarray(index.tree.query(line, predicate="intersects"), dtype=np.int64)
        if hits.size == 0:
            return ()
        unique = np.unique(hits)
        return tuple(index.polygons[int(position)] for position in unique)
    except Exception:
        return _fallback_candidates(index, cleaned)


def _sample_positions(
    cleaned: Sequence[PointXZ],
    cumulative: Sequence[float],
    distances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(cleaned, dtype=np.float64)
    travel = np.asarray(cumulative, dtype=np.float64)
    if points.shape[0] < 2:
        if points.shape[0] == 0:
            return np.zeros_like(distances), np.zeros_like(distances)
        return (
            np.full(distances.shape, float(points[0, 0]), dtype=np.float64),
            np.full(distances.shape, float(points[0, 1]), dtype=np.float64),
        )

    targets = np.clip(distances, 0.0, float(travel[-1]))
    segment = np.searchsorted(travel[1:], targets, side="left")
    segment = np.clip(segment, 0, points.shape[0] - 2)
    start_distance = travel[segment]
    end_distance = travel[segment + 1]
    length = end_distance - start_distance
    fraction = np.divide(
        targets - start_distance,
        length,
        out=np.zeros_like(targets),
        where=length > 1.0e-12,
    )
    start = points[segment, :2]
    end = points[segment + 1, :2]
    positions = start + (end - start) * fraction[:, None]
    return positions[:, 0], positions[:, 1]


def _vectorized_point_in_water(
    xs: np.ndarray,
    zs: np.ndarray,
    polygons: Sequence[_IndexedWaterPolygon],
) -> np.ndarray:
    wet = np.zeros(xs.shape, dtype=np.bool_)
    for polygon in polygons:
        if np.all(wet):
            break
        min_x, min_z, max_x, max_z = polygon.source.bounds
        candidate_mask = (
            (~wet)
            & (xs >= float(min_x))
            & (xs <= float(max_x))
            & (zs >= float(min_z))
            & (zs <= float(max_z))
        )
        positions = np.flatnonzero(candidate_mask)
        if positions.size == 0:
            continue
        points = vectorized_points(xs[positions], zs[positions])
        try:
            inside = np.asarray(vectorized_covers(polygon.outer, points), dtype=np.bool_)
            if np.any(inside) and polygon.holes:
                for hole in polygon.holes:
                    inside &= ~np.asarray(vectorized_covers(hole, points), dtype=np.bool_)
                    if not np.any(inside):
                        break
            if np.any(inside):
                wet[positions[inside]] = True
        except Exception:
            source_polygon = (polygon.source,)
            for position in positions:
                if _source._point_in_water(
                    (float(xs[position]), float(zs[position])),
                    source_polygon,
                ):
                    wet[position] = True
    return wet


def _fast_source_mapped_water_interval(
    points: Sequence[PointXZ],
    context: Any | None = None,
) -> tuple[float, float] | None:
    context = _source._CONTEXT.get() if context is None else context
    if context is None or not context.water:
        return None

    cleaned, cumulative = _source._polyline_measure(points)
    if len(cleaned) < 2 or cumulative[-1] <= 0.01:
        return None

    candidates = _candidate_polygons(context, cleaned)
    if not candidates:
        return None

    total = float(cumulative[-1])
    count = max(1, int(math.ceil(total / _source._SOURCE_WATER_SAMPLE_STEP_METRES)))
    distances = np.asarray(
        [total * index / count for index in range(count + 1)],
        dtype=np.float64,
    )
    diagnostic = distances.size >= _DIAGNOSTIC_SAMPLE_THRESHOLD
    if diagnostic:
        print(
            "[bridge-source-water] batched mapped-water probe: "
            f"length={total:,.1f}m, samples={distances.size:,}, "
            f"candidate_polygons={len(candidates):,}",
            flush=True,
        )

    xs, zs = _sample_positions(cleaned, cumulative, distances)
    wet = _vectorized_point_in_water(xs, zs, candidates)
    indices = np.flatnonzero(wet)
    if indices.size == 0:
        if diagnostic:
            print("[bridge-source-water] mapped-water probe complete: no wet samples", flush=True)
        return None

    first = int(indices[0])
    last = int(indices[-1])
    start = float(distances[first])
    end = float(distances[last])
    source_candidates = tuple(polygon.source for polygon in candidates)

    if first > 0 and not bool(wet[first - 1]):
        low, high = float(distances[first - 1]), float(distances[first])
        for _ in range(_source._SOURCE_WATER_REFINEMENT_STEPS):
            middle = (low + high) * 0.5
            if _source._point_in_water(
                _source._point_at(cleaned, cumulative, middle),
                source_candidates,
            ):
                high = middle
            else:
                low = middle
        start = high

    if last + 1 < len(distances) and not bool(wet[last + 1]):
        low, high = float(distances[last]), float(distances[last + 1])
        for _ in range(_source._SOURCE_WATER_REFINEMENT_STEPS):
            middle = (low + high) * 0.5
            if _source._point_in_water(
                _source._point_at(cleaned, cumulative, middle),
                source_candidates,
            ):
                low = middle
            else:
                high = middle
        end = low

    result = (start, end) if end > start + 1.0e-4 else None
    if diagnostic:
        if result is None:
            detail = "degenerate interval"
        else:
            detail = f"wet_interval={result[0]:,.2f}-{result[1]:,.2f}m"
        print(f"[bridge-source-water] mapped-water probe complete: {detail}", flush=True)
    return result


def install_bridge_source_water_performance_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _source._source_mapped_water_interval = _fast_source_mapped_water_interval
    _INSTALLED = True
