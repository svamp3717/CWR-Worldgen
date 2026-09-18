# SPDX-License-Identifier: GPL-3.0-or-later
"""Cache repeated terrain samples used by stock roads and forest placement.

The object-placement performance layer removes broad geometry scans.  The
remaining hot loops still ask identical small sampling questions many times:
adjacent stock road pieces share endpoints, stock-piece candidates revisit the
same polyline breakpoints, and the regular forest lattice repeatedly rebuilds
the same terrain breakpoint/patch-centre tables.  Cache those pure results while
leaving mutable elevation sequences on the historical uncached path.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

from . import osm as _osm
from . import playability as _playability

_ORIGINAL_OSM_SAMPLE = _osm._sample_elevation
_ORIGINAL_OSM_TRIANGLE = _osm._triangle_elevation_bounds
_ORIGINAL_TERRAIN_AXIS_BREAKPOINTS = _osm._terrain_axis_breakpoints
_ORIGINAL_TERRAIN_PATCH_CENTRES = _osm._terrain_patch_centres
_ORIGINAL_PLAYABILITY_SAMPLE = _playability._sample_elevation
_ORIGINAL_MEASURE_POINT = _playability._PolylineMeasure.point

_INSTALLED = False
_MAX_ELEVATION_OWNERS = 3
_MAX_SAMPLE_ENTRIES = 262_144
_MAX_TRIANGLE_ENTRIES = 524_288
_MAX_MEASURE_OWNERS = 128
_MAX_MEASURE_POINTS = 2_048


@dataclass(slots=True)
class _ElevationCache:
    owner: tuple[Any, ...]
    osm_samples: dict[tuple[int, float, float, float], float]
    playability_samples: dict[tuple[int, float, float, float], float]
    triangles: dict[tuple[int, float, float, float], tuple[float, float]]


@dataclass(slots=True)
class _MeasurePointCache:
    owner: Any
    points: dict[float, tuple[float, float, float]]


_ELEVATION_CACHES: "OrderedDict[int, _ElevationCache]" = OrderedDict()
_MEASURE_CACHES: "OrderedDict[int, _MeasurePointCache]" = OrderedDict()


def _elevation_cache(elevations: Sequence[float]) -> _ElevationCache | None:
    """Return an identity cache only for immutable tuple terrain arrays."""
    if not isinstance(elevations, tuple):
        return None
    identity = id(elevations)
    cache = _ELEVATION_CACHES.get(identity)
    if cache is not None and cache.owner is elevations:
        _ELEVATION_CACHES.move_to_end(identity)
        return cache
    cache = _ElevationCache(elevations, {}, {}, {})
    _ELEVATION_CACHES[identity] = cache
    _ELEVATION_CACHES.move_to_end(identity)
    while len(_ELEVATION_CACHES) > _MAX_ELEVATION_OWNERS:
        _ELEVATION_CACHES.popitem(last=False)
    return cache


def _fast_osm_sample(
    elevations: Sequence[float], cells: int, cell_size: float, x: float, z: float
) -> float:
    cache = _elevation_cache(elevations)
    if cache is None:
        return _ORIGINAL_OSM_SAMPLE(elevations, cells, cell_size, x, z)
    key = (int(cells), float(cell_size), float(x), float(z))
    value = cache.osm_samples.get(key)
    if value is not None:
        return value
    value = _ORIGINAL_OSM_SAMPLE(elevations, cells, cell_size, x, z)
    if len(cache.osm_samples) >= _MAX_SAMPLE_ENTRIES:
        cache.osm_samples.clear()
    cache.osm_samples[key] = value
    return value


def _fast_playability_sample(
    elevations: Sequence[float], cells: int, cell_size: float, x: float, z: float
) -> float:
    cache = _elevation_cache(elevations)
    if cache is None:
        return _ORIGINAL_PLAYABILITY_SAMPLE(elevations, cells, cell_size, x, z)
    key = (int(cells), float(cell_size), float(x), float(z))
    value = cache.playability_samples.get(key)
    if value is not None:
        return value
    value = _ORIGINAL_PLAYABILITY_SAMPLE(elevations, cells, cell_size, x, z)
    if len(cache.playability_samples) >= _MAX_SAMPLE_ENTRIES:
        cache.playability_samples.clear()
    cache.playability_samples[key] = value
    return value


def _fast_triangle_elevation_bounds(
    elevations: Sequence[float], cells: int, cell_size: float, x: float, z: float
) -> tuple[float, float]:
    cache = _elevation_cache(elevations)
    if cache is None:
        return _ORIGINAL_OSM_TRIANGLE(elevations, cells, cell_size, x, z)
    key = (int(cells), float(cell_size), float(x), float(z))
    value = cache.triangles.get(key)
    if value is not None:
        return value
    value = _ORIGINAL_OSM_TRIANGLE(elevations, cells, cell_size, x, z)
    if len(cache.triangles) >= _MAX_TRIANGLE_ENTRIES:
        cache.triangles.clear()
    cache.triangles[key] = value
    return value


@lru_cache(maxsize=16_384)
def _cached_terrain_axis_breakpoints(
    minimum: float, maximum: float, cells: int, cell_size: float
) -> tuple[float, ...]:
    return _ORIGINAL_TERRAIN_AXIS_BREAKPOINTS(
        minimum, maximum, cells, cell_size
    )


def _fast_terrain_axis_breakpoints(
    minimum: float, maximum: float, cells: int, cell_size: float
) -> tuple[float, ...]:
    return _cached_terrain_axis_breakpoints(
        float(minimum), float(maximum), int(cells), float(cell_size)
    )


@lru_cache(maxsize=16_384)
def _cached_terrain_patch_centres(
    minimum: float, maximum: float, cells: int, cell_size: float
) -> tuple[float, ...]:
    return _ORIGINAL_TERRAIN_PATCH_CENTRES(
        minimum, maximum, cells, cell_size
    )


def _fast_terrain_patch_centres(
    minimum: float, maximum: float, cells: int, cell_size: float
) -> tuple[float, ...]:
    return _cached_terrain_patch_centres(
        float(minimum), float(maximum), int(cells), float(cell_size)
    )


def _measure_cache(measure: Any) -> _MeasurePointCache:
    identity = id(measure)
    cache = _MEASURE_CACHES.get(identity)
    if cache is not None and cache.owner is measure:
        _MEASURE_CACHES.move_to_end(identity)
        return cache
    cache = _MeasurePointCache(measure, {})
    _MEASURE_CACHES[identity] = cache
    _MEASURE_CACHES.move_to_end(identity)
    while len(_MEASURE_CACHES) > _MAX_MEASURE_OWNERS:
        _MEASURE_CACHES.popitem(last=False)
    return cache


def _fast_measure_point(self: Any, distance: float) -> tuple[float, float, float]:
    cache = _measure_cache(self)
    key = float(distance)
    value = cache.points.get(key)
    if value is not None:
        return value
    value = _ORIGINAL_MEASURE_POINT(self, distance)
    if len(cache.points) >= _MAX_MEASURE_POINTS:
        cache.points.clear()
    cache.points[key] = value
    return value


def install_object_sampling_cache_policy() -> None:
    """Install exact-result caches around object-placement sampling helpers."""
    global _INSTALLED
    if _INSTALLED:
        return
    _osm._sample_elevation = _fast_osm_sample
    _osm._triangle_elevation_bounds = _fast_triangle_elevation_bounds
    _osm._terrain_axis_breakpoints = _fast_terrain_axis_breakpoints
    _osm._terrain_patch_centres = _fast_terrain_patch_centres
    _playability._sample_elevation = _fast_playability_sample
    _playability._PolylineMeasure.point = _fast_measure_point
    _INSTALLED = True
