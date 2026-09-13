# SPDX-License-Identifier: GPL-3.0-or-later
"""Avoid rebuilding NumPy forest/terrain arrays for every cluster candidate."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from . import forest_vector_performance_policy as _forest

_INSTALLED = False
_ORIGINAL_VECTOR_CLUSTER: Any = None
_MAX_RASTERS = 3
_MAX_TERRAINS = 3


@dataclass(slots=True)
class _RasterArrays:
    owner: Any
    forest: np.ndarray
    water: np.ndarray
    roads: np.ndarray
    buildings: np.ndarray


@dataclass(slots=True)
class _TerrainArray:
    owner: Any
    grid: np.ndarray


_RASTERS: "OrderedDict[int, _RasterArrays]" = OrderedDict()
_TERRAINS: "OrderedDict[int, _TerrainArray]" = OrderedDict()


class _RasterProxy:
    __slots__ = ("_base", "forest", "water", "roads", "buildings")

    def __init__(self, base: Any, arrays: _RasterArrays) -> None:
        self._base = base
        self.forest = arrays.forest
        self.water = arrays.water
        self.roads = arrays.roads
        self.buildings = arrays.buildings

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def _raster_arrays(raster: Any) -> _RasterArrays:
    identity = id(raster)
    cached = _RASTERS.get(identity)
    if cached is not None and cached.owner is raster:
        _RASTERS.move_to_end(identity)
        return cached
    cached = _RasterArrays(
        raster,
        np.asarray(raster.forest, dtype=np.bool_),
        np.asarray(raster.water, dtype=np.bool_),
        np.asarray(raster.roads, dtype=np.bool_),
        np.asarray(raster.buildings, dtype=np.bool_),
    )
    _RASTERS[identity] = cached
    _RASTERS.move_to_end(identity)
    while len(_RASTERS) > _MAX_RASTERS:
        _RASTERS.popitem(last=False)
    return cached


def _terrain_grid(elevations: Sequence[float], cells: int) -> np.ndarray:
    # Mutable terrain is deliberately not identity-cached. Final object placement
    # uses immutable tuples, while solver-side lists must always reflect mutation.
    if not isinstance(elevations, tuple):
        return np.asarray(elevations, dtype=np.float64).reshape(cells, cells)
    identity = id(elevations)
    cached = _TERRAINS.get(identity)
    if cached is not None and cached.owner is elevations:
        _TERRAINS.move_to_end(identity)
        return cached.grid
    value = _TerrainArray(
        elevations,
        np.asarray(elevations, dtype=np.float64).reshape(cells, cells),
    )
    _TERRAINS[identity] = value
    _TERRAINS.move_to_end(identity)
    while len(_TERRAINS) > _MAX_TERRAINS:
        _TERRAINS.popitem(last=False)
    return value.grid


def _cached_triangle_bounds_batch(
    elevations: Sequence[float],
    cells: int,
    cell_size: float,
    xs: np.ndarray,
    zs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fx = np.clip(xs / cell_size, 0.0, cells - 1.0)
    fz = np.clip(zs / cell_size, 0.0, cells - 1.0)
    x0 = np.floor(fx).astype(np.int64)
    z0 = np.floor(fz).astype(np.int64)
    x1 = np.minimum(cells - 1, x0 + 1)
    z1 = np.minimum(cells - 1, z0 + 1)
    tx = fx - x0
    tz = fz - z0
    grid = _terrain_grid(elevations, cells)
    h00 = grid[z0, x0]
    h10 = grid[z0, x1]
    h01 = grid[z1, x0]
    h11 = grid[z1, x1]
    main = np.where(
        tz <= tx,
        h00 + tx * (h10 - h00) + tz * (h11 - h10),
        h00 + tx * (h11 - h01) + tz * (h01 - h00),
    )
    cross = np.where(
        tx + tz <= 1.0,
        h00 + tx * (h10 - h00) + tz * (h01 - h00),
        h11 + (1.0 - tz) * (h10 - h11) + (1.0 - tx) * (h01 - h11),
    )
    return np.minimum(main, cross), np.maximum(main, cross)


def _cached_vector_cluster(*, raster: Any, **kwargs):
    arrays = _raster_arrays(raster)
    return _ORIGINAL_VECTOR_CLUSTER(
        raster=_RasterProxy(raster, arrays),
        **kwargs,
    )


def install_forest_array_cache_policy() -> None:
    global _INSTALLED, _ORIGINAL_VECTOR_CLUSTER
    if _INSTALLED:
        return
    _ORIGINAL_VECTOR_CLUSTER = _forest._vector_place_cluster_at
    _forest._triangle_bounds_batch = _cached_triangle_bounds_batch
    _forest._vector_place_cluster_at = _cached_vector_cluster
    # osm._place_cluster_at was assigned the old function object by the previous
    # installer, so refresh that live binding as well.
    from . import osm as _osm
    _osm._place_cluster_at = _cached_vector_cluster
    _INSTALLED = True
