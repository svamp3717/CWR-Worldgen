# SPDX-License-Identifier: GPL-3.0-or-later
"""Vectorized prefilters for the large forest placement grids.

Primary Everon forest placement can visit tens or hundreds of thousands of
regular lattice cells. Most cells are outside mapped forest, yet the historical
loop still built five probe tuples and converted each probe to raster indices in
Python before learning that fact. Undergrowth/cluster candidates likewise tested
each fixed proxy one at a time.

This policy preserves the exact detailed grounding path for candidates that can
matter while using NumPy for the cheap reject stages.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, Sequence

import numpy as np

from . import generator as _generator
from . import osm as _osm

_INSTALLED = False
_ORIGINAL_GENERATE: Any = None
_ORIGINAL_EDGE_GUARD: Any = None
_ORIGINAL_PLACE_CLUSTER: Any = None


@dataclass(slots=True)
class _ForestVectorContext:
    spacing: float
    world_size: float
    possible_primary: np.ndarray
    primary_active: bool = False


_CONTEXT: ContextVar[_ForestVectorContext | None] = ContextVar(
    "cwr_forest_vector_context", default=None
)


def _primary_forest_possible(raster: Any, spec: Any) -> _ForestVectorContext | None:
    """Bulk-evaluate the exact Everon two-of-five coarse forest prefilter."""

    if str(getattr(spec, "forest_profile", "malden")).casefold() != "everon":
        return None
    if int(getattr(spec, "max_forest_objects", 0)) == 0:
        return None
    spacing = max(1.0e-6, float(getattr(spec, "forest_tree_spacing", 50.0)))
    world_size = float(getattr(spec, "world_size"))
    cells = int(getattr(spec, "cells"))
    cell_size = float(getattr(spec, "cell_size"))
    columns = max(1, int(math.ceil(world_size / spacing)))
    rows = columns
    half = spacing * 0.5
    clearance = min(half * 0.7, max(1.0, cell_size * 0.45))

    xs = np.minimum(
        world_size - 0.001,
        (np.arange(columns, dtype=np.float64) + 0.5) * spacing,
    )
    zs = np.minimum(
        world_size - 0.001,
        (np.arange(rows, dtype=np.float64) + 0.5) * spacing,
    )
    xx, zz = np.meshgrid(xs, zs)
    forest = np.asarray(raster.forest, dtype=np.bool_).reshape(cells, cells)
    forest_count = np.zeros((rows, columns), dtype=np.uint8)
    scale = cells / world_size

    for offset_x, offset_z in (
        (0.0, 0.0),
        (-clearance, -clearance),
        (clearance, -clearance),
        (-clearance, clearance),
        (clearance, clearance),
    ):
        sx = xx + offset_x
        sz = zz + offset_z
        valid = (
            (sx >= 0.0) & (sx < world_size)
            & (sz >= 0.0) & (sz < world_size)
        )
        cols = np.clip((sx * scale).astype(np.int64), 0, cells - 1)
        rows_index = np.clip((sz * scale).astype(np.int64), 0, cells - 1)
        forest_count += valid & forest[rows_index, cols]

    # This is the first semantic reject in the default Everon primary loop. A
    # block with fewer than two forest probes was historically discarded before
    # any road query or diagnostic counter changed, so moving it ahead of tuple
    # construction is result-equivalent.
    possible = forest_count >= 2
    return _ForestVectorContext(spacing, world_size, possible)


def _fast_edge_guard(x: float, z: float, margin: float) -> bool:
    if not _ORIGINAL_EDGE_GUARD(x, z, margin):
        return False
    context = _CONTEXT.get()
    if context is None or not context.primary_active:
        return True

    column = int(math.floor(float(x) / context.spacing))
    row = int(math.floor(float(z) / context.spacing))
    rows, columns = context.possible_primary.shape
    if not (0 <= row < rows and 0 <= column < columns):
        return True
    expected_x = min(
        context.world_size - 0.001,
        (column + 0.5) * context.spacing,
    )
    expected_z = min(
        context.world_size - 0.001,
        (row + 0.5) * context.spacing,
    )
    # The edge-guard helper is reused by later vegetation passes. Only intercept
    # the exact regular primary-block centres; every other call keeps the old
    # behaviour even while a nested helper happens to run during this stage.
    if abs(float(x) - expected_x) > 1.0e-6 or abs(float(z) - expected_z) > 1.0e-6:
        return True
    return bool(context.possible_primary[row, column])


@lru_cache(maxsize=128)
def _proxy_tree_flags(variant_name: str, models: tuple[str, ...]) -> tuple[bool, ...]:
    return tuple(_osm._forest_proxy_is_tree(model) for model in models)


def _triangle_bounds_batch(
    elevations: Sequence[float],
    cells: int,
    cell_size: float,
    xs: np.ndarray,
    zs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized form of ``_triangle_elevation_bounds`` for proxy points."""

    fx = np.clip(xs / cell_size, 0.0, cells - 1.0)
    fz = np.clip(zs / cell_size, 0.0, cells - 1.0)
    x0 = np.floor(fx).astype(np.int64)
    z0 = np.floor(fz).astype(np.int64)
    x1 = np.minimum(cells - 1, x0 + 1)
    z1 = np.minimum(cells - 1, z0 + 1)
    tx = fx - x0
    tz = fz - z0
    grid = np.asarray(elevations, dtype=np.float64).reshape(cells, cells)
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


def _vector_place_cluster_at(
    *,
    variant: Any,
    elevations: Sequence[float],
    raster: Any,
    road_corridors: Sequence[object],
    spec: object,
    x: float,
    z: float,
    heading: float,
    require_forest: bool,
    minimum_forest_fraction: float,
    maximum_relief: float,
    maximum_burial: float,
    maximum_float: float,
    clearance: float,
    avoid_roads: bool = True,
):
    """Ground a fixed proxy cluster with batched transform/mask/terrain checks."""

    world_size = float(getattr(spec, "world_size"))
    cells = int(getattr(spec, "cells"))
    cell_size = float(getattr(spec, "cell_size"))
    world_name = str(getattr(spec, "name"))
    margin = max(
        0.0, float(getattr(spec, "forest_cluster_footprint_margin", 0.75))
    )
    if not (0.0 <= x < world_size and 0.0 <= z < world_size):
        return None

    gradient_x, gradient_z = _osm._local_terrain_gradient(
        elevations, cells, cell_size, x, z
    )
    heading, grade = _osm._cluster_heading_and_grade(
        gradient_x, gradient_z, heading % 360.0, variant.slope_axis
    )
    polygon = _osm._oriented_rectangle(
        x, z, variant.width_m, variant.length_m, heading, margin=margin
    )
    if not all(0.0 <= px < world_size and 0.0 <= pz < world_size for px, pz in polygon):
        return None
    if (
        bool(getattr(spec, "forest_low_anchor", False))
        and world_size >= 200.0
        and variant.category in {"border", "interior", "undergrowth"}
    ):
        proxy_edge_guard = max(8.0, min(18.0, cell_size * 0.55))
        if not all(
            proxy_edge_guard <= px <= world_size - proxy_edge_guard
            and proxy_edge_guard <= pz <= world_size - proxy_edge_guard
            for px, pz in polygon
        ):
            return None

    # Cheap indexed road rejection stays ahead of terrain work.
    if avoid_roads and _osm.forest_block_intersects_road_corridors(
        road_corridors,
        x,
        z,
        block_size=max(variant.width_m, variant.length_m) + 2.0 * margin,
    ):
        return None

    minimum, maximum = _osm._polygon_elevation_extrema(
        elevations, cells, cell_size, polygon
    )
    relief = maximum - minimum
    if relief > min(max(0.0, maximum_relief), variant.maximum_relief_m):
        return None

    layout = tuple(variant.proxy_layout)
    if not layout:
        return None
    local_x = np.fromiter((item[1] for item in layout), dtype=np.float64)
    local_z = np.fromiter((item[2] for item in layout), dtype=np.float64)
    models = tuple(str(item[0]) for item in layout)
    angle = math.radians(heading)
    width_x, width_z = math.cos(angle), -math.sin(angle)
    length_x, length_z = math.sin(angle), math.cos(angle)
    world_x = x + local_x * width_x + local_z * length_x
    world_z = z + local_x * width_z + local_z * length_z
    in_world = (
        (world_x >= 0.0) & (world_x < world_size)
        & (world_z >= 0.0) & (world_z < world_size)
    )
    if not bool(np.all(in_world)):
        return None

    scale = cells / world_size
    cols = np.clip((world_x * scale).astype(np.int64), 0, cells - 1)
    rows = np.clip((world_z * scale).astype(np.int64), 0, cells - 1)
    indices = rows * cells + cols
    forest_mask = np.asarray(raster.forest, dtype=np.bool_)[indices]
    if require_forest and not bool(np.all(forest_mask)):
        return None
    if float(np.count_nonzero(forest_mask)) / len(layout) < minimum_forest_fraction:
        return None
    blocked = (
        np.asarray(raster.water, dtype=np.bool_)[indices]
        | np.asarray(raster.roads, dtype=np.bool_)[indices]
        | np.asarray(raster.buildings, dtype=np.bool_)[indices]
    )
    if bool(np.any(blocked)):
        return None

    model_y = grade * (local_x if variant.slope_axis == "width" else local_z)
    lower_ground, upper_ground = _triangle_bounds_batch(
        elevations, cells, cell_size, world_x, world_z
    )
    lower_support = lower_ground - model_y
    upper_support = upper_ground - model_y
    supports = tuple(
        value
        for pair in zip(lower_support.tolist(), upper_support.tolist())
        for value in pair
    )

    if variant.category in {"border", "undergrowth", "ditch", "rural"}:
        fitted = _osm._non_buried_vegetation_fit(
            supports,
            clearance=clearance,
            maximum_float=maximum_float,
        )
        if fitted is None:
            return None
        anchor, floating = fitted
        burial = 0.0
    else:
        fitted = _osm._terrain_fit_anchor(
            supports,
            clearance=clearance,
            maximum_burial=maximum_burial,
            maximum_float=maximum_float,
        )
        if fitted is None:
            return None
        anchor, burial, floating = fitted

    flags = np.asarray(
        _proxy_tree_flags(variant.name, models), dtype=np.bool_
    )
    proxy_float = np.maximum(0.0, anchor + model_y - lower_ground)
    tree_count = int(np.count_nonzero(flags))
    bush_count = len(flags) - tree_count
    maximum_tree_float = (
        float(np.max(proxy_float[flags])) if tree_count else 0.0
    )
    maximum_bush_float = (
        float(np.max(proxy_float[~flags])) if bush_count else 0.0
    )
    tree_limit = min(
        max(0.0, maximum_float),
        max(0.0, float(getattr(spec, "forest_cluster_tree_maximum_float", 0.20))),
    )
    bush_limit = min(
        max(0.0, maximum_float),
        max(0.0, float(getattr(spec, "forest_cluster_bush_maximum_float", 0.60))),
    )
    if (
        tree_count and maximum_tree_float > tree_limit + 1.0e-9
    ) or (
        bush_count and maximum_bush_float > bush_limit + 1.0e-9
    ):
        return None

    floating = max(floating, maximum_tree_float, maximum_bush_float)
    return (
        _osm.cluster_model_path(world_name, variant.name, grade),
        x,
        anchor,
        z,
        heading,
        variant.name,
        relief,
        burial,
        floating,
    )


def _generate_with_vector_forest_context(*args, **kwargs):
    if len(args) >= 5:
        raster, spec = args[2], args[4]
    else:
        raster, spec = kwargs.get("raster"), kwargs.get("spec")
    context = (
        _primary_forest_possible(raster, spec)
        if raster is not None and spec is not None
        else None
    )
    if context is None:
        return _ORIGINAL_GENERATE(*args, **kwargs)

    original_progress = kwargs.get("progress_callback")

    def progress(value: int, stage: str) -> None:
        text = str(stage)
        context.primary_active = text.startswith("Placing primary forest blocks")
        if original_progress is not None:
            original_progress(value, stage)

    kwargs["progress_callback"] = progress
    token = _CONTEXT.set(context)
    try:
        return _ORIGINAL_GENERATE(*args, **kwargs)
    finally:
        _CONTEXT.reset(token)


def install_forest_vector_performance_policy() -> None:
    global _INSTALLED, _ORIGINAL_GENERATE, _ORIGINAL_EDGE_GUARD, _ORIGINAL_PLACE_CLUSTER
    if _INSTALLED:
        return

    _ORIGINAL_GENERATE = _generator.generate_world_objects
    _ORIGINAL_EDGE_GUARD = _osm.forest_point_inside_edge_guard
    _ORIGINAL_PLACE_CLUSTER = _osm._place_cluster_at
    _osm.forest_point_inside_edge_guard = _fast_edge_guard
    _osm._place_cluster_at = _vector_place_cluster_at
    _generator.generate_world_objects = _generate_with_vector_forest_context
    _INSTALLED = True
