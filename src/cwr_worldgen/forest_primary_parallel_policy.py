# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic multicore evaluation for regular primary forest blocks.

Workers evaluate the expensive read-only terrain support/grounding work for
regular Everon forest blocks. The historical generator still owns row-major
acceptance, counters, limits, object ids and all fallback emission, so parallel
execution cannot reorder the resulting objects.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass
import math
from typing import Any, Iterator, Sequence

import numpy as np

from . import forest_vector_performance_policy as _forest
from . import generator as _generator
from . import object_stage_parallel_policy as _object_parallel
from . import osm as _osm

_INSTALLED = False
_BASE_GENERATE: Any = None
_BASE_SAMPLE: Any = None
_BASE_SQUARE: Any = None
_BASE_TERRAIN_FIT: Any = None
_BASE_ROAD_TEST: Any = None

_W_ELEVATIONS: Sequence[float] | None = None
_W_CELLS = 0
_W_CELL_SIZE = 1.0
_W_WORLD_SIZE = 0.0
_W_SPACING = 1.0
_W_COLUMNS = 1
_W_FLATS: np.ndarray | None = None
_W_LOW_ANCHOR = False
_W_MAXIMUM_RELIEF = 0.0
_W_MAXIMUM_BURIAL = 0.0
_W_MAXIMUM_FLOAT = 0.0
_W_CLEARANCE = 0.0


class _PrimarySupports:
    """Compact replacement for a full terrain-support tuple."""

    __slots__ = ("minimum", "maximum", "fit")

    def __init__(
        self,
        minimum: float,
        maximum: float,
        fit: tuple[float, float, float] | None,
    ) -> None:
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.fit = fit

    def __len__(self) -> int:
        return 2

    def __iter__(self) -> Iterator[float]:
        yield self.minimum
        yield self.maximum

    def __getitem__(self, index: int) -> float:
        if index in (0, -2):
            return self.minimum
        if index in (1, -1):
            return self.maximum
        raise IndexError(index)


@dataclass(slots=True)
class _PrimaryForestContext:
    elevations: Sequence[float]
    cells: int
    cell_size: float
    world_size: float
    spacing: float
    rows: int
    columns: int
    road_hits: np.ndarray
    lookup: np.ndarray
    centre_heights: np.ndarray
    bounds: np.ndarray
    fits: np.ndarray | None
    active: bool = False


_CONTEXT: ContextVar[_PrimaryForestContext | None] = ContextVar(
    "cwr_parallel_primary_forest_context", default=None
)


def _fit_to_row(value: tuple[float, float, float] | None) -> tuple[float, float, float]:
    if value is None:
        return (math.nan, math.nan, math.nan)
    return (float(value[0]), float(value[1]), float(value[2]))


def _fit_from_row(row: np.ndarray) -> tuple[float, float, float] | None:
    if np.isnan(row[0]):
        return None
    return (float(row[0]), float(row[1]), float(row[2]))


def _raw_sample(elevations, cells: int, cell_size: float, x: float, z: float) -> float:
    fn = _object_parallel._BASE_OSM_SAMPLE or _osm._sample_elevation
    return float(fn(elevations, cells, cell_size, x, z))


def _raw_square(elevations, cells: int, cell_size: float, x: float, z: float, size: float):
    fn = _object_parallel._BASE_SQUARE_SAMPLES or _osm._square_elevation_samples
    return fn(elevations, cells, cell_size, x, z, size)


def _raw_fit(supports):
    fn = _BASE_TERRAIN_FIT or _osm._terrain_fit_anchor
    return fn(
        supports,
        clearance=_W_CLEARANCE,
        maximum_burial=_W_MAXIMUM_BURIAL,
        maximum_float=_W_MAXIMUM_FLOAT,
    )


def _init_worker(elevations, spec, columns: int, flats) -> None:
    global _W_ELEVATIONS, _W_CELLS, _W_CELL_SIZE, _W_WORLD_SIZE
    global _W_SPACING, _W_COLUMNS, _W_FLATS, _W_LOW_ANCHOR
    global _W_MAXIMUM_RELIEF, _W_MAXIMUM_BURIAL, _W_MAXIMUM_FLOAT, _W_CLEARANCE
    _W_ELEVATIONS = elevations
    _W_CELLS = int(spec.cells)
    _W_CELL_SIZE = float(spec.cell_size)
    _W_WORLD_SIZE = float(spec.world_size)
    _W_SPACING = float(spec.forest_tree_spacing)
    _W_COLUMNS = int(columns)
    _W_FLATS = flats
    _W_LOW_ANCHOR = bool(getattr(spec, "forest_low_anchor", False))
    _W_MAXIMUM_RELIEF = max(
        0.0, float(getattr(spec, "forest_maximum_block_relief", 8.0))
    )
    _W_MAXIMUM_BURIAL = max(
        0.0, float(getattr(spec, "forest_block_maximum_burial", 8.0))
    )
    _W_MAXIMUM_FLOAT = max(
        0.0, float(getattr(spec, "forest_block_maximum_float", 0.5))
    )
    _W_CLEARANCE = float(spec.forest_ground_clearance)


def _evaluate_batch(job: tuple[int, int]):
    offset, count = job
    results = []
    for flat in _W_FLATS[offset: offset + count]:
        row, column = divmod(int(flat), _W_COLUMNS)
        x = min(_W_WORLD_SIZE - 0.001, (column + 0.5) * _W_SPACING)
        z = min(_W_WORLD_SIZE - 0.001, (row + 0.5) * _W_SPACING)
        centre = _raw_sample(_W_ELEVATIONS, _W_CELLS, _W_CELL_SIZE, x, z)
        supports = _raw_square(
            _W_ELEVATIONS, _W_CELLS, _W_CELL_SIZE, x, z, _W_SPACING
        )
        minimum = min(supports)
        maximum = max(supports)
        fit = None
        if _W_LOW_ANCHOR and maximum - minimum <= _W_MAXIMUM_RELIEF:
            fit = _raw_fit(supports)
        results.append(
            (float(centre), float(minimum), float(maximum), _fit_to_row(fit))
        )
    return offset, tuple(results)


def _batches(total: int, workers: int) -> tuple[tuple[int, int], ...]:
    if total <= 0:
        return ()
    size = max(32, min(256, math.ceil(total / max(1, workers * 12))))
    return tuple(
        (offset, min(size, total - offset))
        for offset in range(0, total, size)
    )


def _terrain_candidates(raster, spec, primary, road_hits: np.ndarray) -> np.ndarray:
    selected = np.flatnonzero(primary.possible_primary.ravel()).astype(np.int64, copy=False)
    if selected.size == 0:
        return selected

    columns = int(primary.possible_primary.shape[1])
    spacing = float(primary.spacing)
    world_size = float(spec.world_size)
    cells = int(spec.cells)
    cell_size = float(spec.cell_size)
    rows = selected // columns
    cols = selected % columns
    xs = np.minimum(world_size - 0.001, (cols.astype(np.float64) + 0.5) * spacing)
    zs = np.minimum(world_size - 0.001, (rows.astype(np.float64) + 0.5) * spacing)

    edge_guard_enabled = bool(getattr(spec, "forest_low_anchor", False)) and world_size >= 200.0
    edge_margin = max(24.0, min(40.0, cell_size * 1.25)) if edge_guard_enabled else 0.0
    block_guard = max(edge_margin, spacing * 0.58)
    eligible = np.ones(selected.shape, dtype=np.bool_)
    if edge_guard_enabled:
        eligible &= (
            (xs >= block_guard)
            & (xs <= world_size - block_guard)
            & (zs >= block_guard)
            & (zs <= world_size - block_guard)
        )
    eligible &= ~road_hits.ravel()[selected]

    forest = np.asarray(raster.forest, dtype=np.bool_)
    water = np.asarray(raster.water, dtype=np.bool_)
    roads = np.asarray(raster.roads, dtype=np.bool_)
    buildings = np.asarray(raster.buildings, dtype=np.bool_)
    scale = cells / world_size
    half = spacing * 0.5
    clearance = min(half * 0.7, max(1.0, cell_size * 0.45))
    for dx, dz in (
        (0.0, 0.0),
        (-clearance, -clearance),
        (clearance, -clearance),
        (-clearance, clearance),
        (clearance, clearance),
    ):
        sx = xs + dx
        sz = zs + dz
        valid = (sx >= 0.0) & (sx < world_size) & (sz >= 0.0) & (sz < world_size)
        cell_x = np.clip((sx * scale).astype(np.int64), 0, cells - 1)
        cell_z = np.clip((sz * scale).astype(np.int64), 0, cells - 1)
        indices = cell_z * cells + cell_x
        eligible &= (
            valid
            & forest[indices]
            & ~water[indices]
            & ~roads[indices]
            & ~buildings[indices]
        )
    return selected[eligible]


def _build_context(dataset, projection, raster, elevations, spec, progress):
    if str(getattr(spec, "forest_profile", "malden")).casefold() != "everon":
        return None
    if int(getattr(spec, "max_forest_objects", 0)) == 0:
        return None
    if bool(getattr(spec, "forest_individual_objects_only", False)):
        return None

    primary = _forest._primary_forest_possible(raster, spec)
    if primary is None:
        return None
    corridors = _osm.project_road_corridors(dataset, projection, spec)
    road_hits = _forest._batch_primary_road_hits(corridors, primary)
    flats = _terrain_candidates(raster, spec, primary, road_hits)
    workers = _object_parallel._worker_count(int(flats.size))
    if workers <= 1 or flats.size == 0:
        return None

    if progress is not None:
        progress(
            52,
            f"Parallel evaluating {int(flats.size):,} primary forest blocks with {workers} workers",
        )

    total = int(flats.size)
    lookup = np.full(int(primary.possible_primary.size), -1, dtype=np.int32)
    lookup[flats] = np.arange(total, dtype=np.int32)
    centres = np.empty(total, dtype=np.float64)
    bounds = np.empty((total, 2), dtype=np.float64)
    low_anchor = bool(getattr(spec, "forest_low_anchor", False))
    fits = np.full((total, 3), np.nan, dtype=np.float64) if low_anchor else None
    batches = _batches(total, workers)
    progress_stride = max(1, len(batches) // 20)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(elevations, spec, int(primary.possible_primary.shape[1]), flats),
    ) as executor:
        for batch_index, (offset, results) in enumerate(
            executor.map(_evaluate_batch, batches, chunksize=1)
        ):
            for local_index, (centre, minimum, maximum, fit_row) in enumerate(results):
                index = offset + local_index
                centres[index] = centre
                bounds[index] = (minimum, maximum)
                if fits is not None:
                    fits[index] = fit_row
            if progress is not None and (
                (batch_index + 1) % progress_stride == 0
                or batch_index + 1 == len(batches)
            ):
                done = min(total, offset + len(results))
                progress(
                    52,
                    f"Parallel evaluating primary forest blocks {done:,}/{total:,} with {workers} workers",
                )

    return _PrimaryForestContext(
        elevations=elevations,
        cells=int(spec.cells),
        cell_size=float(spec.cell_size),
        world_size=float(spec.world_size),
        spacing=float(primary.spacing),
        rows=int(primary.possible_primary.shape[0]),
        columns=int(primary.possible_primary.shape[1]),
        road_hits=np.asarray(road_hits, dtype=np.bool_).reshape(-1),
        lookup=lookup,
        centre_heights=centres,
        bounds=bounds,
        fits=fits,
    )


def _regular_flat(context: _PrimaryForestContext, x: float, z: float) -> int:
    column = int(math.floor(float(x) / context.spacing))
    row = int(math.floor(float(z) / context.spacing))
    if not (0 <= row < context.rows and 0 <= column < context.columns):
        return -1
    expected_x = min(context.world_size - 0.001, (column + 0.5) * context.spacing)
    expected_z = min(context.world_size - 0.001, (row + 0.5) * context.spacing)
    if abs(float(x) - expected_x) > 1.0e-6 or abs(float(z) - expected_z) > 1.0e-6:
        return -1
    return row * context.columns + column


def _cached_sample(elevations, cells, cell_size, x, z):
    context = _CONTEXT.get()
    if context is not None and context.active and elevations is context.elevations:
        flat = _regular_flat(context, x, z)
        if flat >= 0:
            index = int(context.lookup[flat])
            if index >= 0:
                return float(context.centre_heights[index])
    return _BASE_SAMPLE(elevations, cells, cell_size, x, z)


def _cached_square(elevations, cells, cell_size, x, z, size):
    context = _CONTEXT.get()
    if (
        context is not None
        and context.active
        and elevations is context.elevations
        and abs(float(size) - context.spacing) <= 1.0e-6
    ):
        flat = _regular_flat(context, x, z)
        if flat >= 0:
            index = int(context.lookup[flat])
            if index >= 0:
                fit = _fit_from_row(context.fits[index]) if context.fits is not None else None
                return _PrimarySupports(context.bounds[index, 0], context.bounds[index, 1], fit)
    return _BASE_SQUARE(elevations, cells, cell_size, x, z, size)


def _cached_fit(supports, *, clearance, maximum_burial, maximum_float):
    if isinstance(supports, _PrimarySupports):
        return supports.fit
    return _BASE_TERRAIN_FIT(
        supports,
        clearance=clearance,
        maximum_burial=maximum_burial,
        maximum_float=maximum_float,
    )


def _cached_road_test(corridors, x: float, z: float, *, block_size: float) -> bool:
    context = _CONTEXT.get()
    if (
        context is not None
        and context.active
        and abs(float(block_size) - context.spacing) <= 1.0e-6
    ):
        flat = _regular_flat(context, x, z)
        if flat >= 0:
            return bool(context.road_hits[flat])
    return _BASE_ROAD_TEST(corridors, x, z, block_size=block_size)


def _parallel_generate(*args, **kwargs):
    if len(args) >= 5:
        dataset, projection, raster, elevations, spec = args[:5]
    else:
        dataset = kwargs.get("dataset")
        projection = kwargs.get("projection")
        raster = kwargs.get("raster")
        elevations = kwargs.get("elevations")
        spec = kwargs.get("spec")
    if any(value is None for value in (dataset, projection, raster, elevations, spec)):
        return _BASE_GENERATE(*args, **kwargs)

    original_progress = kwargs.get("progress_callback")
    try:
        context = _build_context(
            dataset, projection, raster, elevations, spec, original_progress
        )
    except (OSError, RuntimeError, BrokenPipeError, EOFError, TypeError, ValueError):
        return _BASE_GENERATE(*args, **kwargs)
    if context is None:
        return _BASE_GENERATE(*args, **kwargs)

    def progress(percent: int, stage: str) -> None:
        context.active = str(stage).startswith("Placing primary forest blocks")
        if original_progress is not None:
            original_progress(percent, stage)

    forwarded = dict(kwargs)
    forwarded["progress_callback"] = progress
    token = _CONTEXT.set(context)
    try:
        return _BASE_GENERATE(*args, **forwarded)
    finally:
        _CONTEXT.reset(token)


def install_forest_primary_parallel_policy() -> None:
    """Install multicore regular-block evaluation ahead of the serial emitter."""
    global _INSTALLED, _BASE_GENERATE, _BASE_SAMPLE, _BASE_SQUARE
    global _BASE_TERRAIN_FIT, _BASE_ROAD_TEST
    if _INSTALLED:
        return

    _BASE_GENERATE = _generator.generate_world_objects
    _BASE_SAMPLE = _osm._sample_elevation
    _BASE_SQUARE = _osm._square_elevation_samples
    _BASE_TERRAIN_FIT = _osm._terrain_fit_anchor
    _BASE_ROAD_TEST = _osm.forest_block_intersects_road_corridors

    _osm._sample_elevation = _cached_sample
    _osm._square_elevation_samples = _cached_square
    _osm._terrain_fit_anchor = _cached_fit
    _osm.forest_block_intersects_road_corridors = _cached_road_test

    # The old object-stage prepass stored up to 300k complete support tuples in
    # a Python dict. Compact plans above cover every regular terrain candidate,
    # so doing both would duplicate work and resurrect the memory ceiling.
    _object_parallel._MAX_PRIMARY_JOBS = 0

    _osm.generate_world_objects = _parallel_generate
    _generator.generate_world_objects = _parallel_generate
    _INSTALLED = True
