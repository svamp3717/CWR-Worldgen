# SPDX-License-Identifier: GPL-3.0-or-later
"""Parallel pre-evaluation for Everon primary-forest placement fallbacks."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Iterable, Sequence

import numpy as np

from . import forest_primary_parallel_policy as _primary
from . import forest_vector_performance_policy as _forest
from . import generator as _generator
from . import object_stage_parallel_policy as _object_parallel
from . import osm as _osm

_INSTALLED = False
_BASE_GENERATE: Any = None
_BASE_PRIMARY_BUILD_CONTEXT: Any = None
_BASE_SAMPLE: Any = None
_BASE_SQUARE: Any = None
_BASE_TRIANGLE: Any = None
_BASE_ROAD_TEST: Any = None
_BASE_GRADIENT: Any = None
_BASE_ORIENTED: Any = None

_W_ELEVATIONS: Sequence[float] | None = None
_W_RASTER: Any = None
_W_CORRIDORS: Any = None
_W_SPEC: Any = None
_W_SEED = "cwr-worldgen"
_W_SPACING = 1.0
_W_COLUMNS = 1
_W_FLATS: np.ndarray | None = None
_W_ROAD_HITS: np.ndarray | None = None
_W_GEO_COLUMNS: np.ndarray | None = None
_W_GEO_ROWS: np.ndarray | None = None


def _point_key(x: float, z: float) -> tuple[float, float]:
    return float(x), float(z)


def _square_key(x: float, z: float, size: float) -> tuple[float, float, float]:
    return float(x), float(z), float(size)


def _oriented_key(x, z, width, length, heading):
    return float(x), float(z), float(width), float(length), float(heading)


@dataclass(slots=True)
class _FallbackContext:
    elevations: Sequence[float]
    samples: dict[tuple[float, float], float] = field(default_factory=dict)
    squares: dict[tuple[float, float, float], tuple[float, ...]] = field(default_factory=dict)
    triangles: dict[tuple[float, float], Any] = field(default_factory=dict)
    road_hits: dict[tuple[float, float, float], bool] = field(default_factory=dict)
    gradients: dict[tuple[float, float], tuple[float, float]] = field(default_factory=dict)
    oriented: dict[tuple[float, float, float, float, float], tuple[float, ...]] = field(default_factory=dict)
    active: bool = False


_CONTEXT: ContextVar[_FallbackContext | None] = ContextVar(
    "cwr_parallel_primary_forest_fallback_context", default=None
)


def _raw_sample(x: float, z: float) -> float:
    fn = _object_parallel._BASE_OSM_SAMPLE or _primary._BASE_SAMPLE or _BASE_SAMPLE
    return float(fn(_W_ELEVATIONS, _W_SPEC.cells, _W_SPEC.cell_size, x, z))


def _raw_square(x: float, z: float, size: float) -> tuple[float, ...]:
    fn = _object_parallel._BASE_SQUARE_SAMPLES or _primary._BASE_SQUARE or _BASE_SQUARE
    return tuple(fn(_W_ELEVATIONS, _W_SPEC.cells, _W_SPEC.cell_size, x, z, size))


def _raw_triangle(x: float, z: float):
    return _BASE_TRIANGLE(_W_ELEVATIONS, _W_SPEC.cells, _W_SPEC.cell_size, x, z)


def _raw_road(x: float, z: float, size: float) -> bool:
    fn = _primary._BASE_ROAD_TEST or _BASE_ROAD_TEST
    return bool(fn(_W_CORRIDORS, x, z, block_size=size))


def _raw_gradient(x: float, z: float) -> tuple[float, float]:
    gx, gz = _BASE_GRADIENT(_W_ELEVATIONS, _W_SPEC.cells, _W_SPEC.cell_size, x, z)
    return float(gx), float(gz)


def _raw_oriented(x, z, width, length, heading) -> tuple[float, ...]:
    return tuple(
        _BASE_ORIENTED(
            _W_ELEVATIONS, _W_SPEC.cells, _W_SPEC.cell_size,
            x, z, width, length, heading,
        )
    )


def _mask_ok(x: float, z: float) -> bool:
    if not (0.0 <= x < _W_SPEC.world_size and 0.0 <= z < _W_SPEC.world_size):
        return False
    cells = int(_W_SPEC.cells)
    scale = cells / float(_W_SPEC.world_size)
    col = min(cells - 1, int(x * scale))
    row = min(cells - 1, int(z * scale))
    index = row * cells + col
    return bool(
        _W_RASTER.forest[index]
        and not _W_RASTER.water[index]
        and not _W_RASTER.roads[index]
        and not _W_RASTER.buildings[index]
    )


def _candidate_terrain_entries(x, z, footprint, maximum_relief):
    entries = []
    road = _raw_road(x, z, footprint)
    entries.append(("road", _square_key(x, z, footprint), road))
    if road:
        return entries
    supports = _raw_square(x, z, footprint)
    entries.append(("square", _square_key(x, z, footprint), supports))
    if max(supports) - min(supports) <= maximum_relief:
        entries.append(("triangle", _point_key(x, z), _raw_triangle(x, z)))
    return entries


def _road_cut_entries(geo_col: int, geo_row: int, x: float, z: float):
    entries = []
    tree_footprint = max(1.5, float(getattr(_W_SPEC, "forest_single_tree_footprint", 2.0)))
    tree_relief = max(1.5, float(getattr(_W_SPEC, "forest_single_tree_maximum_relief", 8.0)))
    for tree_x, tree_z, _heading, _variant in _osm._roadside_tree_candidates(
        _W_SEED, geo_col, geo_row, x, z, _W_SPACING
    ):
        if _mask_ok(tree_x, tree_z):
            entries.extend(
                _candidate_terrain_entries(tree_x, tree_z, tree_footprint, tree_relief)
            )

    bushes = tuple(
        getattr(_W_SPEC, "forest_roadside_bush_models", _osm.ROADSIDE_BUSH_MODELS)
    )
    if bushes:
        bush_footprint = max(0.5, float(getattr(_W_SPEC, "forest_roadside_bush_footprint", 1.5)))
        bush_relief = max(1.0, float(getattr(_W_SPEC, "steep_hill_bush_maximum_relief", 8.0)))
        for bush_x, bush_z, _heading, _variant in _osm._roadside_bush_candidates(
            _W_SEED, geo_col, geo_row, x, z, _W_SPACING
        ):
            if _mask_ok(bush_x, bush_z):
                entries.extend(
                    _candidate_terrain_entries(bush_x, bush_z, bush_footprint, bush_relief)
                )
    return entries


def _regular_entries(geo_col: int, geo_row: int, x: float, z: float):
    entries = []
    centre = _raw_sample(x, z)
    supports = _raw_square(x, z, _W_SPACING)
    entries.extend((
        ("sample", _point_key(x, z), centre),
        ("square", _square_key(x, z, _W_SPACING), supports),
    ))
    if not bool(getattr(_W_SPEC, "forest_low_anchor", False)):
        return entries

    relief = max(supports) - min(supports)
    regular_max_relief = max(0.0, float(getattr(_W_SPEC, "forest_maximum_block_relief", 8.0)))
    fit_fn = _primary._BASE_TERRAIN_FIT or _osm._terrain_fit_anchor
    regular_fit = (
        fit_fn(
            supports,
            clearance=float(_W_SPEC.forest_ground_clearance),
            maximum_burial=max(0.0, float(getattr(_W_SPEC, "forest_block_maximum_burial", 8.0))),
            maximum_float=max(0.0, float(getattr(_W_SPEC, "forest_block_maximum_float", 0.5))),
        )
        if relief <= regular_max_relief else None
    )
    if regular_fit is not None:
        return entries

    digest = hashlib.blake2s(
        f"{_W_SEED}:forest:{geo_col}:{geo_row}".encode("utf-8"), digest_size=2
    ).digest()
    heading = float((int.from_bytes(digest, "little") % 4) * 90)
    gradient = _raw_gradient(x, z)
    entries.append(("gradient", _point_key(x, z), gradient))
    gx, gz = gradient
    if abs(gx) + abs(gz) > 1.0e-9:
        heading = math.degrees(math.atan2(-gz, gx)) % 360.0

    footprint = max(8.0, float(getattr(_W_SPEC, "forest_everon_steep_footprint", 35.0)))
    width = footprint * 0.58
    blocked = _raw_road(x, z, footprint)
    steep_supports = _raw_oriented(x, z, width, footprint, heading)
    entries.extend((
        ("road", _square_key(x, z, footprint), blocked),
        ("oriented", _oriented_key(x, z, width, footprint, heading), steep_supports),
    ))

    steep_relief = max(steep_supports) - min(steep_supports)
    steep_fit = (
        fit_fn(
            steep_supports,
            clearance=float(_W_SPEC.forest_ground_clearance),
            maximum_burial=max(0.0, float(getattr(_W_SPEC, "forest_everon_steep_maximum_burial", 18.0))),
            maximum_float=max(0.0, float(getattr(_W_SPEC, "forest_everon_steep_maximum_float", 0.5))),
        )
        if (
            not blocked
            and steep_relief <= max(0.0, float(getattr(_W_SPEC, "forest_everon_steep_maximum_relief", 18.0)))
        ) else None
    )
    if steep_fit is not None:
        return entries

    tree_footprint = max(1.5, float(getattr(_W_SPEC, "forest_single_tree_footprint", 2.0)))
    tree_relief = max(1.5, float(getattr(_W_SPEC, "forest_single_tree_maximum_relief", 8.0)))
    for tree_x, tree_z, _heading in _osm._dense_hillside_tree_candidates(
        f"{_W_SEED}:steep-infill", geo_col, geo_row, x, z, _W_SPACING
    ):
        if _mask_ok(tree_x, tree_z):
            entries.extend(
                _candidate_terrain_entries(tree_x, tree_z, tree_footprint, tree_relief)
            )
    return entries


def _init_worker(
    elevations, raster, corridors, spec, seed, spacing, columns,
    flats, road_hits, geo_columns, geo_rows,
) -> None:
    global _W_ELEVATIONS, _W_RASTER, _W_CORRIDORS, _W_SPEC, _W_SEED
    global _W_SPACING, _W_COLUMNS, _W_FLATS, _W_ROAD_HITS
    global _W_GEO_COLUMNS, _W_GEO_ROWS
    _W_ELEVATIONS = elevations
    _W_RASTER = raster
    _W_CORRIDORS = corridors
    _W_SPEC = spec
    _W_SEED = str(seed)
    _W_SPACING = float(spacing)
    _W_COLUMNS = int(columns)
    _W_FLATS = flats
    _W_ROAD_HITS = road_hits
    _W_GEO_COLUMNS = geo_columns
    _W_GEO_ROWS = geo_rows


def _evaluate_batch(job: tuple[int, int]):
    offset, count = job
    output = []
    world_size = float(_W_SPEC.world_size)
    for flat_value in _W_FLATS[offset: offset + count]:
        flat = int(flat_value)
        row, column = divmod(flat, _W_COLUMNS)
        x = min(world_size - 0.001, (column + 0.5) * _W_SPACING)
        z = min(world_size - 0.001, (row + 0.5) * _W_SPACING)
        geo_col = int(_W_GEO_COLUMNS[flat])
        geo_row = int(_W_GEO_ROWS[flat])
        road_hit = bool(_W_ROAD_HITS[flat])
        output.append(("road", _square_key(x, z, _W_SPACING), road_hit))
        if road_hit:
            output.extend(_road_cut_entries(geo_col, geo_row, x, z))
        else:
            output.extend(_regular_entries(geo_col, geo_row, x, z))
    return tuple(output)


def _forest_probe_counts(raster, spec, primary) -> np.ndarray:
    rows, columns = primary.possible_primary.shape
    flats = np.arange(int(rows * columns), dtype=np.int64)
    rr = flats // int(columns)
    cc = flats % int(columns)
    spacing = float(primary.spacing)
    world_size = float(spec.world_size)
    xs = np.minimum(world_size - 0.001, (cc.astype(np.float64) + 0.5) * spacing)
    zs = np.minimum(world_size - 0.001, (rr.astype(np.float64) + 0.5) * spacing)
    clearance = min(spacing * 0.35, max(1.0, float(spec.cell_size) * 0.45))
    forest = np.asarray(raster.forest, dtype=np.bool_)
    counts = np.zeros(flats.size, dtype=np.uint8)
    scale = int(spec.cells) / world_size
    for dx, dz in ((0.0, 0.0), (-clearance, -clearance), (clearance, -clearance),
                   (-clearance, clearance), (clearance, clearance)):
        sx, sz = xs + dx, zs + dz
        valid = (sx >= 0.0) & (sx < world_size) & (sz >= 0.0) & (sz < world_size)
        cx = np.clip((sx * scale).astype(np.int64), 0, int(spec.cells) - 1)
        cz = np.clip((sz * scale).astype(np.int64), 0, int(spec.cells) - 1)
        counts += (valid & forest[cz * int(spec.cells) + cx]).astype(np.uint8)
    return counts


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
    road_hits = np.asarray(_forest._batch_primary_road_hits(corridors, primary), dtype=np.bool_).reshape(-1)
    regular = _primary._terrain_candidates(raster, spec, primary, road_hits)
    forest_counts = _forest_probe_counts(raster, spec, primary)
    edge_guard = bool(getattr(spec, "forest_low_anchor", False)) and float(spec.world_size) >= 200.0
    rows, columns = primary.possible_primary.shape
    road_cut = np.flatnonzero(road_hits & (forest_counts >= 2)).astype(np.int64, copy=False)
    if edge_guard and road_cut.size:
        spacing = float(primary.spacing)
        margin = max(max(24.0, min(40.0, float(spec.cell_size) * 1.25)), spacing * 0.58)
        rr, cc = road_cut // int(columns), road_cut % int(columns)
        xs, zs = (cc + 0.5) * spacing, (rr + 0.5) * spacing
        road_cut = road_cut[(xs >= margin) & (xs <= spec.world_size - margin) &
                            (zs >= margin) & (zs <= spec.world_size - margin)]
    flats = np.unique(np.concatenate((regular, road_cut))).astype(np.int64, copy=False)
    workers = _object_parallel._worker_count(int(flats.size))
    if workers <= 1 or flats.size == 0:
        return None

    geo_columns = np.zeros(int(rows * columns), dtype=np.int64)
    geo_rows = np.zeros(int(rows * columns), dtype=np.int64)
    spacing = float(primary.spacing)
    world_size = float(spec.world_size)
    for flat_value in flats:
        flat = int(flat_value)
        row, column = divmod(flat, int(columns))
        x = min(world_size - 0.001, (column + 0.5) * spacing)
        z = min(world_size - 0.001, (row + 0.5) * spacing)
        geo_columns[flat], geo_rows[flat] = _osm._geographic_lattice_identity(
            projection, x, z, spacing
        )

    if progress is not None:
        progress(57, f"Parallel planning {int(flats.size):,} primary forest cells with {workers} workers")
    context = _FallbackContext(elevations=elevations)
    batches = _batches(int(flats.size), workers)
    stride = max(1, len(batches) // 20)
    completed = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(
            elevations, raster, corridors, spec,
            str(getattr(spec, "deterministic_seed", "cwr-worldgen")), spacing,
            int(columns), flats, road_hits, geo_columns, geo_rows,
        ),
    ) as executor:
        for batch_index, results in enumerate(executor.map(_evaluate_batch, batches, chunksize=1)):
            _store(context, results)
            completed += batches[batch_index][1]
            if progress is not None and ((batch_index + 1) % stride == 0 or batch_index + 1 == len(batches)):
                progress(57, f"Parallel planning primary forest cells {completed:,}/{int(flats.size):,} with {workers} workers")
    return context


def _batches(total: int, workers: int):
    size = max(8, min(96, math.ceil(total / max(1, workers * 24))))
    return tuple((offset, min(size, total - offset)) for offset in range(0, total, size))


def _store(context: _FallbackContext, results: Iterable[tuple[str, Any, Any]]) -> None:
    for kind, key, value in results:
        target = getattr(context, {
            "sample": "samples", "square": "squares", "triangle": "triangles",
            "road": "road_hits", "gradient": "gradients", "oriented": "oriented",
        }[kind])
        target[key] = tuple(value) if kind in {"square", "oriented", "gradient"} else value


def _cached_sample(elevations, cells, cell_size, x, z):
    context = _CONTEXT.get()
    if context is not None and context.active and elevations is context.elevations:
        value = context.samples.get(_point_key(x, z))
        if value is not None:
            return value
    return _BASE_SAMPLE(elevations, cells, cell_size, x, z)


def _cached_square(elevations, cells, cell_size, x, z, size):
    context = _CONTEXT.get()
    if context is not None and context.active and elevations is context.elevations:
        value = context.squares.get(_square_key(x, z, size))
        if value is not None:
            return value
    return _BASE_SQUARE(elevations, cells, cell_size, x, z, size)


def _cached_triangle(elevations, cells, cell_size, x, z):
    context = _CONTEXT.get()
    key = _point_key(x, z)
    if context is not None and context.active and elevations is context.elevations and key in context.triangles:
        return context.triangles[key]
    return _BASE_TRIANGLE(elevations, cells, cell_size, x, z)


def _cached_road(corridors, x, z, *, block_size):
    context = _CONTEXT.get()
    key = _square_key(x, z, block_size)
    if context is not None and context.active and key in context.road_hits:
        return context.road_hits[key]
    return _BASE_ROAD_TEST(corridors, x, z, block_size=block_size)


def _cached_gradient(elevations, cells, cell_size, x, z):
    context = _CONTEXT.get()
    value = context.gradients.get(_point_key(x, z)) if context is not None and context.active and elevations is context.elevations else None
    return value if value is not None else _BASE_GRADIENT(elevations, cells, cell_size, x, z)


def _cached_oriented(elevations, cells, cell_size, x, z, width, length, heading):
    context = _CONTEXT.get()
    key = _oriented_key(x, z, width, length, heading)
    value = context.oriented.get(key) if context is not None and context.active and elevations is context.elevations else None
    return value if value is not None else _BASE_ORIENTED(elevations, cells, cell_size, x, z, width, length, heading)


def _primary_build_context_bridge(*args, **kwargs):
    return None if _CONTEXT.get() is not None else _BASE_PRIMARY_BUILD_CONTEXT(*args, **kwargs)


def _parallel_generate(*args, **kwargs):
    if len(args) >= 5:
        dataset, projection, raster, elevations, spec = args[:5]
    else:
        dataset, projection = kwargs.get("dataset"), kwargs.get("projection")
        raster, elevations, spec = kwargs.get("raster"), kwargs.get("elevations"), kwargs.get("spec")
    if any(value is None for value in (dataset, projection, raster, elevations, spec)):
        return _BASE_GENERATE(*args, **kwargs)
    original_progress = kwargs.get("progress_callback")
    try:
        context = _build_context(dataset, projection, raster, elevations, spec, original_progress)
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


def install_forest_primary_fallback_parallel_policy() -> None:
    global _INSTALLED, _BASE_GENERATE, _BASE_PRIMARY_BUILD_CONTEXT
    global _BASE_SAMPLE, _BASE_SQUARE, _BASE_TRIANGLE, _BASE_ROAD_TEST
    global _BASE_GRADIENT, _BASE_ORIENTED
    if _INSTALLED:
        return
    _BASE_GENERATE = _generator.generate_world_objects
    _BASE_PRIMARY_BUILD_CONTEXT = _primary._build_context
    _BASE_SAMPLE = _osm._sample_elevation
    _BASE_SQUARE = _osm._square_elevation_samples
    _BASE_TRIANGLE = _osm._triangle_elevation_bounds
    _BASE_ROAD_TEST = _osm.forest_block_intersects_road_corridors
    _BASE_GRADIENT = _osm._local_terrain_gradient
    _BASE_ORIENTED = _osm._oriented_footprint_elevation_samples
    _osm._sample_elevation = _cached_sample
    _osm._square_elevation_samples = _cached_square
    _osm._triangle_elevation_bounds = _cached_triangle
    _osm.forest_block_intersects_road_corridors = _cached_road
    _osm._local_terrain_gradient = _cached_gradient
    _osm._oriented_footprint_elevation_samples = _cached_oriented
    _primary._build_context = _primary_build_context_bridge
    _osm.generate_world_objects = _parallel_generate
    _generator.generate_world_objects = _parallel_generate
    _INSTALLED = True
