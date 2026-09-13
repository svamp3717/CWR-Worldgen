# SPDX-License-Identifier: GPL-3.0-or-later
"""Parallel precomputation for the remaining large object-placement stages.

The stock-road chain planner already benefits from a process pool.  The later
object stages still repeat independent CPU-heavy terrain/geometry queries in one
Python process.  This policy keeps final mutation, object budgets and object-id
assignment in the parent process, but precomputes the pure work for:

* final stock-road endpoint terrain sampling;
* building footprint grounding/submergence checks;
* primary forest terrain-support samples;
* forest undergrowth cluster grounding;
* barrier line segmentation; and
* rural vegetation candidate grids, tree-row chunks and cluster grounding.

Results are keyed by the exact arguments used by the historical helpers.  The
original serial generation code therefore remains authoritative: it simply hits
precomputed values when available and falls back to the previous helper for any
candidate not covered by the bounded prepass.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import math
import multiprocessing
import os
from typing import Any, Iterable, Sequence

from . import forest_vector_performance_policy as _forest
from . import generator as _generator
from . import osm as _osm
from . import playability as _playability
from . import road_chain_parallel_policy as _road_parallel

_INSTALLED = False

_BASE_GENERATE: Any = None
_BASE_OSM_SAMPLE: Any = None
_BASE_PLAYABILITY_SAMPLE: Any = None
_BASE_POLYGON_EXTREMA: Any = None
_BASE_SQUARE_SAMPLES: Any = None
_BASE_BUILDING_SUBMERGED: Any = None
_BASE_LINE_CHUNKS: Any = None
_BASE_POLYGON_GRID: Any = None
_BASE_PLACE_CLUSTER: Any = None
_BASE_STOCK_FIT: Any = None
_BASE_EXECUTE_RUN_JOBS: Any = None

# Worker state.  Large immutable inputs are transferred once per worker through
# the executor initializer rather than once per candidate.
_W_ELEVATIONS: Sequence[float] | None = None
_W_RASTER: Any = None
_W_CORRIDORS: Any = None
_W_SPEC: Any = None
_W_CELLS = 0
_W_CELL_SIZE = 1.0

_MAX_PRIMARY_JOBS = 300_000
_MAX_UNDERGROWTH_JOBS = 250_000
_MAX_RURAL_CLUSTER_JOBS = 160_000


@dataclass(slots=True)
class _ObjectContext:
    elevations: Sequence[float]
    cells: int
    cell_size: float
    samples: dict[tuple[float, float], float] = field(default_factory=dict)
    polygon_extrema: dict[tuple[tuple[float, float], ...], tuple[float, float]] = field(default_factory=dict)
    square_samples: dict[tuple[float, float, float], tuple[float, ...]] = field(default_factory=dict)
    building_submerged: dict[tuple[Any, ...], bool] = field(default_factory=dict)
    line_chunks: dict[tuple[Any, ...], tuple[Any, ...]] = field(default_factory=dict)
    polygon_grids: dict[tuple[Any, ...], tuple[Any, ...]] = field(default_factory=dict)
    clusters: dict[tuple[Any, ...], Any] = field(default_factory=dict)


@dataclass(slots=True)
class _RoadContext:
    elevations: Sequence[float]
    cells: int
    cell_size: float
    samples: dict[tuple[float, float], float] = field(default_factory=dict)
    progress_callback: Any = None


_OBJECT_CONTEXT: ContextVar[_ObjectContext | None] = ContextVar(
    "cwr_parallel_object_context", default=None
)
_ROAD_CONTEXT: ContextVar[_RoadContext | None] = ContextVar(
    "cwr_parallel_road_finish_context", default=None
)


def _worker_count(job_count: int) -> int:
    if job_count < 128 or multiprocessing.current_process().daemon:
        return 1
    raw = os.environ.get("CWR_WORLDGEN_OBJECT_WORKERS", "").strip()
    if not raw:
        raw = os.environ.get("CWR_WORLDGEN_ROAD_WORKERS", "").strip()
    if raw:
        try:
            requested = max(1, int(raw))
        except ValueError:
            requested = 1
    else:
        cpu = os.cpu_count() or 1
        requested = max(1, min(8, cpu - 1 if cpu > 2 else cpu))
    return min(requested, max(1, job_count // 48))


def _init_worker(elevations, raster, corridors, spec) -> None:
    global _W_ELEVATIONS, _W_RASTER, _W_CORRIDORS, _W_SPEC, _W_CELLS, _W_CELL_SIZE
    _W_ELEVATIONS = elevations
    _W_RASTER = raster
    _W_CORRIDORS = corridors
    _W_SPEC = spec
    _W_CELLS = int(getattr(spec, "cells", 0) or 0) if spec is not None else 0
    _W_CELL_SIZE = float(getattr(spec, "cell_size", 1.0) or 1.0) if spec is not None else 1.0


def _sample_exact(elevations: Sequence[float], cells: int, cell_size: float, x: float, z: float) -> float:
    fx = max(0.0, min(cells - 1.0, float(x) / cell_size))
    fz = max(0.0, min(cells - 1.0, float(z) / cell_size))
    x0 = int(math.floor(fx))
    z0 = int(math.floor(fz))
    x1 = min(cells - 1, x0 + 1)
    z1 = min(cells - 1, z0 + 1)
    tx = fx - x0
    tz = fz - z0
    a = elevations[z0 * cells + x0] * (1.0 - tx) + elevations[z0 * cells + x1] * tx
    b = elevations[z1 * cells + x0] * (1.0 - tx) + elevations[z1 * cells + x1] * tx
    return float(a * (1.0 - tz) + b * tz)


def _building_key(plan: Any) -> tuple[Any, ...]:
    return (
        str(getattr(plan, "osm_key", "")),
        int(getattr(plan, "geometry_index", 0) or 0),
        tuple((float(x), float(z)) for x, z in getattr(plan, "support_polygon", ())),
    )


def _line_key(points, target_length: float, endpoint_trim: float) -> tuple[Any, ...]:
    return (
        tuple((float(x), float(z)) for x, z in points),
        float(target_length),
        float(endpoint_trim),
    )


def _grid_key(
    points,
    spacing: float,
    seed: str,
    jitter_fraction: float,
    heading_jitter_degrees: float,
) -> tuple[Any, ...]:
    return (
        tuple((float(x), float(z)) for x, z in points),
        float(spacing),
        str(seed),
        float(jitter_fraction),
        float(heading_jitter_degrees),
    )


def _cluster_key(
    variant: Any,
    x: float,
    z: float,
    heading: float,
    require_forest: bool,
    minimum_forest_fraction: float,
    maximum_relief: float,
    maximum_burial: float,
    maximum_float: float,
    clearance: float,
    avoid_roads: bool,
) -> tuple[Any, ...]:
    return (
        str(getattr(variant, "name", "")),
        float(x), float(z), float(heading), bool(require_forest),
        float(minimum_forest_fraction), float(maximum_relief),
        float(maximum_burial), float(maximum_float), float(clearance),
        bool(avoid_roads),
    )


def _worker_batch(batch: tuple[tuple[str, Any, Any], ...]):
    results = []
    elevations = _W_ELEVATIONS
    for kind, key, payload in batch:
        if kind == "sample":
            x, z = payload
            value = _sample_exact(elevations, _W_CELLS, _W_CELL_SIZE, x, z)
        elif kind == "polygon":
            points = payload
            fn = _BASE_POLYGON_EXTREMA or _osm._polygon_elevation_extrema
            value = fn(elevations, _W_CELLS, _W_CELL_SIZE, points)
        elif kind == "square":
            x, z, size = payload
            fn = _BASE_SQUARE_SAMPLES or _osm._square_elevation_samples
            value = fn(elevations, _W_CELLS, _W_CELL_SIZE, x, z, size)
        elif kind == "submerged":
            fn = _BASE_BUILDING_SUBMERGED or _osm._building_plan_fully_submerged
            value = bool(fn(payload, elevations, _W_RASTER, _W_SPEC))
        elif kind == "line":
            points, target, trim = payload
            fn = _BASE_LINE_CHUNKS or _osm._line_chunks
            value = fn(points, target, endpoint_trim=trim)
        elif kind == "grid":
            points, spacing, seed, jitter, heading_jitter = payload
            fn = _BASE_POLYGON_GRID or _osm._polygon_grid_candidates
            value = fn(
                points,
                spacing,
                seed,
                jitter_fraction=jitter,
                heading_jitter_degrees=heading_jitter,
            )
        elif kind == "cluster":
            (
                variant, x, z, heading, require_forest, minimum_forest_fraction,
                maximum_relief, maximum_burial, maximum_float, clearance,
                avoid_roads,
            ) = payload
            fn = _BASE_PLACE_CLUSTER or _osm._place_cluster_at
            value = fn(
                variant=variant,
                elevations=elevations,
                raster=_W_RASTER,
                road_corridors=_W_CORRIDORS,
                spec=_W_SPEC,
                x=x,
                z=z,
                heading=heading,
                require_forest=require_forest,
                minimum_forest_fraction=minimum_forest_fraction,
                maximum_relief=maximum_relief,
                maximum_burial=maximum_burial,
                maximum_float=maximum_float,
                clearance=clearance,
                avoid_roads=avoid_roads,
            )
        else:
            raise ValueError(f"unknown object-stage worker job {kind!r}")
        results.append((kind, key, value))
    return tuple(results)


def _batches(items: Sequence[Any], workers: int) -> tuple[tuple[Any, ...], ...]:
    if not items:
        return ()
    size = max(16, min(256, math.ceil(len(items) / max(1, workers * 10))))
    return tuple(
        tuple(items[offset: offset + size])
        for offset in range(0, len(items), size)
    )


def _store_results(context: _ObjectContext, results: Iterable[tuple[str, Any, Any]]) -> None:
    for kind, key, value in results:
        if kind == "sample":
            context.samples[key] = float(value)
        elif kind == "polygon":
            context.polygon_extrema[key] = tuple(value)
        elif kind == "square":
            context.square_samples[key] = tuple(value)
        elif kind == "submerged":
            context.building_submerged[key] = bool(value)
        elif kind == "line":
            context.line_chunks[key] = tuple(value)
        elif kind == "grid":
            context.polygon_grids[key] = tuple(value)
        elif kind == "cluster":
            context.clusters[key] = value


def _execute_jobs(
    executor: ProcessPoolExecutor,
    context: _ObjectContext,
    jobs: Sequence[tuple[str, Any, Any]],
    workers: int,
) -> None:
    if not jobs:
        return
    for results in executor.map(_worker_batch, _batches(jobs, workers), chunksize=1):
        _store_results(context, results)


def _rural_category(feature: Any) -> str:
    tags = getattr(feature, "tags", {}) or {}
    natural = str(tags.get("natural", "")).casefold()
    landuse = str(tags.get("landuse", "")).casefold()
    rural_kind = str(tags.get("rural_kind", "")).casefold()
    if not natural and rural_kind in {"scrub", "bare_rock", "rock", "scree", "wetland"}:
        natural = rural_kind
    if not landuse and rural_kind in {"orchard", "vineyard"}:
        landuse = rural_kind
    if natural == "wetland":
        return "wetland"
    if natural in {"bare_rock", "rock", "scree"}:
        return "rock"
    if natural == "scrub":
        return "scrub"
    if landuse == "orchard":
        return "orchard"
    if landuse == "vineyard":
        return "vineyard"
    return ""


def _prepare_object_context(dataset, projection, raster, elevations, spec, building_plans) -> _ObjectContext:
    context = _ObjectContext(elevations, int(spec.cells), float(spec.cell_size))
    seed = str(getattr(spec, "deterministic_seed", "cwr-worldgen"))
    initial: list[tuple[str, Any, Any]] = []

    # Buildings: the same submerged predicate can be called by planning and final
    # emission, while footprint extrema and centre height are pure read-only work.
    for plan in tuple(building_plans or ()):
        polygon = tuple((float(x), float(z)) for x, z in getattr(plan, "support_polygon", ()))
        if len(polygon) >= 3:
            initial.append(("polygon", polygon, polygon))
        initial.append(("sample", (float(plan.x), float(plan.z)), (float(plan.x), float(plan.z))))
        initial.append(("submerged", _building_key(plan), plan))

    # Primary forest: precompute expensive terrain support only for cells that can
    # pass the already-vectorized Everon forest mask. Row-major truncation keeps
    # memory bounded on continent-sized worlds.
    try:
        primary = _forest._primary_forest_possible(raster, spec)
    except Exception:
        primary = None
    if primary is not None:
        selected = list(map(int, __import__("numpy").flatnonzero(primary.possible_primary.ravel())[:_MAX_PRIMARY_JOBS]))
        columns = primary.possible_primary.shape[1]
        spacing = float(primary.spacing)
        for flat in selected:
            row, column = divmod(flat, columns)
            x = min(float(spec.world_size) - 0.001, (column + 0.5) * spacing)
            z = min(float(spec.world_size) - 0.001, (row + 0.5) * spacing)
            key = (float(x), float(z), spacing)
            initial.append(("square", key, key))

    # Barrier segmentation is pure geometry and feature-local.
    if bool(getattr(spec, "barriers_enabled", False)):
        barrier_length = max(2.0, float(getattr(spec, "barrier_segment_length", 6.0)))
        wall_length = float(getattr(_osm, "STOCK_WALL_EFFECTIVE_LENGTH_METRES", 2.45))
        for feature in sorted(getattr(dataset, "barriers", ()), key=lambda item: item.osm_key):
            subtype = str(feature.tags.get("barrier", "fence")).casefold()
            subtype = "wall" if subtype in {"wall", "retaining_wall"} else "hedge" if subtype == "hedge" else "fence"
            target = wall_length if subtype == "wall" else barrier_length
            points = tuple(projection.to_world(point) for point in feature.points)
            key = _line_key(points, target, 0.0)
            initial.append(("line", key, (points, target, 0.0)))

    rural_spacing = max(10.0, float(getattr(spec, "rural_vegetation_spacing", 28.0)))
    # Tree-row chunks are later consumed by cluster grounding.
    for feature in sorted(getattr(dataset, "tree_rows", ()), key=lambda item: item.osm_key):
        points = tuple(projection.to_world(point) for point in feature.points)
        key = _line_key(points, rural_spacing, 2.0)
        initial.append(("line", key, (points, rural_spacing, 2.0)))

    # Rural polygon lattice generation is independent per polygon.
    wetland_spacing = max(8.0, float(getattr(spec, "wetland_reed_spacing", 20.0)))
    rural_grid_meta: list[tuple[Any, int, str, tuple[Any, ...]]] = []
    for feature in sorted(getattr(dataset, "rural_vegetation", ()), key=lambda item: item.osm_key):
        category = _rural_category(feature)
        if not category:
            continue
        for polygon_index, polygon in enumerate(feature.polygons):
            projected = tuple(projection.to_world(point) for point in polygon.outer[:-1])
            if len(projected) < 3:
                continue
            spacing = wetland_spacing if category == "wetland" else max(16.0, rural_spacing) if category == "scrub" else rural_spacing
            jitter = 0.70 if category == "scrub" else 0.0
            heading_jitter = 180.0 if category == "scrub" else 0.0
            grid_seed = f"{seed}:{feature.osm_key}:{polygon_index}"
            key = _grid_key(projected, spacing, grid_seed, jitter, heading_jitter)
            initial.append(("grid", key, (projected, spacing, grid_seed, jitter, heading_jitter)))
            rural_grid_meta.append((feature, polygon_index, category, key))

    # Forest undergrowth cluster fits are independent and substantially more
    # expensive than the hash/grid walk that produces their coordinates.
    undergrowth_jobs = 0
    if bool(getattr(spec, "forest_undergrowth_enabled", False)):
        spacing = max(10.0, float(getattr(spec, "forest_undergrowth_spacing", 30.0)))
        columns = max(1, int(math.ceil(float(spec.world_size) / spacing)))
        configured_limit = max(0, int(getattr(spec, "forest_undergrowth_maximum_objects", 120000)))
        job_limit = min(_MAX_UNDERGROWTH_JOBS, max(8192, configured_limit * 4 if configured_limit else 8192))
        variants = tuple(getattr(_osm, "FOREST_UNDERGROWTH_VARIANTS", ()))
        if variants:
            maximum_relief = max(0.0, float(getattr(spec, "forest_undergrowth_maximum_relief", 20.0)))
            maximum_burial = max(0.0, float(getattr(spec, "forest_undergrowth_maximum_burial", 0.8)))
            maximum_float = max(0.0, float(getattr(spec, "forest_undergrowth_maximum_float", 0.8)))
            clearance = float(getattr(spec, "forest_undergrowth_ground_clearance", 0.03))
            for grid_index in _osm._distributed_grid_indices(columns, seed, "forest-undergrowth"):
                row, column = divmod(grid_index, columns)
                digest = hashlib.blake2s(
                    f"{seed}:forest-undergrowth:{column}:{row}".encode("utf-8"), digest_size=8
                ).digest()
                jitter_x = (int.from_bytes(digest[:2], "little") / 65535.0 - 0.5) * spacing * 0.42
                jitter_z = (int.from_bytes(digest[2:4], "little") / 65535.0 - 0.5) * spacing * 0.42
                x = min(float(spec.world_size) - 0.001, max(0.0, (column + 0.5) * spacing + jitter_x))
                z = min(float(spec.world_size) - 0.001, max(0.0, (row + 0.5) * spacing + jitter_z))
                if not _osm._mask_at(raster.forest, spec.cells, spec.world_size, x, z):
                    continue
                variant = variants[int.from_bytes(digest[4:6], "little") % len(variants)]
                heading = float(int.from_bytes(digest[6:], "little") % 360)
                key = _cluster_key(variant, x, z, heading, True, 0.80, maximum_relief, maximum_burial, maximum_float, clearance, True)
                payload = (variant, x, z, heading, True, 0.80, maximum_relief, maximum_burial, maximum_float, clearance, True)
                initial.append(("cluster", key, payload))
                undergrowth_jobs += 1
                if undergrowth_jobs >= job_limit:
                    break

    road_corridors = _osm.project_road_corridors(dataset, projection, spec)
    workers = _worker_count(len(initial))
    if workers <= 1:
        _init_worker(elevations, raster, road_corridors, spec)
        _store_results(context, _worker_batch(tuple(initial)))
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(elevations, raster, road_corridors, spec),
        )
        try:
            _execute_jobs(executor, context, initial, workers)
        except (OSError, RuntimeError, BrokenPipeError, EOFError):
            executor.shutdown(wait=False, cancel_futures=True)
            executor = None
            _init_worker(elevations, raster, road_corridors, spec)
            _store_results(context, _worker_batch(tuple(initial)))

    # Second phase: use the parallel candidate-grid/chunk results to ground rural
    # clusters in parallel as well. Rock patches use the same terrain-sample cache.
    second: list[tuple[str, Any, Any]] = []
    variants = {variant.name: variant for variant in getattr(_osm, "RURAL_VEGETATION_VARIANTS", ())}
    tree_variant = variants.get("tree_row")
    if tree_variant is not None:
        for feature in sorted(getattr(dataset, "tree_rows", ()), key=lambda item: item.osm_key):
            points = tuple(projection.to_world(point) for point in feature.points)
            chunks = context.line_chunks.get(_line_key(points, rural_spacing, 2.0), ())
            for x, z, heading, _length, _x0, _z0, _x1, _z1 in chunks:
                key = _cluster_key(tree_variant, x, z, heading, False, 0.0, 24.0, 1.0, 1.0, 0.03, False)
                second.append(("cluster", key, (tree_variant, x, z, heading, False, 0.0, 24.0, 1.0, 1.0, 0.03, False)))
                if len(second) >= _MAX_RURAL_CLUSTER_JOBS:
                    break
            if len(second) >= _MAX_RURAL_CLUSTER_JOBS:
                break

    if len(second) < _MAX_RURAL_CLUSTER_JOBS:
        ditch_variants = tuple(getattr(_osm, "DITCH_GRASS_VARIANTS", ()))
        for feature, _polygon_index, category, grid_key in rural_grid_meta:
            candidates = context.polygon_grids.get(grid_key, ())
            for candidate_index, (x, z, heading) in enumerate(candidates):
                if category == "rock":
                    key = (float(x), float(z), 9.0)
                    second.append(("square", key, key))
                elif category in {"scrub", "orchard", "vineyard"}:
                    if category == "scrub":
                        variant = ditch_variants[0] if ditch_variants and candidate_index % 4 == 0 else variants.get("scrub_patch")
                    else:
                        variant = variants.get("orchard_row" if category == "orchard" else "vineyard_row")
                    if variant is None:
                        continue
                    key = _cluster_key(variant, x, z, heading, False, 0.0, 28.0, 1.0, 1.0, 0.03, True)
                    second.append(("cluster", key, (variant, x, z, heading, False, 0.0, 28.0, 1.0, 1.0, 0.03, True)))
                if len(second) >= _MAX_RURAL_CLUSTER_JOBS:
                    break
            if len(second) >= _MAX_RURAL_CLUSTER_JOBS:
                break

    if second:
        if executor is None:
            _store_results(context, _worker_batch(tuple(second)))
        else:
            try:
                _execute_jobs(executor, context, second, workers)
            except (OSError, RuntimeError, BrokenPipeError, EOFError):
                _init_worker(elevations, raster, road_corridors, spec)
                _store_results(context, _worker_batch(tuple(second)))
    if executor is not None:
        executor.shutdown(wait=True)
    return context


def _cached_osm_sample(elevations, cells, cell_size, x, z):
    context = _OBJECT_CONTEXT.get()
    if context is not None and elevations is context.elevations and int(cells) == context.cells and float(cell_size) == context.cell_size:
        key = (float(x), float(z))
        if key in context.samples:
            return context.samples[key]
    return _BASE_OSM_SAMPLE(elevations, cells, cell_size, x, z)


def _cached_polygon_extrema(elevations, cells, cell_size, polygon):
    context = _OBJECT_CONTEXT.get()
    if context is not None and elevations is context.elevations and int(cells) == context.cells and float(cell_size) == context.cell_size:
        key = tuple((float(x), float(z)) for x, z in polygon)
        if key in context.polygon_extrema:
            return context.polygon_extrema[key]
    return _BASE_POLYGON_EXTREMA(elevations, cells, cell_size, polygon)


def _cached_square_samples(elevations, cells, cell_size, x, z, size):
    context = _OBJECT_CONTEXT.get()
    if context is not None and elevations is context.elevations and int(cells) == context.cells and float(cell_size) == context.cell_size:
        key = (float(x), float(z), float(size))
        if key in context.square_samples:
            return context.square_samples[key]
    return _BASE_SQUARE_SAMPLES(elevations, cells, cell_size, x, z, size)


def _cached_building_submerged(plan, elevations, raster, spec):
    context = _OBJECT_CONTEXT.get()
    if context is not None and elevations is context.elevations:
        key = _building_key(plan)
        if key in context.building_submerged:
            return context.building_submerged[key]
    return _BASE_BUILDING_SUBMERGED(plan, elevations, raster, spec)


def _cached_line_chunks(points, target_length, *, endpoint_trim=0.0):
    context = _OBJECT_CONTEXT.get()
    if context is not None:
        key = _line_key(points, target_length, endpoint_trim)
        if key in context.line_chunks:
            return context.line_chunks[key]
    return _BASE_LINE_CHUNKS(points, target_length, endpoint_trim=endpoint_trim)


def _cached_polygon_grid(points, spacing, seed, *, jitter_fraction=0.0, heading_jitter_degrees=0.0):
    context = _OBJECT_CONTEXT.get()
    if context is not None:
        key = _grid_key(points, spacing, seed, jitter_fraction, heading_jitter_degrees)
        if key in context.polygon_grids:
            return context.polygon_grids[key]
    return _BASE_POLYGON_GRID(
        points, spacing, seed,
        jitter_fraction=jitter_fraction,
        heading_jitter_degrees=heading_jitter_degrees,
    )


def _cached_place_cluster_at(*, variant, elevations, raster, road_corridors, spec, x, z, heading, require_forest, minimum_forest_fraction, maximum_relief, maximum_burial, maximum_float, clearance, avoid_roads=True):
    context = _OBJECT_CONTEXT.get()
    if context is not None and elevations is context.elevations:
        key = _cluster_key(
            variant, x, z, heading, require_forest, minimum_forest_fraction,
            maximum_relief, maximum_burial, maximum_float, clearance, avoid_roads,
        )
        if key in context.clusters:
            return context.clusters[key]
    return _BASE_PLACE_CLUSTER(
        variant=variant, elevations=elevations, raster=raster,
        road_corridors=road_corridors, spec=spec, x=x, z=z, heading=heading,
        require_forest=require_forest,
        minimum_forest_fraction=minimum_forest_fraction,
        maximum_relief=maximum_relief,
        maximum_burial=maximum_burial,
        maximum_float=maximum_float,
        clearance=clearance,
        avoid_roads=avoid_roads,
    )


def _parallel_generate_world_objects(dataset, projection, raster, elevations, spec, *args, **kwargs):
    building_plans = kwargs.get("building_placement_plans")
    if building_plans is None:
        building_plans = ()
    progress = kwargs.get("progress_callback")
    # Do not pay process startup for tiny worlds where serial helper calls are
    # already below measurement noise.
    estimated = len(tuple(building_plans)) + len(getattr(dataset, "barriers", ())) + len(getattr(dataset, "rural_vegetation", ()))
    if float(getattr(spec, "world_size", 0.0)) >= 1000.0:
        estimated += 256
    if _worker_count(estimated) <= 1:
        return _BASE_GENERATE(dataset, projection, raster, elevations, spec, *args, **kwargs)
    if progress is not None:
        workers = _worker_count(max(estimated, 128))
        progress(53, f"Parallel precomputing building/forest/barrier/rural placement with {workers} workers")
    try:
        context = _prepare_object_context(dataset, projection, raster, elevations, spec, building_plans)
    except (OSError, RuntimeError, BrokenPipeError, EOFError, TypeError):
        return _BASE_GENERATE(dataset, projection, raster, elevations, spec, *args, **kwargs)
    token = _OBJECT_CONTEXT.set(context)
    try:
        return _BASE_GENERATE(dataset, projection, raster, elevations, spec, *args, **kwargs)
    finally:
        _OBJECT_CONTEXT.reset(token)


def _cached_playability_sample(elevations, cells, cell_size, x, z):
    context = _ROAD_CONTEXT.get()
    if context is not None and elevations is context.elevations and int(cells) == context.cells and float(cell_size) == context.cell_size:
        key = (float(x), float(z))
        if key in context.samples:
            return context.samples[key]
    return _BASE_PLAYABILITY_SAMPLE(elevations, cells, cell_size, x, z)


def _parallel_execute_run_jobs(jobs, progress_callback=None):
    plans = _BASE_EXECUTE_RUN_JOBS(jobs, progress_callback=progress_callback)
    context = _ROAD_CONTEXT.get()
    if context is None or not plans:
        return plans
    points: dict[tuple[float, float], None] = {}
    for plan in plans:
        for _piece, start, end in getattr(plan, "fitted_pieces", ()):
            points[(float(start[0]), float(start[1]))] = None
            points[(float(end[0]), float(end[1]))] = None
    missing = tuple(point for point in points if point not in context.samples)
    workers = _worker_count(len(missing))
    if workers <= 1 or not missing:
        return plans
    if context.progress_callback is not None:
        context.progress_callback(
            58,
            f"Pre-sampling {len(missing):,} stock-road endpoints with {workers} workers",
        )
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(context.elevations, None, None, type("_Spec", (), {"cells": context.cells, "cell_size": context.cell_size})()),
        ) as executor:
            batches = _batches(missing, workers)
            for values in executor.map(_road_sample_batch, batches, chunksize=1):
                context.samples.update(values)
    except (OSError, RuntimeError, BrokenPipeError, EOFError, TypeError):
        # The serial fitter remains authoritative if a frozen runtime cannot
        # create a second pool after road-chain planning.
        pass
    return plans


def _road_sample_batch(points: tuple[tuple[float, float], ...]):
    elevations = _W_ELEVATIONS
    return tuple(
        (point, _sample_exact(elevations, _W_CELLS, _W_CELL_SIZE, point[0], point[1]))
        for point in points
    )


def _parallel_stock_fit(dataset, projection, elevations, spec, *args, **kwargs):
    context = _RoadContext(
        elevations=elevations,
        cells=int(spec.cells),
        cell_size=float(spec.cell_size),
        progress_callback=kwargs.get("progress_callback"),
    )
    token = _ROAD_CONTEXT.set(context)
    try:
        return _BASE_STOCK_FIT(dataset, projection, elevations, spec, *args, **kwargs)
    finally:
        _ROAD_CONTEXT.reset(token)


def install_object_stage_parallel_policy() -> None:
    """Install deterministic multicore precomputation around final object stages."""
    global _INSTALLED
    global _BASE_GENERATE, _BASE_OSM_SAMPLE, _BASE_PLAYABILITY_SAMPLE
    global _BASE_POLYGON_EXTREMA, _BASE_SQUARE_SAMPLES, _BASE_BUILDING_SUBMERGED
    global _BASE_LINE_CHUNKS, _BASE_POLYGON_GRID, _BASE_PLACE_CLUSTER
    global _BASE_STOCK_FIT, _BASE_EXECUTE_RUN_JOBS
    if _INSTALLED:
        return

    # Capture late so array/vector forest policies and bridge source-water policy
    # remain inside the chain rather than being bypassed.
    _BASE_GENERATE = _generator.generate_world_objects
    _BASE_OSM_SAMPLE = _osm._sample_elevation
    _BASE_PLAYABILITY_SAMPLE = _playability._sample_elevation
    _BASE_POLYGON_EXTREMA = _osm._polygon_elevation_extrema
    _BASE_SQUARE_SAMPLES = _osm._square_elevation_samples
    _BASE_BUILDING_SUBMERGED = _osm._building_plan_fully_submerged
    _BASE_LINE_CHUNKS = _osm._line_chunks
    _BASE_POLYGON_GRID = _osm._polygon_grid_candidates
    _BASE_PLACE_CLUSTER = _osm._place_cluster_at
    _BASE_STOCK_FIT = _playability._fit_stock_piece_road_objects
    _BASE_EXECUTE_RUN_JOBS = _road_parallel._execute_run_jobs

    _osm._sample_elevation = _cached_osm_sample
    _osm._polygon_elevation_extrema = _cached_polygon_extrema
    _osm._square_elevation_samples = _cached_square_samples
    _osm._building_plan_fully_submerged = _cached_building_submerged
    _osm._line_chunks = _cached_line_chunks
    _osm._polygon_grid_candidates = _cached_polygon_grid
    _osm._place_cluster_at = _cached_place_cluster_at

    _playability._sample_elevation = _cached_playability_sample
    _road_parallel._execute_run_jobs = _parallel_execute_run_jobs
    _playability._fit_stock_piece_road_objects = _parallel_stock_fit

    # Bind both names because late wrapper policies capture osm.generate_world_objects
    # while generator._load_nonroad_objects resolves the generator module global.
    _osm.generate_world_objects = _parallel_generate_world_objects
    _generator.generate_world_objects = _parallel_generate_world_objects
    _INSTALLED = True
