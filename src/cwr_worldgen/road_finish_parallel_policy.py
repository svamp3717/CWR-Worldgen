# SPDX-License-Identifier: GPL-3.0-or-later
"""Parallelize the terrain-sampling half of final stock-road fitting.

Chain geometry is already planned in parallel.  Once those plans are known the
parent still has to create tens of thousands of road objects, and each object
samples the same immutable terrain at its two endpoints.  Pre-sample every unique
planned endpoint in worker processes, then let the historical object-construction
loop consume the exact cached heights in deterministic order.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field
import math
import multiprocessing
import os
from typing import Any, Sequence

from . import object_stage_parallel_policy as _object_stage
from . import playability as _playability
from . import road_chain_parallel_policy as _road_parallel

_INSTALLED = False
_BASE_STOCK_FIT: Any = None
_BASE_EXECUTE: Any = None
_BASE_SAMPLE: Any = None

_W_ELEVATIONS: Sequence[float] | None = None
_W_CELLS = 0
_W_CELL_SIZE = 1.0


@dataclass(slots=True)
class _Context:
    elevations: Sequence[float]
    cells: int
    cell_size: float
    progress_callback: Any = None
    samples: dict[tuple[float, float], float] = field(default_factory=dict)


_CONTEXT: ContextVar[_Context | None] = ContextVar(
    "cwr_parallel_road_finish", default=None
)


def _workers(job_count: int) -> int:
    if job_count < 256 or multiprocessing.current_process().daemon:
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
    return min(requested, max(1, job_count // 128))


def _init_worker(elevations, cells: int, cell_size: float) -> None:
    global _W_ELEVATIONS, _W_CELLS, _W_CELL_SIZE
    _W_ELEVATIONS = elevations
    _W_CELLS = int(cells)
    _W_CELL_SIZE = float(cell_size)


def _sample(x: float, z: float) -> float:
    elevations = _W_ELEVATIONS
    cells = _W_CELLS
    cell_size = _W_CELL_SIZE
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


def _sample_batch(points: tuple[tuple[float, float], ...]):
    return tuple((point, _sample(point[0], point[1])) for point in points)


def _batches(points: Sequence[tuple[float, float]], workers: int):
    size = max(128, min(4096, math.ceil(len(points) / max(1, workers * 8))))
    return tuple(
        tuple(points[offset: offset + size])
        for offset in range(0, len(points), size)
    )


def _execute(jobs, progress_callback=None):
    plans = _BASE_EXECUTE(jobs, progress_callback=progress_callback)
    context = _CONTEXT.get()
    if context is None or not plans:
        return plans

    unique: dict[tuple[float, float], None] = {}
    for plan in plans:
        for _piece, start, end in getattr(plan, "fitted_pieces", ()):
            unique[(float(start[0]), float(start[1]))] = None
            unique[(float(end[0]), float(end[1]))] = None
    missing = tuple(point for point in unique if point not in context.samples)
    workers = _workers(len(missing))
    if not missing:
        return plans
    if workers <= 1:
        _init_worker(context.elevations, context.cells, context.cell_size)
        context.samples.update(_sample_batch(missing))
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
            initargs=(context.elevations, context.cells, context.cell_size),
        ) as executor:
            for values in executor.map(_sample_batch, _batches(missing, workers), chunksize=1):
                context.samples.update(values)
    except (OSError, RuntimeError, BrokenPipeError, EOFError, TypeError):
        # Frozen Python builds can reject a second process pool. Preserve output
        # exactly and use the same batched scalar sampler in the parent.
        _init_worker(context.elevations, context.cells, context.cell_size)
        context.samples.update(_sample_batch(missing))
    return plans


def _cached_sample(elevations, cells, cell_size, x, z):
    context = _CONTEXT.get()
    if (
        context is not None
        and elevations is context.elevations
        and int(cells) == context.cells
        and float(cell_size) == context.cell_size
    ):
        key = (float(x), float(z))
        if key in context.samples:
            return context.samples[key]
    return _BASE_SAMPLE(elevations, cells, cell_size, x, z)


def _fit(dataset, projection, elevations, spec, *args, **kwargs):
    context = _Context(
        elevations=elevations,
        cells=int(spec.cells),
        cell_size=float(spec.cell_size),
        progress_callback=kwargs.get("progress_callback"),
    )
    token = _CONTEXT.set(context)
    try:
        return _BASE_STOCK_FIT(dataset, projection, elevations, spec, *args, **kwargs)
    finally:
        _CONTEXT.reset(token)


def install_road_finish_parallel_policy() -> None:
    global _INSTALLED, _BASE_STOCK_FIT, _BASE_EXECUTE, _BASE_SAMPLE
    if _INSTALLED:
        return

    # object_stage_parallel_policy is installed first.  Its non-road precompute
    # remains live, but supersede its road-only wrappers with this Windows-safe
    # implementation.  The captured stock fitter is the bridge-aware wrapper.
    _BASE_STOCK_FIT = _object_stage._BASE_STOCK_FIT or _playability._fit_stock_piece_road_objects
    _BASE_EXECUTE = _object_stage._BASE_EXECUTE_RUN_JOBS or _road_parallel._execute_run_jobs
    _BASE_SAMPLE = _object_stage._BASE_PLAYABILITY_SAMPLE or _playability._sample_elevation

    _road_parallel._execute_run_jobs = _execute
    _playability._sample_elevation = _cached_sample
    _playability._fit_stock_piece_road_objects = _fit
    _INSTALLED = True
