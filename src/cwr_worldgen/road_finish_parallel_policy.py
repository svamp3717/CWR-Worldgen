# SPDX-License-Identifier: GPL-3.0-or-later
"""Parallelize the transform-heavy half of final stock-road fitting.

Chain geometry is already planned in worker processes.  The historical final pass
then revisited every fitted piece in the parent to choose a curved gravel sibling,
sample terrain, calculate heading/pitch, build model axes and finally construct a
WorldObject.  On dense worlds that means roughly one hundred thousand repetitions
of the same small Python geometry pipeline after the expensive planning work has
already finished.

This policy keeps object ids, ordering, object budgets and shared junction checks
in the parent process, but moves the independent chain transform work out of the
serial hot loop:

* every unique road endpoint is sampled in one NumPy bilinear interpolation pass;
* each planned chain is finalized in a bounded process pool;
* curved gravel model choice, heading, pitch and model axes are cached per piece;
* the historical fitter consumes those exact precomputed values in deterministic
  order and remains authoritative for final ids and junction validation.

If process creation is unavailable (notably some frozen/embedded Python runtimes),
the same chain-finalization function runs serially and preserves output semantics.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextvars import ContextVar
from dataclasses import dataclass, field
import math
import multiprocessing
import os
from typing import Any, Sequence

import numpy as np

from . import object_stage_parallel_policy as _object_stage
from . import playability as _playability
from . import road_chain_parallel_policy as _road_parallel

_INSTALLED = False
_BASE_STOCK_FIT: Any = None
_BASE_EXECUTE: Any = None
_BASE_SAMPLE: Any = None
_BASE_CURVED_MODEL: Any = None
_BASE_ROAD_OBJECT: Any = None
_BASE_MODEL_AXIS: Any = None


Point = tuple[float, float]
Axis = tuple[Point, Point]


@dataclass(slots=True)
class _Context:
    elevations: Sequence[float]
    cells: int
    cell_size: float
    progress_callback: Any = None
    samples: dict[Point, float] = field(default_factory=dict)
    curved_models: dict[tuple[Any, ...], str] = field(default_factory=dict)
    geometry: dict[tuple[float, float, float, float], tuple[float, float, float, float, float]] = field(default_factory=dict)
    axes: dict[tuple[float, float, float, float], Axis] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _FinalizeJob:
    order: int
    run: tuple[Point, ...]
    pieces: tuple[
        tuple[str, float, int, Point, Point, float, float], ...
    ]


@dataclass(frozen=True, slots=True)
class _FinalizeResult:
    order: int
    curved_models: tuple[tuple[tuple[Any, ...], str], ...]
    geometry: tuple[
        tuple[
            tuple[float, float, float, float],
            tuple[float, float, float, float, float],
        ], ...
    ]
    axes: tuple[tuple[tuple[float, float, float, float], Axis], ...]


_CONTEXT: ContextVar[_Context | None] = ContextVar(
    "cwr_parallel_road_finish", default=None
)


def _workers(job_count: int) -> int:
    if job_count < 96 or multiprocessing.current_process().daemon:
        return 1
    raw = os.environ.get("CWR_WORLDGEN_ROAD_WORKERS", "").strip()
    if not raw:
        raw = os.environ.get("CWR_WORLDGEN_OBJECT_WORKERS", "").strip()
    if raw:
        try:
            requested = max(1, int(raw))
        except ValueError:
            requested = 1
    else:
        cpu = os.cpu_count() or 1
        requested = max(1, min(8, cpu - 1 if cpu > 2 else cpu))
    return min(requested, max(1, job_count // 24))


def _point_key(point: Point) -> Point:
    return float(point[0]), float(point[1])


def _geometry_key(start: Point, end: Point) -> tuple[float, float, float, float]:
    return float(start[0]), float(start[1]), float(end[0]), float(end[1])


def _curve_key(model_path: str, run: Sequence[Point], start: Point, end: Point) -> tuple[Any, ...]:
    return (
        str(model_path),
        tuple((float(x), float(z)) for x, z in run),
        _point_key(start),
        _point_key(end),
    )


def _axis_key(x: float, z: float, heading: float, length: float) -> tuple[float, float, float, float]:
    return float(x), float(z), float(heading), float(length)


def _vector_sample_points(
    elevations: Sequence[float],
    cells: int,
    cell_size: float,
    points: Sequence[Point],
) -> dict[Point, float]:
    """Sample all points with the same bilinear formula as ``_sample_elevation``."""
    if not points:
        return {}
    coordinates = np.asarray(points, dtype=np.float64)
    fx = np.clip(coordinates[:, 0] / float(cell_size), 0.0, cells - 1.0)
    fz = np.clip(coordinates[:, 1] / float(cell_size), 0.0, cells - 1.0)
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
    first = h00 * (1.0 - tx) + h10 * tx
    second = h01 * (1.0 - tx) + h11 * tx
    values = first * (1.0 - tz) + second * tz
    return {
        _point_key(point): float(value)
        for point, value in zip(points, values.tolist())
    }


def _piece_geometry(
    start: Point,
    end: Point,
    start_height: float,
    end_height: float,
) -> tuple[float, float, float, float, float]:
    dx = float(end[0]) - float(start[0])
    dz = float(end[1]) - float(start[1])
    horizontal = max(0.01, math.hypot(dx, dz))
    heading = math.degrees(math.atan2(dx, dz)) % 360.0
    pitch = math.degrees(math.atan2(float(end_height) - float(start_height), horizontal))
    pitch = max(-35.0, min(35.0, pitch))
    return (
        (float(start[0]) + float(end[0])) * 0.5,
        (float(start_height) + float(end_height)) * 0.5,
        (float(start[1]) + float(end[1])) * 0.5,
        heading,
        pitch,
    )


def _axis_from_geometry(
    geometry: tuple[float, float, float, float, float],
    length: float,
) -> Axis:
    x, _base_y, z, heading, pitch = geometry
    # Delegate the exact endpoint convention to the historical helper once in
    # the worker.  The parent hot loop later receives this axis from the cache.
    obj = _playability.WorldObject(0, "", x, 0.0, z, heading, pitch)
    fn = _BASE_MODEL_AXIS or _playability._model_axis
    return fn(obj, float(length))


def _finalize_job(job: _FinalizeJob) -> _FinalizeResult:
    curved: list[tuple[tuple[Any, ...], str]] = []
    geometry_values: list[
        tuple[
            tuple[float, float, float, float],
            tuple[float, float, float, float, float],
        ]
    ] = []
    axes: list[tuple[tuple[float, float, float, float], Axis]] = []
    curved_fn = _BASE_CURVED_MODEL or _playability._curved_gravel_model_for_run

    for model_path, length, _nominal, start, end, start_height, end_height in job.pieces:
        placed_model = curved_fn(model_path, job.run, start, end)
        curve_key = _curve_key(model_path, job.run, start, end)
        geometry = _piece_geometry(start, end, start_height, end_height)
        geometry_key = _geometry_key(start, end)
        axis = _axis_from_geometry(geometry, length)
        axis_key = _axis_key(geometry[0], geometry[2], geometry[3], length)
        curved.append((curve_key, placed_model))
        geometry_values.append((geometry_key, geometry))
        axes.append((axis_key, axis))

    return _FinalizeResult(
        job.order,
        tuple(curved),
        tuple(geometry_values),
        tuple(axes),
    )


def _finalize_batch(batch: tuple[_FinalizeJob, ...]) -> tuple[_FinalizeResult, ...]:
    return tuple(_finalize_job(job) for job in batch)


def _batches(jobs: Sequence[_FinalizeJob], workers: int) -> tuple[tuple[_FinalizeJob, ...], ...]:
    if not jobs:
        return ()
    size = max(8, min(64, math.ceil(len(jobs) / max(1, workers * 12))))
    return tuple(
        tuple(jobs[offset: offset + size])
        for offset in range(0, len(jobs), size)
    )


def _store_result(context: _Context, result: _FinalizeResult) -> None:
    context.curved_models.update(result.curved_models)
    context.geometry.update(result.geometry)
    context.axes.update(result.axes)


def _prepare_finalize_jobs(plans, context: _Context) -> tuple[_FinalizeJob, ...]:
    unique: dict[Point, None] = {}
    for plan in plans:
        for _piece, start, end in getattr(plan, "fitted_pieces", ()):
            unique[_point_key(start)] = None
            unique[_point_key(end)] = None

    if unique:
        context.samples.update(
            _vector_sample_points(
                context.elevations,
                context.cells,
                context.cell_size,
                tuple(unique),
            )
        )

    jobs: list[_FinalizeJob] = []
    for plan in plans:
        pieces = []
        for piece, start, end in getattr(plan, "fitted_pieces", ()):
            start_key = _point_key(start)
            end_key = _point_key(end)
            pieces.append((
                str(piece.model_path),
                float(piece.length_metres),
                int(piece.nominal_length),
                start_key,
                end_key,
                context.samples[start_key],
                context.samples[end_key],
            ))
        if pieces:
            jobs.append(_FinalizeJob(
                int(getattr(plan, "order", len(jobs))),
                tuple((float(x), float(z)) for x, z in plan.run),
                tuple(pieces),
            ))
    return tuple(jobs)


def _execute(jobs, progress_callback=None):
    plans = _BASE_EXECUTE(jobs, progress_callback=progress_callback)
    context = _CONTEXT.get()
    if context is None or not plans:
        return plans

    finalize_jobs = _prepare_finalize_jobs(plans, context)
    if not finalize_jobs:
        return plans
    workers = _workers(len(finalize_jobs))
    if context.progress_callback is not None:
        context.progress_callback(
            58,
            f"Finalizing {len(finalize_jobs):,} stock-road chains"
            + (f" with {workers} workers" if workers > 1 else ""),
        )

    if workers <= 1:
        for job in finalize_jobs:
            _store_result(context, _finalize_job(job))
        return plans

    completed = 0
    batches = _batches(finalize_jobs, workers)
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_finalize_batch, batch): len(batch)
                for batch in batches
            }
            for future in as_completed(futures):
                values = future.result()
                for result in values:
                    _store_result(context, result)
                completed += len(values)
                if context.progress_callback is not None:
                    local = 58 + round(1.0 * completed / len(finalize_jobs))
                    context.progress_callback(
                        min(59, local),
                        f"Finalized stock-road chains {completed:,}/{len(finalize_jobs):,} "
                        f"with {workers} workers",
                    )
    except (OSError, RuntimeError, BrokenPipeError, EOFError, TypeError):
        # Frozen Python can reject a second pool after chain planning.  The same
        # deterministic calculations still run here in the parent rather than
        # falling back to the old repeated helper path.
        context.curved_models.clear()
        context.geometry.clear()
        context.axes.clear()
        for job in finalize_jobs:
            _store_result(context, _finalize_job(job))
    return plans


def _cached_sample(elevations, cells, cell_size, x, z):
    context = _CONTEXT.get()
    if (
        context is not None
        and elevations is context.elevations
        and int(cells) == context.cells
        and float(cell_size) == context.cell_size
    ):
        key = float(x), float(z)
        if key in context.samples:
            return context.samples[key]
    return _BASE_SAMPLE(elevations, cells, cell_size, x, z)


def _cached_curved_model(model_path, run, start, end):
    context = _CONTEXT.get()
    if context is not None:
        key = _curve_key(model_path, run, start, end)
        value = context.curved_models.get(key)
        if value is not None:
            return value
    return _BASE_CURVED_MODEL(model_path, run, start, end)


def _cached_road_object(
    object_id,
    model_path,
    start,
    end,
    elevations,
    spec,
    *,
    vertical_offset,
):
    context = _CONTEXT.get()
    if (
        context is not None
        and elevations is context.elevations
        and int(spec.cells) == context.cells
        and float(spec.cell_size) == context.cell_size
    ):
        geometry = context.geometry.get(_geometry_key(start, end))
        if geometry is not None:
            x, base_y, z, heading, pitch = geometry
            placement_offset = float(vertical_offset)
            if (
                _playability.is_generated_gravel_road_model(model_path)
                or _playability.is_generated_gravel_junction_model(model_path)
            ):
                placement_offset = -float(
                    _playability.GENERATED_GRAVEL_VISUAL_TOP_METRES
                ) * math.cos(math.radians(pitch))
            return _playability.WorldObject(
                int(object_id),
                str(model_path),
                x,
                base_y + placement_offset,
                z,
                heading,
                pitch,
            )
    return _BASE_ROAD_OBJECT(
        object_id,
        model_path,
        start,
        end,
        elevations,
        spec,
        vertical_offset=vertical_offset,
    )


def _cached_model_axis(obj, length):
    context = _CONTEXT.get()
    if context is not None:
        value = context.axes.get(
            _axis_key(obj.x, obj.z, obj.heading_degrees, length)
        )
        if value is not None:
            return value
    return _BASE_MODEL_AXIS(obj, length)


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
    global _INSTALLED
    global _BASE_STOCK_FIT, _BASE_EXECUTE, _BASE_SAMPLE
    global _BASE_CURVED_MODEL, _BASE_ROAD_OBJECT, _BASE_MODEL_AXIS
    if _INSTALLED:
        return

    # object_stage_parallel_policy is installed first.  Keep its non-road work,
    # but supersede its road-only pre-sampler with this chain-level finalizer.
    # The captured fitter is the source-water bridge-aware wrapper.
    _BASE_STOCK_FIT = (
        _object_stage._BASE_STOCK_FIT
        or _playability._fit_stock_piece_road_objects
    )
    _BASE_EXECUTE = (
        _object_stage._BASE_EXECUTE_RUN_JOBS
        or _road_parallel._execute_run_jobs
    )
    _BASE_SAMPLE = (
        _object_stage._BASE_PLAYABILITY_SAMPLE
        or _playability._sample_elevation
    )
    _BASE_CURVED_MODEL = _playability._curved_gravel_model_for_run
    _BASE_ROAD_OBJECT = _playability._road_object_on_slope
    _BASE_MODEL_AXIS = _playability._model_axis

    _road_parallel._execute_run_jobs = _execute
    _playability._sample_elevation = _cached_sample
    _playability._curved_gravel_model_for_run = _cached_curved_model
    _playability._road_object_on_slope = _cached_road_object
    _playability._model_axis = _cached_model_axis
    _playability._fit_stock_piece_road_objects = _fit
    _INSTALLED = True
