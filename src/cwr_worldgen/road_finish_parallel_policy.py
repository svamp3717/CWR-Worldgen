# SPDX-License-Identifier: GPL-3.0-or-later
"""Vectorize the transform-heavy half of final stock-road fitting.

Road-chain geometry is already planned in worker processes.  A second process
pool for final transforms proved counterproductive on Windows: process startup,
pickling thousands of chains and returning tens of thousands of tiny transform
records cost more than the arithmetic itself.

This layer keeps the useful half of that experiment and removes the paperwork:

* all fitted road endpoints are sampled from the immutable terrain in NumPy;
* centre coordinates, heading, pitch and model axes are calculated in arrays;
* chain gap/overlap diagnostics are reduced from those arrays in one pass;
* curved-gravel selection is cached only for the pieces that actually need it;
* the historical fitter still constructs WorldObjects, assigns deterministic
  object ids and performs shared junction validation in its existing order.

The old per-chain adjacency diagnostic loop is skipped while this context is
active and the report receives the equivalent vectorized maxima afterwards.
"""
from __future__ import annotations

import builtins
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
import math
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
_ORIGINAL_ZIP = builtins.zip

Point = tuple[float, float]
Axis = tuple[Point, Point]
Geometry = tuple[float, float, float, float, float]


@dataclass(slots=True)
class _Context:
    elevations: Sequence[float]
    cells: int
    cell_size: float
    progress_callback: Any = None
    curved_models: dict[tuple[Any, ...], str] = field(default_factory=dict)
    geometry: dict[tuple[float, float, float, float], Geometry] = field(default_factory=dict)
    axes: dict[tuple[float, float, float, float], Axis] = field(default_factory=dict)
    maximum_chain_gap: float = 0.0
    maximum_model_overlap: float = 0.0
    prepared: bool = False


_CONTEXT: ContextVar[_Context | None] = ContextVar(
    "cwr_vector_road_finish", default=None
)


def _point_key(point: Point) -> Point:
    return float(point[0]), float(point[1])


def _geometry_key(start: Point, end: Point) -> tuple[float, float, float, float]:
    return float(start[0]), float(start[1]), float(end[0]), float(end[1])


def _curve_key(
    model_path: str,
    run: Sequence[Point],
    start: Point,
    end: Point,
) -> tuple[Any, ...]:
    return (
        str(model_path),
        tuple((float(x), float(z)) for x, z in run),
        _point_key(start),
        _point_key(end),
    )


def _axis_key(
    x: float,
    z: float,
    heading: float,
    length: float,
) -> tuple[float, float, float, float]:
    return float(x), float(z), float(heading), float(length)


def _vector_sample_values(
    elevations: Sequence[float],
    cells: int,
    cell_size: float,
    points: Sequence[Point] | np.ndarray,
) -> np.ndarray:
    """Bilinear terrain sampling for an arbitrary point array."""
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.size == 0:
        return np.empty(0, dtype=np.float64)
    coordinates = coordinates.reshape(-1, 2)
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
    return first * (1.0 - tz) + second * tz


def _vector_sample_points(
    elevations: Sequence[float],
    cells: int,
    cell_size: float,
    points: Sequence[Point],
) -> dict[Point, float]:
    """Compatibility helper used by focused equivalence tests."""
    values = _vector_sample_values(elevations, cells, cell_size, points)
    return {
        _point_key(point): float(value)
        for point, value in zip(points, values.tolist())
    }


def _point_segment_distance_array(
    points: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    delta = ends - starts
    denominator = np.sum(delta * delta, axis=1)
    relative = points - starts
    fraction = np.zeros(len(points), dtype=np.float64)
    usable = denominator > 1.0e-12
    fraction[usable] = (
        np.sum(relative[usable] * delta[usable], axis=1)
        / denominator[usable]
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    nearest = starts + delta * fraction[:, None]
    return np.hypot(points[:, 0] - nearest[:, 0], points[:, 1] - nearest[:, 1])


def _vector_prepare(plans, context: _Context) -> int:
    """Precompute all final piece transforms and chain diagnostics in arrays."""
    starts_list: list[Point] = []
    ends_list: list[Point] = []
    lengths: list[float] = []
    plan_ids: list[int] = []
    records: list[tuple[Any, tuple[Point, ...], Point, Point]] = []

    for plan_index, plan in enumerate(plans):
        run = tuple((float(x), float(z)) for x, z in getattr(plan, "run", ()))
        for piece, start, end in getattr(plan, "fitted_pieces", ()):
            start_key = _point_key(start)
            end_key = _point_key(end)
            starts_list.append(start_key)
            ends_list.append(end_key)
            lengths.append(float(piece.length_metres))
            plan_ids.append(plan_index)
            records.append((piece, run, start_key, end_key))

    count = len(records)
    if count == 0:
        context.prepared = True
        return 0

    starts = np.asarray(starts_list, dtype=np.float64)
    ends = np.asarray(ends_list, dtype=np.float64)
    piece_lengths = np.asarray(lengths, dtype=np.float64)
    chain_ids = np.asarray(plan_ids, dtype=np.int64)

    start_heights = _vector_sample_values(
        context.elevations, context.cells, context.cell_size, starts
    )
    end_heights = _vector_sample_values(
        context.elevations, context.cells, context.cell_size, ends
    )

    dx = ends[:, 0] - starts[:, 0]
    dz = ends[:, 1] - starts[:, 1]
    horizontal = np.maximum(0.01, np.hypot(dx, dz))
    headings = np.mod(np.degrees(np.arctan2(dx, dz)), 360.0)
    pitches = np.clip(
        np.degrees(np.arctan2(end_heights - start_heights, horizontal)),
        -35.0,
        35.0,
    )
    centres_x = (starts[:, 0] + ends[:, 0]) * 0.5
    centres_z = (starts[:, 1] + ends[:, 1]) * 0.5
    base_y = (start_heights + end_heights) * 0.5

    angles = np.radians(headings)
    half_lengths = piece_lengths * 0.5
    axis_dx = np.sin(angles) * half_lengths
    axis_dz = np.cos(angles) * half_lengths
    axis_starts = np.column_stack((centres_x - axis_dx, centres_z - axis_dz))
    axis_ends = np.column_stack((centres_x + axis_dx, centres_z + axis_dz))

    for index, (_piece, _run, start, end) in enumerate(records):
        geometry = (
            float(centres_x[index]),
            float(base_y[index]),
            float(centres_z[index]),
            float(headings[index]),
            float(pitches[index]),
        )
        context.geometry[_geometry_key(start, end)] = geometry
        context.axes[_axis_key(
            geometry[0], geometry[2], geometry[3], piece_lengths[index]
        )] = (
            (float(axis_starts[index, 0]), float(axis_starts[index, 1])),
            (float(axis_ends[index, 0]), float(axis_ends[index, 1])),
        )

    # Curved model selection only applies to generated gravel. Paved/dirt stock
    # pieces can return their original model immediately without a 96k-entry map.
    for piece, run, start, end in records:
        if not _playability.is_generated_gravel_road_model(piece.model_path):
            continue
        key = _curve_key(piece.model_path, run, start, end)
        context.curved_models[key] = _BASE_CURVED_MODEL(
            piece.model_path, run, start, end
        )

    # Vector form of the historical adjacent-piece diagnostics. Chain ids make
    # sure the flattened arrays never compare the end of one run to the start of
    # the next run.
    if count > 1:
        adjacent = chain_ids[1:] == chain_ids[:-1]
        if bool(np.any(adjacent)):
            previous_end = axis_ends[:-1][adjacent]
            current_start = axis_starts[1:][adjacent]
            offset = current_start - previous_end
            gaps = np.hypot(offset[:, 0], offset[:, 1])
            context.maximum_chain_gap = float(np.max(gaps)) if gaps.size else 0.0

            previous_angles = angles[:-1][adjacent]
            direction_x = np.sin(previous_angles)
            direction_z = np.cos(previous_angles)
            lateral = np.abs(direction_x * offset[:, 1] - direction_z * offset[:, 0])
            longitudinal = direction_x * offset[:, 0] + direction_z * offset[:, 1]
            overlaps = np.where(
                (lateral <= 0.10) & (longitudinal < 0.0),
                -longitudinal,
                0.0,
            )
            context.maximum_model_overlap = (
                float(np.max(overlaps)) if overlaps.size else 0.0
            )

    context.prepared = True
    return count


def _diagnostic_zip(*iterables, **kwargs):
    """Skip only the old adjacent-road diagnostic loop after vector reduction."""
    context = _CONTEXT.get()
    if context is not None and context.prepared and len(iterables) == 2:
        left, right = iterables
        if (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) >= 2
            and len(right) == len(left) - 1
            and right
            and right[0] is left[1]
            and isinstance(left[0], tuple)
            and len(left[0]) == 2
            and isinstance(left[0][0], _playability.WorldObject)
        ):
            return _ORIGINAL_ZIP((), ())
    return _ORIGINAL_ZIP(*iterables, **kwargs)


def _execute(jobs, progress_callback=None):
    plans = _BASE_EXECUTE(jobs, progress_callback=progress_callback)
    context = _CONTEXT.get()
    if context is None or not plans:
        return plans

    if context.progress_callback is not None:
        piece_count = sum(
            len(getattr(plan, "fitted_pieces", ())) for plan in plans
        )
        context.progress_callback(
            58,
            f"Vectorizing {piece_count:,} final stock-road transforms and diagnostics",
        )
    _vector_prepare(plans, context)
    return plans


def _cached_curved_model(model_path, run, start, end):
    if not _playability.is_generated_gravel_road_model(model_path):
        return model_path
    context = _CONTEXT.get()
    if context is not None:
        value = context.curved_models.get(_curve_key(model_path, run, start, end))
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
        report = _BASE_STOCK_FIT(
            dataset, projection, elevations, spec, *args, **kwargs
        )
        if context.prepared:
            report = replace(
                report,
                maximum_chain_gap=context.maximum_chain_gap,
                maximum_model_overlap_metres=context.maximum_model_overlap,
            )
        return report
    finally:
        _CONTEXT.reset(token)


def install_road_finish_parallel_policy() -> None:
    global _INSTALLED
    global _BASE_STOCK_FIT, _BASE_EXECUTE, _BASE_SAMPLE
    global _BASE_CURVED_MODEL, _BASE_ROAD_OBJECT, _BASE_MODEL_AXIS
    if _INSTALLED:
        return

    # object_stage_parallel_policy is installed first. Keep its non-road work,
    # but bypass its old road endpoint worker pool. The captured fitter remains
    # the source-water bridge-aware wrapper around the quality-aware road planner.
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
    _road_parallel.zip = _diagnostic_zip
    # Restore the direct fast sampler instead of retaining the object-stage road
    # ContextVar wrapper. Final chain terrain is already sampled in one array.
    _playability._sample_elevation = _BASE_SAMPLE
    _playability._curved_gravel_model_for_run = _cached_curved_model
    _playability._road_object_on_slope = _cached_road_object
    _playability._model_axis = _cached_model_axis
    _playability._fit_stock_piece_road_objects = _fit
    _INSTALLED = True
