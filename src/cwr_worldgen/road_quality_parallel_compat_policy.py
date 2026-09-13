# SPDX-License-Identifier: GPL-3.0-or-later
"""Preserve road-quality semantics inside the parallel stock-chain planner.

``road_quality_policy`` is installed before Milestone 9 and stores its terrain and
junction state in a ContextVar. Windows process workers do not inherit that
ContextVar. Without an explicit bridge, parallel road planning would silently
fall back to the simpler base piece scorer and trade road quality for speed.

This layer copies the immutable quality context to each worker once and evaluates
the exact existing quality score while batching the 25/12/6/3 m chord searches
into one source-segment scan per chain step. Results are still merged in original
feature/run order by ``road_chain_parallel_policy``.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from typing import Any, Sequence

from . import playability as _playability
from . import road_chain_parallel_policy as _parallel
from . import road_quality_policy as _quality

_INSTALLED = False
_ORIGINAL_PARALLEL_PLAN_RUN: Any = None
_ORIGINAL_PARALLEL_EXECUTE: Any = None


def _batched_quality_chain(
    measure: Any,
    pieces: Sequence[Any],
    *,
    start_distance: float,
    preferred_end_distance: float,
    minimum_end_distance: float,
    maximum_end_distance: float,
):
    """Exact ``road_quality_policy._quality_chain`` with batched chord queries."""

    context = _quality._CONTEXT.get()
    if context is None:
        return _parallel._batched_stock_piece_chain(
            measure,
            pieces,
            start_distance=start_distance,
            preferred_end_distance=preferred_end_distance,
            minimum_end_distance=minimum_end_distance,
            maximum_end_distance=maximum_end_distance,
        )

    (
        start_distance,
        preferred_end_distance,
        minimum_end_distance,
        maximum_end_distance,
    ) = _quality._quality_window(
        measure,
        pieces,
        start_distance,
        preferred_end_distance,
        minimum_end_distance,
        maximum_end_distance,
        context,
    )
    if not pieces or preferred_end_distance <= start_distance + 0.05:
        return ()

    ordered = tuple(sorted(
        pieces,
        key=lambda piece: (-piece.length_metres, piece.model_path.casefold()),
    ))
    shortest = min(piece.length_metres for piece in ordered)
    longest = max(piece.length_metres for piece in ordered)
    current = start_distance
    fitted: list[tuple[Any, tuple[float, float], tuple[float, float]]] = []
    maximum_objects = max(
        1,
        int(math.ceil((maximum_end_distance - start_distance) / shortest)) + 2,
    )

    for _ in range(maximum_objects):
        if current >= preferred_end_distance - 0.05:
            break
        remaining = preferred_end_distance - current
        if current >= minimum_end_distance - 0.05 and remaining < shortest * 0.45:
            break

        preferred = _playability._road_piece_sequence(remaining, ordered)
        preferred_piece = preferred[0] if preferred else ordered[-1]
        start_x, start_z, start_heading = measure.point(current)
        near_end = remaining <= longest * 2.25
        endpoints = _parallel._batch_chord_endpoints(
            measure,
            current,
            tuple(piece.length_metres for piece in ordered),
            maximum_end_distance,
        )
        candidates = []

        for piece in ordered:
            endpoint = endpoints.get(float(piece.length_metres))
            if endpoint is None:
                continue
            end_distance, end_x, end_z, chord_heading = endpoint
            end_heading = measure.point(end_distance)[2]
            turn = max(
                _playability._heading_difference(chord_heading, start_heading),
                _playability._heading_difference(chord_heading, end_heading),
            )
            deviation = measure.maximum_chord_deviation(
                current,
                end_distance,
                (start_x, start_z),
                (end_x, end_z),
            )
            gravel = _playability.is_generated_gravel_road_model(piece.model_path)
            if gravel:
                if piece.nominal_length >= 25:
                    turn_limit, deviation_limit = 15.0, 0.85
                elif piece.nominal_length >= 12:
                    turn_limit, deviation_limit = 22.0, 0.55
                elif piece.nominal_length >= 6:
                    turn_limit, deviation_limit = 30.0, 0.35
                else:
                    turn_limit, deviation_limit = 42.0, 0.20
            elif piece.nominal_length >= 25:
                turn_limit, deviation_limit = 7.0, 0.45
            elif piece.nominal_length >= 12:
                turn_limit, deviation_limit = 11.0, 0.30
            else:
                turn_limit, deviation_limit = 18.0, 0.22

            fidelity_penalty = int(
                turn > turn_limit or deviation > deviation_limit
            )
            bulge = _quality._terrain_bulge(
                context,
                (start_x, start_z),
                (end_x, end_z),
                piece.nominal_length,
            )
            terrain_limit = (
                _quality._GRAVEL_BULGE_LIMIT
                if gravel else _quality._STOCK_BULGE_LIMIT
            )
            terrain_penalty = int(bulge > terrain_limit)
            terrain_ratio = bulge / max(0.001, terrain_limit)
            tail_error = (
                _quality._tail_error(
                    measure,
                    ordered,
                    end_distance,
                    preferred_end_distance,
                    maximum_end_distance,
                    _quality._LOOKAHEAD_DEPTH,
                )
                if near_end else 0.0
            )
            tail_tolerance = max(
                0.20,
                float(getattr(context.spec, "road_connection_tolerance", 0.35)),
            )
            tail_penalty = int(near_end and tail_error > tail_tolerance)

            if gravel:
                score = (
                    fidelity_penalty,
                    tail_penalty,
                    terrain_penalty,
                    tail_error if near_end else 0.0,
                    0 if piece == preferred_piece else 1,
                    terrain_ratio,
                    max(turn / turn_limit, deviation / deviation_limit),
                    abs(preferred_end_distance - end_distance),
                    -piece.length_metres,
                )
            else:
                score = (
                    fidelity_penalty,
                    tail_penalty,
                    terrain_penalty,
                    max(turn / turn_limit, deviation / deviation_limit),
                    terrain_ratio,
                    tail_error,
                    0 if piece == preferred_piece else 1,
                    abs(preferred_end_distance - end_distance),
                    -piece.length_metres,
                )
            candidates.append((score, piece, endpoint))

        if not candidates:
            if current >= minimum_end_distance - 0.05:
                break
            piece = ordered[-1]
            target_distance = min(preferred_end_distance, measure.total)
            target_x, target_z, target_heading = measure.point(target_distance)
            dx, dz = target_x - start_x, target_z - start_z
            length = math.hypot(dx, dz)
            if length <= 1.0e-9:
                angle = math.radians(target_heading)
                dx, dz, length = math.sin(angle), math.cos(angle), 1.0
            fitted.append((
                piece,
                (start_x, start_z),
                (
                    start_x + dx / length * piece.length_metres,
                    start_z + dz / length * piece.length_metres,
                ),
            ))
            break

        _score, piece, endpoint = min(candidates, key=lambda item: item[0])
        end_distance, end_x, end_z, _heading = endpoint
        fitted.append((piece, (start_x, start_z), (end_x, end_z)))
        if end_distance <= current + 1.0e-7:
            break
        current = end_distance

    return tuple(fitted)


def _quality_aware_plan_run(job: Any):
    run = job.run
    measure = _playability._PolylineMeasure.create(run)
    total_length = measure.total
    start_key = _playability._road_node_key(run[0])
    end_key = _playability._road_node_key(run[-1])
    if total_length <= 0.05:
        return _parallel._RunPlan(
            job.order,
            job.feature_index,
            job.run_index,
            run,
            start_key,
            end_key,
            job.start_cover,
            job.end_cover,
            (),
            False,
            1,
        )

    start_distance = min(total_length, job.start_trim)
    preferred_end = max(start_distance, total_length - job.end_trim)
    minimum_end = max(start_distance, total_length - job.end_cover)
    shortest = min(piece.length_metres for piece in job.variants)
    maximum_end = total_length + (
        0.70 if job.end_cover > 0.0 else shortest * 0.5
    )
    fitted_pieces = _batched_quality_chain(
        measure,
        job.variants,
        start_distance=start_distance,
        preferred_end_distance=preferred_end,
        minimum_end_distance=minimum_end,
        maximum_end_distance=maximum_end,
    )

    covered_by_hubs = False
    skipped = 0
    if not fitted_pieces:
        covered_by_hubs = (
            total_length <= job.start_cover + job.end_cover + 1.0e-6
        )
        if not covered_by_hubs or job.cap_surface_mismatch:
            fitted_pieces = _playability._short_run_fallback_piece(
                measure,
                job.variants,
                start_trim=job.start_trim,
                end_trim=job.end_trim,
            )
            if fitted_pieces:
                covered_by_hubs = False
        if not fitted_pieces:
            skipped = 1

    placeable = []
    for piece, start_point, end_point in fitted_pieces:
        start_x, start_z = start_point
        end_x, end_z = end_point
        if math.hypot(end_x - start_x, end_z - start_z) <= 0.05:
            continue
        centre_x = (start_x + end_x) * 0.5
        centre_z = (start_z + end_z) * 0.5
        if not (
            0.0 <= centre_x < job.world_size
            and 0.0 <= centre_z < job.world_size
        ):
            continue
        placeable.append((piece, start_point, end_point))

    return _parallel._RunPlan(
        job.order,
        job.feature_index,
        job.run_index,
        run,
        start_key,
        end_key,
        job.start_cover,
        job.end_cover,
        tuple(placeable),
        covered_by_hubs,
        skipped,
    )


def _install_worker_quality_context(context: Any) -> None:
    if context is not None:
        _quality._CONTEXT.set(context)


def _quality_aware_execute_run_jobs(jobs, progress_callback=None):
    if not jobs:
        return ()
    workers = _parallel._worker_count(len(jobs))
    if workers <= 1:
        return tuple(_quality_aware_plan_run(job) for job in jobs)

    batch_size = max(
        8,
        min(48, math.ceil(len(jobs) / (workers * 12))),
    )
    batches = tuple(
        tuple(jobs[offset:offset + batch_size])
        for offset in range(0, len(jobs), batch_size)
    )
    ordered = [None] * len(jobs)
    completed_jobs = 0
    quality_context = _quality._CONTEXT.get()

    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_install_worker_quality_context,
            initargs=(quality_context,),
        ) as executor:
            futures = {
                executor.submit(_parallel._plan_run_batch, batch): len(batch)
                for batch in batches
            }
            for future in as_completed(futures):
                plans = future.result()
                for plan in plans:
                    ordered[plan.order] = plan
                completed_jobs += len(plans)
                if progress_callback is not None:
                    local = 35 + round(completed_jobs / len(jobs) * 23)
                    progress_callback(
                        min(58, local),
                        f"Planning stock road lines {completed_jobs:,}/{len(jobs):,} chains "
                        f"with {workers} workers",
                    )
    except (OSError, RuntimeError, BrokenPipeError):
        return tuple(_quality_aware_plan_run(job) for job in jobs)

    return tuple(plan for plan in ordered if plan is not None)


def install_road_quality_parallel_compat_policy() -> None:
    """Make the parallel planner quality-aware before bridge wrappers capture it."""

    global _INSTALLED, _ORIGINAL_PARALLEL_PLAN_RUN, _ORIGINAL_PARALLEL_EXECUTE
    if _INSTALLED:
        return
    _ORIGINAL_PARALLEL_PLAN_RUN = _parallel._plan_run
    _ORIGINAL_PARALLEL_EXECUTE = _parallel._execute_run_jobs
    _parallel._plan_run = _quality_aware_plan_run
    _parallel._execute_run_jobs = _quality_aware_execute_run_jobs
    _playability._stock_piece_chain = _batched_quality_chain
    _INSTALLED = True
