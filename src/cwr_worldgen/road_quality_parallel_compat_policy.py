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

Paved-junction fallback can refit the same network several times. Consecutive
passes therefore retain exact per-run plans when the run geometry, stock variants,
terrain/spec identity and endpoint junction state are unchanged. Changed junction
arms are replanned while unrelated chains are reused in the parent process.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import math
from typing import Any, Sequence

from . import playability as _playability
from . import road_chain_parallel_policy as _parallel
from . import road_quality_policy as _quality

_INSTALLED = False
_ORIGINAL_PARALLEL_PLAN_RUN: Any = None
_ORIGINAL_PARALLEL_EXECUTE: Any = None

# Reuse only across consecutive refits of the same immutable terrain/spec objects.
# Holding one previous pass is enough for the bounded paved-junction stabilization
# loop and avoids turning this into an accidental long-lived world cache.
_PLAN_CACHE_ELEVATIONS: Any = None
_PLAN_CACHE_SPEC: Any = None
_RUN_PLAN_CACHE: dict[tuple[Any, ...], Any] = {}

# Terrain sampling is batched per chain decision and deduplicated in a chain-local
# cache. Tiny 8-20 point batches are deliberately kept on the scalar sampler: the
# NumPy setup cost is larger than the bilinear arithmetic at this scale.


def _clear_run_plan_cache() -> None:
    global _PLAN_CACHE_ELEVATIONS, _PLAN_CACHE_SPEC, _RUN_PLAN_CACHE
    _PLAN_CACHE_ELEVATIONS = None
    _PLAN_CACHE_SPEC = None
    _RUN_PLAN_CACHE = {}


def _sample_terrain_points(
    context: Any,
    points: Sequence[tuple[float, float]],
    cache: dict[tuple[float, float], float],
) -> None:
    """Fill ``cache`` once for every unique terrain coordinate in a batch."""

    missing = tuple(
        point
        for point in dict.fromkeys(
            (float(point[0]), float(point[1])) for point in points
        )
        if point not in cache
    )
    if not missing:
        return

    elevations = context.elevations
    cells = int(context.spec.cells)
    cell_size = float(context.spec.cell_size)
    for point in missing:
        cache[point] = _playability._sample_elevation(
            elevations, cells, cell_size, point[0], point[1]
        )


def _batched_terrain_bulges(
    context: Any,
    start: tuple[float, float],
    candidates: Sequence[tuple[tuple[float, float], int]],
    sample_cache: dict[tuple[float, float], float] | None = None,
) -> tuple[float, ...]:
    """Evaluate candidate terrain bulges with one deduplicated height batch.

    The scalar quality scorer samples the same start point once per candidate and
    often samples a previous endpoint again at the next chain step. Keep those
    exact coordinates in a chain-local cache and evaluate each missing point only
    once while preserving the historical scalar bilinear sampler exactly.
    """

    if not candidates:
        return ()
    cache = {} if sample_cache is None else sample_cache
    start = float(start[0]), float(start[1])
    sample_points: list[tuple[float, float]] = []
    candidate_points: list[
        tuple[tuple[float, float], tuple[tuple[float, float], ...]]
    ] = []

    for end, nominal in candidates:
        end = float(end[0]), float(end[1])
        nominal = int(nominal)
        if nominal <= 6:
            candidate_points.append((end, ()))
            continue
        fractions = (0.50,) if nominal <= 12 else (0.25, 0.50, 0.75)
        interior = tuple(
            (
                start[0] + (end[0] - start[0]) * fraction,
                start[1] + (end[1] - start[1]) * fraction,
            )
            for fraction in fractions
        )
        candidate_points.append((end, interior))
        sample_points.extend((start, end, *interior))

    _sample_terrain_points(context, sample_points, cache)
    result: list[float] = []
    for (end, interior), (_candidate_end, nominal) in zip(
        candidate_points, candidates
    ):
        if int(nominal) <= 6:
            result.append(0.0)
            continue
        h0 = cache[start]
        h1 = cache[end]
        bulge = 0.0
        fractions = (0.50,) if int(nominal) <= 12 else (0.25, 0.50, 0.75)
        for fraction, point in zip(fractions, interior):
            terrain = cache[point]
            plane = h0 * (1.0 - fraction) + h1 * fraction
            bulge = max(bulge, terrain - plane)
        result.append(bulge)
    return tuple(result)


def _batched_tail_error(
    measure: Any,
    pieces: Sequence[Any],
    current: float,
    preferred_end: float,
    maximum_end: float,
    depth: int,
    cache: dict[tuple[float, int], float],
) -> float:
    """Exact tail lookahead using memoized, batched chord endpoint searches."""

    key = float(current), int(depth)
    cached = cache.get(key)
    if cached is not None:
        return cached

    best = abs(preferred_end - current)
    if depth <= 0 or current >= preferred_end - 0.05:
        cache[key] = best
        return best

    endpoints = _parallel._batch_chord_endpoints(
        measure,
        current,
        tuple(piece.length_metres for piece in pieces),
        maximum_end,
    )
    for piece in pieces:
        endpoint = endpoints.get(float(piece.length_metres))
        if endpoint is None or endpoint[0] <= current + 1.0e-7:
            continue
        best = min(
            best,
            _batched_tail_error(
                measure,
                pieces,
                endpoint[0],
                preferred_end,
                maximum_end,
                depth - 1,
                cache,
            ),
        )
    cache[key] = best
    return best


def _junction_signature(context: Any, key: tuple[int, int]) -> Any:
    junction = context.junctions.get(key)
    if junction is None:
        return None
    return (
        tuple(float(value) for value in junction.point),
        tuple(float(value) for value in junction.axis),
        float(junction.half_length),
        float(junction.half_width),
        tuple(
            tuple(float(value) for value in direction)
            for direction in junction.directions
        ),
    )


def _variant_signature(variants: Sequence[Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            str(piece.model_path),
            float(piece.length_metres),
            int(piece.nominal_length),
        )
        for piece in variants
    )


def _run_plan_key(job: Any, context: Any) -> tuple[Any, ...]:
    start_key = _playability._road_node_key(job.run[0])
    end_key = _playability._road_node_key(job.run[-1])
    return (
        tuple((float(x), float(z)) for x, z in job.run),
        _variant_signature(job.variants),
        float(job.start_trim),
        float(job.end_trim),
        float(job.start_cover),
        float(job.end_cover),
        bool(job.cap_surface_mismatch),
        bool(getattr(job, "suppress_short_fallback", False)),
        bool(getattr(job, "hard_stop_at_preferred_end", False)),
        float(job.world_size),
        _junction_signature(context, start_key),
        _junction_signature(context, end_key),
    )


def _rebind_cached_plan(plan: Any, job: Any) -> Any:
    """Copy cached geometry while preserving this pass's ordering metadata."""

    return replace(
        plan,
        order=job.order,
        feature_index=job.feature_index,
        run_index=job.run_index,
        run=job.run,
        start_cover=job.start_cover,
        end_cover=job.end_cover,
    )


def _batched_quality_chain(
    measure: Any,
    pieces: Sequence[Any],
    *,
    start_distance: float,
    preferred_end_distance: float,
    minimum_end_distance: float,
    maximum_end_distance: float,
):
    """Exact ``road_quality_policy._quality_chain`` with batched hot paths."""

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
    terrain_samples: dict[tuple[float, float], float] = {}
    tail_cache: dict[tuple[float, int], float] = {}

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
        prepared = []

        for piece in ordered:
            endpoint = endpoints.get(float(piece.length_metres))
            if endpoint is None:
                continue
            end_distance, end_x, end_z, chord_heading = endpoint
            end_heading = measure.heading_before(end_distance)
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
            joint_turn = 0.0
            joint_edge = 0.0
            joint_penalty = 0
            if (
                fitted
                and _quality._is_stock_paved_piece(piece)
                and _quality._is_stock_paved_piece(fitted[-1][0])
            ):
                previous_heading = _quality._piece_chord_heading(
                    fitted[-1][1], fitted[-1][2]
                )
                joint_turn = _playability._heading_difference(
                    previous_heading, chord_heading
                )
                joint_edge = _quality._stock_paved_joint_edge_discontinuity(
                    fitted[-1][0],
                    previous_heading,
                    piece,
                    chord_heading,
                )
                joint_penalty = int(
                    joint_edge
                    > _quality._STOCK_PAVED_MAX_EDGE_DISCONTINUITY_METRES
                )
            prepared.append((
                piece,
                endpoint,
                gravel,
                turn_limit,
                deviation_limit,
                fidelity_penalty,
                turn,
                deviation,
                joint_penalty,
                joint_turn,
                joint_edge,
            ))

        bulges = _batched_terrain_bulges(
            context,
            (start_x, start_z),
            tuple(
                ((row[1][1], row[1][2]), row[0].nominal_length)
                for row in prepared
            ),
            terrain_samples,
        )
        candidates = []
        for row, bulge in zip(prepared, bulges):
            (
                piece,
                endpoint,
                gravel,
                turn_limit,
                deviation_limit,
                fidelity_penalty,
                turn,
                deviation,
                joint_penalty,
                joint_turn,
                joint_edge,
            ) = row
            end_distance, _end_x, _end_z, _chord_heading = endpoint
            terrain_limit = (
                _quality._GRAVEL_BULGE_LIMIT
                if gravel else _quality._STOCK_BULGE_LIMIT
            )
            terrain_penalty = int(bulge > terrain_limit)
            terrain_ratio = bulge / max(0.001, terrain_limit)
            tail_error = (
                _batched_tail_error(
                    measure,
                    ordered,
                    end_distance,
                    preferred_end_distance,
                    maximum_end_distance,
                    _quality._LOOKAHEAD_DEPTH,
                    tail_cache,
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
                    joint_penalty,
                    tail_penalty,
                    terrain_penalty,
                    max(turn / turn_limit, deviation / deviation_limit),
                    (
                        joint_edge
                        / _quality._STOCK_PAVED_MAX_EDGE_DISCONTINUITY_METRES
                        if joint_penalty else 0.0
                    ),
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
            if maximum_end_distance <= preferred_end_distance + 1.0e-7:
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
            job.order, job.feature_index, job.run_index, run,
            start_key, end_key, job.start_cover, job.end_cover,
            (), False, 1,
        )

    start_distance = min(total_length, job.start_trim)
    preferred_end = max(start_distance, total_length - job.end_trim)
    minimum_end = max(start_distance, total_length - job.end_cover)
    shortest = min(piece.length_metres for piece in job.variants)
    maximum_end = (
        preferred_end
        if getattr(job, "hard_stop_at_preferred_end", False)
        else total_length + (
            0.70 if job.end_cover > 0.0 else shortest * 0.5
        )
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
        suppress_short_fallback = bool(
            getattr(job, "suppress_short_fallback", False)
        )
        covered_by_hubs = (
            not suppress_short_fallback
            and total_length <= job.start_cover + job.end_cover + 1.0e-6
        )
        if (
            not suppress_short_fallback
            and (not covered_by_hubs or job.cap_surface_mismatch)
        ):
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
    global _PLAN_CACHE_ELEVATIONS, _PLAN_CACHE_SPEC, _RUN_PLAN_CACHE

    if not jobs:
        return ()

    quality_context = _quality._CONTEXT.get()
    use_cache = quality_context is not None
    if use_cache and (
        _PLAN_CACHE_ELEVATIONS is not quality_context.elevations
        or _PLAN_CACHE_SPEC is not quality_context.spec
    ):
        _clear_run_plan_cache()
        _PLAN_CACHE_ELEVATIONS = quality_context.elevations
        _PLAN_CACHE_SPEC = quality_context.spec

    ordered = [None] * len(jobs)
    pending = []
    current_keys: dict[int, tuple[Any, ...]] = {}
    reused = 0

    for job in jobs:
        key = _run_plan_key(job, quality_context) if use_cache else None
        if key is not None:
            current_keys[job.order] = key
            cached = _RUN_PLAN_CACHE.get(key)
            if cached is not None:
                ordered[job.order] = _rebind_cached_plan(cached, job)
                reused += 1
                continue
        pending.append(job)

    if reused and progress_callback is not None:
        progress_callback(
            35 + round(reused / len(jobs) * 23),
            f"Reusing {reused:,}/{len(jobs):,} unchanged stock road chain plans; "
            f"planning {len(pending):,} changed chains",
        )

    if pending:
        workers = _parallel._worker_count(len(pending))
        if workers <= 1:
            plans = tuple(_quality_aware_plan_run(job) for job in pending)
            for plan in plans:
                ordered[plan.order] = plan
        else:
            batch_size = max(
                8,
                min(48, math.ceil(len(pending) / (workers * 12))),
            )
            batches = tuple(
                tuple(pending[offset:offset + batch_size])
                for offset in range(0, len(pending), batch_size)
            )
            completed_jobs = reused
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
                for job in pending:
                    ordered[job.order] = _quality_aware_plan_run(job)

    result = tuple(plan for plan in ordered if plan is not None)
    if use_cache and len(result) == len(jobs):
        # Keep only the current pass. This is enough for the next stabilization
        # refit and bounds memory even when endpoint junction state keeps changing.
        _RUN_PLAN_CACHE = {
            current_keys[plan.order]: plan
            for plan in result
            if plan.order in current_keys
        }
        _PLAN_CACHE_ELEVATIONS = quality_context.elevations
        _PLAN_CACHE_SPEC = quality_context.spec
        if reused == len(jobs) and progress_callback is not None:
            progress_callback(
                58,
                f"Reused all {len(jobs):,} stock road chain plans",
            )
    return result


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
