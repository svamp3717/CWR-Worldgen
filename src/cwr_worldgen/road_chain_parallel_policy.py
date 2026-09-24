# SPDX-License-Identifier: GPL-3.0-or-later
"""Batch and parallelize independent stock-road chain planning.

The stock road planner is deterministic but historically planned every split road
run serially. Dense worlds can contain thousands of independent runs and each
run tests the same 3/4 stock piece lengths against the same polyline segments.
This policy keeps final object ordering and IDs in the parent process while:

* scanning each polyline segment once for all candidate piece lengths;
* planning independent runs in a bounded process pool;
* collecting results by their original feature/run order before object emission.

Only pure geometry planning is sent to workers. Terrain arrays stay in the parent
process so a large height grid is not copied into every worker.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import math
import multiprocessing
import os
from typing import Any, Sequence

from . import playability as _playability

_INSTALLED = False
_ORIGINAL_STOCK_FIT: Any = None
_ORIGINAL_STOCK_CHAIN: Any = None


@dataclass(frozen=True, slots=True)
class _RunJob:
    order: int
    feature_index: int
    run_index: int
    run: tuple[tuple[float, float], ...]
    variants: tuple[Any, ...]
    start_trim: float
    end_trim: float
    start_cover: float
    end_cover: float
    cap_surface_mismatch: bool
    world_size: float


@dataclass(frozen=True, slots=True)
class _RunPlan:
    order: int
    feature_index: int
    run_index: int
    run: tuple[tuple[float, float], ...]
    start_key: tuple[int, int]
    end_key: tuple[int, int]
    start_cover: float
    end_cover: float
    fitted_pieces: tuple[Any, ...]
    covered_by_hubs: bool
    skipped_short_runs: int


def _ordered_pieces(pieces: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(sorted(
        pieces,
        key=lambda piece: (-piece.length_metres, piece.model_path.casefold()),
    ))


def _preferred_piece(remaining: float, ordered: Sequence[Any]) -> Any:
    shortest = min(piece.length_metres for piece in ordered)
    for piece in ordered:
        if piece.length_metres <= remaining + shortest * 0.5:
            return piece
    return ordered[-1]


def _batch_chord_endpoints(
    measure: Any,
    start_distance: float,
    chord_lengths: Sequence[float],
    maximum_distance: float,
) -> dict[float, tuple[float, float, float, float]]:
    """Find the first circle/polyline hit for every requested chord in one scan.

    ``_PolylineMeasure.chord_endpoint`` historically walks the same breakpoints
    once for every 25/12/6/3 m candidate. Here each source segment is visited
    once and all still-unresolved radii are tested against it. The quadratic and
    root ordering are intentionally identical to the scalar implementation.
    """

    lengths = tuple(dict.fromkeys(
        float(value) for value in chord_lengths if float(value) > 0.0
    ))
    if not lengths or maximum_distance <= start_distance + 1.0e-9:
        return {}

    from bisect import bisect_left, bisect_right

    origin_x, origin_z, _ = measure.point(start_distance)
    lower = bisect_right(measure.cumulative, start_distance + 1.0e-9)
    upper = bisect_left(measure.cumulative, maximum_distance - 1.0e-9)
    breakpoints = (
        start_distance,
        *measure.cumulative[lower:upper],
        maximum_distance,
    )
    unresolved = {length: length * length for length in lengths}
    results: dict[float, tuple[float, float, float, float]] = {}

    for distance0, distance1 in zip(breakpoints, breakpoints[1:]):
        if not unresolved:
            break
        ax, az, _ = measure.point(distance0)
        bx, bz, _ = measure.point(distance1)
        vx, vz = bx - ax, bz - az
        denominator = vx * vx + vz * vz
        if denominator <= 1.0e-12:
            continue
        ox, oz = ax - origin_x, az - origin_z
        linear = 2.0 * (ox * vx + oz * vz)
        base_constant = ox * ox + oz * oz

        for chord_length, radius_squared in tuple(unresolved.items()):
            constant = base_constant - radius_squared
            discriminant = linear * linear - 4.0 * denominator * constant
            if discriminant < -1.0e-8:
                continue
            root = math.sqrt(max(0.0, discriminant))
            fractions = sorted((
                (-linear - root) / (2.0 * denominator),
                (-linear + root) / (2.0 * denominator),
            ))
            for fraction in fractions:
                if fraction < -1.0e-8 or fraction > 1.0 + 1.0e-8:
                    continue
                fraction = max(0.0, min(1.0, fraction))
                distance = distance0 + (distance1 - distance0) * fraction
                if distance <= start_distance + 1.0e-7:
                    continue
                x = ax + vx * fraction
                z = az + vz * fraction
                heading = math.degrees(
                    math.atan2(x - origin_x, z - origin_z)
                ) % 360.0
                results[chord_length] = (distance, x, z, heading)
                unresolved.pop(chord_length, None)
                break
    return results


def _batched_stock_piece_chain(
    measure: Any,
    pieces: Sequence[Any],
    *,
    start_distance: float,
    preferred_end_distance: float,
    minimum_end_distance: float,
    maximum_end_distance: float,
):
    """Exact stock-piece chain selection with batched chord intersection scans."""

    if not pieces or preferred_end_distance <= start_distance + 0.05:
        return ()
    ordered = _ordered_pieces(pieces)
    shortest = min(piece.length_metres for piece in ordered)
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

        preferred_piece = _preferred_piece(remaining, ordered)
        start_x, start_z, start_heading = measure.point(current)
        endpoints = _batch_chord_endpoints(
            measure,
            current,
            tuple(piece.length_metres for piece in ordered),
            maximum_end_distance,
        )
        candidates: list[tuple[tuple[float, ...], Any, tuple[float, float, float, float]]] = []

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
            if _playability.is_generated_gravel_road_model(piece.model_path):
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
                not (turn <= turn_limit and deviation <= deviation_limit)
            )
            if _playability.is_generated_gravel_road_model(piece.model_path):
                score = (
                    fidelity_penalty,
                    0 if piece == preferred_piece else 1,
                    -piece.length_metres,
                    max(turn / turn_limit, deviation / deviation_limit),
                    turn,
                    deviation,
                    abs(preferred_end_distance - end_distance),
                )
            else:
                score = (
                    fidelity_penalty,
                    max(turn / turn_limit, deviation / deviation_limit),
                    0 if piece == preferred_piece else 1,
                    turn,
                    deviation,
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
            end = (
                start_x + dx / length * piece.length_metres,
                start_z + dz / length * piece.length_metres,
            )
            fitted.append((piece, (start_x, start_z), end))
            break

        _score, piece, endpoint = min(candidates, key=lambda item: item[0])
        end_distance, end_x, end_z, _heading = endpoint
        fitted.append((piece, (start_x, start_z), (end_x, end_z)))
        if end_distance <= current + 1.0e-7:
            break
        current = end_distance
    return tuple(fitted)


def _plan_run(job: _RunJob) -> _RunPlan:
    run = job.run
    measure = _playability._PolylineMeasure.create(run)
    total_length = measure.total
    start_key = _playability._road_node_key(run[0])
    end_key = _playability._road_node_key(run[-1])
    if total_length <= 0.05:
        return _RunPlan(
            job.order, job.feature_index, job.run_index, run,
            start_key, end_key, job.start_cover, job.end_cover,
            (), False, 1,
        )

    start_distance = min(total_length, job.start_trim)
    preferred_end = max(start_distance, total_length - job.end_trim)
    minimum_end = max(start_distance, total_length - job.end_cover)
    shortest = min(piece.length_metres for piece in job.variants)
    maximum_end = total_length + (0.70 if job.end_cover > 0.0 else shortest * 0.5)
    fitted_pieces = _batched_stock_piece_chain(
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
        covered_by_hubs = total_length <= job.start_cover + job.end_cover + 1.0e-6
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

    return _RunPlan(
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


def _plan_run_batch(batch: tuple[_RunJob, ...]) -> tuple[_RunPlan, ...]:
    return tuple(_plan_run(job) for job in batch)


def _worker_count(job_count: int) -> int:
    raw = os.environ.get("CWR_WORLDGEN_ROAD_WORKERS", "").strip()
    if raw:
        try:
            requested = max(1, int(raw))
        except ValueError:
            requested = 1
    else:
        cpu = os.cpu_count() or 1
        requested = max(1, min(8, cpu - 1 if cpu > 2 else cpu))
    if job_count < 96 or multiprocessing.current_process().daemon:
        return 1
    return min(requested, max(1, job_count // 24))


def _execute_run_jobs(
    jobs: Sequence[_RunJob],
    progress_callback=None,
) -> tuple[_RunPlan, ...]:
    if not jobs:
        return ()
    workers = _worker_count(len(jobs))
    if workers <= 1:
        return tuple(_plan_run(job) for job in jobs)

    batch_size = max(8, min(48, math.ceil(len(jobs) / (workers * 12))))
    batches = tuple(
        tuple(jobs[offset:offset + batch_size])
        for offset in range(0, len(jobs), batch_size)
    )
    ordered: list[_RunPlan | None] = [None] * len(jobs)
    completed_jobs = 0
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_plan_run_batch, batch): len(batch)
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
        # Frozen/embedded Python can forbid child creation. Fall back to the same
        # deterministic batched planner rather than failing the world build.
        return tuple(_plan_run(job) for job in jobs)

    return tuple(plan for plan in ordered if plan is not None)


def _fit_stock_piece_road_objects_parallel(
    dataset,
    projection,
    elevations,
    spec,
    *,
    starting_id: int = 1,
    progress_callback=None,
):
    """Drop-in stock-road fitter with parallel pure-geometry run planning."""

    if spec.max_road_objects == 0:
        if progress_callback is not None:
            progress_callback(100, "Stock road placement disabled by zero object budget")
        return _playability.RoadFitReport(
            objects=(), chain_count=0, connection_count=0,
            failed_connections=0, maximum_connection_gap=0.0,
            maximum_chain_gap=0.0, truncated=False,
        )

    projected_features = []
    incidents: dict[tuple[int, int], list[Any]] = {}
    node_positions: dict[tuple[int, int], tuple[float, float]] = {}
    bend_keys: set[tuple[int, int]] = set()

    total_roads = len(dataset.roads)
    if progress_callback is not None:
        progress_callback(0, f"Projecting {total_roads:,} normalized road lines")
    projection_step = max(1, total_roads // 20)
    road_polylines = _playability.projected_road_polylines(dataset, projection)
    for feature_index, (feature, projected_points) in enumerate(
        zip(dataset.roads, road_polylines), start=1
    ):
        if progress_callback is not None and (
            feature_index == total_roads or feature_index % projection_step == 0
        ):
            progress_callback(
                min(20, round(feature_index / max(1, total_roads) * 20)),
                f"Projected road lines {feature_index:,}/{total_roads:,}",
            )
        if not _playability.road_is_supported(
            feature.tags, include_minor=spec.include_minor_roads
        ):
            continue
        points = tuple(_playability._clean_road_points(projected_points))
        if len(points) < 2:
            continue
        if (
            bool(getattr(spec, "bridges_enabled", True))
            and not bool(getattr(spec, "procedural_bridges", True))
            and _playability._road_is_explicit_bridge(feature.tags)
            and not _playability.road_bridge_crosses_ditch_only(
                feature, dataset, projection
            )
            and _playability.road_span_has_in_game_water(
                points,
                elevations,
                cells=spec.cells,
                cell_size=spec.cell_size,
                sea_level=spec.sea_level,
                width=max(6.0, _playability.road_width_metres(feature.tags)),
            )
        ):
            continue
        dirt = _playability.road_is_dirt(feature.tags)
        model = _playability.road_model_for_tags(spec, feature.tags)
        width = max(6.0, _playability.road_width_metres(feature.tags))
        projected_features.append((feature, model, dirt, width, points))
        for index, (start_point, end_point) in enumerate(zip(points, points[1:])):
            if math.dist(start_point, end_point) <= 0.05:
                continue
            start_key = _playability._road_node_key(start_point)
            end_key = _playability._road_node_key(end_point)
            segment_key = f"{feature.osm_key}/{index:06d}"
            forward = _playability._normalised_direction(start_point, end_point)
            reverse = (-forward[0], -forward[1])
            incidents.setdefault(start_key, []).append(
                (forward, dirt, model, segment_key, feature.osm_key)
            )
            incidents.setdefault(end_key, []).append(
                (reverse, dirt, model, segment_key, feature.osm_key)
            )
            node_positions.setdefault(start_key, start_point)
            node_positions.setdefault(end_key, end_point)
        for index in range(1, len(points) - 1):
            if _playability._turn_degrees(
                points[index - 1], points[index], points[index + 1]
            ) >= 15.0:
                bend_keys.add(_playability._road_node_key(points[index]))

    effective_incidents = {
        key: _playability._unique_incidents(values)
        for key, values in incidents.items()
    }
    degree_two_turn_keys: set[tuple[int, int]] = set()
    for key, values in effective_incidents.items():
        if len(values) != 2:
            continue
        first, second = values[0][0], values[1][0]
        dot = max(-1.0, min(1.0, first[0] * second[0] + first[1] * second[1]))
        outgoing_angle = math.degrees(math.acos(dot))
        if abs(180.0 - outgoing_angle) >= 15.0:
            degree_two_turn_keys.add(key)

    true_junction_keys = {
        key for key, values in effective_incidents.items() if 3 <= len(values) <= 4
    }
    complex_keys = {
        key for key, values in effective_incidents.items() if len(values) > 4
    }
    candidate_cap_keys = true_junction_keys - complex_keys
    if progress_callback is not None:
        progress_callback(
            24,
            f"Classified {len(candidate_cap_keys):,} real road junctions; "
            f"{len(degree_two_turn_keys | bend_keys):,} ordinary bends use rounded piece chains",
        )

    cap_keys = set(candidate_cap_keys)
    suppressed_nearby_hubs = 0
    degree_two_keys = {
        key for key, values in effective_incidents.items() if len(values) == 2
    }
    suppressed_degree_two_caps = len(degree_two_keys)
    variant_cache: dict[str, tuple[Any, ...]] = {}

    def variants_for(model_path: str) -> tuple[Any, ...]:
        variants = variant_cache.get(model_path)
        if variants is None:
            variants = _playability.road_model_variants(
                model_path, spec.road_segment_length
            )
            if _playability.is_generated_gravel_road_model(model_path):
                variants = tuple(
                    piece for piece in variants
                    if piece.nominal_length in {12, 6, 3}
                ) or variants[-1:]
            variant_cache[model_path] = variants
        return variants

    cap_plans: dict[tuple[int, int], Any] = {}
    cap_trim_lengths: dict[tuple[int, int], float] = {}
    cap_cover_lengths: dict[tuple[int, int], float] = {}
    for key in sorted(cap_keys):
        values = effective_incidents[key]
        use_dirt = all(value[1] for value in values)
        all_gravel = all(
            _playability.is_generated_gravel_road_model(value[2])
            for value in values
        )
        all_paved = all(not value[1] for value in values) and not all_gravel
        incident_models = {value[2].casefold(): value[2] for value in values}
        axis_override = None
        if all_gravel:
            degree = len(values)
            base_model = _playability.gravel_junction_model_path(
                spec.name, degree
            )
            hub_length = 5.4 if degree == 3 else 6.0
            cap_piece = _playability._RoadPiece(base_model, hub_length, 6)
        elif all_paved:
            headings, axis_override = (
                _playability.paved_junction_signature_for_directions(
                    tuple(value[0] for value in values)
                )
            )
            width = _playability.paved_junction_width_for_models(
                tuple(value[2] for value in values)
            )
            base_model = _playability.paved_junction_model_path(
                spec.name,
                width,
                headings,
            )
            hub_length = (
                _playability.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES * 2.0
            )
            cap_piece = _playability._RoadPiece(base_model, hub_length, 6)
        else:
            if len(incident_models) == 1:
                base_model = next(iter(incident_models.values()))
            else:
                base_model = (
                    spec.dirt_road_model if use_dirt else spec.paved_road_model
                )
            variants = variants_for(base_model)
            cap_piece = next(
                (piece for piece in variants if piece.nominal_length == 6),
                variants[-1],
            )
        dominant_values = tuple(
            (value[0], value[1], value[2], value[3]) for value in values
        )
        axis = (
            axis_override
            if axis_override is not None
            else _playability._dominant_node_axis(dominant_values)
        )
        node = node_positions[key]
        half = cap_piece.length_metres * 0.5
        start_point = (node[0] - axis[0] * half, node[1] - axis[1] * half)
        end_point = (node[0] + axis[0] * half, node[1] + axis[1] * half)
        cap_plans[key] = (cap_piece, start_point, end_point)
        if all_paved:
            cap_trim_lengths[key] = half
            cap_cover_lengths[key] = half + 0.05
        else:
            cap_trim_lengths[key] = max(0.40, half - 0.70)
            cap_cover_lengths[key] = half + 0.15

    virtual_short_length = max(
        6.25,
        float(spec.road_segment_length) * 6.0 / 25.0,
    )
    virtual_cover_lengths = {
        key: virtual_short_length * 0.5 + 0.15
        for key in complex_keys
    }
    virtual_trim_lengths = {
        key: max(0.40, cover - 0.85)
        for key, cover in virtual_cover_lengths.items()
    }
    split_keys = cap_keys | complex_keys

    jobs: list[_RunJob] = []
    feature_job_counts = [0] * len(projected_features)
    order = 0
    for feature_index, (feature, model, _dirt, _width, points) in enumerate(
        projected_features
    ):
        variants = variants_for(model)
        variant_paths = {piece.model_path.casefold() for piece in variants}
        for run_index, raw_run in enumerate(
            _playability._split_polyline_at_keys(points, split_keys)
        ):
            run = tuple(_playability._rounded_road_run(raw_run))
            if len(run) < 2:
                continue
            start_key = _playability._road_node_key(run[0])
            end_key = _playability._road_node_key(run[-1])
            start_trim = cap_trim_lengths.get(
                start_key, virtual_trim_lengths.get(start_key, 0.0)
            )
            end_trim = cap_trim_lengths.get(
                end_key, virtual_trim_lengths.get(end_key, 0.0)
            )
            start_cover = cap_cover_lengths.get(
                start_key, virtual_cover_lengths.get(start_key, 0.0)
            )
            end_cover = cap_cover_lengths.get(
                end_key, virtual_cover_lengths.get(end_key, 0.0)
            )
            cap_surface_mismatch = any(
                key in cap_plans
                and cap_plans[key][0].model_path.casefold() not in variant_paths
                for key in (start_key, end_key)
            )
            jobs.append(_RunJob(
                order=order,
                feature_index=feature_index,
                run_index=run_index,
                run=run,
                variants=variants,
                start_trim=start_trim,
                end_trim=end_trim,
                start_cover=start_cover,
                end_cover=end_cover,
                cap_surface_mismatch=cap_surface_mismatch,
                world_size=float(spec.world_size),
            ))
            feature_job_counts[feature_index] += 1
            order += 1

    if progress_callback is not None:
        workers = _worker_count(len(jobs))
        progress_callback(
            35,
            f"Planning stock road lines across {len(jobs):,} independent chains"
            + (f" with {workers} workers" if workers > 1 else ""),
        )
    run_plans = _execute_run_jobs(jobs, progress_callback=progress_callback)

    grouped: list[list[_RunPlan]] = [[] for _ in projected_features]
    skipped_short_runs = 0
    required_chain_objects = 0
    for plan in run_plans:
        grouped[plan.feature_index].append(plan)
        skipped_short_runs += plan.skipped_short_runs
        required_chain_objects += len(plan.fitted_pieces)
    for values in grouped:
        values.sort(key=lambda plan: plan.run_index)

    planned_features = []
    for feature_index, (feature, _model, _dirt, _width, _points) in enumerate(
        projected_features
    ):
        feature_plans = tuple(
            (
                plan.run,
                plan.start_key,
                plan.end_key,
                plan.start_cover,
                plan.end_cover,
                plan.fitted_pieces,
                plan.covered_by_hubs,
            )
            for plan in grouped[feature_index]
        )
        planned_features.append((feature, feature_plans))

    planned_features.sort(
        key=lambda item: (
            _playability._road_surface_priority(item[0].tags),
            item[0].osm_key,
        )
    )
    required_objects = len(cap_plans) + required_chain_objects
    if progress_callback is not None:
        progress_callback(
            59,
            f"Road object plan requires {required_objects:,}; budget {spec.max_road_objects:,}",
        )
    if required_objects > spec.max_road_objects:
        if bool(getattr(spec, "advisory_object_limits", False)):
            if progress_callback is not None:
                progress_callback(
                    59,
                    "WARNING: road object warning threshold exceeded: "
                    f"requires {required_objects:,} objects, configured threshold is "
                    f"{spec.max_road_objects:,}. Continuing with the complete road network.",
                )
        else:
            raise ValueError(
                "road object budget is too small for a complete network: "
                f"requires {required_objects:,} objects, limit is {spec.max_road_objects:,}; "
                f"increase --max-road-objects to at least {required_objects:,}. "
                "Partial road networks are not emitted."
            )

    objects = []
    next_id = starting_id
    cap_objects: dict[tuple[int, int], Any] = {}
    maximum_pitch = 0.0
    if progress_callback is not None:
        progress_callback(61, f"Placing {len(cap_plans):,} junction caps")
    for key in sorted(cap_plans):
        cap_piece, start_point, end_point = cap_plans[key]
        obj = _playability._road_object_on_slope(
            next_id,
            cap_piece.model_path,
            start_point,
            end_point,
            elevations,
            spec,
            vertical_offset=0.060,
        )
        next_id += 1
        objects.append(obj)
        cap_objects[key] = obj
        maximum_pitch = max(maximum_pitch, abs(obj.pitch_degrees))

    maximum_chain_gap = 0.0
    maximum_model_overlap = 0.0
    maximum_endpoint_gap = 0.0
    short_piece_objects = 0
    chain_count = 0
    endpoint_axes: dict[tuple[int, int], list[Any]] = {}

    fitting_step = max(1, len(planned_features) // 30)
    for feature_index, (feature, feature_plans) in enumerate(planned_features, start=1):
        if progress_callback is not None and (
            feature_index == len(planned_features)
            or feature_index % fitting_step == 0
        ):
            local = 63 + round(feature_index / max(1, len(planned_features)) * 33)
            progress_callback(
                min(96, local),
                f"Fitting stock road lines {feature_index:,}/{len(planned_features):,}; "
                f"{len(objects):,}/{required_objects:,} objects",
            )
        for (
            run,
            start_key,
            end_key,
            start_cover,
            end_cover,
            fitted_pieces,
            covered_by_hubs,
        ) in feature_plans:
            if not fitted_pieces:
                if covered_by_hubs and start_key in cap_keys:
                    node = node_positions[start_key]
                    endpoint_axes.setdefault(start_key, []).append((node, node))
                if covered_by_hubs and end_key in cap_keys:
                    node = node_positions[end_key]
                    endpoint_axes.setdefault(end_key, []).append((node, node))
                continue
            chain_count += 1
            chain: list[tuple[Any, float]] = []
            for piece, start_point, end_point in fitted_pieces:
                placed_model = _playability._curved_gravel_model_for_run(
                    piece.model_path, run, start_point, end_point
                )
                obj = _playability._road_object_on_slope(
                    next_id,
                    placed_model,
                    start_point,
                    end_point,
                    elevations,
                    spec,
                    vertical_offset=_playability._road_vertical_offset(feature.tags),
                )
                next_id += 1
                objects.append(obj)
                chain.append((obj, piece.length_metres))
                if piece.nominal_length != 25:
                    short_piece_objects += 1
                maximum_pitch = max(maximum_pitch, abs(obj.pitch_degrees))

            for (previous, previous_length), (current, current_length) in zip(
                chain, chain[1:]
            ):
                previous_axis = _playability._model_axis(previous, previous_length)
                current_axis = _playability._model_axis(current, current_length)
                previous_end = previous_axis[1]
                current_start = current_axis[0]
                gap = math.dist(previous_end, current_start)
                maximum_chain_gap = max(maximum_chain_gap, gap)
                angle = math.radians(previous.heading_degrees)
                direction = (math.sin(angle), math.cos(angle))
                offset = (
                    current_start[0] - previous_end[0],
                    current_start[1] - previous_end[1],
                )
                lateral = abs(
                    direction[0] * offset[1] - direction[1] * offset[0]
                )
                longitudinal = direction[0] * offset[0] + direction[1] * offset[1]
                if lateral <= 0.10 and longitudinal < 0.0:
                    maximum_model_overlap = max(
                        maximum_model_overlap, -longitudinal
                    )

            first_axis = _playability._model_axis(chain[0][0], chain[0][1])
            last_axis = _playability._model_axis(chain[-1][0], chain[-1][1])
            endpoint_axes.setdefault(start_key, []).append(first_axis)
            endpoint_axes.setdefault(end_key, []).append(last_axis)
            maximum_endpoint_gap = max(
                maximum_endpoint_gap,
                _playability._point_segment_distance(
                    run[0], first_axis[0], first_axis[1]
                ),
                _playability._point_segment_distance(
                    run[-1], last_axis[0], last_axis[1]
                ),
            )

    failed_connections = 0
    maximum_connection_gap = 0.0
    maximum_clearance = 0.0
    for key in sorted(cap_keys):
        cover = cap_cover_lengths.get(key, 0.0)
        maximum_clearance = max(maximum_clearance, cover)
        if key not in cap_objects:
            failed_connections += 1
            continue
        node = node_positions[key]
        axes = endpoint_axes.get(key, [])
        expected = len(effective_incidents[key])
        if len(axes) < expected:
            failed_connections += expected - len(axes)
        for axis in axes:
            uncovered = max(
                0.0,
                _playability._point_segment_distance(node, axis[0], axis[1]) - cover,
            )
            maximum_connection_gap = max(maximum_connection_gap, uncovered)
            if uncovered > spec.road_connection_tolerance:
                failed_connections += 1

    if progress_callback is not None:
        progress_callback(
            100,
            f"Stock road fitting complete: {len(objects):,} objects in {chain_count:,} chains",
        )
    return _playability.RoadFitReport(
        objects=tuple(objects),
        chain_count=chain_count,
        connection_count=len(cap_keys),
        failed_connections=failed_connections,
        maximum_connection_gap=maximum_connection_gap,
        maximum_chain_gap=maximum_chain_gap,
        truncated=False,
        trimmed_junctions=0,
        skipped_short_runs=skipped_short_runs,
        maximum_model_overlap_metres=maximum_model_overlap,
        maximum_junction_clearance_metres=maximum_clearance,
        maximum_terrain_patch_radius_metres=0.0,
        junction_cap_objects=len(cap_objects),
        short_piece_objects=short_piece_objects,
        maximum_endpoint_gap_metres=maximum_endpoint_gap,
        maximum_road_pitch_degrees=maximum_pitch,
        suppressed_degree_two_caps=suppressed_degree_two_caps,
        terrain_filled_junctions=len(complex_keys),
        complex_junctions_without_caps=len(complex_keys),
        road_connection_slot_risk_nodes=0,
        suppressed_nearby_hubs=suppressed_nearby_hubs,
    )


def install_road_chain_parallel_policy() -> None:
    """Install before bridge wrappers so they capture this fitter as their base."""

    global _INSTALLED, _ORIGINAL_STOCK_FIT, _ORIGINAL_STOCK_CHAIN
    if _INSTALLED:
        return
    _ORIGINAL_STOCK_FIT = _playability._fit_stock_piece_road_objects
    _ORIGINAL_STOCK_CHAIN = _playability._stock_piece_chain
    _playability._stock_piece_chain = _batched_stock_piece_chain
    _playability._fit_stock_piece_road_objects = _fit_stock_piece_road_objects_parallel
    _INSTALLED = True
