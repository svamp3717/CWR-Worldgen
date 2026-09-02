# SPDX-License-Identifier: GPL-3.0-or-later
"""Production refinements for stock-road seams, junction overlap and terrain fit."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, replace
import math
import re
from typing import Mapping, Sequence

from . import generator as _generator
from . import playability as _p

_JUNCTION_OVERLAP = 0.22
_JUNCTION_MARGIN = 0.14
_JUNCTION_MIN_TRIM = 0.40
_HUB_HALF_WIDTH = 3.0
_STOCK_BULGE_LIMIT = 0.10
_GRAVEL_BULGE_LIMIT = 0.075
_LOOKAHEAD_DEPTH = 2
_AUDIT_BUCKET_METRES = 32.0
_AUDIT_ALIGNMENT_COSINE = math.cos(math.radians(28.0))
_PIECE_LENGTH_PATTERN = re.compile(r"(25|12|6|3)(?:_[lr](?:05|10|15|20|30|45))?\.p3d$")


@dataclass(frozen=True, slots=True)
class _Junction:
    point: tuple[float, float]
    axis: tuple[float, float]
    half_length: float
    half_width: float
    directions: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class _Context:
    elevations: Sequence[float]
    spec: object
    junctions: Mapping[tuple[int, int], _Junction]


_CONTEXT: ContextVar[_Context | None] = ContextVar("cwr_road_quality", default=None)
_ORIGINAL_FIT = _p.fit_road_objects
_ORIGINAL_CHAIN = _p._stock_piece_chain
_INSTALLED = False


def _unit(start, end) -> tuple[float, float]:
    dx, dz = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dz)
    return (0.0, 1.0) if length <= 1e-9 else (dx / length, dz / length)


def _junction_geometry(dataset, projection, spec) -> dict[tuple[int, int], _Junction]:
    incidents: dict[tuple[int, int], list[tuple[tuple[float, float], bool, str, str, str]]] = {}
    positions: dict[tuple[int, int], tuple[float, float]] = {}
    for feature, projected in zip(dataset.roads, _p.projected_road_polylines(dataset, projection)):
        if not _p.road_is_supported(feature.tags, include_minor=spec.include_minor_roads):
            continue
        points = tuple(_p._clean_road_points(projected))
        if len(points) < 2:
            continue
        dirt = _p.road_is_dirt(feature.tags)
        model = _p.road_model_for_tags(spec, feature.tags)
        for index, (start, end) in enumerate(zip(points, points[1:])):
            if math.dist(start, end) <= 0.05:
                continue
            skey, ekey = _p._road_node_key(start), _p._road_node_key(end)
            segment = f"{feature.osm_key}/{index:06d}"
            forward = _unit(start, end)
            reverse = (-forward[0], -forward[1])
            incidents.setdefault(skey, []).append((forward, dirt, model, segment, feature.osm_key))
            incidents.setdefault(ekey, []).append((reverse, dirt, model, segment, feature.osm_key))
            positions.setdefault(skey, start)
            positions.setdefault(ekey, end)

    result: dict[tuple[int, int], _Junction] = {}
    for key, raw in incidents.items():
        values = _p._unique_incidents(raw)
        if not 3 <= len(values) <= 4:
            continue
        all_gravel = all(_p.is_generated_gravel_road_model(v[2]) for v in values)
        if all_gravel:
            hub_length = 5.4 if len(values) == 3 else 6.0
        else:
            models = {v[2].casefold(): v[2] for v in values}
            if len(models) == 1:
                base_model = next(iter(models.values()))
            else:
                base_model = spec.dirt_road_model if all(v[1] for v in values) else spec.paved_road_model
            variants = _p.road_model_variants(base_model, spec.road_segment_length)
            cap = next((piece for piece in variants if piece.nominal_length == 6), variants[-1])
            hub_length = cap.length_metres
        axis = _p._dominant_node_axis(tuple((v[0], v[1], v[2], v[3]) for v in values))
        result[key] = _Junction(
            positions[key], axis, hub_length * 0.5, _HUB_HALF_WIDTH, tuple(v[0] for v in values)
        )
    return result


def _exit_distance(junction: _Junction, direction: tuple[float, float]) -> float:
    dx, dz = direction
    ax, az = junction.axis
    along = abs(dx * ax + dz * az)
    across = abs(dx * -az + dz * ax)
    values = []
    if along > 1e-8:
        values.append(junction.half_length / along)
    if across > 1e-8:
        values.append(junction.half_width / across)
    return min(values) if values else max(junction.half_length, junction.half_width)


def _end_direction(measure, *, start: bool) -> tuple[float, float]:
    points = measure.points
    node = points[0] if start else points[-1]
    candidates = points[1:] if start else reversed(points[:-1])
    for point in candidates:
        if math.dist(node, point) > 0.05:
            return _unit(node, point)
    return (0.0, 1.0)


def _quality_window(measure, pieces, start_distance, preferred_end, minimum_end, maximum_end, context):
    if not pieces:
        return start_distance, preferred_end, minimum_end, maximum_end
    shortest = min(piece.length_metres for piece in pieces)
    start_junction = context.junctions.get(_p._road_node_key(measure.points[0]))
    end_junction = context.junctions.get(_p._road_node_key(measure.points[-1]))
    desired_start = start_distance
    desired_end_trim = max(0.0, measure.total - preferred_end)
    desired_end_cover = max(0.0, measure.total - minimum_end)
    adjusted_maximum = maximum_end
    if start_junction is not None:
        desired_start = max(
            _JUNCTION_MIN_TRIM,
            _exit_distance(start_junction, _end_direction(measure, start=True)) - _JUNCTION_OVERLAP,
        )
    if end_junction is not None:
        exit_distance = _exit_distance(end_junction, _end_direction(measure, start=False))
        desired_end_trim = max(_JUNCTION_MIN_TRIM, exit_distance - _JUNCTION_OVERLAP)
        desired_end_cover = exit_distance + _JUNCTION_MARGIN
        adjusted_maximum = min(maximum_end, measure.total + _JUNCTION_OVERLAP)
    # Leave extremely short hub-to-hub runs to the fitter's existing short-run fallback.
    if (start_junction is not None or end_junction is not None) and measure.total >= (
        desired_start + desired_end_trim + shortest * 0.60
    ):
        start_distance = desired_start
        if end_junction is not None:
            preferred_end = max(start_distance, measure.total - desired_end_trim)
            minimum_end = max(start_distance, measure.total - desired_end_cover)
            maximum_end = max(preferred_end, adjusted_maximum)
    return start_distance, preferred_end, minimum_end, maximum_end


def _terrain_bulge(context: _Context, start, end, nominal: int) -> float:
    if nominal <= 6:
        return 0.0
    spec = context.spec
    h0 = _p._sample_elevation(context.elevations, spec.cells, spec.cell_size, *start)
    h1 = _p._sample_elevation(context.elevations, spec.cells, spec.cell_size, *end)
    fractions = (0.50,) if nominal <= 12 else (0.25, 0.50, 0.75)
    bulge = 0.0
    for fraction in fractions:
        x = start[0] + (end[0] - start[0]) * fraction
        z = start[1] + (end[1] - start[1]) * fraction
        terrain = _p._sample_elevation(context.elevations, spec.cells, spec.cell_size, x, z)
        plane = h0 * (1.0 - fraction) + h1 * fraction
        bulge = max(bulge, terrain - plane)
    return bulge


def _tail_error(measure, pieces, current, preferred_end, maximum_end, depth: int) -> float:
    """Estimate the best short tail without recursively fitting hypothetical geometry.

    The tail score is only a lookahead heuristic. Actual road pieces still pass
    through ``chord_endpoint`` and the normal fidelity checks when they are
    selected on a later fitting iteration. Running those full geometric searches
    recursively here multiplied the expensive chord work for every candidate and
    made dense stock-road planning dramatically slower than 0.9.257.

    For the bounded lookahead, fixed model lengths are enough to tell whether a
    combination such as 12+12+6 can finish close to the requested endpoint.
    Deduplicating equal accumulated distances also keeps the work bounded by the
    small set of reachable tail lengths rather than the number of piece orders.
    """

    _ = measure  # Kept in the signature for callers and tests; geometry is intentionally not sampled here.
    best = abs(preferred_end - current)
    if depth <= 0 or current >= preferred_end - 0.05:
        return best

    frontier = {current}
    for _step in range(depth):
        following: set[float] = set()
        for distance in frontier:
            for piece in pieces:
                candidate = distance + piece.length_metres
                if candidate > maximum_end + 1e-7:
                    continue
                best = min(best, abs(preferred_end - candidate))
                if best <= 1e-7:
                    return 0.0
                following.add(candidate)
        if not following:
            break
        frontier = following
    return best


def _quality_chain(measure, pieces, *, start_distance, preferred_end_distance, minimum_end_distance, maximum_end_distance):
    context = _CONTEXT.get()
    if context is None:
        return _ORIGINAL_CHAIN(
            measure, pieces,
            start_distance=start_distance,
            preferred_end_distance=preferred_end_distance,
            minimum_end_distance=minimum_end_distance,
            maximum_end_distance=maximum_end_distance,
        )
    start_distance, preferred_end_distance, minimum_end_distance, maximum_end_distance = _quality_window(
        measure, pieces, start_distance, preferred_end_distance, minimum_end_distance, maximum_end_distance, context
    )
    if not pieces or preferred_end_distance <= start_distance + 0.05:
        return ()
    ordered = tuple(sorted(pieces, key=lambda piece: (-piece.length_metres, piece.model_path.casefold())))
    shortest = min(piece.length_metres for piece in ordered)
    longest = max(piece.length_metres for piece in ordered)
    current = start_distance
    fitted = []
    maximum_objects = max(1, int(math.ceil((maximum_end_distance - start_distance) / shortest)) + 2)
    for _ in range(maximum_objects):
        if current >= preferred_end_distance - 0.05:
            break
        remaining = preferred_end_distance - current
        if current >= minimum_end_distance - 0.05 and remaining < shortest * 0.45:
            break
        preferred = _p._road_piece_sequence(remaining, ordered)
        preferred_piece = preferred[0] if preferred else ordered[-1]
        start_x, start_z, start_heading = measure.point(current)
        near_end = remaining <= longest * 2.25
        candidates = []
        for piece in ordered:
            endpoint = measure.chord_endpoint(current, piece.length_metres, maximum_end_distance)
            if endpoint is None:
                continue
            end_distance, end_x, end_z, chord_heading = endpoint
            end_heading = measure.point(end_distance)[2]
            turn = max(_p._heading_difference(chord_heading, start_heading), _p._heading_difference(chord_heading, end_heading))
            deviation = measure.maximum_chord_deviation(current, end_distance, (start_x, start_z), (end_x, end_z))
            gravel = _p.is_generated_gravel_road_model(piece.model_path)
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
            fidelity_penalty = int(turn > turn_limit or deviation > deviation_limit)
            bulge = _terrain_bulge(context, (start_x, start_z), (end_x, end_z), piece.nominal_length)
            terrain_limit = _GRAVEL_BULGE_LIMIT if gravel else _STOCK_BULGE_LIMIT
            terrain_penalty = int(bulge > terrain_limit)
            terrain_ratio = bulge / max(0.001, terrain_limit)
            tail_error = _tail_error(
                measure, ordered, end_distance, preferred_end_distance, maximum_end_distance, _LOOKAHEAD_DEPTH
            ) if near_end else 0.0
            tail_tolerance = max(0.20, float(getattr(context.spec, "road_connection_tolerance", 0.35)))
            tail_penalty = int(near_end and tail_error > tail_tolerance)
            if gravel:
                score = (
                    fidelity_penalty, tail_penalty, terrain_penalty,
                    tail_error if near_end else 0.0,
                    0 if piece == preferred_piece else 1,
                    terrain_ratio,
                    max(turn / turn_limit, deviation / deviation_limit),
                    abs(preferred_end_distance - end_distance), -piece.length_metres,
                )
            else:
                score = (
                    fidelity_penalty, tail_penalty, terrain_penalty,
                    max(turn / turn_limit, deviation / deviation_limit),
                    terrain_ratio, tail_error,
                    0 if piece == preferred_piece else 1,
                    abs(preferred_end_distance - end_distance), -piece.length_metres,
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
            if length <= 1e-9:
                angle = math.radians(target_heading)
                dx, dz, length = math.sin(angle), math.cos(angle), 1.0
            fitted.append((piece, (start_x, start_z), (
                start_x + dx / length * piece.length_metres,
                start_z + dz / length * piece.length_metres,
            )))
            break
        _score, piece, endpoint = min(candidates, key=lambda item: item[0])
        end_distance, end_x, end_z, _heading = endpoint
        fitted.append((piece, (start_x, start_z), (end_x, end_z)))
        if end_distance <= current + 1e-7:
            break
        current = end_distance
    return tuple(fitted)


def _piece_length(model_path: str, configured_long_length: float) -> float:
    filename = model_path.replace("/", "\\").rsplit("\\", 1)[-1].casefold()
    match = _PIECE_LENGTH_PATTERN.search(filename)
    return configured_long_length if match is None else configured_long_length * int(match.group(1)) / 25.0


def _audit(report, context: _Context):
    """Re-audit junction coverage without scanning the whole road network per arm.

    The first road-quality version compared every junction direction against every
    emitted road object. On large maps that made the UI appear to hang *after*
    the base fitter had already logged "Stock road fitting complete". A junction
    only needs nearby road axes, so index axis midpoints in a small spatial hash.
    Any axis close enough to satisfy the connection tolerance is guaranteed to
    have its midpoint inside ``cover + tolerance + half_length`` of the node.
    """
    if not context.junctions or not report.objects:
        return report
    spec = context.spec
    bucket_size = _AUDIT_BUCKET_METRES
    length_cache: dict[str, float] = {}
    buckets: dict[tuple[int, int], list[tuple[float, float, tuple[tuple[float, float], tuple[float, float]]]]] = {}
    maximum_half_length = 0.0

    for obj in report.objects[report.junction_cap_objects:]:
        model_key = obj.model_path.casefold()
        length = length_cache.get(model_key)
        if length is None:
            length = _piece_length(obj.model_path, spec.road_segment_length)
            length_cache[model_key] = length
        axis = _p._model_axis(obj, length)
        dx = axis[1][0] - axis[0][0]
        dz = axis[1][1] - axis[0][1]
        axis_length = math.hypot(dx, dz)
        if axis_length <= 1.0e-9:
            continue
        maximum_half_length = max(maximum_half_length, axis_length * 0.5)
        ux, uz = dx / axis_length, dz / axis_length
        midpoint_x = (axis[0][0] + axis[1][0]) * 0.5
        midpoint_z = (axis[0][1] + axis[1][1]) * 0.5
        key = (math.floor(midpoint_x / bucket_size), math.floor(midpoint_z / bucket_size))
        buckets.setdefault(key, []).append((ux, uz, axis))

    failed = 0
    maximum_gap = report.maximum_connection_gap
    maximum_cover = 0.0
    tolerance = float(spec.road_connection_tolerance)
    for junction in context.junctions.values():
        point_x, point_z = junction.point
        for direction in junction.directions:
            cover = _exit_distance(junction, direction) + _JUNCTION_MARGIN
            maximum_cover = max(maximum_cover, cover)
            # Searching to this radius is sufficient for every axis that could
            # possibly pass the connection tolerance, regardless of piece length.
            radius = cover + tolerance + maximum_half_length + 0.05
            bx0 = math.floor((point_x - radius) / bucket_size)
            bx1 = math.floor((point_x + radius) / bucket_size)
            bz0 = math.floor((point_z - radius) / bucket_size)
            bz1 = math.floor((point_z + radius) / bucket_size)
            best = math.inf
            dx, dz = direction
            for bx in range(bx0, bx1 + 1):
                for bz in range(bz0, bz1 + 1):
                    for ux, uz, axis in buckets.get((bx, bz), ()):
                        if abs(ux * dx + uz * dz) < _AUDIT_ALIGNMENT_COSINE:
                            continue
                        distance = _p._point_segment_distance(junction.point, axis[0], axis[1])
                        if distance < best:
                            best = distance
                            if best <= cover:
                                break
                    if best <= cover:
                        break
                if best <= cover:
                    break
            if not math.isfinite(best):
                failed += 1
                maximum_gap = max(maximum_gap, tolerance + 1.0e-6)
                continue
            uncovered = max(0.0, best - cover)
            maximum_gap = max(maximum_gap, uncovered)
            if uncovered > tolerance:
                failed += 1
    return replace(
        report,
        failed_connections=failed,
        maximum_connection_gap=maximum_gap,
        maximum_junction_clearance_metres=max(report.maximum_junction_clearance_metres, maximum_cover),
    )


def _fit(dataset, projection, elevations, spec, *, starting_id: int = 1, progress_callback=None):
    if not bool(getattr(spec, "stock_road_piece_fitting", False)):
        return _ORIGINAL_FIT(
            dataset, projection, elevations, spec,
            starting_id=starting_id, progress_callback=progress_callback,
        )
    context = _Context(elevations, spec, _junction_geometry(dataset, projection, spec))
    deferred_completion: tuple[int, str] | None = None

    def _progress(value: int, message: str) -> None:
        nonlocal deferred_completion
        if value >= 100 and message.startswith("Stock road fitting complete:"):
            deferred_completion = (value, message)
            return
        if progress_callback is not None:
            progress_callback(value, message)

    token = _CONTEXT.set(context)
    try:
        report = _ORIGINAL_FIT(
            dataset, projection, elevations, spec,
            starting_id=starting_id,
            progress_callback=_progress if progress_callback is not None else None,
        )
    finally:
        _CONTEXT.reset(token)

    if progress_callback is not None and context.junctions:
        progress_callback(99, f"Auditing {len(context.junctions):,} road junctions")
    report = _audit(report, context)
    if progress_callback is not None and deferred_completion is not None:
        progress_callback(*deferred_completion)
    return report


def install_road_quality_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _p._stock_piece_chain = _quality_chain
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True