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
    best = abs(preferred_end - current)
    if depth <= 0 or current >= preferred_end - 0.05:
        return best
    for piece in pieces:
        endpoint = measure.chord_endpoint(current, piece.length_metres, maximum_end)
        if endpoint is not None and endpoint[0] > current + 1e-7:
            best = min(best, _tail_error(measure, pieces, endpoint[0], preferred_end, maximum_end, depth - 1))
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
    match = re.search(r"(25|12|6|3)(?:_[lr](?:05|10|15|20|30|45))?\.p3d$", filename)
    return configured_long_length if match is None else configured_long_length * int(match.group(1)) / 25.0


def _audit(report, context: _Context):
    if not context.junctions or not report.objects:
        return report
    spec = context.spec
    axes = [
        (obj, _p._model_axis(obj, _piece_length(obj.model_path, spec.road_segment_length)))
        for obj in report.objects[report.junction_cap_objects:]
    ]
    failed = 0
    maximum_gap = 0.0
    maximum_cover = 0.0
    for junction in context.junctions.values():
        for direction in junction.directions:
            cover = _exit_distance(junction, direction) + _JUNCTION_MARGIN
            maximum_cover = max(maximum_cover, cover)
            heading = math.degrees(math.atan2(direction[0], direction[1])) % 360.0
            best = math.inf
            for obj, axis in axes:
                forward = _p._heading_difference(obj.heading_degrees, heading)
                reverse = _p._heading_difference((obj.heading_degrees + 180.0) % 360.0, heading)
                if min(forward, reverse) > 28.0:
                    continue
                best = min(best, _p._point_segment_distance(junction.point, axis[0], axis[1]))
            if not math.isfinite(best):
                failed += 1
                continue
            uncovered = max(0.0, best - cover)
            maximum_gap = max(maximum_gap, uncovered)
            if uncovered > spec.road_connection_tolerance:
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
    token = _CONTEXT.set(context)
    try:
        report = _ORIGINAL_FIT(
            dataset, projection, elevations, spec,
            starting_id=starting_id, progress_callback=progress_callback,
        )
    finally:
        _CONTEXT.reset(token)
    return _audit(report, context)


def install_road_quality_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _p._stock_piece_chain = _quality_chain
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
