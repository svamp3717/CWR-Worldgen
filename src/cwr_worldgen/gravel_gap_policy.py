# SPDX-License-Identifier: GPL-3.0-or-later
"""Close short visual gaps between separately normalized generated gravel ways."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import wraps
import math

from . import generator as _generator
from . import playability as _p
from . import road_quality_policy as _rq
from .procedural_infrastructure import (
    GENERATED_GRAVEL_HALF_WIDTH_METRES,
    GENERATED_GRAVEL_VISUAL_OVERLAP_METRES,
    gravel_road_model_path,
)

_GRAVEL_GAP_MAX_METRES = 8.0
_GRAVEL_GAP_ALIGNMENT_COSINE = math.cos(math.radians(20.0))
_GRAVEL_GAP_EPSILON = 0.20
_ORIGINAL_FIT = _p.fit_road_objects
_INSTALLED = False


@dataclass(frozen=True, slots=True)
class _GravelEndpoint:
    point: tuple[float, float]
    inward: tuple[float, float]
    feature_key: str

    @property
    def outward(self) -> tuple[float, float]:
        return (-self.inward[0], -self.inward[1])


def _unit(start, end) -> tuple[float, float]:
    dx, dz = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dz)
    return (0.0, 1.0) if length <= 1.0e-9 else (dx / length, dz / length)


def _dot(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1]


def _gravel_endpoints(dataset, projection, spec) -> tuple[_GravelEndpoint, ...]:
    result: list[_GravelEndpoint] = []
    projected = _p.projected_road_polylines(dataset, projection)
    for feature, raw_points in zip(dataset.roads, projected):
        if not _p.road_is_supported(feature.tags, include_minor=spec.include_minor_roads):
            continue
        model = _p.road_model_for_tags(spec, feature.tags)
        if not _p.is_generated_gravel_road_model(model):
            continue
        points = tuple(_p._clean_road_points(raw_points))
        if len(points) < 2:
            continue
        if math.dist(points[0], points[1]) > 0.05:
            result.append(_GravelEndpoint(points[0], _unit(points[0], points[1]), feature.osm_key))
        if math.dist(points[-1], points[-2]) > 0.05:
            result.append(_GravelEndpoint(points[-1], _unit(points[-1], points[-2]), feature.osm_key))
    return tuple(result)


def _gravel_piece_length(spec, nominal: int) -> float:
    return float(spec.road_segment_length) * float(nominal) / 25.0


def _gravel_bridge_nominal(spec, span: float, *, endpoint_pair: bool) -> int | None:
    # Existing gravel ribbons extend 0.90 m beyond their fitted axes. The filler
    # has the same lowered tips. Count those overlaps before selecting a physical
    # piece so the repair stays short instead of burying a long slab underneath.
    tips = 4 if endpoint_pair else 3
    allowance = tips * GENERATED_GRAVEL_VISUAL_OVERLAP_METRES
    for nominal in (3, 6, 12):
        if _gravel_piece_length(spec, nominal) + allowance >= span + _GRAVEL_GAP_EPSILON:
            return nominal
    return None


def _gravel_hub_geometry(junction) -> bool:
    return math.isclose(
        float(junction.half_width),
        float(GENERATED_GRAVEL_HALF_WIDTH_METRES),
        rel_tol=0.0,
        abs_tol=1.0e-6,
    )


def _bridge_short_gravel_gaps(report, dataset, projection, elevations, spec, context):
    """Add small filler ribbons only where short gravel endpoints clearly align."""

    endpoints = _gravel_endpoints(dataset, projection, spec)
    if not endpoints:
        return report, 0

    # Endpoint-to-endpoint repair. A spatial hash avoids an O(n^2) dead-end scan.
    bucket_size = _GRAVEL_GAP_MAX_METRES
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, endpoint in enumerate(endpoints):
        key = (
            math.floor(endpoint.point[0] / bucket_size),
            math.floor(endpoint.point[1] / bucket_size),
        )
        buckets.setdefault(key, []).append(index)

    candidates: list[tuple[float, int, int]] = []
    minimum_gap = 2.0 * GENERATED_GRAVEL_VISUAL_OVERLAP_METRES + _GRAVEL_GAP_EPSILON
    for left_index, left in enumerate(endpoints):
        bx = math.floor(left.point[0] / bucket_size)
        bz = math.floor(left.point[1] / bucket_size)
        for nx in range(bx - 1, bx + 2):
            for nz in range(bz - 1, bz + 2):
                for right_index in buckets.get((nx, nz), ()):
                    if right_index <= left_index:
                        continue
                    right = endpoints[right_index]
                    if left.feature_key == right.feature_key:
                        continue
                    dx = right.point[0] - left.point[0]
                    dz = right.point[1] - left.point[1]
                    distance = math.hypot(dx, dz)
                    if not (minimum_gap < distance <= _GRAVEL_GAP_MAX_METRES):
                        continue
                    direction = (dx / distance, dz / distance)
                    if _dot(left.outward, direction) < _GRAVEL_GAP_ALIGNMENT_COSINE:
                        continue
                    if _dot(right.outward, (-direction[0], -direction[1])) < _GRAVEL_GAP_ALIGNMENT_COSINE:
                        continue
                    candidates.append((distance, left_index, right_index))

    fillers: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
    used: set[int] = set()
    for distance, left_index, right_index in sorted(candidates):
        if left_index in used or right_index in used:
            continue
        nominal = _gravel_bridge_nominal(spec, distance, endpoint_pair=True)
        if nominal is None:
            continue
        left, right = endpoints[left_index], endpoints[right_index]
        dx = right.point[0] - left.point[0]
        dz = right.point[1] - left.point[1]
        length = math.hypot(dx, dz)
        ux, uz = dx / length, dz / length
        model_length = _gravel_piece_length(spec, nominal)
        centre = (
            (left.point[0] + right.point[0]) * 0.5,
            (left.point[1] + right.point[1]) * 0.5,
        )
        start = (centre[0] - ux * model_length * 0.5, centre[1] - uz * model_length * 0.5)
        end = (centre[0] + ux * model_length * 0.5, centre[1] + uz * model_length * 0.5)
        fillers.append((gravel_road_model_path(spec.name, nominal), start, end))
        used.update((left_index, right_index))

    # Endpoint-to-hub repair. Generated 3-way gravel hubs deliberately render all
    # four short arms, so a separately normalized continuation can look broken
    # even though the hub itself is correct. Only connect an endpoint that faces
    # a rendered hub axis within 20 degrees and leaves <=8 m of visible gap.
    gravel_junctions = tuple(
        junction for junction in context.junctions.values() if _gravel_hub_geometry(junction)
    )
    junction_bucket_size = _GRAVEL_GAP_MAX_METRES + 4.0
    junction_buckets: dict[tuple[int, int], list[object]] = {}
    for junction in gravel_junctions:
        key = (
            math.floor(junction.point[0] / junction_bucket_size),
            math.floor(junction.point[1] / junction_bucket_size),
        )
        junction_buckets.setdefault(key, []).append(junction)

    for endpoint_index, endpoint in enumerate(endpoints):
        if endpoint_index in used:
            continue
        bx = math.floor(endpoint.point[0] / junction_bucket_size)
        bz = math.floor(endpoint.point[1] / junction_bucket_size)
        nearby_junctions = []
        for nx in range(bx - 1, bx + 2):
            for nz in range(bz - 1, bz + 2):
                nearby_junctions.extend(junction_buckets.get((nx, nz), ()))

        best = None
        for junction in nearby_junctions:
            dx = junction.point[0] - endpoint.point[0]
            dz = junction.point[1] - endpoint.point[1]
            node_distance = math.hypot(dx, dz)
            if node_distance <= 1.0e-9:
                continue
            to_junction = (dx / node_distance, dz / node_distance)
            if _dot(endpoint.outward, to_junction) < _GRAVEL_GAP_ALIGNMENT_COSINE:
                continue
            from_junction = (-to_junction[0], -to_junction[1])
            ax, az = junction.axis
            perpendicular = (-az, ax)
            if max(
                abs(_dot(from_junction, (ax, az))),
                abs(_dot(from_junction, perpendicular)),
            ) < _GRAVEL_GAP_ALIGNMENT_COSINE:
                continue
            hub_exit = _rq._exit_distance(junction, from_junction)
            span = node_distance - hub_exit
            visible_gap = span - GENERATED_GRAVEL_VISUAL_OVERLAP_METRES
            if not (_GRAVEL_GAP_EPSILON < visible_gap <= _GRAVEL_GAP_MAX_METRES):
                continue
            if best is None or span < best[0]:
                best = (span, junction, from_junction, hub_exit)

        if best is None:
            continue
        span, junction, from_junction, hub_exit = best
        nominal = _gravel_bridge_nominal(spec, span, endpoint_pair=False)
        if nominal is None:
            continue
        ux, uz = from_junction
        hub_edge = (
            junction.point[0] + ux * hub_exit,
            junction.point[1] + uz * hub_exit,
        )
        centre = (
            (hub_edge[0] + endpoint.point[0]) * 0.5,
            (hub_edge[1] + endpoint.point[1]) * 0.5,
        )
        model_length = _gravel_piece_length(spec, nominal)
        start = (centre[0] - ux * model_length * 0.5, centre[1] - uz * model_length * 0.5)
        end = (centre[0] + ux * model_length * 0.5, centre[1] + uz * model_length * 0.5)
        fillers.append((gravel_road_model_path(spec.name, nominal), start, end))
        used.add(endpoint_index)

    if not fillers:
        return report, 0

    next_id = max((obj.object_id for obj in report.objects), default=0) + 1
    objects = list(report.objects)
    for model_path, start, end in fillers:
        centre_x = (start[0] + end[0]) * 0.5
        centre_z = (start[1] + end[1]) * 0.5
        if not (0.0 <= centre_x < spec.world_size and 0.0 <= centre_z < spec.world_size):
            continue
        objects.append(
            _p._road_object_on_slope(
                next_id,
                model_path,
                start,
                end,
                elevations,
                spec,
                vertical_offset=_p._STOCK_GRAVEL_VERTICAL_OFFSET_METRES,
            )
        )
        next_id += 1

    added = len(objects) - len(report.objects)
    if not added:
        return report, 0
    return replace(
        report,
        objects=tuple(objects),
        chain_count=report.chain_count + added,
        short_piece_objects=report.short_piece_objects + added,
    ), added


@wraps(_ORIGINAL_FIT)
def _fit(dataset, projection, elevations, spec, *, starting_id: int = 1, progress_callback=None):
    if not bool(getattr(spec, "stock_road_piece_fitting", False)):
        return _ORIGINAL_FIT(
            dataset,
            projection,
            elevations,
            spec,
            starting_id=starting_id,
            progress_callback=progress_callback,
        )

    deferred_completion: tuple[int, str] | None = None

    def progress(value: int, message: str) -> None:
        nonlocal deferred_completion
        if value >= 100 and message.startswith("Stock road fitting complete:"):
            deferred_completion = (value, message)
            return
        if progress_callback is not None:
            progress_callback(value, message)

    report = _ORIGINAL_FIT(
        dataset,
        projection,
        elevations,
        spec,
        starting_id=starting_id,
        progress_callback=progress if progress_callback is not None else None,
    )
    context = _rq._Context(elevations, spec, _rq._junction_geometry(dataset, projection, spec))
    report, added = _bridge_short_gravel_gaps(
        report,
        dataset,
        projection,
        elevations,
        spec,
        context,
    )
    if progress_callback is not None and added:
        progress_callback(99, f"Closed {added:,} short gravel road gaps")
    if progress_callback is not None and deferred_completion is not None:
        # Rebuild the count because the base fitter formed this message before
        # the tiny visual bridge pass appended its connector objects.
        progress_callback(
            deferred_completion[0],
            f"Stock road fitting complete: {len(report.objects):,} objects in {report.chain_count:,} chains",
        )
    return report


def install_gravel_gap_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
