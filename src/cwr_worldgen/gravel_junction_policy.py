# SPDX-License-Identifier: GPL-3.0-or-later
"""Trim generated gravel chains against the reusable junction model family."""
from __future__ import annotations

from dataclasses import replace
import math

from .gravel_family_policy import (
    GRAVEL_JUNCTION_ARM_EXTENT_METRES,
    gravel_junction_ray_exit_distance,
    gravel_junction_variant_for_directions,
)
from .procedural_infrastructure import (
    GENERATED_GRAVEL_HALF_WIDTH_METRES,
    GENERATED_GRAVEL_VISUAL_OVERLAP_METRES,
)

_GRAVEL_JUNCTION_OVERLAP = min(0.70, GENERATED_GRAVEL_VISUAL_OVERLAP_METRES)
_RQ = None
_ORIGINAL_JUNCTION_GEOMETRY = None
_ORIGINAL_EXIT_DISTANCE = None
_INSTALLED = False


def _is_gravel_junction(junction) -> bool:
    return math.isclose(
        float(junction.half_width),
        float(GENERATED_GRAVEL_HALF_WIDTH_METRES),
        rel_tol=0.0,
        abs_tol=1.0e-7,
    )


def _junction_geometry(dataset, projection, spec):
    rq = _RQ
    result = dict(_ORIGINAL_JUNCTION_GEOMETRY(dataset, projection, spec))
    if not result:
        return result

    models_by_key: dict[tuple[int, int], list[str]] = {}
    for feature, projected in zip(dataset.roads, rq._p.projected_road_polylines(dataset, projection)):
        if not rq._p.road_is_supported(feature.tags, include_minor=spec.include_minor_roads):
            continue
        points = tuple(rq._p._clean_road_points(projected))
        if len(points) < 2:
            continue
        model = rq._p.road_model_for_tags(spec, feature.tags)
        for start, end in zip(points, points[1:]):
            if math.dist(start, end) <= 0.05:
                continue
            models_by_key.setdefault(rq._p._road_node_key(start), []).append(model)
            models_by_key.setdefault(rq._p._road_node_key(end), []).append(model)

    for key, junction in tuple(result.items()):
        models = models_by_key.get(key, ())
        if not models or not all(rq._p.is_generated_gravel_road_model(model) for model in models):
            continue
        if len(junction.directions) not in {3, 4}:
            continue
        _variant, axis = gravel_junction_variant_for_directions(junction.directions)
        result[key] = replace(
            junction,
            axis=axis,
            half_length=GRAVEL_JUNCTION_ARM_EXTENT_METRES,
            half_width=GENERATED_GRAVEL_HALF_WIDTH_METRES,
        )
    return result


def _exit_distance(junction, direction: tuple[float, float]) -> float:
    if not _is_gravel_junction(junction):
        return _ORIGINAL_EXIT_DISTANCE(junction, direction)
    variant, axis = gravel_junction_variant_for_directions(junction.directions)
    return gravel_junction_ray_exit_distance(variant, axis, direction)


def _overlap_for(junction) -> float:
    if junction is not None and _is_gravel_junction(junction):
        return _GRAVEL_JUNCTION_OVERLAP
    return float(_RQ._JUNCTION_OVERLAP)


def _quality_window(measure, pieces, start_distance, preferred_end, minimum_end, maximum_end, context):
    if not pieces:
        return start_distance, preferred_end, minimum_end, maximum_end
    rq = _RQ
    shortest = min(piece.length_metres for piece in pieces)
    start_junction = context.junctions.get(rq._p._road_node_key(measure.points[0]))
    end_junction = context.junctions.get(rq._p._road_node_key(measure.points[-1]))
    desired_start = start_distance
    desired_end_trim = max(0.0, measure.total - preferred_end)
    desired_end_cover = max(0.0, measure.total - minimum_end)
    adjusted_maximum = maximum_end

    if start_junction is not None:
        desired_start = max(
            rq._JUNCTION_MIN_TRIM,
            _exit_distance(start_junction, rq._end_direction(measure, start=True))
            - _overlap_for(start_junction),
        )
    if end_junction is not None:
        exit_distance = _exit_distance(end_junction, rq._end_direction(measure, start=False))
        overlap = _overlap_for(end_junction)
        desired_end_trim = max(rq._JUNCTION_MIN_TRIM, exit_distance - overlap)
        desired_end_cover = exit_distance + rq._JUNCTION_MARGIN
        adjusted_maximum = min(maximum_end, measure.total + overlap)

    if (start_junction is not None or end_junction is not None) and measure.total >= (
        desired_start + desired_end_trim + shortest * 0.60
    ):
        start_distance = desired_start
        if end_junction is not None:
            preferred_end = max(start_distance, measure.total - desired_end_trim)
            minimum_end = max(start_distance, measure.total - desired_end_cover)
            maximum_end = max(preferred_end, adjusted_maximum)
    return start_distance, preferred_end, minimum_end, maximum_end


def install_gravel_junction_policy() -> None:
    global _RQ, _ORIGINAL_JUNCTION_GEOMETRY, _ORIGINAL_EXIT_DISTANCE, _INSTALLED
    if _INSTALLED:
        return
    from . import road_quality_policy as rq

    _RQ = rq
    _ORIGINAL_JUNCTION_GEOMETRY = rq._junction_geometry
    _ORIGINAL_EXIT_DISTANCE = rq._exit_distance
    rq._junction_geometry = _junction_geometry
    rq._exit_distance = _exit_distance
    rq._quality_window = _quality_window
    _INSTALLED = True
