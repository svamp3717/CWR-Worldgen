# SPDX-License-Identifier: GPL-3.0-or-later
"""Close generated-gravel seams at skewed three/four-way junctions.

The road-quality policy models ordinary stock junction caps as oriented
rectangles. Generated gravel hubs are different: they are compact plus-shaped
meshes with a narrower 2.3 m half-width and a 0.9 m lowered visual overhang on
the incident road ribbons. Treating those hubs as 3 m rectangles can trim a
skewed branch almost a metre too early, leaving bare terrain between the branch
and the hub.
"""
from __future__ import annotations

from dataclasses import replace
import math

from .procedural_infrastructure import (
    GENERATED_GRAVEL_HALF_WIDTH_METRES,
    GENERATED_GRAVEL_VISUAL_OVERLAP_METRES,
)

# Restore the deliberately hidden overlap used by the original junction fitter
# for generated gravel only. The ribbon itself has a 0.90 m lowered visual tip,
# so 0.70 m of centreline overlap is generous enough to hide seams without
# forcing stock asphalt branches back into deep cap clipping.
_GRAVEL_JUNCTION_OVERLAP = min(0.70, GENERATED_GRAVEL_VISUAL_OVERLAP_METRES)
_GRAVEL_J3_LENGTH_METRES = 5.4
_GRAVEL_J4_LENGTH_METRES = 6.0
_GRAVEL_J3_ARM_INSET_METRES = 0.25
_EPSILON = 1.0e-8

_RQ = None
_ORIGINAL_JUNCTION_GEOMETRY = None
_ORIGINAL_EXIT_DISTANCE = None
_INSTALLED = False


def _ray_limit(extent: float, component: float) -> float:
    return math.inf if abs(component) <= _EPSILON else extent / abs(component)


def gravel_hub_exit_distance(
    axis: tuple[float, float],
    direction: tuple[float, float],
    *,
    extent: float,
    half_width: float = GENERATED_GRAVEL_HALF_WIDTH_METRES,
) -> float:
    """Return where a ray leaves the actual plus-shaped gravel hub footprint."""

    dx, dz = direction
    ax, az = axis
    along = abs(dx * ax + dz * az)
    across = abs(dx * -az + dz * ax)

    # The visual hub is the union of one longitudinal and one transverse arm:
    #   |along| <= extent, |across| <= half_width
    #   |along| <= half_width, |across| <= extent
    # A ray remains covered while it is inside either arm, so its true exit is
    # the farther of the two individual arm exits.
    longitudinal = min(_ray_limit(extent, along), _ray_limit(half_width, across))
    transverse = min(_ray_limit(half_width, along), _ray_limit(extent, across))
    return max(longitudinal, transverse)


def _is_gravel_junction(junction) -> bool:
    # road_quality_policy currently gives stock caps a fixed 3.0 m half-width.
    # Rewritten generated hubs carry their real 2.30 m width, which is also the
    # exact source constant used to build their P3D visual mesh.
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
    for feature, projected in zip(
        dataset.roads,
        rq._p.projected_road_polylines(dataset, projection),
    ):
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
        degree = len(junction.directions)
        if degree not in {3, 4}:
            continue
        hub_length = _GRAVEL_J3_LENGTH_METRES if degree == 3 else _GRAVEL_J4_LENGTH_METRES
        extent = hub_length * 0.5
        if degree == 3:
            # _gravel_junction_lods deliberately shortens all four visual arms by
            # 0.25 m for the compact T-junction variant.
            extent -= _GRAVEL_J3_ARM_INSET_METRES
        result[key] = replace(
            junction,
            half_length=extent,
            half_width=GENERATED_GRAVEL_HALF_WIDTH_METRES,
        )
    return result


def _exit_distance(junction, direction: tuple[float, float]) -> float:
    if not _is_gravel_junction(junction):
        return _ORIGINAL_EXIT_DISTANCE(junction, direction)
    return gravel_hub_exit_distance(
        junction.axis,
        direction,
        extent=float(junction.half_length),
        half_width=float(junction.half_width),
    )


def _overlap_for(junction) -> float:
    if junction is not None and _is_gravel_junction(junction):
        return _GRAVEL_JUNCTION_OVERLAP
    return float(_RQ._JUNCTION_OVERLAP)


def _quality_window(measure, pieces, start_distance, preferred_end, minimum_end, maximum_end, context):
    """Use the real gravel hub edge and gravel-only overlap for endpoint trims."""

    rq = _RQ
    if not pieces:
        return start_distance, preferred_end, minimum_end, maximum_end
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

    # Preserve the stock policy's short hub-to-hub fallback. Generated gravel
    # has a 3 m sibling and a lowered 0.9 m visual overhang, so once this window
    # is based on the true hub edge the existing chain fitter can close the seam
    # without adding a duplicate patch object.
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
    """Layer gravel-specific hub geometry onto the general road-quality policy."""

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
