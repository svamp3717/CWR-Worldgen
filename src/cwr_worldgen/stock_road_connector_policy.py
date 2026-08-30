# SPDX-License-Identifier: GPL-3.0-or-later
"""Locally relax skewed mixed-road approaches onto native junction connectors.

Native CWA junction meshes have fixed connector directions. Merely replacing a
legacy straight cap with a native T mesh is not enough when the source arms are
noticeably skewed: the first road pieces are still fitted to the original source
tangents and therefore meet the native mesh at the wrong angle.

This policy adjusts only the first few metres of eligible mixed paved/gravel T
junction arms before the ordinary road fitter runs. The junction center stays
fixed. A synthetic point is inserted just beyond the measured 6.25 m connector
radius in the exact connector direction, after which the polyline returns to the
original geometry. The adjustment is bounded to two metres laterally, which is
inside the 2.30 m half-width of the generated gravel road.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import math

from . import generator as _generator
from . import playability as _p
from . import stock_road_junction_policy as _junction
from . import stock_road_skew_policy as _skew

MAXIMUM_APPROACH_LATERAL_RELAXATION_METRES = 2.0
APPROACH_CONNECTOR_MARGIN_METRES = 0.20

_ORIGINAL_FIT = None
_ORIGINAL_PROJECTED_ROADS = None
_INSTALLED = False
_RELAXED_PROJECTED_ROADS: ContextVar[object | None] = ContextVar(
    "cwr_relaxed_stock_road_polylines", default=None
)


@dataclass(frozen=True, slots=True)
class _Occurrence:
    feature_index: int
    node_index: int
    neighbour_index: int
    segment_key: str
    direction: tuple[float, float]
    model_path: str


def _heading(direction: tuple[float, float]) -> float:
    return math.degrees(math.atan2(direction[0], direction[1])) % 360.0


def _angular_distance(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _native_t_targets(incidents, native) -> tuple[float, ...] | None:
    """Map three incident arms to the measured native T connector headings."""
    if len(incidents) != 3:
        return None
    pair = _junction._dominant_pair(incidents)
    if pair is None:
        return None
    first, second = pair
    branch = next(index for index in range(3) if index not in pair)
    rotation = float(native.heading_degrees) % 360.0
    target_zero = rotation
    target_180 = (rotation + 180.0) % 360.0
    # stock_road_junction_policy fits native T assets with their branch on the
    # model-local +90 degree connector. Keep the relaxation pass on that same
    # signed connector. Using +270 here mirrors the branch through the node and
    # folds the first branch piece across the junction, creating the large
    # in-game overlap that is especially obvious on skewed sil/ces T nodes.
    target_branch = (rotation + 90.0) % 360.0

    actual_first = _heading(incidents[first].direction)
    actual_second = _heading(incidents[second].direction)
    direct = (
        _angular_distance(actual_first, target_zero)
        + _angular_distance(actual_second, target_180)
    )
    swapped = (
        _angular_distance(actual_first, target_180)
        + _angular_distance(actual_second, target_zero)
    )
    targets = [0.0, 0.0, 0.0]
    if direct <= swapped:
        targets[first], targets[second] = target_zero, target_180
    else:
        targets[first], targets[second] = target_180, target_zero
    targets[branch] = target_branch
    return tuple(targets)


def _relaxed_arm_point(
    node: tuple[float, float],
    neighbour: tuple[float, float],
    target_heading: float,
    connector_half_extent: float,
) -> tuple[float, float] | None:
    """Return a connector-aligned synthetic point while staying in-road."""
    distance = math.dist(node, neighbour)
    desired = connector_half_extent + APPROACH_CONNECTOR_MARGIN_METRES
    if distance <= desired * 1.35:
        return None

    original_heading = math.degrees(
        math.atan2(neighbour[0] - node[0], neighbour[1] - node[1])
    ) % 360.0
    error = _angular_distance(original_heading, target_heading)
    if error <= 1.0e-6:
        length = min(desired, distance * 0.45)
    else:
        safe_length = MAXIMUM_APPROACH_LATERAL_RELAXATION_METRES / max(
            1.0e-6, math.sin(math.radians(min(89.0, error)))
        )
        length = min(desired, safe_length, distance * 0.45)
    if length <= connector_half_extent + 0.03:
        return None
    angle = math.radians(target_heading)
    return (
        node[0] + math.sin(angle) * length,
        node[1] + math.cos(angle) * length,
    )


def _collect_relaxations(dataset, projection, projected, spec):
    raw = {}
    occurrences = {}
    positions = {}
    for feature_index, (feature, points_raw) in enumerate(zip(dataset.roads, projected)):
        if not _p.road_is_supported(feature.tags, include_minor=spec.include_minor_roads):
            continue
        points = tuple(points_raw)
        if len(points) < 2:
            continue
        model = _p.road_model_for_tags(spec, feature.tags)
        dirt = _p.road_is_dirt(feature.tags)
        for segment_index, (start, end) in enumerate(zip(points, points[1:])):
            if math.dist(start, end) <= 0.05:
                continue
            segment_key = f"{feature.osm_key}/{segment_index:06d}"
            forward = _p._normalised_direction(start, end)
            reverse = (-forward[0], -forward[1])
            start_key, end_key = _p._road_node_key(start), _p._road_node_key(end)
            raw.setdefault(start_key, []).append(
                (forward, dirt, model, segment_key, feature.osm_key)
            )
            raw.setdefault(end_key, []).append(
                (reverse, dirt, model, segment_key, feature.osm_key)
            )
            occurrences[segment_key] = (
                _Occurrence(feature_index, segment_index, segment_index + 1, segment_key, forward, model),
                _Occurrence(feature_index, segment_index + 1, segment_index, segment_key, reverse, model),
            )
            positions.setdefault(start_key, start)
            positions.setdefault(end_key, end)

    connector_half = _junction._connector_half_extent(spec)
    result = {}
    for key, values in raw.items():
        unique = _p._unique_incidents(values)
        if len(unique) != 3:
            continue
        incidents = tuple(
            _junction._Incident(value[0], _junction._family(value[2]), value[2])
            for value in unique
        )
        if not _skew._eligible_relaxed_mixed_t(incidents):
            continue
        native = _junction._native_junction_for_incidents(incidents)
        if native is None:
            continue
        targets = _native_t_targets(incidents, native)
        if targets is None:
            continue

        node = positions[key]
        for value, target_heading in zip(unique, targets):
            segment_key = value[3]
            pair = occurrences.get(segment_key)
            if pair is None:
                continue
            occurrence = next(
                (
                    item
                    for item in pair
                    if _p._road_node_key(
                        tuple(projected[item.feature_index][item.node_index])
                    )
                    == key
                ),
                None,
            )
            if occurrence is None:
                continue
            neighbour = tuple(
                projected[occurrence.feature_index][occurrence.neighbour_index]
            )
            point = _relaxed_arm_point(node, neighbour, target_heading, connector_half)
            if point is None:
                continue
            result[(occurrence.feature_index, occurrence.node_index, occurrence.neighbour_index)] = point
    return result


def _apply_relaxations(projected, relaxations):
    if not relaxations:
        return projected
    output = []
    for feature_index, points_raw in enumerate(projected):
        points = list(points_raw)
        before = {}
        after = {}
        for (fi, node_index, neighbour_index), point in relaxations.items():
            if fi != feature_index:
                continue
            if neighbour_index < node_index:
                before[node_index] = point
            else:
                after[node_index] = point
        relaxed = []
        for index, point in enumerate(points):
            if index in before:
                relaxed.append(before[index])
            relaxed.append(point)
            if index in after:
                relaxed.append(after[index])
        output.append(tuple(relaxed))
    return tuple(output)


def _projected_road_polylines(dataset, projection):
    relaxed = _RELAXED_PROJECTED_ROADS.get()
    if relaxed is not None:
        return relaxed
    if _ORIGINAL_PROJECTED_ROADS is None:
        raise RuntimeError("stock road connector policy is not installed")
    return _ORIGINAL_PROJECTED_ROADS(dataset, projection)


def _fit(dataset, projection, elevations, spec, *, starting_id=1, progress_callback=None):
    if _ORIGINAL_FIT is None or _ORIGINAL_PROJECTED_ROADS is None:
        raise RuntimeError("stock road connector policy is not installed")
    projected = _ORIGINAL_PROJECTED_ROADS(dataset, projection)
    relaxations = _collect_relaxations(dataset, projection, projected, spec)
    relaxed = _apply_relaxations(projected, relaxations)
    token = _RELAXED_PROJECTED_ROADS.set(relaxed)
    try:
        return _ORIGINAL_FIT(
            dataset,
            projection,
            elevations,
            spec,
            starting_id=starting_id,
            progress_callback=progress_callback,
        )
    finally:
        _RELAXED_PROJECTED_ROADS.reset(token)


def install_stock_road_connector_policy() -> None:
    global _ORIGINAL_FIT, _ORIGINAL_PROJECTED_ROADS, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_FIT = _p.fit_road_objects
    _ORIGINAL_PROJECTED_ROADS = _p.projected_road_polylines
    _p.projected_road_polylines = _projected_road_polylines
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
