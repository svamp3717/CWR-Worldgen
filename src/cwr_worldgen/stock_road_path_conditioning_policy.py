# SPDX-License-Identifier: GPL-3.0-or-later
"""Condition stock-road centerlines before rigid CWA P3Ds are fitted.

Arma-style terrain pipelines get markedly better roads when source geometry is
cleaned before the game-specific representation is generated. CWA cannot hand a
polyline to the engine like newer Arma versions can; it must place rigid road P3Ds
itself. That makes pre-fitting conditioning even more valuable:

* compatible OSM way fragments are merged into one continuous path where their
  endpoints have one unambiguous same-type continuation;
* sub-metre source noise may be simplified before curve selection rather than
  being serialized as several rotated 6.25 m straight pieces; and
* real junctions, surface transitions and sharp corners remain explicit anchors.

The simplifier never moves an endpoint and only replaces an existing sub-path by
its chord. The chord must remain within a 0.50 m source corridor and pass the
same source-backed obstacle check used by the later road-relaxation policy.
"""
from __future__ import annotations

from contextvars import ContextVar
import math
from typing import Sequence

from . import generator as _generator
from . import playability as _p
from . import stock_road_connector_policy as _connector
from . import stock_road_model_geometry as _geometry
from . import stock_road_obstacle_policy as _obstacles
from . import stock_road_relaxation_policy as _relax

MAXIMUM_PRE_FIT_DEVIATION_METRES = 0.50
MAXIMUM_PRE_FIT_CHORD_METRES = 100.0
MAXIMUM_PRE_FIT_LOOKAHEAD_POINTS = 48
SHARP_CORNER_ANCHOR_DEGREES = 45.0

_SPEC: ContextVar[object | None] = ContextVar(
    "cwr_stock_road_path_conditioning_spec", default=None
)
_ORIGINAL_FIT = None
_ORIGINAL_PROJECTED_ROADS = None
_INSTALLED = False


def _normalised_special(tags) -> tuple[str, str, str, str]:
    """Keep bridge/tunnel/layer semantics from being merged across each other."""

    def value(name: str) -> str:
        return str(tags.get(name, "")).strip().casefold()

    return value("bridge"), value("tunnel"), value("layer"), value("ford")


def _compatibility_key(feature, spec):
    """Return the visual/fitting class used for deterministic path merging."""

    if not _p.road_is_supported(
        feature.tags, include_minor=bool(getattr(spec, "include_minor_roads", False))
    ):
        return None
    model = _p.road_model_for_tags(spec, feature.tags)
    # Only the verified stock straight families participate. Generated gravel has
    # its own flexible curve family, and custom models must not silently inherit
    # assumptions measured from the CWA O\Road assets.
    stock = _geometry.stock_straight_match(str(model))
    if stock is None:
        return None
    width = float(_p.road_width_metres(feature.tags))
    return (
        stock.group("family").casefold(),
        str(model).replace("/", "\\").casefold(),
        round(width, 3),
        bool(_p.road_is_dirt(feature.tags)),
        _normalised_special(feature.tags),
    )


def _clean(points: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    return tuple(_p._clean_road_points(points))


def _merge_two(first_points, first_side: int, second_points, second_side: int):
    """Join two paths at one endpoint while preserving every source vertex."""

    a = list(first_points)
    b = list(second_points)
    if first_side == 0:
        a.reverse()
    if second_side == 1:
        b.reverse()
    if not a or not b:
        return tuple(a or b)
    if _p._road_node_key(a[-1]) != _p._road_node_key(b[0]):
        raise ValueError("road merge endpoints do not share a node")
    # Projection of the same OSM node should be identical. When floating point
    # conversion leaves a tiny discrepancy, keep the first path's exact endpoint
    # rather than introducing a new averaged source position.
    b[0] = a[-1]
    return tuple(a + b[1:])


def _merge_compatible_paths(projected, compatibility):
    """Merge chains at nodes with exactly one compatible continuation.

    The lowest source index owns the merged chain and absorbed entries become
    empty. Junction vertices are not discarded; a branch of another road type
    can therefore still meet an intermediate node on the merged main road.
    """

    output = [tuple(points) for points in projected]
    chains: dict[int, tuple[tuple[int, ...], tuple[tuple[float, float], ...], object]] = {}
    next_id = 0
    for index, (points_raw, key) in enumerate(zip(projected, compatibility)):
        points = _clean(points_raw)
        if key is None or len(points) < 2:
            continue
        chains[next_id] = ((index,), points, key)
        next_id += 1

    while True:
        endpoints = {}
        for chain_id, (_indices, points, key) in chains.items():
            endpoints.setdefault((key, _p._road_node_key(points[0])), []).append(
                (chain_id, 0)
            )
            endpoints.setdefault((key, _p._road_node_key(points[-1])), []).append(
                (chain_id, 1)
            )

        candidate = None
        for endpoint_key in sorted(endpoints, key=lambda value: repr(value)):
            values = endpoints[endpoint_key]
            if len(values) != 2 or values[0][0] == values[1][0]:
                continue
            candidate = values
            break
        if candidate is None:
            break

        (first_id, first_side), (second_id, second_side) = candidate
        first_indices, first_points, key = chains.pop(first_id)
        second_indices, second_points, second_key = chains.pop(second_id)
        if key != second_key:
            raise AssertionError("incompatible road chains selected for merge")
        merged_points = _merge_two(
            first_points, first_side, second_points, second_side
        )
        merged_indices = tuple(sorted((*first_indices, *second_indices)))
        chains[next_id] = (merged_indices, merged_points, key)
        next_id += 1

    for indices, points, _key in chains.values():
        owner = min(indices)
        output[owner] = points
        for index in indices:
            if index != owner:
                output[index] = ()
    return tuple(output)


def _point_segment_distance(point, start, end) -> float:
    dx, dz = end[0] - start[0], end[1] - start[1]
    length2 = dx * dx + dz * dz
    if length2 <= 1.0e-12:
        return math.dist(point, start)
    fraction = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dz
    ) / length2
    fraction = max(0.0, min(1.0, fraction))
    nearest = (start[0] + dx * fraction, start[1] + dz * fraction)
    return math.dist(point, nearest)


def _protected_node_keys(paths, compatibility) -> set[tuple[int, int]]:
    """Protect real network decisions and sharp corners from simplification."""

    incidents: dict[tuple[int, int], list[object]] = {}
    protected: set[tuple[int, int]] = set()
    for points_raw, key in zip(paths, compatibility):
        points = _clean(points_raw)
        if len(points) < 2:
            continue
        for start, end in zip(points, points[1:]):
            if math.dist(start, end) <= 0.05:
                continue
            incidents.setdefault(_p._road_node_key(start), []).append(key)
            incidents.setdefault(_p._road_node_key(end), []).append(key)
        for index in range(1, len(points) - 1):
            if (
                _p._turn_degrees(points[index - 1], points[index], points[index + 1])
                >= SHARP_CORNER_ANCHOR_DEGREES
            ):
                protected.add(_p._road_node_key(points[index]))

    for node, values in incidents.items():
        # Degree-one endpoints, T/X junctions, and changes of fitting class are
        # anchors. Only a degree-two continuation of one road class is eligible
        # to disappear as harmless source-line detail.
        if len(values) != 2 or len(set(values)) != 1:
            protected.add(node)
    return protected


def _shortcut_is_safe(points, first, last, protected, obstacles) -> bool:
    if last <= first + 1:
        return True
    start, end = points[first], points[last]
    if math.dist(start, end) > MAXIMUM_PRE_FIT_CHORD_METRES:
        return False
    for index in range(first + 1, last):
        point = points[index]
        if _p._road_node_key(point) in protected:
            return False
        if (
            _point_segment_distance(point, start, end)
            > MAXIMUM_PRE_FIT_DEVIATION_METRES
        ):
            return False
    if obstacles is None:
        return False
    return _relax._shortcut_clear(obstacles, start, end)


def _simplify_path(points_raw, protected, obstacles):
    """Greedily replace harmless source noise by the longest safe chord."""

    points = _clean(points_raw)
    if len(points) < 3:
        return points
    result = [points[0]]
    first = 0
    while first < len(points) - 1:
        best = first + 1
        limit = min(len(points) - 1, first + MAXIMUM_PRE_FIT_LOOKAHEAD_POINTS)
        for last in range(first + 2, limit + 1):
            if math.dist(points[first], points[last]) > MAXIMUM_PRE_FIT_CHORD_METRES:
                break
            if _shortcut_is_safe(points, first, last, protected, obstacles):
                best = last
        result.append(points[best])
        first = best
    return tuple(result)


def _condition_paths(projected, compatibility, obstacles):
    merged = _merge_compatible_paths(projected, compatibility)
    protected = _protected_node_keys(merged, compatibility)
    output = []
    for points, key in zip(merged, compatibility):
        if key is None or len(points) < 3:
            output.append(tuple(points))
        else:
            output.append(_simplify_path(points, protected, obstacles))
    return tuple(output)


def _projected_road_polylines(dataset, projection):
    if _ORIGINAL_PROJECTED_ROADS is None:
        raise RuntimeError("stock road path conditioning policy is not installed")
    projected = _ORIGINAL_PROJECTED_ROADS(dataset, projection)
    spec = _SPEC.get()
    if spec is None:
        return projected
    compatibility = tuple(_compatibility_key(feature, spec) for feature in dataset.roads)
    context = _relax._CONTEXT.get()
    obstacles = None if context is None else context.obstacles
    return _condition_paths(projected, compatibility, obstacles)


def _fit(dataset, projection, elevations, spec, *, starting_id=1, progress_callback=None):
    if _ORIGINAL_FIT is None:
        raise RuntimeError("stock road path conditioning policy is not installed")
    token = _SPEC.set(spec)
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
        _SPEC.reset(token)


def install_stock_road_path_conditioning_policy() -> None:
    global _ORIGINAL_FIT, _ORIGINAL_PROJECTED_ROADS, _INSTALLED
    if _INSTALLED:
        return
    if _connector._ORIGINAL_PROJECTED_ROADS is None:
        raise RuntimeError("stock road connector policy must be installed first")

    # Add mapped fences/walls/hedges and tree rows to the same obstacle index
    # before any pre-fit or junction relaxation can consume it.
    _obstacles.install_stock_road_obstacle_policy()
    _ORIGINAL_FIT = _p.fit_road_objects
    _ORIGINAL_PROJECTED_ROADS = _connector._ORIGINAL_PROJECTED_ROADS
    _connector._ORIGINAL_PROJECTED_ROADS = _projected_road_polylines
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
