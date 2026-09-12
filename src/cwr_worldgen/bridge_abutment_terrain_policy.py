# SPDX-License-Identifier: GPL-3.0-or-later
"""Raise the immediate road approach so it physically meets a stock bridge.

The stock bridge span remains water-authoritative. Fixed 50.190 m bridge modules
can nevertheless end a short distance offshore on CWA's coarse terrain grid,
leaving the terminal ordinary-road piece submerged between dry land and the
bridge deck.

Solve that as a terrain problem, not a bridge-length problem: after the wet span
has been captured, flatten the single coarse terrain cell supporting each bridge
endpoint to the road-approach height. This applies even when the endpoint itself
is currently underwater. On the usual 50 m WRP grid that creates the smallest
possible embankment needed to expose the terminal road piece and join it to the
bridge without moving or extending the bridge.

The approach terrain retains the existing empirical 0.85 m road-side correction.
Bridge position, pitch, length and module alignment remain untouched.

The whole four-vertex support cell is flattened rather than merely raising low
corners. Otherwise bilinear terrain interpolation can leave a ramp or water notch
through the road-to-bridge joint.

Because this local grading changes the terrain water test, cache the bridge span
computed immediately before grading. Later road cleanup and bridge rendering
reuse that pre-grade span, preventing the new embankment from shortening or
moving the bridge on the next planning pass. Final rendering may ask with a
synthetic two-point component corridor instead of the original OSM polyline, so
the cache also supports a tightly bounded aligned-corridor lookup.
"""
from __future__ import annotations

from dataclasses import replace
from functools import wraps
import math

from . import bridge_or_causeway_terrain_policy as _terrain_policy
from . import bridge_render_policy as _bridge
from . import bridge_source_water_policy as _source
from . import bridge_water_deck_clamp_policy as _clamp
from . import osm as _osm
from . import terrain_solver as _terrain

_INSTALLED = False
_ORIGINAL_SOLVE = None
_PLAN_CACHE: dict[tuple[object, ...], object] = {}

# Empirical road-side correction from atinybridgetest22. Keep this on the
# terrain side of the bridge/road joint so stock bridge transforms remain
# completely unchanged.
_ROAD_APPROACH_RAISE_METRES = 0.85
_CACHE_CORRIDOR_HEADING_TOLERANCE_DEGREES = 12.0


def _plan_world_key(spec) -> tuple[object, ...]:
    return (
        int(getattr(spec, "cells", 0)),
        round(float(getattr(spec, "cell_size", 0.0)), 6),
        round(float(getattr(spec, "world_size", 0.0) or 0.0), 3),
    )


def _plan_key(points, spec) -> tuple[object, ...]:
    cleaned = _bridge._clean_points(points)
    coordinates = tuple(
        (round(float(point[0]), 3), round(float(point[1]), 3))
        for point in cleaned
    )
    reversed_coordinates = tuple(reversed(coordinates))
    canonical = min(coordinates, reversed_coordinates) if coordinates else coordinates
    return (*_plan_world_key(spec), canonical)


def _cached_bridge_plan(points, spec):
    """Return the exact wet span captured before abutment terrain was graded."""
    return _PLAN_CACHE.get(_plan_key(points, spec))


def _cached_bridge_plan_for_corridor(points, spec):
    """Match a final physical bridge corridor to its pre-fill wet plan.

    Final component reconciliation works from the already-emitted stock bridge
    objects, so it no longer has the original OSM polyline that keyed the cache.
    Reuse a cached plan only when its centre lies within the physical component,
    its axis closely matches the component axis, and its lateral displacement is
    small. This keeps the terrain fill and final bridge placement tied to the
    same plan without letting unrelated nearby bridges steal one another's span.
    """
    exact = _cached_bridge_plan(points, spec)
    if exact is not None:
        return exact

    cleaned = _bridge._clean_points(points)
    if len(cleaned) < 2:
        return None
    first = cleaned[0]
    last = cleaned[-1]
    dx = float(last[0]) - float(first[0])
    dz = float(last[1]) - float(first[1])
    length = math.hypot(dx, dz)
    if length <= 0.1:
        return None

    ux, uz = dx / length, dz / length
    midpoint = (
        (float(first[0]) + float(last[0])) * 0.5,
        (float(first[1]) + float(last[1])) * 0.5,
    )
    half_length = length * 0.5
    spacing = float(_bridge._STOCK_MODULE_SPACING_METRES)
    cell_size = max(0.0, float(getattr(spec, "cell_size", 0.0) or 0.0))
    lateral_limit = max(5.0, min(spacing * 0.5, cell_size * 0.5 or spacing * 0.5))
    minimum_alignment = math.cos(
        math.radians(_CACHE_CORRIDOR_HEADING_TOLERANCE_DEGREES)
    )
    world_key = _plan_world_key(spec)
    candidates: list[tuple[float, object]] = []

    for key, plan in _PLAN_CACHE.items():
        if tuple(key[:3]) != world_key:
            continue
        try:
            plan_first, plan_last = plan.points
        except (AttributeError, TypeError, ValueError):
            continue
        pdx = float(plan_last[0]) - float(plan_first[0])
        pdz = float(plan_last[1]) - float(plan_first[1])
        plan_length = math.hypot(pdx, pdz)
        if plan_length <= 0.1:
            continue
        pux, puz = pdx / plan_length, pdz / plan_length
        alignment = abs(pux * ux + puz * uz)
        if alignment < minimum_alignment:
            continue

        plan_midpoint = (
            (float(plan_first[0]) + float(plan_last[0])) * 0.5,
            (float(plan_first[1]) + float(plan_last[1])) * 0.5,
        )
        rel_x = plan_midpoint[0] - midpoint[0]
        rel_z = plan_midpoint[1] - midpoint[1]
        along = rel_x * ux + rel_z * uz
        lateral = abs(rel_x * uz - rel_z * ux)
        if lateral > lateral_limit:
            continue
        if abs(along) > half_length + spacing * 0.5:
            continue

        # The cached bridge itself must fit inside (or at most one stock module
        # beyond) the physical corridor supplied by the final component pass.
        fits_corridor = True
        for endpoint in (plan_first, plan_last):
            endpoint_rel_x = float(endpoint[0]) - midpoint[0]
            endpoint_rel_z = float(endpoint[1]) - midpoint[1]
            projection = endpoint_rel_x * ux + endpoint_rel_z * uz
            if abs(projection) > half_length + spacing:
                fits_corridor = False
                break
        if not fits_corridor:
            continue

        score = lateral + abs(along) * 0.01 + (1.0 - alignment) * spacing
        candidates.append((score, plan))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _endpoint_cell_vertices(point, spec) -> tuple[int, ...]:
    """Return the four WRP terrain vertices that bilinearly support one endpoint."""
    cell = float(spec.cell_size)
    fx = max(0.0, min(float(spec.cells - 1), float(point[0]) / cell))
    fz = max(0.0, min(float(spec.cells - 1), float(point[1]) / cell))
    x0 = int(fx)
    z0 = int(fz)
    x1 = min(spec.cells - 1, x0 + 1)
    z1 = min(spec.cells - 1, z0 + 1)
    return tuple(
        sorted(
            {
                z0 * spec.cells + x0,
                z0 * spec.cells + x1,
                z1 * spec.cells + x0,
                z1 * spec.cells + x1,
            }
        )
    )


def _abutment_ground_target(spec) -> float:
    """Return the raised terrain level for the ordinary-road bridge approach."""
    deck = float(_clamp._minimum_final_deck(spec))
    road_offset = max(
        0.0,
        float(getattr(_osm, "NOGOVA_BRIDGE_APPROACH_OFFSET_METRES", 0.0)),
    )
    return deck - road_offset + _ROAD_APPROACH_RAISE_METRES


def _explicit_bridge_plans(dataset, projection, elevations, spec):
    """Return wet-only stock plans for explicit bridge-tagged source roads."""
    context = _source._make_context(dataset, projection)
    result = []
    for feature, raw_points in zip(
        dataset.roads,
        _osm.projected_road_polylines(dataset, projection),
    ):
        if not _terrain_policy._explicit_bridge(feature):
            continue
        if _osm.road_bridge_crosses_ditch_only(feature, dataset, projection):
            continue
        points = tuple((float(x), float(z)) for x, z in raw_points)
        if len(points) < 2:
            continue

        token = _source._CONTEXT.set(context)
        try:
            plan = _bridge.stock_bridge_span_plan(
                points,
                elevations,
                _bridge._stock_bridge_spec(spec),
            )
        finally:
            _source._CONTEXT.reset(token)
        if plan is None or int(plan.module_count) <= 0:
            continue
        result.append((points, plan))
    return tuple(result)


def _raise_bridge_abutments(report, dataset, projection, spec):
    """Raise one coarse support cell at each bridge end to the road-join height."""
    plans = _explicit_bridge_plans(
        dataset,
        projection,
        report.elevations,
        spec,
    )
    if not plans:
        return report

    values = list(report.elevations)
    target = _abutment_ground_target(spec)
    touched: set[int] = set()

    for points, plan in plans:
        # Capture the water-authoritative span before this local embankment
        # changes the terrain sampler used by subsequent planning calls.
        _PLAN_CACHE[_plan_key(points, spec)] = plan

        for endpoint in plan.points:
            ground = float(
                _osm._sample_elevation(
                    values,
                    spec.cells,
                    spec.cell_size,
                    float(endpoint[0]),
                    float(endpoint[1]),
                )
            )
            # A fixed stock module can legitimately end just offshore. That is
            # exactly where the terminal road piece otherwise disappears below
            # water. Raise that one support cell too; only already-high banks are
            # left untouched.
            if ground >= target - 1.0e-6:
                continue

            for index in _endpoint_cell_vertices(endpoint, spec):
                if abs(float(values[index]) - target) > 1.0e-7:
                    values[index] = target
                    touched.add(index)

    if not touched:
        return report

    return replace(
        report,
        elevations=tuple(values),
        changed_cells=int(getattr(report, "changed_cells", 0)) + len(touched),
    )


def install_bridge_abutment_terrain_policy() -> None:
    """Apply one-cell road-approach fill after bridge-water reopening."""
    global _INSTALLED, _ORIGINAL_SOLVE
    if _INSTALLED:
        return

    _ORIGINAL_SOLVE = _terrain.solve_terrain_constraints

    @wraps(_ORIGINAL_SOLVE)
    def raised_bridge_abutment_solve(
        elevations, dataset, projection, raster, spec, *args, **kwargs
    ):
        # Never let a previous world/build influence planning inside this solve.
        _PLAN_CACHE.clear()
        report = _ORIGINAL_SOLVE(
            elevations,
            dataset,
            projection,
            raster,
            spec,
            *args,
            **kwargs,
        )
        return _raise_bridge_abutments(
            report,
            dataset,
            projection,
            _bridge._stock_bridge_spec(spec),
        )

    _terrain.solve_terrain_constraints = raised_bridge_abutment_solve
    _INSTALLED = True
