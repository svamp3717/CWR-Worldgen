# SPDX-License-Identifier: GPL-3.0-or-later
"""Grade only the immediate ordinary-road terrain at low stock-bridge abutments.

CWA's tide can cover nominally dry two- or three-metre shoreline terrain. The
bridge itself must still be planned from real water, so do not solve that visual
road-to-bridge gap by extending or moving the stock bridge. Instead, after mapped
water has been reopened, grade the single coarse terrain cell containing each low
but nominally dry bridge endpoint to a raised ordinary-road approach level.

The approach terrain gets an additional 0.85 m lift. This is deliberately a
terrain/road correction only: bridge position, pitch, length and module alignment
are left untouched. Test22 showed both endpoint terrain cells at about 5.45 m and
the first ordinary road pieces at about 5.48 m, leaving the stock bridge deck
visually roughly a metre above the road. The extra lift brings the approach into
the requested 0.7-1.0 m correction range.

The whole four-vertex support cell is flattened, rather than merely raising low
corners. On a 50 m WRP grid, leaving one seven-metre corner beside three lower
corners creates a bilinear terrain ramp that can cut directly through the stock
bridge deck even though the bridge modules themselves are perfectly joined.

Because this local grading changes the terrain water test, cache the bridge span
computed immediately before the grading. Later road cleanup and bridge rendering
reuse that pre-grade span, preventing the abutment from shortening the bridge on
the next planning pass.
"""
from __future__ import annotations

from dataclasses import replace
from functools import wraps

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


def _plan_key(points, spec) -> tuple[object, ...]:
    cleaned = _bridge._clean_points(points)
    coordinates = tuple(
        (round(float(point[0]), 3), round(float(point[1]), 3))
        for point in cleaned
    )
    reversed_coordinates = tuple(reversed(coordinates))
    canonical = min(coordinates, reversed_coordinates) if coordinates else coordinates
    return (
        int(getattr(spec, "cells", 0)),
        round(float(getattr(spec, "cell_size", 0.0)), 6),
        round(float(getattr(spec, "world_size", 0.0) or 0.0), 3),
        canonical,
    )


def _cached_bridge_plan(points, spec):
    """Return the wet span captured before abutment terrain was graded."""
    return _PLAN_CACHE.get(_plan_key(points, spec))


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
    """Grade at most one coarse terrain cell at each low, nominally dry endpoint."""
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
    epsilon = float(getattr(_osm, "BRIDGE_WATER_EPSILON_METRES", 0.05))
    nominal_dry_floor = float(spec.sea_level) - epsilon
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
            # Do not turn genuinely underwater bridge endpoints into causeways.
            # High banks also remain untouched. This pass exists only for the
            # low nominally-dry bank cell that CWA tide would otherwise flood or
            # leave visibly below the fixed stock bridge deck.
            if ground < nominal_dry_floor or ground >= target - 1.0e-6:
                continue

            # Grade all four support vertices to one road-ground plane. This
            # raises the road approach without changing any bridge transform.
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
    """Apply one-cell raised road grading after bridge-water reopening."""
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
