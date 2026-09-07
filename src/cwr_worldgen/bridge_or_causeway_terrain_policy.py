# SPDX-License-Identifier: GPL-3.0-or-later
"""Make explicit stock bridges and terrain causeways mutually exclusive.

Source-backed bridge detection can correctly recover a short ``bridge=yes`` that
falls between coarse terrain vertices.  The ordinary road-water causeway pass,
however, can still raise those same shared vertices to the tide-safe road floor
through adjacent approach roads.  The result is both a stock bridge and a filled
embankment beneath it.

This policy keeps tide-safe causeways for genuinely ordinary water crossings, but
for explicit mapped-water bridges it reopens a coarse-grid water channel beneath
the stock span after terrain solving.  It also runs the terrain solver with the
same stock-bridge spec used by rendering so procedural bridge-underfill never
fires on a bridge that will actually be emitted as a stock object.
"""
from __future__ import annotations

from dataclasses import replace
from functools import wraps
import math

from . import bridge_render_policy as _bridge
from . import bridge_source_water_policy as _source
from . import osm as _osm
from . import terrain_solver as _terrain

_INSTALLED = False
_ORIGINAL_SOLVE = None


def _explicit_bridge(feature) -> bool:
    tags = feature.tags
    bridge = str(tags.get("bridge", "")).strip().casefold()
    return (
        bridge not in {"", "no", "false", "0", "none"}
        or str(tags.get("man_made", "")).strip().casefold() == "bridge"
        or str(tags.get("special", "")).strip().casefold() == "bridge"
    )


def _coarse_source_bridge_channels(dataset, projection, elevations, spec):
    """Return stock-plan centre samples whose mapped water is missed by terrain."""
    context = _source._make_context(dataset, projection)
    channels: list[tuple[tuple[float, float], tuple[float, float]]] = []

    for feature, raw_points in zip(
        dataset.roads,
        _osm.projected_road_polylines(dataset, projection),
    ):
        if not _explicit_bridge(feature):
            continue
        if _osm.road_bridge_crosses_ditch_only(feature, dataset, projection):
            continue
        points = tuple((float(x), float(z)) for x, z in raw_points)
        if len(points) < 2:
            continue

        width = max(6.0, _osm.road_width_metres(feature.tags))
        if _source._ORIGINAL_WATER_TEST(
            points,
            elevations,
            cells=spec.cells,
            cell_size=spec.cell_size,
            sea_level=spec.sea_level,
            width=width,
        ):
            # Existing terrain already preserves real water. Do not touch it.
            continue

        interval = _source._source_mapped_water_interval(points, context)
        if interval is None:
            continue

        token = _source._CONTEXT.set(context)
        try:
            plan = _source._mapped_water_stock_plan(
                points,
                elevations,
                _bridge._stock_bridge_spec(spec),
            )
        finally:
            _source._CONTEXT.reset(token)
        if plan is None or plan.module_count <= 0:
            continue

        start, end = plan.points
        dx = float(end[0]) - float(start[0])
        dz = float(end[1]) - float(start[1])
        length = math.hypot(dx, dz)
        if length <= 0.1:
            continue
        axis = (dx / length, dz / length)
        for index in range(plan.module_count):
            fraction = (index + 0.5) / plan.module_count
            centre = (
                float(start[0]) + dx * fraction,
                float(start[1]) + dz * fraction,
            )
            channels.append((centre, axis))
    return tuple(channels)


def _nearest_crossing_vertices(
    centre: tuple[float, float],
    axis: tuple[float, float],
    spec,
) -> tuple[int, ...]:
    """Choose the nearest terrain-vertex row crossing the bridge centre."""
    cell = float(spec.cell_size)
    fx = max(0.0, min(float(spec.cells - 1), float(centre[0]) / cell))
    fz = max(0.0, min(float(spec.cells - 1), float(centre[1]) / cell))
    x0 = int(math.floor(fx))
    z0 = int(math.floor(fz))
    x1 = min(spec.cells - 1, x0 + 1)
    z1 = min(spec.cells - 1, z0 + 1)

    candidates: list[tuple[float, float, int]] = []
    ux, uz = axis
    nx, nz = -uz, ux
    for x_index, z_index in {
        (x0, z0), (x1, z0), (x0, z1), (x1, z1)
    }:
        x = x_index * cell
        z = z_index * cell
        dx = x - float(centre[0])
        dz = z - float(centre[1])
        longitudinal = abs(dx * ux + dz * uz)
        lateral = abs(dx * nx + dz * nz)
        candidates.append(
            (longitudinal, lateral, z_index * spec.cells + x_index)
        )

    candidates.sort()
    # Two vertices are enough to make one coarse terrain row cross the road
    # without sinking the next road segment beyond the stock bridge end.
    return tuple(item[2] for item in candidates[:2])


def _water_target(spec) -> float:
    depth = max(0.35, float(getattr(spec, "water_depth", 0.35) or 0.35))
    # A roughly three-metre centre channel keeps the interpolated shoreline
    # inside a 50 m stock bridge on the usual 50 m terrain grid. Going to the
    # full five-metre bed can move the shoreline beyond the bridge abutment.
    return float(spec.sea_level) - min(3.0, depth)


def _reopen_bridge_water(report, elevations, dataset, projection, spec):
    channels = _coarse_source_bridge_channels(
        dataset, projection, elevations, spec
    )
    if not channels:
        return report

    values = list(report.elevations)
    target = _water_target(spec)
    touched: set[int] = set()
    for centre, axis in channels:
        for index in _nearest_crossing_vertices(centre, axis, spec):
            if values[index] > target:
                values[index] = target
                touched.add(index)

    if not touched:
        return report

    changed_cells = sum(
        1
        for before, after in zip(elevations, values)
        if abs(float(before) - float(after)) > 1.0e-7
    )
    return replace(
        report,
        elevations=tuple(values),
        changed_cells=changed_cells,
    )


def install_bridge_or_causeway_terrain_policy() -> None:
    """Ensure an emitted stock bridge keeps water instead of a filled causeway."""
    global _INSTALLED, _ORIGINAL_SOLVE
    if _INSTALLED:
        return

    _ORIGINAL_SOLVE = _terrain.solve_terrain_constraints

    @wraps(_ORIGINAL_SOLVE)
    def bridge_or_causeway_solve(
        elevations, dataset, projection, raster, spec, *args, **kwargs
    ):
        # Rendering is forced to stock bridges on this branch. Give terrain the
        # same spec so the old procedural bridge-underfill path cannot run first.
        stock_spec = _bridge._stock_bridge_spec(spec)
        report = _ORIGINAL_SOLVE(
            elevations,
            dataset,
            projection,
            raster,
            stock_spec,
            *args,
            **kwargs,
        )
        return _reopen_bridge_water(
            report,
            elevations,
            dataset,
            projection,
            stock_spec,
        )

    _terrain.solve_terrain_constraints = bridge_or_causeway_solve
    _INSTALLED = True
