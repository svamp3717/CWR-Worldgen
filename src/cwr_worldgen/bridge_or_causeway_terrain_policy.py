# SPDX-License-Identifier: GPL-3.0-or-later
"""Make explicit stock bridges and terrain causeways mutually exclusive.

Source-backed bridge detection can correctly recover a short ``bridge=yes`` that
falls between coarse terrain vertices.  The ordinary road-water causeway pass,
however, can still raise those same shared vertices to the tide-safe road floor
through adjacent approach roads.  The result is both a stock bridge and a filled
embankment beneath it.

This policy keeps tide-safe causeways for genuinely ordinary water crossings, but
for explicit mapped-water bridges it reopens a coarse-grid water channel beneath
the actual mapped wet interval after terrain solving.  It also runs the terrain
solver with the same stock-bridge spec used by rendering so procedural bridge
underfill never fires on a bridge that will actually be emitted as a stock object.
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
    """Return mapped wet centres for stock bridges missed by coarse terrain."""
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
            # Use the final installed stock-plan chain, including connected-road
            # endpoint extension. The water opening still belongs at the mapped
            # wet interval, not beneath every bridge module sitting on dry bank.
            plan = _bridge.stock_bridge_span_plan(
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
        wet_centre = (
            (float(plan.wet_start[0]) + float(plan.wet_end[0])) * 0.5,
            (float(plan.wet_start[1]) + float(plan.wet_end[1])) * 0.5,
        )
        channels.append((wet_centre, axis))
    return tuple(channels)


def _nearest_crossing_vertices(
    centre: tuple[float, float],
    axis: tuple[float, float],
    spec,
) -> tuple[int, ...]:
    """Choose the nearest terrain-vertex row crossing the mapped wet centre."""
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
    # Two vertices make one coarse row cross the road. Do not lower the next
    # terrain row merely because a second stock module extends onto dry land.
    return tuple(item[2] for item in candidates[:2])


def _water_target(spec) -> float:
    depth = max(0.35, float(getattr(spec, "water_depth", 0.35) or 0.35))
    # A roughly three-metre centre channel keeps the interpolated shoreline
    # inside a stock bridge on the usual 50 m terrain grid. Going to the full
    # five-metre bed can move the shoreline beyond the bridge abutment.
    return float(spec.sea_level) - min(3.0, depth)


def _reopen_bridge_water(report, elevations, dataset, projection, spec):
    # Plan against the solved pre-reopen terrain, the same surface the road
    # fitter and later bridge renderer will see. Using raw DEM elevations here
    # can choose different approach endpoints and reopen water under the wrong
    # stock span.
    channels = _coarse_source_bridge_channels(
        dataset, projection, report.elevations, spec
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
    """Ensure an emitted stock bridge keeps water and reaches real road ends."""
    global _INSTALLED, _ORIGINAL_SOLVE
    if _INSTALLED:
        return

    # Horizontal stock span selection must know the real connected road on both
    # banks before this terrain wrapper decides where the water opening belongs.
    from .bridge_road_connection_policy import install_bridge_road_connection_policy

    install_bridge_road_connection_policy()

    # Once bridge endpoints come from actual connected roads, finish the job:
    # cancel the legacy 0.7 m deck lowering at those sampled road heights and
    # remove ordinary road centres across the full measured stock-bridge width.
    from .bridge_final_alignment_policy import install_bridge_final_alignment_policy

    install_bridge_final_alignment_policy()

    # The deck can be vertically correct and still expose a wall-like bridge
    # corner when the fitted road arrives at a different heading. Replace the
    # straight-only abutment filler with stock 10-degree tangent transitions.
    from .bridge_tangent_transition_policy import (
        install_bridge_tangent_transition_policy,
    )

    install_bridge_tangent_transition_policy()

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
