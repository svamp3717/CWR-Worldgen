# SPDX-License-Identifier: GPL-3.0-or-later
"""Make explicit stock bridges and terrain causeways mutually exclusive.

Source-backed bridge detection can correctly recover a short ``bridge=yes`` that
falls between coarse terrain vertices.  The ordinary road-water causeway pass,
however, can still raise those same shared vertices to the tide-safe road floor
through adjacent approach roads.  The result is both a stock bridge and a filled
embankment beneath it.

This policy keeps tide-safe causeways for genuinely ordinary water crossings, but
for explicit mapped-water bridges it reopens coarse-grid water beneath every
contiguous mapped wet run after terrain solving.  It also runs the terrain solver
with the same stock-bridge spec used by rendering so procedural bridge underfill
never fires on a bridge that will actually be emitted as a stock object.
"""
from __future__ import annotations

from dataclasses import replace
from functools import wraps
import math

from . import bridge_render_policy as _bridge
from . import bridge_source_water_policy as _source
from . import bridge_water_deck_clamp_policy as _clamp
from . import osm as _osm
from . import terrain_solver as _terrain

_INSTALLED = False
_ORIGINAL_SOLVE = None


def _empirically_lowered_pre_tuning_height(height: float, spec) -> float:
    """Restore the tested 0.7 m lowering while retaining the tide-safe floor.

    ``bridge_final_alignment_policy`` currently returns an approach height with
    the bridge render offset already added back, so the renderer's later
    subtraction cancels the old in-game tuning.  Remove that compensation here.
    Low-bank bridges still reserve one render offset above the final 5.5 m tide
    floor so applying the renderer offset cannot put the deck back under water.
    """
    offset = float(_bridge._BRIDGE_WORLD_DOWNWARD_OFFSET_METRES)
    minimum_pre_tuning = float(_clamp._minimum_final_deck(spec)) + offset
    return max(float(height) - offset, minimum_pre_tuning)


def _explicit_bridge(feature) -> bool:
    tags = feature.tags
    bridge = str(tags.get("bridge", "")).strip().casefold()
    return (
        bridge not in {"", "no", "false", "0", "none"}
        or str(tags.get("man_made", "")).strip().casefold() == "bridge"
        or str(tags.get("special", "")).strip().casefold() == "bridge"
    )


def _mapped_water_runs(points, context):
    """Return contiguous along-road mapped-water intervals.

    Bridge planning intentionally uses a first-to-last wet envelope so one stock
    chain can span several mapped water polygons.  Terrain repair must be more
    precise: lowering that whole envelope would turn real islands or dry gaps
    into water.  Keep the wet runs separate here and refine each shoreline.
    """
    if context is None or not context.water:
        return ()
    cleaned, cumulative = _source._polyline_measure(points)
    if len(cleaned) < 2 or cumulative[-1] <= 0.01:
        return ()

    min_x = min(point[0] for point in cleaned)
    min_z = min(point[1] for point in cleaned)
    max_x = max(point[0] for point in cleaned)
    max_z = max(point[1] for point in cleaned)
    candidates = tuple(
        polygon
        for polygon in context.water
        if not (
            polygon.bounds[2] < min_x
            or polygon.bounds[0] > max_x
            or polygon.bounds[3] < min_z
            or polygon.bounds[1] > max_z
        )
    )
    if not candidates:
        return ()

    total = float(cumulative[-1])
    step = max(
        0.1,
        float(getattr(_source, "_SOURCE_WATER_SAMPLE_STEP_METRES", 0.5)),
    )
    count = max(1, int(math.ceil(total / step)))
    distances = tuple(total * index / count for index in range(count + 1))
    wet = tuple(
        _source._point_in_water(
            _source._point_at(cleaned, cumulative, distance), candidates
        )
        for distance in distances
    )

    refinement_steps = max(
        0,
        int(getattr(_source, "_SOURCE_WATER_REFINEMENT_STEPS", 12)),
    )
    runs: list[tuple[float, float]] = []
    index = 0
    while index < len(wet):
        if not wet[index]:
            index += 1
            continue
        first = index
        while index + 1 < len(wet) and wet[index + 1]:
            index += 1
        last = index

        start = distances[first]
        end = distances[last]
        if first > 0 and not wet[first - 1]:
            low, high = distances[first - 1], distances[first]
            for _ in range(refinement_steps):
                middle = (low + high) * 0.5
                point = _source._point_at(cleaned, cumulative, middle)
                if _source._point_in_water(point, candidates):
                    high = middle
                else:
                    low = middle
            start = high
        if last + 1 < len(distances) and not wet[last + 1]:
            low, high = distances[last], distances[last + 1]
            for _ in range(refinement_steps):
                middle = (low + high) * 0.5
                point = _source._point_at(cleaned, cumulative, middle)
                if _source._point_in_water(point, candidates):
                    low = middle
                else:
                    high = middle
            end = low

        if end > start + 1.0e-4:
            runs.append((start, end))
        index += 1
    return tuple(runs)


def _coarse_source_bridge_channels(dataset, projection, elevations, spec):
    """Return actual mapped wet runs beneath explicit stock bridges."""
    context = _source._make_context(dataset, projection)
    channels: list[
        tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ]
    ] = []

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

        intervals = _mapped_water_runs(points, context)
        if not intervals:
            continue

        token = _source._CONTEXT.set(context)
        try:
            # Use the final installed stock-plan chain, including connected-road
            # endpoint extension.  Its axis tells us where the rendered bridge
            # lies, while the source runs tell us which parts should remain wet.
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

        cleaned, cumulative = _source._polyline_measure(points)
        if len(cleaned) < 2:
            continue
        for wet_start_distance, wet_end_distance in intervals:
            wet_start = _source._point_at(
                cleaned, cumulative, wet_start_distance
            )
            wet_end = _source._point_at(
                cleaned, cumulative, wet_end_distance
            )
            if math.dist(wet_start, wet_end) <= 0.05:
                continue
            channels.append((wet_start, wet_end, axis))
    return tuple(channels)


def _nearest_crossing_vertices(
    centre: tuple[float, float],
    axis: tuple[float, float],
    spec,
) -> tuple[int, ...]:
    """Choose the nearest terrain-vertex row crossing one mapped wet point."""
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
    # Two vertices form the narrow coarse row crossing the bridge at this point.
    return tuple(item[2] for item in candidates[:2])


def _wet_interval_crossing_vertices(
    wet_start: tuple[float, float],
    wet_end: tuple[float, float],
    axis: tuple[float, float],
    spec,
) -> tuple[int, ...]:
    """Cover one complete mapped wet run with coarse crossing rows."""
    dx = float(wet_end[0]) - float(wet_start[0])
    dz = float(wet_end[1]) - float(wet_start[1])
    length = math.hypot(dx, dz)
    if length <= 0.05:
        centre = (
            (float(wet_start[0]) + float(wet_end[0])) * 0.5,
            (float(wet_start[1]) + float(wet_end[1])) * 0.5,
        )
        return _nearest_crossing_vertices(centre, axis, spec)

    # Half-cell longitudinal sampling guarantees that adjacent 50 m terrain rows
    # cannot be skipped even when the source shoreline falls between vertices.
    sample_step = max(1.0, float(spec.cell_size) * 0.5)
    count = max(1, int(math.ceil(length / sample_step)))
    indices: set[int] = set()
    for index in range(count + 1):
        fraction = index / count
        point = (
            float(wet_start[0]) + dx * fraction,
            float(wet_start[1]) + dz * fraction,
        )
        indices.update(_nearest_crossing_vertices(point, axis, spec))
    return tuple(sorted(indices))


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
    for wet_start, wet_end, axis in channels:
        for index in _wet_interval_crossing_vertices(
            wet_start, wet_end, axis, spec
        ):
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

    # Once bridge endpoints come from actual connected roads, retain the tested
    # 0.7 m downward visual tuning while keeping the tide-safe low-bank floor and
    # full-width ordinary-road cleanup.
    from .bridge_final_alignment_policy import install_bridge_final_alignment_policy

    install_bridge_final_alignment_policy()

    final_approach_height = _bridge._dry_approach_height

    @wraps(final_approach_height)
    def empirically_lowered_approach_height(
        endpoint, outward, raster, elevations, spec
    ):
        height = final_approach_height(
            endpoint,
            outward,
            raster,
            elevations,
            spec,
        )
        if height is None:
            return None
        return _empirically_lowered_pre_tuning_height(height, spec)

    _bridge._dry_approach_height = empirically_lowered_approach_height

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
