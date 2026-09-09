# SPDX-License-Identifier: GPL-3.0-or-later
"""Install the complete stock-bridge policy chain used by real builds.

Bridge length is water-authoritative, but CWA's sea surface can rise roughly five
metres above nominal terrain sea level.  A bridge clipped only to terrain below
zero can therefore stop on a nominally dry two-metre bank that is visibly under
water in game.

The final runtime planner keeps the nominal/source-water bridge as its base, then
extends only far enough to cover the nearest maximum-tide shoreline on each
connected approach.  It never walks hundreds of metres inland merely to find a
road surface at bridge-deck height.
"""
from __future__ import annotations

from dataclasses import replace
import math

_INSTALLED = False

_OFP_MAX_TIDE_METRES = 5.0
_TIDE_SHORE_CLEARANCE_METRES = 0.05
_TIDE_SHORE_SAMPLE_STEP_METRES = 0.5
_MAX_TIDE_BANK_PROBE_MODULES = 1.25


def _first_max_tide_dry_point(path, elevations, spec):
    """Return the first connected-road point whose terrain stays above max tide."""
    from . import bridge_render_policy as _bridge
    from . import osm as _osm

    cleaned, cumulative = _bridge._polyline_measure(path)
    if len(cleaned) < 2 or not cumulative or cumulative[-1] <= 0.01:
        return None

    threshold = (
        float(spec.sea_level)
        + _OFP_MAX_TIDE_METRES
        + _TIDE_SHORE_CLEARANCE_METRES
    )
    maximum_probe = min(
        float(cumulative[-1]),
        float(_bridge._STOCK_MODULE_SPACING_METRES)
        * _MAX_TIDE_BANK_PROBE_MODULES,
    )
    if maximum_probe <= 0.01:
        return None

    distance = 0.0
    while distance < maximum_probe - 1.0e-9:
        point = _bridge._point_at_measure(cleaned, cumulative, distance)
        ground = float(
            _osm._sample_elevation(
                elevations,
                spec.cells,
                spec.cell_size,
                float(point[0]),
                float(point[1]),
            )
        )
        if ground >= threshold:
            return float(point[0]), float(point[1])
        distance += _TIDE_SHORE_SAMPLE_STEP_METRES

    point = _bridge._point_at_measure(cleaned, cumulative, maximum_probe)
    ground = float(
        _osm._sample_elevation(
            elevations,
            spec.cells,
            spec.cell_size,
            float(point[0]),
            float(point[1]),
        )
    )
    if ground >= threshold:
        return float(point[0]), float(point[1])
    return None


def _maximum_tide_stock_plan(plan, points, elevations, spec):
    """Expand a wet-only stock chain just enough to cover CWA's maximum tide.

    The expansion follows the real connected road at each source bridge end only
    to locate the first terrain above maximum tide.  Those points are projected
    onto the already-selected straight stock-bridge axis, preserving alignment.
    Whole-module rounding can add at most the normal stock-model excess; it does
    not search for an exact endpoint match farther inland.
    """
    if plan is None:
        return None

    from . import bridge_render_policy as _bridge
    from . import bridge_road_connection_policy as _road
    from . import bridge_source_water_policy as _source
    from . import osm as _osm

    context = _source._CONTEXT.get()
    if context is None:
        return plan

    matched = _road._matching_bridge_feature(points, context, spec)
    if matched is None:
        return plan
    feature, projected = matched
    if len(projected) < 2:
        return plan

    probe = min(
        float(_road._APPROACH_PROBE_METRES),
        float(_bridge._STOCK_MODULE_SPACING_METRES)
        * _MAX_TIDE_BANK_PROBE_MODULES,
    )
    start_path = _osm._connected_bridge_approach_path(
        feature,
        context.dataset,
        context.projection,
        projected[0],
        projected[1],
        spec,
        probe,
    )
    end_path = _osm._connected_bridge_approach_path(
        feature,
        context.dataset,
        context.projection,
        projected[-1],
        projected[-2],
        spec,
        probe,
    )

    start_dry = _first_max_tide_dry_point(
        start_path, elevations, spec
    )
    end_dry = _first_max_tide_dry_point(
        end_path, elevations, spec
    )
    if start_dry is None and end_dry is None:
        return plan

    bridge_start, bridge_end = plan.points
    dx = float(bridge_end[0]) - float(bridge_start[0])
    dz = float(bridge_end[1]) - float(bridge_start[1])
    current_length = math.hypot(dx, dz)
    if current_length <= 0.1:
        return plan
    axis = (dx / current_length, dz / current_length)

    def projection(point):
        return (
            (float(point[0]) - float(bridge_start[0])) * axis[0]
            + (float(point[1]) - float(bridge_start[1])) * axis[1]
        )

    target_min = 0.0
    target_max = current_length
    tide_min = projection(plan.wet_start)
    tide_max = projection(plan.wet_end)

    if start_dry is not None:
        value = projection(start_dry)
        target_min = min(target_min, value)
        tide_min = min(tide_min, value)
    if end_dry is not None:
        value = projection(end_dry)
        target_max = max(target_max, value)
        tide_max = max(tide_max, value)

    required_length = target_max - target_min
    module_length = float(_bridge._STOCK_MODULE_SPACING_METRES)
    tolerance = max(1.0e-6, module_length * 1.0e-9)
    module_count = max(
        int(plan.module_count),
        1,
        int(math.ceil((required_length - tolerance) / module_length)),
    )

    # Nothing lies outside the existing chain; keep its exact placement.
    if (
        module_count == int(plan.module_count)
        and target_min >= -1.0e-6
        and target_max <= current_length + 1.0e-6
    ):
        return plan

    target_length = module_count * module_length
    centre_measure = (target_min + target_max) * 0.5
    centre = (
        float(bridge_start[0]) + axis[0] * centre_measure,
        float(bridge_start[1]) + axis[1] * centre_measure,
    )
    new_start = (
        centre[0] - axis[0] * target_length * 0.5,
        centre[1] - axis[1] * target_length * 0.5,
    )
    new_end = (
        centre[0] + axis[0] * target_length * 0.5,
        centre[1] + axis[1] * target_length * 0.5,
    )

    world_size = float(getattr(spec, "world_size", 0.0) or 0.0)
    if world_size > 0.0 and not all(
        0.0 <= coordinate < world_size
        for point in (new_start, new_end)
        for coordinate in point
    ):
        return plan

    tide_start = (
        float(bridge_start[0]) + axis[0] * tide_min,
        float(bridge_start[1]) + axis[1] * tide_min,
    )
    tide_end = (
        float(bridge_start[0]) + axis[0] * tide_max,
        float(bridge_start[1]) + axis[1] * tide_max,
    )
    return replace(
        plan,
        points=(new_start, new_end),
        module_count=module_count,
        wet_start=tide_start,
        wet_end=tide_end,
        wet_length=max(0.0, tide_max - tide_min),
    )


def _runtime_stock_bridge_span_plan(points, elevations, spec):
    """Use nominal/source water, then cover only the nearby max-tide shoreline."""
    from . import bridge_source_water_policy as _source

    plan = _source._mapped_water_stock_plan(points, elevations, spec)
    return _maximum_tide_stock_plan(plan, points, elevations, spec)


def install_bridge_runtime_policy() -> None:
    """Install bridge policies and finish with the minimal max-tide stock planner."""
    global _INSTALLED
    if _INSTALLED:
        return

    # Rendering must capture the unwrapped OSM/generator functions first.
    from .bridge_render_policy import install_bridge_render_policy

    install_bridge_render_policy()

    # Retain the tide-safe vertical deck sampler. Its historical horizontal
    # extension is bypassed by the final planner below.
    from .bridge_water_deck_clamp_policy import (
        install_bridge_water_deck_clamp_policy,
    )

    install_bridge_water_deck_clamp_policy()

    # Underlay cleanup must precede final alignment, which wraps its road fitter
    # and footprint test. Source-water then makes cleanup source-aware as well.
    from .bridge_underlay_cleanup_policy import install_bridge_underlay_cleanup_policy

    install_bridge_underlay_cleanup_policy()

    from .bridge_source_water_policy import install_bridge_source_water_policy

    install_bridge_source_water_policy()

    # Install diagnostics/alignment and post-solver water reopening. Their old
    # connected-road horizontal planner is intentionally superseded below.
    from .bridge_or_causeway_terrain_policy import (
        install_bridge_or_causeway_terrain_policy,
    )

    install_bridge_or_causeway_terrain_policy()

    from . import bridge_render_policy as _bridge
    from . import bridge_source_water_policy as _source
    from . import bridge_water_deck_clamp_policy as _clamp

    # Start source-water fallback directly from the original wet-only renderer,
    # bypassing the old "find a 5.5 m road and make the bridge reach it" wrapper.
    base_wet_plan = _clamp._ORIGINAL_STOCK_BRIDGE_SPAN_PLAN
    if base_wet_plan is None:
        raise RuntimeError("bridge tide policy did not capture the wet stock planner")
    _source._ORIGINAL_STOCK_PLAN = base_wet_plan
    _bridge.stock_bridge_span_plan = _runtime_stock_bridge_span_plan

    # Placement/terrain cache keys predate these runtime wrappers. Test19 used
    # the v3 wet-only planner, so force a fresh placement/terrain generation.
    from . import build_cache_policy as _build_cache

    _build_cache.BUILD_CACHE_REVISION = "v4-maximum-tide-bridge-envelope"
    _INSTALLED = True
