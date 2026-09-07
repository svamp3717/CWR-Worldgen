# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep clipped stock bridges above OFP/CWA tides while preserving dry road.

OFP's published tide formula uses ``maxTide = 5`` metres.  A bridge deck merely
above terrain sea level can therefore be submerged in game even though its WRP
coordinates are positive.  The wet-only bridge clipping policy exposed exactly
that problem by moving bridge ends down to low shoreline terrain.

This policy keeps the clipped-bridge design, but expands each wet span only as
far onto each bank as needed to reach a tide-safe ordinary-road elevation.  Dry
bridge-tagged road beyond those points remains fitted with normal road pieces.
The final deck floor also reserves the existing bridge-render tuning offset so
that offset can never lower a stock bridge into the tidal range.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import bridge_render_policy as _bridge
from . import osm as _osm

_ORIGINAL_DRY_APPROACH_HEIGHT = None
_ORIGINAL_STOCK_BRIDGE_SPAN_PLAN = None
_INSTALLED = False

# BIS/OFP published tide code uses ``const float maxTide = 5``.  Leave another
# half metre for wave crest/rendering variation so bridges remain visibly dry.
_OFP_MAX_TIDE_METRES = 5.0
_TIDE_WAVE_MARGIN_METRES = 0.5
_BANK_SEARCH_STEP_METRES = 2.0


def _minimum_final_deck(spec) -> float:
    return (
        float(spec.sea_level)
        + _OFP_MAX_TIDE_METRES
        + _TIDE_WAVE_MARGIN_METRES
    )


def _minimum_pre_tuning_deck(spec) -> float:
    return (
        _minimum_final_deck(spec)
        + float(_bridge._BRIDGE_WORLD_DOWNWARD_OFFSET_METRES)
    )


def _safe_dry_approach_height(
    endpoint,
    outward,
    raster,
    elevations,
    spec,
):
    height = _ORIGINAL_DRY_APPROACH_HEIGHT(
        endpoint,
        outward,
        raster,
        elevations,
        spec,
    )
    if height is None:
        return None
    return max(float(height), _minimum_pre_tuning_deck(spec))


def _nearest_measure(points, cumulative, point) -> float:
    """Return along-polyline distance of the closest point on ``points``."""

    if len(points) < 2:
        return 0.0
    px, pz = float(point[0]), float(point[1])
    best: tuple[float, float] | None = None
    for index, (start, end) in enumerate(zip(points, points[1:])):
        dx = float(end[0]) - float(start[0])
        dz = float(end[1]) - float(start[1])
        length2 = dx * dx + dz * dz
        if length2 <= 1.0e-12:
            continue
        fraction = (
            (px - float(start[0])) * dx
            + (pz - float(start[1])) * dz
        ) / length2
        fraction = max(0.0, min(1.0, fraction))
        nearest_x = float(start[0]) + dx * fraction
        nearest_z = float(start[1]) + dz * fraction
        distance = math.hypot(px - nearest_x, pz - nearest_z)
        along = float(cumulative[index]) + math.sqrt(length2) * fraction
        candidate = (distance, along)
        if best is None or candidate < best:
            best = candidate
    return best[1] if best is not None else 0.0


def _bank_is_tide_safe(points, cumulative, distance, elevations, spec) -> bool:
    ground = _bridge._ground_at_measure(
        points, cumulative, distance, elevations, spec
    )
    road_surface = ground + float(_osm.NOGOVA_BRIDGE_APPROACH_OFFSET_METRES)
    return road_surface >= _minimum_pre_tuning_deck(spec)


def _search_tide_safe_bank(
    points,
    cumulative,
    start_distance: float,
    direction: int,
    elevations,
    spec,
) -> float | None:
    """Walk away from water to the nearest bank point safe above maximum tide."""

    total = float(cumulative[-1])
    distance = max(0.0, min(total, float(start_distance)))
    direction = -1 if direction < 0 else 1
    while 0.0 <= distance <= total:
        if _bank_is_tide_safe(
            points, cumulative, distance, elevations, spec
        ):
            return distance
        next_distance = distance + direction * _BANK_SEARCH_STEP_METRES
        if direction < 0:
            if distance <= 0.0:
                break
            distance = max(0.0, next_distance)
        else:
            if distance >= total:
                break
            distance = min(total, next_distance)
    return None


def _tide_safe_stock_bridge_span_plan(points, elevations, spec):
    """Expand the wet-only stock span only to the nearest tide-safe bank road."""

    plan = _ORIGINAL_STOCK_BRIDGE_SPAN_PLAN(points, elevations, spec)
    if plan is None:
        return None

    cleaned, cumulative = _bridge._polyline_measure(points)
    if len(cleaned) < 2 or not cumulative or cumulative[-1] <= 0.1:
        return plan

    wet_start_distance = _nearest_measure(
        cleaned, cumulative, plan.wet_start
    )
    wet_end_distance = _nearest_measure(
        cleaned, cumulative, plan.wet_end
    )
    if wet_end_distance < wet_start_distance:
        wet_start_distance, wet_end_distance = (
            wet_end_distance,
            wet_start_distance,
        )

    safe_start_distance = _search_tide_safe_bank(
        cleaned,
        cumulative,
        wet_start_distance,
        -1,
        elevations,
        spec,
    )
    safe_end_distance = _search_tide_safe_bank(
        cleaned,
        cumulative,
        wet_end_distance,
        1,
        elevations,
        spec,
    )
    if safe_start_distance is None or safe_end_distance is None:
        # Keep the existing wet-only geometry.  The final deck-height clamp still
        # prevents submersion, although a source way that ends at the shoreline
        # cannot provide a perfectly graded dry approach without invented road.
        return plan

    safe_start = _bridge._point_at_measure(
        cleaned, cumulative, safe_start_distance
    )
    safe_end = _bridge._point_at_measure(
        cleaned, cumulative, safe_end_distance
    )
    dx = float(safe_end[0]) - float(safe_start[0])
    dz = float(safe_end[1]) - float(safe_start[1])
    safe_length = math.hypot(dx, dz)
    if safe_length <= 0.1:
        return plan

    unit_x, unit_z = dx / safe_length, dz / safe_length
    module_length = float(_bridge._STOCK_MODULE_SPACING_METRES)
    tolerance = max(1.0e-6, module_length * 1.0e-9)
    module_count = max(
        1,
        int(math.ceil((safe_length - tolerance) / module_length)),
    )
    target_length = module_count * module_length
    half_target = target_length * 0.5
    centre = (
        (float(safe_start[0]) + float(safe_end[0])) * 0.5,
        (float(safe_start[1]) + float(safe_end[1])) * 0.5,
    )

    # Keep the complete fixed-length chain inside the source bridge way whenever
    # the source contains enough dry approach.  Excess module length therefore
    # moves farther onto land, where ordinary road can meet it, rather than back
    # toward the water interval we just made tide-safe.
    projections = [
        (float(point[0]) - centre[0]) * unit_x
        + (float(point[1]) - centre[1]) * unit_z
        for point in cleaned
    ]
    source_min = min(projections)
    source_max = max(projections)
    if source_max - source_min >= target_length:
        minimum_shift = source_min + half_target
        maximum_shift = source_max - half_target
        shift = max(minimum_shift, min(0.0, maximum_shift))
        centre = (
            centre[0] + unit_x * shift,
            centre[1] + unit_z * shift,
        )

    start = (
        centre[0] - unit_x * half_target,
        centre[1] - unit_z * half_target,
    )
    end = (
        centre[0] + unit_x * half_target,
        centre[1] + unit_z * half_target,
    )
    world_size = float(getattr(spec, "world_size", 0.0) or 0.0)
    if world_size > 0.0 and not all(
        0.0 <= coordinate < world_size
        for point in (start, end)
        for coordinate in point
    ):
        return plan

    return replace(
        plan,
        points=(start, end),
        module_count=module_count,
    )


def install_bridge_water_deck_clamp_policy() -> None:
    """Make clipped bridge length and vertical anchoring safe for OFP tides."""

    global _ORIGINAL_DRY_APPROACH_HEIGHT
    global _ORIGINAL_STOCK_BRIDGE_SPAN_PLAN
    global _INSTALLED
    if _INSTALLED:
        return

    _ORIGINAL_DRY_APPROACH_HEIGHT = _bridge._dry_approach_height
    _ORIGINAL_STOCK_BRIDGE_SPAN_PLAN = _bridge.stock_bridge_span_plan
    _bridge._dry_approach_height = _safe_dry_approach_height
    _bridge.stock_bridge_span_plan = _tide_safe_stock_bridge_span_plan
    _INSTALLED = True

    # Explicit bridges over narrow mapped water can be missed entirely by a
    # coarse WRP terrain grid. Install the shared source-water fallback only
    # after the tide-safe stock-plan wrapper above is authoritative.
    from .bridge_source_water_policy import install_bridge_source_water_policy

    install_bridge_source_water_policy()
