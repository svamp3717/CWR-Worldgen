# SPDX-License-Identifier: GPL-3.0-or-later
"""Adjust stock bridges to nearby fitted-road approaches without redefining crossings.

The water-derived stock bridge plan is authoritative.  Connected non-bridge roads
may move an abutment enough to make a stock module meet the ordinary road cleanly,
but they must not turn dry approach terrain into additional bridge span.  This is
especially important on coarse terrain where requiring tide-safe road elevation can
otherwise expand a short water crossing by hundreds of metres.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import bridge_render_policy as _bridge
from . import bridge_source_water_policy as _source
from . import osm as _osm

_INSTALLED = False
_ORIGINAL_PLAN = None

# Keep this aligned with the final-deck safety target used by
# bridge_water_deck_clamp_policy.  The empirical 0.7 m stock-model tuning is
# applied later by the bridge anchor; endpoint selection only needs to reach a
# dry road surface at the final safe deck level.
_TIDE_SAFE_ROAD_SURFACE_ABOVE_SEA_METRES = 5.5
_MINIMUM_DRY_INSET_CELLS = 0.55
_APPROACH_PROBE_METRES = 220.0
_SAFE_SAMPLE_STEP_METRES = 1.0
_MATCH_SAMPLE_STEP_METRES = 1.0
_MATCH_REFINE_STEP_METRES = 0.10
_MAXIMUM_ENDPOINT_LENGTH_ERROR_METRES = 1.0
_MAXIMUM_EXTRA_MODULES_TO_SEARCH = 2

# Road connection is a fit adjustment, not a second bridge planner.  The
# water-derived plan may grow by at most one stock module, and neither endpoint
# may move more than one module from the original wet-authoritative plan.
_MAXIMUM_CONNECTED_MODULE_COUNT_INCREASE = 1
_MAXIMUM_CONNECTED_ENDPOINT_SHIFT_MODULES = 1.0


def _explicit_bridge(feature) -> bool:
    tags = feature.tags
    bridge = str(tags.get("bridge", "")).strip().casefold()
    return (
        bridge not in {"", "no", "false", "0", "none"}
        or str(tags.get("man_made", "")).strip().casefold() == "bridge"
        or str(tags.get("special", "")).strip().casefold() == "bridge"
    )


def _matching_bridge_feature(points, context, spec):
    """Find the explicit source bridge represented by one projected point run."""
    cleaned = _bridge._clean_points(points)
    if len(cleaned) < 2:
        return None
    tolerance = max(2.0, min(8.0, float(spec.cell_size) * 0.20))
    best = None
    for feature in getattr(context.dataset, "roads", ()):
        if not _explicit_bridge(feature) or len(feature.points) < 2:
            continue
        if _osm.road_bridge_crosses_ditch_only(
            feature, context.dataset, context.projection
        ):
            continue
        projected = tuple(
            context.projection.to_world(point) for point in feature.points
        )
        if len(projected) < 2:
            continue
        direct = (
            math.dist(cleaned[0], projected[0])
            + math.dist(cleaned[-1], projected[-1])
        )
        reverse = (
            math.dist(cleaned[0], projected[-1])
            + math.dist(cleaned[-1], projected[0])
        )
        score = min(direct, reverse)
        candidate = (score, feature.osm_key, feature, projected, reverse < direct)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None or best[0] > tolerance * 2.0:
        return None
    _score, _key, feature, projected, reversed_match = best
    if reversed_match:
        projected = tuple(reversed(projected))
    return feature, projected


def _path_measure(path):
    cleaned, cumulative = _bridge._polyline_measure(path)
    return cleaned, cumulative


def _road_surface_at(path, cumulative, distance, elevations, spec) -> float:
    x, z = _bridge._point_at_measure(path, cumulative, distance)
    ground = float(
        _osm._sample_elevation(
            elevations, spec.cells, spec.cell_size, x, z
        )
    )
    return ground + float(_osm.NOGOVA_BRIDGE_APPROACH_OFFSET_METRES)


def _first_safe_distance(path, cumulative, elevations, spec) -> float | None:
    if len(path) < 2 or not cumulative or cumulative[-1] <= 0.1:
        return None
    minimum_distance = min(
        float(cumulative[-1]),
        max(5.0, float(spec.cell_size) * _MINIMUM_DRY_INSET_CELLS),
    )
    threshold = (
        float(spec.sea_level) + _TIDE_SAFE_ROAD_SURFACE_ABOVE_SEA_METRES
    )
    distance = minimum_distance
    while distance <= float(cumulative[-1]) + 1.0e-9:
        if _road_surface_at(
            path, cumulative, distance, elevations, spec
        ) >= threshold:
            return distance
        distance += _SAFE_SAMPLE_STEP_METRES
    # Always test the exact path end; the stepping loop can overshoot it.
    end = float(cumulative[-1])
    if _road_surface_at(path, cumulative, end, elevations, spec) >= threshold:
        return end
    return None


def _safe_candidates(path, cumulative, start_distance, elevations, spec):
    threshold = (
        float(spec.sea_level) + _TIDE_SAFE_ROAD_SURFACE_ABOVE_SEA_METRES
    )
    end = float(cumulative[-1])
    values = []
    distance = float(start_distance)
    while distance < end - 1.0e-9:
        if _road_surface_at(
            path, cumulative, distance, elevations, spec
        ) >= threshold:
            values.append(
                (distance, _bridge._point_at_measure(path, cumulative, distance))
            )
        distance += _MATCH_SAMPLE_STEP_METRES
    if _road_surface_at(path, cumulative, end, elevations, spec) >= threshold:
        values.append((end, _bridge._point_at_measure(path, cumulative, end)))
    return tuple(values)


def _refine_pair(
    start_path,
    start_cumulative,
    end_path,
    end_cumulative,
    start_distance,
    end_distance,
    target_length,
    elevations,
    spec,
):
    threshold = (
        float(spec.sea_level) + _TIDE_SAFE_ROAD_SURFACE_ABOVE_SEA_METRES
    )
    best = None
    for start_step in range(-12, 13):
        sd = max(
            0.0,
            min(
                float(start_cumulative[-1]),
                float(start_distance) + start_step * _MATCH_REFINE_STEP_METRES,
            ),
        )
        if _road_surface_at(
            start_path, start_cumulative, sd, elevations, spec
        ) < threshold:
            continue
        sp = _bridge._point_at_measure(start_path, start_cumulative, sd)
        for end_step in range(-12, 13):
            ed = max(
                0.0,
                min(
                    float(end_cumulative[-1]),
                    float(end_distance) + end_step * _MATCH_REFINE_STEP_METRES,
                ),
            )
            if _road_surface_at(
                end_path, end_cumulative, ed, elevations, spec
            ) < threshold:
                continue
            ep = _bridge._point_at_measure(end_path, end_cumulative, ed)
            error = abs(math.dist(sp, ep) - target_length)
            candidate = (error, sd + ed, sd, ed, sp, ep)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
    return best


def _connected_adjustment_is_bounded(plan, module_count, start, end) -> bool:
    """Return whether road fitting stayed a small adjustment to the wet plan."""
    original_modules = max(1, int(plan.module_count))
    if int(module_count) > original_modules + _MAXIMUM_CONNECTED_MODULE_COUNT_INCREASE:
        return False

    original_points = tuple(plan.points)
    if len(original_points) < 2:
        return False

    maximum_shift = (
        float(_bridge._STOCK_MODULE_SPACING_METRES)
        * _MAXIMUM_CONNECTED_ENDPOINT_SHIFT_MODULES
        + _MAXIMUM_ENDPOINT_LENGTH_ERROR_METRES
    )
    direct = max(
        math.dist(start, original_points[0]),
        math.dist(end, original_points[-1]),
    )
    reverse = max(
        math.dist(start, original_points[-1]),
        math.dist(end, original_points[0]),
    )
    return min(direct, reverse) <= maximum_shift


def _connected_stock_plan(plan, points, elevations, spec):
    context = _source._CONTEXT.get()
    if context is None or plan is None:
        return plan

    matched = _matching_bridge_feature(points, context, spec)
    if matched is None:
        return plan
    feature, projected = matched
    if len(projected) < 2:
        return plan

    start_path = _osm._connected_bridge_approach_path(
        feature,
        context.dataset,
        context.projection,
        projected[0],
        projected[1],
        spec,
        _APPROACH_PROBE_METRES,
    )
    end_path = _osm._connected_bridge_approach_path(
        feature,
        context.dataset,
        context.projection,
        projected[-1],
        projected[-2],
        spec,
        _APPROACH_PROBE_METRES,
    )
    start_path, start_cumulative = _path_measure(start_path)
    end_path, end_cumulative = _path_measure(end_path)
    if len(start_path) < 2 or len(end_path) < 2:
        return plan

    safe_start = _first_safe_distance(
        start_path, start_cumulative, elevations, spec
    )
    safe_end = _first_safe_distance(
        end_path, end_cumulative, elevations, spec
    )
    if safe_start is None or safe_end is None:
        return plan

    start_candidates = _safe_candidates(
        start_path, start_cumulative, safe_start, elevations, spec
    )
    end_candidates = _safe_candidates(
        end_path, end_cumulative, safe_end, elevations, spec
    )
    if not start_candidates or not end_candidates:
        return plan

    minimum_span = math.dist(
        start_candidates[0][1], end_candidates[0][1]
    )
    module_length = float(_bridge._STOCK_MODULE_SPACING_METRES)
    minimum_modules = max(
        int(plan.module_count),
        1,
        int(math.ceil((minimum_span - 1.0e-6) / module_length)),
    )
    maximum_modules = max(1, int(plan.module_count)) + _MAXIMUM_CONNECTED_MODULE_COUNT_INCREASE

    # If the connected-road safety rule would require more than one extra stock
    # module, keep the water-derived plan.  Dry approach terrain is handled by
    # the abutment/road grading policies instead of being converted into bridge.
    if minimum_modules > maximum_modules:
        return plan

    selected = None
    search_stop = min(
        maximum_modules,
        minimum_modules + _MAXIMUM_EXTRA_MODULES_TO_SEARCH,
    )
    for module_count in range(minimum_modules, search_stop + 1):
        target = module_count * module_length
        coarse_best = None
        for sd, sp in start_candidates:
            for ed, ep in end_candidates:
                error = abs(math.dist(sp, ep) - target)
                candidate = (
                    error,
                    (sd - safe_start) + (ed - safe_end),
                    sd,
                    ed,
                    sp,
                    ep,
                )
                if coarse_best is None or candidate[:2] < coarse_best[:2]:
                    coarse_best = candidate
        if coarse_best is None:
            continue
        refined = _refine_pair(
            start_path,
            start_cumulative,
            end_path,
            end_cumulative,
            coarse_best[2],
            coarse_best[3],
            target,
            elevations,
            spec,
        )
        if refined is None:
            continue
        error, _distance_sum, _sd, _ed, start, end = refined
        if error <= _MAXIMUM_ENDPOINT_LENGTH_ERROR_METRES:
            selected = (module_count, error, start, end)
            break
        if selected is None or error < selected[1]:
            selected = (module_count, error, start, end)

    if selected is None or selected[1] > _MAXIMUM_ENDPOINT_LENGTH_ERROR_METRES:
        return plan

    module_count, _error, start, end = selected
    if not _connected_adjustment_is_bounded(plan, module_count, start, end):
        return plan

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


def install_bridge_road_connection_policy() -> None:
    """Allow only bounded stock-bridge adjustments onto connected roads."""
    global _INSTALLED, _ORIGINAL_PLAN
    if _INSTALLED:
        return

    _ORIGINAL_PLAN = _bridge.stock_bridge_span_plan

    def road_connected_plan(points, elevations, spec):
        plan = _ORIGINAL_PLAN(points, elevations, spec)
        return _connected_stock_plan(plan, points, elevations, spec)

    _bridge.stock_bridge_span_plan = road_connected_plan
    _INSTALLED = True
