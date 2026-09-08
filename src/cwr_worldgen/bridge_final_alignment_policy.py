# SPDX-License-Identifier: GPL-3.0-or-later
"""Finish stock-bridge alignment against real roads and the bridge footprint.

Road-connected bridge planning now places abutments on actual ordinary-road
approaches. Three legacy/discrete-model effects can still spoil that result:

* the historical 0.7 m bridge tuning lowers the visible deck below the fitted
  road surface even after horizontal endpoints are correctly connected;
* underlay cleanup can miss aligned road centres inside the stock bridge width;
  and
* fixed 6/12/25 m stock road pieces can stop a few metres short of an otherwise
  correctly placed fixed-length bridge abutment.

Keep the historical tuning constant intact for compatibility, compensate it when
sampling final bridge approach heights, clean the measured stock bridge width,
and add a short buried-overlap stock-road connector only where the fitted road
piece phase leaves a visible abutment gap.
"""
from __future__ import annotations

from dataclasses import replace
import math
import re

from . import bridge_render_policy as _bridge
from . import bridge_underlay_cleanup_policy as _cleanup
from . import bridge_water_deck_clamp_policy as _clamp
from . import generator as _generator
from . import playability as _playability
from .model import WorldObject

_INSTALLED = False
_ORIGINAL_DRY_APPROACH_HEIGHT = None
_ORIGINAL_ROAD_OBJECT_UNDER_BRIDGE = None
_ORIGINAL_FINAL_FIT = None

# Measured from O\Hous\most_stred30.p3d Roadway LOD. The outer roadway strips
# reach x = +/-6.642317 m. A small tolerance covers WRP float/heading rounding.
_STOCK_BRIDGE_OUTER_HALF_WIDTH_METRES = 6.642317295074463
_STOCK_BRIDGE_FOOTPRINT_MARGIN_METRES = 0.15

# Stock paved roads have no shorter straight than the 6 m sibling. If the final
# fitted chain stops short of a bridge, place one 6 m approach slab with its
# bridgeward edge exactly on the abutment and bury its overlap beneath the
# existing road. This exposes only the missing gap and avoids coplanar z-fighting.
_APPROACH_FILL_MIN_GAP_METRES = 0.20
_APPROACH_FILL_MAX_GAP_METRES = 5.75
_APPROACH_FILL_NOMINAL_LENGTH_METRES = 6.0
_APPROACH_FILL_OVERLAP_BURY_METRES = 0.02
_APPROACH_FILL_HEADING_TOLERANCE_DEGREES = 30.0
_APPROACH_FILL_MAX_PITCH_DEGREES = 5.0
_APPROACH_CHAIN_JOIN_TOLERANCE_METRES = 0.75
_APPROACH_MAX_VERTICAL_MISMATCH_METRES = 1.0
_STRAIGHT_ROAD_LENGTH_RE = re.compile(
    r"(?P<length>25|12|6)(?P<suffix>\.p3d)$",
    re.IGNORECASE,
)


def _final_road_approach_height(
    endpoint,
    outward,
    raster,
    elevations,
    spec,
):
    """Return the pre-tuning height that makes the final deck match the road."""
    raw_sampler = getattr(_clamp, "_ORIGINAL_DRY_APPROACH_HEIGHT", None)
    if raw_sampler is None:
        raw_sampler = _ORIGINAL_DRY_APPROACH_HEIGHT
    height = raw_sampler(endpoint, outward, raster, elevations, spec)
    if height is None:
        return None

    # _anchor_stock_bridge_chains subtracts the historical tuning offset after
    # this function returns. Add it back here so a road-connected endpoint lands
    # at the actual road surface. Low banks still clamp to the 5.5 m tide-safe
    # final deck used by bridge_water_deck_clamp_policy.
    final_target = max(
        float(height),
        float(_clamp._minimum_final_deck(spec)),
    )
    return final_target + float(_bridge._BRIDGE_WORLD_DOWNWARD_OFFSET_METRES)


def _inside_emitted_stock_footprint(obj, span) -> bool:
    """Return whether an aligned road centre lies inside the stock bridge body."""
    if _cleanup._paved._family(obj.model_path) is None:
        return False
    if len(span.points) < 2:
        return False

    distance, heading, along, total = _cleanup._nearest_polyline_measure(
        span.points,
        (float(obj.x), float(obj.z)),
    )
    if total <= 1.0e-9:
        return False

    endpoint_keep = min(
        float(_cleanup._ENDPOINT_KEEP_METRES),
        total * 0.02,
    )
    if along <= endpoint_keep or along >= total - endpoint_keep:
        return False

    corridor = (
        _STOCK_BRIDGE_OUTER_HALF_WIDTH_METRES
        + _STOCK_BRIDGE_FOOTPRINT_MARGIN_METRES
    )
    if distance > corridor:
        return False
    if (
        _cleanup._undirected_heading_difference(
            obj.heading_degrees,
            heading,
        )
        > _cleanup._ALIGNMENT_TOLERANCE_DEGREES
    ):
        return False
    return True


def _road_object_under_bridge(obj, spans) -> bool:
    for span in spans:
        if _inside_emitted_stock_footprint(obj, span):
            return True
    return _ORIGINAL_ROAD_OBJECT_UNDER_BRIDGE(obj, spans)


def _straight_road_nominal_length(model_path: str) -> float | None:
    filename = str(model_path).replace("/", "\\").rsplit("\\", 1)[-1]
    match = _STRAIGHT_ROAD_LENGTH_RE.search(filename)
    return float(match.group("length")) if match else None


def _six_metre_sibling(model_path: str) -> str | None:
    path = str(model_path)
    match = _STRAIGHT_ROAD_LENGTH_RE.search(path)
    if match is None:
        return None
    start, end = match.span("length")
    return path[:start] + "6" + path[end:]


def _road_endpoints(
    obj,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Return approximate stock-road surface endpoints from its WRP transform."""
    length = _straight_road_nominal_length(obj.model_path)
    if length is None:
        return None
    heading = math.radians(float(obj.heading_degrees))
    pitch = math.radians(float(obj.pitch_degrees))
    ux = math.sin(heading) * math.cos(pitch)
    uy = math.sin(pitch)
    uz = math.cos(heading) * math.cos(pitch)
    half = length * 0.5
    return (
        (
            float(obj.x) - ux * half,
            float(obj.y) - uy * half,
            float(obj.z) - uz * half,
        ),
        (
            float(obj.x) + ux * half,
            float(obj.y) + uy * half,
            float(obj.z) + uz * half,
        ),
    )


def _axial_heading_difference(first: float, second: float) -> float:
    delta = abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)
    return min(delta, abs(180.0 - delta))


def _road_has_outward_neighbour(obj, bridge_point, road_objects) -> bool:
    """Prefer a real continuing road chain over an isolated cap near an abutment."""
    endpoints = _road_endpoints(obj)
    if endpoints is None:
        return False
    far = max(
        endpoints,
        key=lambda point: math.dist(
            (point[0], point[2]),
            (float(bridge_point[0]), float(bridge_point[1])),
        ),
    )
    for other in road_objects:
        if int(other.object_id) == int(obj.object_id):
            continue
        other_endpoints = _road_endpoints(other)
        if other_endpoints is None:
            continue
        if (
            _axial_heading_difference(
                obj.heading_degrees,
                other.heading_degrees,
            )
            > _APPROACH_FILL_HEADING_TOLERANCE_DEGREES
        ):
            continue
        if (
            min(math.dist(far, point) for point in other_endpoints)
            <= _APPROACH_CHAIN_JOIN_TOLERANCE_METRES
        ):
            return True
    return False


def _approach_candidate(bridge_point, bridge_heading, road_objects):
    candidates = []
    for obj in road_objects:
        if _cleanup._paved._family(obj.model_path) is None:
            continue
        endpoints = _road_endpoints(obj)
        if endpoints is None or _six_metre_sibling(obj.model_path) is None:
            continue
        if (
            _axial_heading_difference(
                obj.heading_degrees,
                bridge_heading,
            )
            > _APPROACH_FILL_HEADING_TOLERANCE_DEGREES
        ):
            continue
        near = min(
            endpoints,
            key=lambda point: math.dist(
                (point[0], point[2]),
                (float(bridge_point[0]), float(bridge_point[1])),
            ),
        )
        gap = math.dist(
            (near[0], near[2]),
            (float(bridge_point[0]), float(bridge_point[1])),
        )
        if gap > _APPROACH_FILL_MAX_GAP_METRES:
            continue
        connected = _road_has_outward_neighbour(
            obj,
            bridge_point,
            road_objects,
        )
        candidates.append(
            (
                0 if connected else 1,
                gap,
                int(obj.object_id),
                obj,
                near,
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[:3])
    return candidates[0][3], candidates[0][4], candidates[0][1]


def _bridge_endpoint_deck_height(
    endpoint: tuple[float, float],
    outward: tuple[float, float],
    candidate_surface_y: float,
    elevations,
    spec,
) -> float:
    pre_tuning = _bridge._dry_approach_height(
        endpoint,
        outward,
        None,
        elevations,
        spec,
    )
    if pre_tuning is None:
        return float(candidate_surface_y)
    return (
        float(pre_tuning)
        - float(_bridge._BRIDGE_WORLD_DOWNWARD_OFFSET_METRES)
    )


def _approach_filler(
    object_id: int,
    bridge_point: tuple[float, float],
    outward: tuple[float, float],
    bridge_heading: float,
    candidate,
    candidate_near,
    gap: float,
    bridge_deck_y: float,
) -> WorldObject | None:
    if gap <= _APPROACH_FILL_MIN_GAP_METRES:
        return None
    if gap > _APPROACH_FILL_MAX_GAP_METRES:
        return None

    model = _six_metre_sibling(candidate.model_path)
    if model is None:
        return None

    dx = float(candidate_near[0]) - float(bridge_point[0])
    dz = float(candidate_near[2]) - float(bridge_point[1])
    distance = math.hypot(dx, dz)
    if distance <= 1.0e-6:
        ux, uz = float(outward[0]), float(outward[1])
    else:
        ux, uz = dx / distance, dz / distance

    filler_heading = math.degrees(math.atan2(ux, uz)) % 360.0
    if (
        _axial_heading_difference(filler_heading, bridge_heading)
        > _APPROACH_FILL_HEADING_TOLERANCE_DEGREES
    ):
        return None

    candidate_y = float(candidate_near[1])
    if (
        abs(float(bridge_deck_y) - candidate_y)
        > _APPROACH_MAX_VERTICAL_MISMATCH_METRES
    ):
        return None

    # The exposed portion ends at candidate_near. Force the connector surface
    # two centimetres below the existing road exactly there, then let the unused
    # remainder of the 6 m slab continue underneath the existing road.
    buried_join_y = candidate_y - _APPROACH_FILL_OVERLAP_BURY_METRES
    pitch = math.degrees(
        math.atan2(
            buried_join_y - float(bridge_deck_y),
            max(gap, 1.0e-9),
        )
    )
    if abs(pitch) > _APPROACH_FILL_MAX_PITCH_DEGREES:
        return None

    pitch_radians = math.radians(pitch)
    length = _APPROACH_FILL_NOMINAL_LENGTH_METRES
    horizontal = length * math.cos(pitch_radians)
    end_x = float(bridge_point[0]) + ux * horizontal
    end_z = float(bridge_point[1]) + uz * horizontal
    end_y = float(bridge_deck_y) + length * math.sin(pitch_radians)

    return WorldObject(
        int(object_id),
        model,
        (float(bridge_point[0]) + end_x) * 0.5,
        (float(bridge_deck_y) + end_y) * 0.5,
        (float(bridge_point[1]) + end_z) * 0.5,
        filler_heading,
        pitch,
    )


def _add_bridge_approach_fillers(report, spans, elevations, spec):
    """Fill only the residual stock-road phase gap at each bridge abutment."""
    if not spans or not getattr(report, "objects", ()):
        return report, 0

    objects = list(report.objects)
    road_objects = tuple(objects)
    next_id = max((int(obj.object_id) for obj in objects), default=0) + 1
    added = 0

    for span in spans:
        if len(span.points) < 2:
            continue
        start = (float(span.points[0][0]), float(span.points[0][1]))
        end = (float(span.points[-1][0]), float(span.points[-1][1]))
        dx = end[0] - start[0]
        dz = end[1] - start[1]
        length = math.hypot(dx, dz)
        if length <= 0.1:
            continue
        axis = (dx / length, dz / length)
        bridge_heading = math.degrees(math.atan2(axis[0], axis[1])) % 360.0

        for bridge_point, outward in (
            (start, (-axis[0], -axis[1])),
            (end, axis),
        ):
            selected = _approach_candidate(
                bridge_point,
                bridge_heading,
                road_objects,
            )
            if selected is None:
                continue
            candidate, candidate_near, gap = selected
            deck_y = _bridge_endpoint_deck_height(
                bridge_point,
                outward,
                float(candidate_near[1]),
                elevations,
                spec,
            )
            filler = _approach_filler(
                next_id,
                bridge_point,
                outward,
                bridge_heading,
                candidate,
                candidate_near,
                gap,
                deck_y,
            )
            if filler is None:
                continue
            objects.append(filler)
            next_id += 1
            added += 1

    if not added:
        return report, 0

    # Generator assembly derives the first non-road ID from the road object count.
    # Cleanup wrappers can leave holes, so re-pack the final road IDs contiguously
    # from the report's original starting ID after appending the connectors.
    first_id = min(int(obj.object_id) for obj in objects)
    renumbered = tuple(
        replace(obj, object_id=first_id + index)
        for index, obj in enumerate(objects)
    )
    return replace(report, objects=renumbered), added


def _final_fit(
    dataset,
    projection,
    elevations,
    spec,
    *,
    starting_id: int = 1,
    progress_callback=None,
):
    report = _ORIGINAL_FINAL_FIT(
        dataset,
        projection,
        elevations,
        spec,
        starting_id=starting_id,
        progress_callback=progress_callback,
    )
    spans = _cleanup._bridge_spans(
        dataset,
        projection,
        elevations,
        spec,
    )
    filled, added = _add_bridge_approach_fillers(
        report,
        spans,
        elevations,
        spec,
    )
    if added and progress_callback is not None:
        progress_callback(
            99,
            f"Added {added:,} stock road approach connector(s) at bridge abutments",
        )
    return filled


def install_bridge_final_alignment_policy() -> None:
    """Align stock decks, bridge footprints and final ordinary-road abutments."""
    global _INSTALLED
    global _ORIGINAL_DRY_APPROACH_HEIGHT
    global _ORIGINAL_ROAD_OBJECT_UNDER_BRIDGE
    global _ORIGINAL_FINAL_FIT

    if _INSTALLED:
        return

    _ORIGINAL_DRY_APPROACH_HEIGHT = _bridge._dry_approach_height
    _ORIGINAL_ROAD_OBJECT_UNDER_BRIDGE = _cleanup._road_object_under_bridge
    _ORIGINAL_FINAL_FIT = _generator.fit_road_objects

    _bridge._dry_approach_height = _final_road_approach_height
    _cleanup._road_object_under_bridge = _road_object_under_bridge
    _playability.fit_road_objects = _final_fit
    _generator.fit_road_objects = _final_fit
    _INSTALLED = True
