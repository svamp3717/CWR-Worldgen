# SPDX-License-Identifier: GPL-3.0-or-later
"""Prefer continuous stock-road surfaces over literal small source-line kinks.

The measured stock-road policies make connector geometry exact, but two visual
failure modes can still remain in game:

* a shallow dog-leg can be represented by several 6.25 m straight slabs whose
  centreline endpoints touch exactly while their rectangular surface edges open
  a triangular grass seam; and
* a skewed paved T can fall back to a straight six-metre cap, after which the
  post-fit seam repair may try to bridge a mostly lateral connector mismatch by
  placing another short road slab across the carriageway.

Use the small source-line deviation budget before resorting to repair objects.
Open-road micro-bends may be simplified inside the existing 0.75 m corridor,
and same-family paved T approaches may relax onto a measured native junction.
Every junction relaxation is vetoed when the moved approach would overlap a
source-backed building, utility object, or mapped individual tree. Legacy caps
remain a safe fallback: their approaches continue underneath the cap and the cap
is raised slightly so its surface wins without z-fighting.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import gravel_junction_policy as _gravel_junction
from . import playability as _p
from . import road_quality_policy as _quality
from . import stock_road_connector_policy as _connector
from . import stock_road_junction_policy as _junction
from . import stock_road_model_geometry as _geometry
from . import stock_road_relaxation_policy as _relax
from . import stock_road_skew_policy as _skew
from . import stock_road_surface_overlap_policy as _surface

MAXIMUM_OPEN_ROAD_HEADING_RELAXATION_DEGREES = 14.0
MAXIMUM_PAVED_T_HEADING_ERROR_DEGREES = 14.0
LEGACY_CAP_VERTICAL_BIAS_METRES = 0.006
MINIMUM_REPAIR_ALIGNMENT_COSINE = math.cos(math.radians(35.0))

_ORIGINAL_MIXED_T_ELIGIBLE = None
_ORIGINAL_COLLECT_RELAXATIONS = None
_ORIGINAL_QUALITY_WINDOW = None
_ORIGINAL_LOWER_LEGACY_CAP = None
_INSTALLED = False


def _same_family_paved_t(incidents) -> bool:
    if len(incidents) != 3:
        return False
    if any(_skew._is_generated_gravel_model(incident.model_path) for incident in incidents):
        return False
    families = tuple(incident.family for incident in incidents)
    return (
        None not in families
        and len(set(families)) == 1
        and families[0] in {"sil", "asf", "kos"}
    )


def _eligible_relaxed_t(incidents) -> bool:
    if _ORIGINAL_MIXED_T_ELIGIBLE is None:
        raise RuntimeError("local stock-road fit policy is not installed")
    return _ORIGINAL_MIXED_T_ELIGIBLE(incidents) or _same_family_paved_t(incidents)


def _native_junction_for_incidents(incidents):
    """Permit measured native T geometry when local relaxation can absorb it."""

    original = _skew._ORIGINAL_NATIVE_JUNCTION_FOR_INCIDENTS
    if original is None:
        raise RuntimeError("stock road skew policy is not installed")

    native = original(incidents)
    if native is not None:
        return native

    if _ORIGINAL_MIXED_T_ELIGIBLE is not None and _ORIGINAL_MIXED_T_ELIGIBLE(incidents):
        limit = _skew.MAXIMUM_RELAXED_JUNCTION_HEADING_ERROR_DEGREES
    elif _same_family_paved_t(incidents):
        limit = MAXIMUM_PAVED_T_HEADING_ERROR_DEGREES
    else:
        return None

    previous = _junction.MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES
    try:
        _junction.MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES = limit
        return original(incidents)
    finally:
        _junction.MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES = previous


def _collect_relaxations(dataset, projection, projected, spec):
    """Filter connector-aligned approach edits through the obstacle corridor."""

    if _ORIGINAL_COLLECT_RELAXATIONS is None:
        raise RuntimeError("local stock-road fit policy is not installed")
    relaxations = _ORIGINAL_COLLECT_RELAXATIONS(dataset, projection, projected, spec)
    if not relaxations:
        return relaxations

    context = _relax._CONTEXT.get()
    if context is None:
        return relaxations

    safe = {}
    for key, point in relaxations.items():
        feature_index, node_index, neighbour_index = key
        node = tuple(projected[feature_index][node_index])
        neighbour = tuple(projected[feature_index][neighbour_index])
        if not _relax._shortcut_clear(context.obstacles, node, point):
            continue
        if not _relax._shortcut_clear(context.obstacles, point, neighbour):
            continue
        safe[key] = point
    return safe


def _quality_window(
    measure,
    pieces,
    start_distance,
    preferred_end,
    minimum_end,
    maximum_end,
    context,
):
    """Let ordinary stock approaches continue underneath any stock junction cap."""

    if _ORIGINAL_QUALITY_WINDOW is None:
        raise RuntimeError("local stock-road fit policy is not installed")
    start_distance, preferred_end, minimum_end, maximum_end = _ORIGINAL_QUALITY_WINDOW(
        measure,
        pieces,
        start_distance,
        preferred_end,
        minimum_end,
        maximum_end,
        context,
    )
    if not pieces:
        return start_distance, preferred_end, minimum_end, maximum_end

    start_junction = context.junctions.get(_p._road_node_key(measure.points[0]))
    end_junction = context.junctions.get(_p._road_node_key(measure.points[-1]))
    shortest = min(float(piece.length_metres) for piece in pieces)

    if start_junction is not None and not _gravel_junction._is_gravel_junction(start_junction):
        start_distance = 0.0
    if end_junction is not None and not _gravel_junction._is_gravel_junction(end_junction):
        preferred_end = max(start_distance, measure.total)
        minimum_end = max(start_distance, measure.total - 0.10)
        maximum_end = max(maximum_end, measure.total + shortest * 0.5)
    return start_distance, preferred_end, minimum_end, maximum_end


def _lower_legacy_stock_cap(old, elevations, spec):
    """Keep a fallback cap above the approaches that now continue beneath it."""

    if _ORIGINAL_LOWER_LEGACY_CAP is None:
        raise RuntimeError("local stock-road fit policy is not installed")
    lowered = _ORIGINAL_LOWER_LEGACY_CAP(old, elevations, spec)
    if _junction._STOCK_CAP_MODEL.fullmatch(str(lowered.model_path).replace("/", "\\")) is None:
        return lowered
    return replace(lowered, y=float(lowered.y) + LEGACY_CAP_VERTICAL_BIAS_METRES)


def _connector_cover_plans(report):
    """Use repair underlays only for real longitudinal native-connector gaps.

    A normal straight cap already owns the road surface between its two ends, so
    adding another six-metre slab beneath one of those ends is redundant and can
    create the conspicuous cross-road clipping seen in game. Native T/X meshes
    may still need an underlay, but only when the uncovered gap points along the
    connector rather than mostly sideways across the road.
    """

    cap_count = min(int(getattr(report, "junction_cap_objects", 0)), len(report.objects))
    if cap_count <= 0:
        return ()
    caps = report.objects[:cap_count]
    chains = report.objects[cap_count:]
    endpoints = _surface._chain_endpoints(chains)
    if not endpoints:
        return ()
    endpoint_buckets = _surface._endpoint_index(endpoints)
    axis_buckets = _relax._axis_index(chains)

    used_endpoints: set[tuple[int, int]] = set()
    plans = []
    for cap in caps:
        # Legacy/mixed straight caps cover their own connector-to-centre span.
        # Their approaches are now fitted underneath to the node, so no extra
        # repair slab belongs here.
        if _geometry.stock_straight_match(cap.model_path) is not None:
            continue
        for connector in _surface._native_cap_connectors(cap):
            if connector.family not in _surface._STOCK_FAMILIES:
                continue
            if _relax._connector_already_covered(axis_buckets, connector):
                continue
            nearest = _surface._nearest_endpoint(endpoint_buckets, connector)
            if nearest is None:
                continue
            gap, endpoint = nearest
            endpoint_key = (endpoint.object_id, endpoint.endpoint_index)
            if endpoint_key in used_endpoints:
                continue
            if not (
                _surface.MINIMUM_CONNECTOR_COVER_GAP_METRES
                <= gap
                <= _surface.MAXIMUM_CONNECTOR_COVER_GAP_METRES
            ):
                continue

            vector = (
                endpoint.point[0] - connector.point[0],
                endpoint.point[1] - connector.point[1],
            )
            direction = _surface._normalised(vector)
            alignment = abs(
                direction[0] * connector.outward[0]
                + direction[1] * connector.outward[1]
            )
            if alignment < MINIMUM_REPAIR_ALIGNMENT_COSINE:
                continue

            centre = (
                (connector.point[0] + endpoint.point[0]) * 0.5,
                (connector.point[1] + endpoint.point[1]) * 0.5,
            )
            plans.append(
                _surface._CoverPlan(
                    _surface._cover_model(connector.family),
                    centre,
                    direction,
                )
            )
            used_endpoints.add(endpoint_key)
    return tuple(plans)


def install_stock_road_local_fit_policy() -> None:
    global _ORIGINAL_MIXED_T_ELIGIBLE, _ORIGINAL_COLLECT_RELAXATIONS
    global _ORIGINAL_QUALITY_WINDOW, _ORIGINAL_LOWER_LEGACY_CAP, _INSTALLED
    if _INSTALLED:
        return

    _ORIGINAL_MIXED_T_ELIGIBLE = _skew._eligible_relaxed_mixed_t
    _ORIGINAL_COLLECT_RELAXATIONS = _connector._collect_relaxations
    _ORIGINAL_QUALITY_WINDOW = _quality._quality_window
    _ORIGINAL_LOWER_LEGACY_CAP = _junction._lower_legacy_stock_cap

    # A shallow 10-14 degree dog-leg can still remain within the already-bounded
    # 0.75 m source corridor. The geometric deviation and obstacle checks remain
    # authoritative, so increasing the heading gate does not flatten real bends
    # that wander farther away from their source line.
    _relax.MAXIMUM_RELAXED_HEADING_CHANGE_DEGREES = (
        MAXIMUM_OPEN_ROAD_HEADING_RELAXATION_DEGREES
    )

    _skew._eligible_relaxed_mixed_t = _eligible_relaxed_t
    _junction._native_junction_for_incidents = _native_junction_for_incidents
    _connector._collect_relaxations = _collect_relaxations

    _surface._quality_window = _quality_window
    _quality._quality_window = _quality_window
    _junction._lower_legacy_stock_cap = _lower_legacy_stock_cap

    _surface._connector_cover_plans = _connector_cover_plans
    _relax._connector_cover_plans = _connector_cover_plans
    _INSTALLED = True
