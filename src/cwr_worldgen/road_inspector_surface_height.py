# SPDX-License-Identifier: GPL-3.0-or-later
"""Evaluate legacy intersection-cap height at the actual WRP road plane.

``RoadObject.y`` is the model origin height, not necessarily the road-surface
height at a nearby connector. On sloped roads the old intersection diagnostic
compared the cap's node-centred origin with approach origins several metres away.
Lundby24 consequently reported a fallback cap as 25 mm above its approaches even
though projecting those pitched P3Ds to the logical node put the cap below them.

Keep this correction inspector-only. Re-evaluate ``turning_intersection_cap``
findings at their source node using the same yaw/pitch transform as RVW4. A cap
that is safely below every nearby approach is no longer a turning-cap defect. If
its approaches still miss the source tangents, report that separately as an
``intersection_approach_mismatch`` instead of letting one category hide the other.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

from . import road_inspector as _core

_MINIMUM_HIDDEN_CAP_MARGIN_METRES = 0.002
_MAXIMUM_MATCHED_APPROACH_ERROR_DEGREES = 2.0
_MINIMUM_APPROACH_MISMATCH_DEGREES = 3.0
_APPROACH_ENDPOINT_RADIUS_METRES = 0.90
_CAP_CENTER_RADIUS_METRES = 0.90

_ORIGINAL_INSPECT = None
_INSTALLED = False


def _surface_height_at(road, point: tuple[float, float]) -> float:
    """Return the model road-plane height at one world X/Z coordinate."""

    heading = math.radians(float(road.heading_degrees))
    pitch = math.radians(float(road.pitch_degrees))
    cosine_pitch = math.cos(pitch)
    if abs(cosine_pitch) <= 1.0e-9:
        return float(road.y)

    dx = float(point[0]) - float(road.x)
    dz = float(point[1]) - float(road.z)
    # Invert the horizontal part of the RVW4 yaw/pitch transform. Local X does
    # not contribute to vertical displacement; local Z does by sin(pitch).
    local_z = (
        dx * math.sin(heading) + dz * math.cos(heading)
    ) / cosine_pitch
    return float(road.y) + local_z * math.sin(pitch)


def _near_node_endpoint(road, node: tuple[float, float]) -> bool:
    return any(
        math.dist(endpoint.point, node) <= _APPROACH_ENDPOINT_RADIUS_METRES
        for endpoint in road.endpoints
    )


def _approach_mismatch(issue, *, maximum_approach_error: float, metrics):
    score = min(100.0, 35.0 + maximum_approach_error * 5.0)
    return replace(
        issue,
        severity=_core._severity(score),
        score=score,
        category="intersection_approach_mismatch",
        message=(
            "The legacy intersection cap is safely below the visible approaches, "
            "but one or more emitted approaches miss the normalized intersection "
            f"tangent by up to {maximum_approach_error:.2f}°."
        ),
        candidate_fix=(
            "Refit the final stock piece on each incident road to the logical node, "
            "using a native curve before the intersection when the source road is "
            "already turning. Keep the existing low cap only as central fill."
        ),
        metrics=metrics,
    )


def _correct_turning_cap_issue(issue, roads_by_id):
    if issue.category != "turning_intersection_cap":
        return issue

    node = (float(issue.x), float(issue.z))
    roads = tuple(
        roads_by_id[object_id]
        for object_id in issue.object_ids
        if object_id in roads_by_id
    )
    caps = tuple(
        road
        for road in roads
        if (
            road.kind == "straight"
            and float(road.nominal_length_metres) <= 6.26
            and math.dist(road.logical_center, node) <= _CAP_CENTER_RADIUS_METRES
        )
    )
    if not caps:
        return issue
    cap = min(caps, key=lambda road: math.dist(road.logical_center, node))
    approaches = tuple(
        road
        for road in roads
        if int(road.object_id) != int(cap.object_id) and _near_node_endpoint(road, node)
    )
    if not approaches:
        return issue

    cap_height = _surface_height_at(cap, node)
    approach_heights = tuple(_surface_height_at(road, node) for road in approaches)
    visible_margin = min(approach_heights) - cap_height

    metrics = dict(issue.metrics)
    metrics["cap_below_approach_margin_metres"] = round(visible_margin, 5)
    metrics["cap_height_detector"] = "wrp_pitch_projected_surface"
    maximum_approach_error = float(
        metrics.get("maximum_approach_heading_error_degrees", 180.0)
    )
    through_turn = float(metrics.get("through_turn_degrees", 0.0))

    cap_is_hidden = (
        through_turn >= 1.0
        and visible_margin >= _MINIMUM_HIDDEN_CAP_MARGIN_METRES
    )
    if cap_is_hidden:
        if maximum_approach_error >= _MINIMUM_APPROACH_MISMATCH_DEGREES:
            return _approach_mismatch(
                issue,
                maximum_approach_error=maximum_approach_error,
                metrics=metrics,
            )
        # The cap is physically hidden and the residual approach error is below
        # the Inspector's ordinary three-degree mismatch threshold. There is no
        # remaining source-intersection defect to report here.
        return None

    edge_estimate = float(metrics.get("estimated_edge_offset_metres", 0.0))
    message = (
        f"The through road turns {through_turn:.2f}° at a legacy straight "
        f"intersection cap. Pitch-projected cap/approach vertical margin at the "
        f"logical node is {visible_margin:.3f} m and the estimated edge mismatch "
        f"is {edge_estimate:.3f} m."
    )
    return replace(issue, message=message, metrics=metrics)


def inspect_road_geometry(
    input_path: Path,
    *,
    roads_geojson: Path | None = None,
    endpoint_tolerance: float = _core.DEFAULT_ENDPOINT_TOLERANCE_METRES,
    minimum_edge_gap: float = _core.DEFAULT_MINIMUM_EDGE_GAP_METRES,
    minimum_tangent_error: float = _core.DEFAULT_MINIMUM_TANGENT_ERROR_DEGREES,
    junction_match_tolerance: float = _core.DEFAULT_JUNCTION_MATCH_TOLERANCE_METRES,
):
    if _ORIGINAL_INSPECT is None:
        raise RuntimeError("road inspector surface-height policy is not installed")
    result = _ORIGINAL_INSPECT(
        input_path,
        roads_geojson=roads_geojson,
        endpoint_tolerance=endpoint_tolerance,
        minimum_edge_gap=minimum_edge_gap,
        minimum_tangent_error=minimum_tangent_error,
        junction_match_tolerance=junction_match_tolerance,
    )
    roads_by_id = {int(road.object_id): road for road in result.road_objects}
    corrected = []
    changed = False
    for issue in result.issues:
        value = _correct_turning_cap_issue(issue, roads_by_id)
        if value is None:
            changed = True
            continue
        if value != issue:
            changed = True
        corrected.append(value)
    if not changed:
        return result
    return replace(result, issues=_core._number_issues(corrected))


def install() -> None:
    """Install pitch-projected cap visibility correction in the inspector only."""

    global _ORIGINAL_INSPECT, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_INSPECT = _core.inspect_road_geometry
    _core.inspect_road_geometry = inspect_road_geometry
    _INSTALLED = True
