# SPDX-License-Identifier: GPL-3.0-or-later
"""Extend the final terrain-clear wedge pass to every paved stock-road family.

The emitted-seam policy originally grew its terrain-clear triangle around the
``sil`` family because that was the first asphalt case reproduced in Lundby.
That leaves the same physical outside-miter failure possible on ``asf`` and
``kos`` roads.  Keep dirt/gravel completely out of this policy: only the three
paved stock families participate.

This module patches the narrow helper predicates used by the already-installed
final emitted-seam wrapper.  The wrapper remains the owner of object budgeting,
placement and generated P3D registration.
"""
from __future__ import annotations

import math

from . import stock_road_emitted_seam_policy as _emitted
from . import stock_road_visual_finish_policy as _finish


_INSTALLED = False


def _generated_wedge_contains(obj, point: tuple[float, float]) -> bool:
    turn = _emitted.paved_wedge_angle_degrees(str(obj.model_path))
    if turn is None:
        return False
    dx = float(point[0]) - float(obj.x)
    dz = float(point[1]) - float(obj.z)
    heading = math.radians(float(obj.heading_degrees))
    local_x = dx * math.cos(heading) - dz * math.sin(heading)
    cosine_pitch = math.cos(math.radians(float(obj.pitch_degrees)))
    if abs(cosine_pitch) <= 1.0e-9:
        return False
    local_z = (dx * math.sin(heading) + dz * math.cos(heading)) / cosine_pitch
    points = _emitted.paved_wedge_local_points(turn)
    depth = float(points[0][2])
    base_half_width = abs(float(points[1][0]))
    margin = _emitted._SURFACE_MARGIN_METRES
    if local_z < -margin or local_z > depth + margin:
        return False
    fraction = max(0.0, min(1.0, 1.0 - local_z / max(1.0e-9, depth)))
    return abs(local_x) <= base_half_width * fraction + margin


def _surface_contains(obj, point: tuple[float, float]) -> bool:
    if _emitted._straight_contains(obj, point):
        return True
    filename = str(obj.model_path).replace("/", "\\").rsplit("\\", 1)[-1].casefold()
    if filename == "paved_fill.p3d":
        return math.dist((float(obj.x), float(obj.z)), point) <= (
            _emitted.GENERATED_PAVED_FILL_RADIUS_METRES + 0.001
        )
    if _emitted.paved_miter_angle_degrees(filename) is not None:
        return _emitted._generated_miter_contains(obj, point)
    return _generated_wedge_contains(obj, point)


def _surface_is_paved(obj) -> bool:
    match = _emitted._geometry.stock_straight_match(str(obj.model_path))
    if match is not None:
        return match.group("family").casefold() in _emitted._PAVED_FAMILIES
    filename = str(obj.model_path).replace("/", "\\").rsplit("\\", 1)[-1].casefold()
    return (
        filename == "paved_fill.p3d"
        or _emitted.paved_miter_angle_degrees(filename) is not None
        or _emitted.paved_wedge_angle_degrees(filename) is not None
    )


def _terrain_wedge_cover_plans(report, elevations=None, spec=None):
    """Plan terrain-clear outside triangles for sil/asf/kos seams only."""

    endpoints = tuple(
        endpoint
        for endpoint in _finish._seam_endpoints(report)
        if endpoint.family in _emitted._PAVED_FAMILIES
    )
    plans = []
    for distance, first, second, _pair_key in _emitted._nearest_endpoint_pairs(endpoints):
        if first.family != second.family:
            continue
        if not _emitted._pair_is_unambiguous(endpoints, first, second, distance):
            continue
        turn = _finish._axis_heading_difference(
            first.tangent_axis_degrees,
            second.tangent_axis_degrees,
        )
        if turn < _emitted.MINIMUM_EMITTED_TANGENT_ERROR_DEGREES:
            continue
        if turn > _emitted.MAXIMUM_EMITTED_STRAIGHT_TANGENT_ERROR_DEGREES:
            continue
        seam_centre = (
            (float(first.point[0]) + float(second.point[0])) * 0.5,
            (float(first.point[1]) + float(second.point[1])) * 0.5,
        )
        involved_ids = {int(first.object_id), int(second.object_id)}
        if any(
            int(candidate.object_id) not in involved_ids
            and math.dist(candidate.point, seam_centre)
            <= _emitted.TERRAIN_WEDGE_JUNCTION_EXCLUSION_METRES
            for candidate in endpoints
        ):
            continue
        geometry = _emitted._outer_miter_geometry(first, second)
        if geometry is None:
            continue
        _area, apex, centroid = geometry
        coverage_samples = _emitted._gap_samples(first, second) + (apex, centroid)
        if _emitted._terrain_wedge_already_visible(
            report,
            first,
            second,
            coverage_samples,
            elevations,
            spec,
        ):
            continue
        plans.append(
            _finish._SeamCoverPlan(
                model_path=rf"o\road\{first.family}6.p3d",
                centre=seam_centre,
                tangent_axis_degrees=_emitted._plan_heading(first, second),
                turn_degrees=turn,
                outer_miter_apex=apex,
            )
        )
    return tuple(plans)


def install_stock_road_paved_wedge_policy() -> None:
    """Make the final wedge audit paved-family complete without touching dirt."""

    global _INSTALLED
    if _INSTALLED:
        return
    _emitted._surface_contains = _surface_contains
    # Keep the historical private name because the emitted policy calls it.
    _emitted._surface_is_sil = _surface_is_paved
    _emitted._terrain_wedge_cover_plans = _terrain_wedge_cover_plans
    _INSTALLED = True
