# SPDX-License-Identifier: GPL-3.0-or-later
"""Refine final emitted paved-seam coverage from Lundby25 evidence.

The outer emitted-seam pass removed nearly every paved seam in Lundby25, leaving
three small aligned connector holes, three larger straight mitres whose single
average-heading underlay did not cover the full outside wedge, and one small
legacy-cap-to-approach seam.  Keep the successful final pass and narrow only
those cases:

* a real physical gap may need coverage even when tangent error is almost zero;
* a straight mitre above five degrees is safer with two low underlays following
  the two visible road tangents than one average slab; and
* a paved six-metre legacy cap may participate when one endpoint has an
  unambiguous same-family mate.

Near-coincident overlapping road pieces are not seams and are deliberately
ignored here.  Road Inspector has a matching read-only overlap filter.
"""
from __future__ import annotations

import math

from . import playability as _p
from . import stock_road_emitted_seam_policy as _emitted
from . import stock_road_model_geometry as _geometry
from . import stock_road_visual_finish_policy as _finish

MINIMUM_PHYSICAL_OPEN_GAP_METRES = 0.04
DUAL_UNDERLAY_TANGENT_ERROR_DEGREES = 5.0
OVERLAPPING_CENTRE_DISTANCE_METRES = 0.50
OVERLAPPING_TANGENT_ERROR_DEGREES = 2.0
OVERLAPPING_VERTICAL_DISTANCE_METRES = 0.15

_ORIGINAL_PLANNER = None
_INSTALLED = False


def _paved_cap_endpoints(report):
    cap_count = min(
        int(getattr(report, "junction_cap_objects", 0)),
        len(report.objects),
    )
    result = []
    for obj in report.objects[:cap_count]:
        match = _geometry.stock_straight_match(str(obj.model_path))
        if match is None or int(match.group("length")) != 6:
            continue
        family = match.group("family").casefold()
        if family not in _emitted._PAVED_FAMILIES:
            continue
        length = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6])
        axis = _p._model_axis(obj, length)
        for endpoint_index, point in enumerate(axis):
            result.append(
                _finish._SeamEndpoint(
                    point=(float(point[0]), float(point[1])),
                    object_id=int(obj.object_id),
                    endpoint_index=endpoint_index,
                    family=family,
                    tangent_axis_degrees=float(obj.heading_degrees) % 180.0,
                    is_curve=False,
                )
            )
    return tuple(result)


def _emitted_endpoints(report):
    chain = tuple(
        endpoint
        for endpoint in _finish._seam_endpoints(report)
        if endpoint.family in _emitted._PAVED_FAMILIES
    )
    return chain + _paved_cap_endpoints(report)


def _axis_error(first: float, second: float) -> float:
    return _finish._axis_heading_difference(first, second)


def _overlapping_pair(report, first, second) -> bool:
    road_by_id = {int(obj.object_id): obj for obj in report.objects}
    first_obj = road_by_id.get(int(first.object_id))
    second_obj = road_by_id.get(int(second.object_id))
    if first_obj is None or second_obj is None:
        return False
    first_match = _geometry.stock_straight_match(str(first_obj.model_path))
    second_match = _geometry.stock_straight_match(str(second_obj.model_path))
    if first_match is None or second_match is None:
        return False
    if first_match.group("family").casefold() != second_match.group("family").casefold():
        return False
    if (
        math.dist(
            (float(first_obj.x), float(first_obj.z)),
            (float(second_obj.x), float(second_obj.z)),
        )
        > OVERLAPPING_CENTRE_DISTANCE_METRES
    ):
        return False
    if (
        _axis_error(first_obj.heading_degrees, second_obj.heading_degrees)
        > OVERLAPPING_TANGENT_ERROR_DEGREES
    ):
        return False
    return (
        abs(float(first_obj.y) - float(second_obj.y))
        <= OVERLAPPING_VERTICAL_DISTANCE_METRES
    )


def _plan(model_path: str, centre, heading: float):
    return _finish._SeamCoverPlan(
        model_path=model_path,
        centre=(float(centre[0]), float(centre[1])),
        tangent_axis_degrees=float(heading) % 180.0,
    )


def _refined_emitted_seam_cover_plans(report):
    endpoints = _emitted_endpoints(report)
    if not endpoints:
        return ()

    plans = []
    for distance, first, second, _pair_key in _emitted._nearest_endpoint_pairs(endpoints):
        if first.family != second.family:
            continue
        if not _emitted._pair_is_unambiguous(endpoints, first, second, distance):
            continue
        if _overlapping_pair(report, first, second):
            continue

        tangent_error = _axis_error(
            first.tangent_axis_degrees,
            second.tangent_axis_degrees,
        )
        curve_seam = bool(first.is_curve or second.is_curve)
        if curve_seam:
            if (
                tangent_error < _emitted.MINIMUM_EMITTED_TANGENT_ERROR_DEGREES
                or distance > _emitted.MAXIMUM_EMITTED_CURVE_GAP_METRES + 1.0e-9
                or tangent_error > _emitted.MAXIMUM_EMITTED_CURVE_TANGENT_ERROR_DEGREES
            ):
                continue
        else:
            if (
                distance > _emitted.MAXIMUM_EMITTED_STRAIGHT_GAP_METRES + 1.0e-9
                or tangent_error > _emitted.MAXIMUM_EMITTED_STRAIGHT_TANGENT_ERROR_DEGREES
            ):
                continue
            # Exact/near-exact seams need no helper, but a real horizontal hole
            # does, even when both road tangents are essentially identical.
            if (
                distance < MINIMUM_PHYSICAL_OPEN_GAP_METRES
                and tangent_error < _emitted.MINIMUM_EMITTED_TANGENT_ERROR_DEGREES
            ):
                continue

        if _emitted._covered_by_existing_surface(report, first, second):
            continue

        centre = (
            (float(first.point[0]) + float(second.point[0])) * 0.5,
            (float(first.point[1]) + float(second.point[1])) * 0.5,
        )
        model_path = rf"o\road\{first.family}6.p3d"
        if (
            not curve_seam
            and tangent_error >= DUAL_UNDERLAY_TANGENT_ERROR_DEGREES
        ):
            headings = []
            for value in (
                first.tangent_axis_degrees,
                second.tangent_axis_degrees,
            ):
                if not any(_axis_error(value, existing) <= 0.25 for existing in headings):
                    headings.append(float(value))
            plans.extend(_plan(model_path, centre, heading) for heading in headings)
        else:
            plans.append(
                _plan(
                    model_path,
                    centre,
                    _emitted._plan_heading(first, second),
                )
            )
    return tuple(plans)


def install_stock_road_emitted_seam_refinement_policy() -> None:
    """Refine the already-outermost emitted-seam planner in place."""

    global _ORIGINAL_PLANNER, _INSTALLED
    if _INSTALLED:
        return
    if not _emitted._INSTALLED:
        raise RuntimeError("final emitted seam policy must install first")
    _ORIGINAL_PLANNER = _emitted._emitted_seam_cover_plans
    _emitted._emitted_seam_cover_plans = _refined_emitted_seam_cover_plans
    _INSTALLED = True
