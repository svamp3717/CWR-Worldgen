# SPDX-License-Identifier: GPL-3.0-or-later
"""Hide triangular terrain wedges at unavoidable paved straight-piece mitres.

Native curve selection remains the preferred way to build a bend.  Some source
geometry still cannot be represented by the available ten-degree stock curves,
so the final fallback is a sequence of short straight P3Ds.  Two such rectangles
can share the exact same centreline connector while their outside edges separate
as soon as their headings differ.  CWA then shows a triangular grass wedge over
roughly half of the carriageway.

Keep those visible road pieces unchanged and place one same-family ``6`` road
piece one centimetre lower beneath an isolated angled seam.  The low piece is
only an underlay: it fills the exposed triangle without replacing the bend,
raising the road, or re-enabling the curve-repair slabs deliberately disabled by
the final continuity policy.  Junction areas are excluded by requiring exactly
two physical endpoints at the seam.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import playability as _p
from . import stock_road_model_geometry as _model_geometry
from . import stock_road_visual_finish_policy as _finish

MINIMUM_STRAIGHT_SEAM_TANGENT_ERROR_DEGREES = 0.75
MAXIMUM_STRAIGHT_SEAM_TANGENT_ERROR_DEGREES = 35.0
_STRAIGHT_SEAM_PAVED_FAMILIES = frozenset({"sil", "asf", "kos"})

_ORIGINAL_FINISH = None
_INSTALLED = False


def _straight_seam_cover_plans(report):
    """Return low underlays for isolated paved straight-to-straight mitres."""

    endpoints = _finish._seam_endpoints(report)
    if not endpoints:
        return ()

    buckets = {}
    for endpoint in endpoints:
        buckets.setdefault(_finish._endpoint_bucket(endpoint.point), []).append(endpoint)

    plans = []
    used_pairs = set()
    tolerance = _finish.CURVE_SEAM_ENDPOINT_TOLERANCE_METRES
    for endpoint in endpoints:
        bx, bz = _finish._endpoint_bucket(endpoint.point)
        neighbours = []
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for candidate in buckets.get((bx + dx, bz + dz), ()):
                    if math.dist(endpoint.point, candidate.point) <= tolerance:
                        neighbours.append(candidate)

        # Two endpoints identify one ordinary chain seam.  Three or more is a
        # junction or overlap area and remains entirely owned by junction policy.
        unique = {
            (candidate.object_id, candidate.endpoint_index): candidate
            for candidate in neighbours
        }
        if len(unique) != 2:
            continue
        first, second = sorted(
            unique.values(), key=lambda item: (item.object_id, item.endpoint_index)
        )
        if first.object_id == second.object_id or first.family != second.family:
            continue
        if first.is_curve or second.is_curve:
            continue
        if first.family not in _STRAIGHT_SEAM_PAVED_FAMILIES:
            continue

        pair_key = (
            first.object_id,
            first.endpoint_index,
            second.object_id,
            second.endpoint_index,
        )
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)

        error = _finish._axis_heading_difference(
            first.tangent_axis_degrees, second.tangent_axis_degrees
        )
        if not (
            MINIMUM_STRAIGHT_SEAM_TANGENT_ERROR_DEGREES
            <= error
            <= MAXIMUM_STRAIGHT_SEAM_TANGENT_ERROR_DEGREES
        ):
            continue

        plans.append(
            _finish._SeamCoverPlan(
                model_path=rf"o\road\{first.family}6.p3d",
                centre=(
                    (first.point[0] + second.point[0]) * 0.5,
                    (first.point[1] + second.point[1]) * 0.5,
                ),
                tangent_axis_degrees=_finish._average_axis_heading(
                    first.tangent_axis_degrees, second.tangent_axis_degrees
                ),
            )
        )
    return tuple(plans)


def _apply_straight_seam_covers(report, elevations, spec):
    if _ORIGINAL_FINISH is None:
        raise RuntimeError("stock road straight-seam policy is not installed")

    # Preserve whatever the final continuity layer currently wants to do with
    # curve seams.  On the present stack that deliberately does nothing.
    report = _ORIGINAL_FINISH(report, elevations, spec)
    plans = _straight_seam_cover_plans(report)
    if not plans:
        return report

    required = len(report.objects) + len(plans)
    if (
        required > int(spec.max_road_objects)
        and not bool(getattr(spec, "advisory_object_limits", False))
    ):
        raise ValueError(
            "road object budget is too small after straight seam coverage: "
            f"requires {required:,} objects, limit is {spec.max_road_objects:,}"
        )

    objects = list(report.objects)
    next_id = max((int(obj.object_id) for obj in objects), default=0) + 1
    half = _model_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6] * 0.5
    for plan in plans:
        angle = math.radians(plan.tangent_axis_degrees)
        direction = (math.sin(angle), math.cos(angle))
        start = (
            plan.centre[0] - direction[0] * half,
            plan.centre[1] - direction[1] * half,
        )
        end = (
            plan.centre[0] + direction[0] * half,
            plan.centre[1] + direction[1] * half,
        )
        objects.append(
            _p._road_object_on_slope(
                next_id,
                plan.model_path,
                start,
                end,
                elevations,
                spec,
                vertical_offset=(
                    _p._STOCK_ROAD_VERTICAL_OFFSET_METRES
                    + _finish.CURVE_SEAM_COVER_VERTICAL_BIAS_METRES
                ),
            )
        )
        next_id += 1

    return replace(
        report,
        objects=tuple(objects),
        short_piece_objects=(
            int(getattr(report, "short_piece_objects", 0)) + len(plans)
        ),
    )


def install_stock_road_straight_seam_policy() -> None:
    """Install the straight-facet underlay after final curve continuity policy."""

    global _ORIGINAL_FINISH, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_FINISH = _finish._apply_curve_seam_covers
    _finish._apply_curve_seam_covers = _apply_straight_seam_covers
    _INSTALLED = True
