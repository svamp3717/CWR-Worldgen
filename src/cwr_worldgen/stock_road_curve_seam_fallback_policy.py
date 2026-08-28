# SPDX-License-Identifier: GPL-3.0-or-later
"""Cover residual paved curve seams that cannot be made tangent-continuous.

Native curve promotion is the preferred fix for triangular asphalt wedges.  Real
source geometry can still force a curve-to-straight boundary whose physical
centreline connectors coincide while the rendered tangents differ by a few
degrees.  The final-continuity policy used to disable the visual curve-seam
underlay entirely, assuming coherent selection had eliminated every such case.
The Lundby in-game screenshots and compiled WRP disprove that assumption.

Keep the promoted curve chain intact and add the existing same-family six-metre
underlay only at isolated paved curve seams.  The underlay is placed below the
visible road, so it fills the exposed triangle without changing the fitted
centreline, terrain, junction geometry, or any non-road object placement.
"""
from __future__ import annotations

from dataclasses import replace
import math
import re

from . import playability as _p
from . import stock_road_model_geometry as _model_geometry
from . import stock_road_visual_finish_policy as _finish

MAXIMUM_PAVED_CURVE_SEAM_TANGENT_ERROR_DEGREES = 8.0
_PAVED_COVER = re.compile(
    r"^(?:.*[\\/])(?P<family>sil|asf|kos)6\.p3d$",
    re.IGNORECASE,
)

_ORIGINAL_FINISH = None
_INSTALLED = False


def _paved_curve_seam_cover_plans(report):
    """Return only isolated paved curve-seam underlays."""

    plans = _finish._curve_seam_cover_plans(report)
    return tuple(
        plan
        for plan in plans
        if _PAVED_COVER.fullmatch(str(plan.model_path).replace("/", "\\")) is not None
    )


def _apply_paved_curve_seam_fallback(report, elevations, spec):
    if _ORIGINAL_FINISH is None:
        raise RuntimeError("stock road curve-seam fallback policy is not installed")

    # Compute curve plans before the straight-seam wrapper adds its own
    # underlays.  Those helper pieces must not turn an ordinary two-piece curve
    # seam into a synthetic three-endpoint cluster and suppress the curve cover.
    plans = _paved_curve_seam_cover_plans(report)
    report = _ORIGINAL_FINISH(report, elevations, spec)
    if not plans:
        return report

    required = len(report.objects) + len(plans)
    if (
        required > int(spec.max_road_objects)
        and not bool(getattr(spec, "advisory_object_limits", False))
    ):
        raise ValueError(
            "road object budget is too small after paved curve seam coverage: "
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


def install_stock_road_curve_seam_fallback_policy() -> None:
    """Install paved curve-seam coverage after straight-seam coverage."""

    global _ORIGINAL_FINISH, _INSTALLED
    if _INSTALLED:
        return

    # The old 3.25-degree ceiling misses the 6.01-degree curve/straight seam
    # measured in the supplied Lundby18 WRP.  Eight degrees remains deliberately
    # below one whole stock-curve turn, so larger geometry errors still have to
    # be solved by fitting rather than hidden with a slab.
    _finish.MAXIMUM_CURVE_SEAM_TANGENT_ERROR_DEGREES = (
        MAXIMUM_PAVED_CURVE_SEAM_TANGENT_ERROR_DEGREES
    )
    _ORIGINAL_FINISH = _finish._apply_curve_seam_covers
    _finish._apply_curve_seam_covers = _apply_paved_curve_seam_fallback
    _INSTALLED = True
