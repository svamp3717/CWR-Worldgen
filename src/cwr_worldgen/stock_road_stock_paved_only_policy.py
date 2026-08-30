# SPDX-License-Identifier: GPL-3.0-or-later
"""Use only stock CWA road P3Ds for late paved-road fallbacks.

The earlier paved seam work introduced world-local ``paved_fill``,
``paved_miter`` and ``paved_wedge`` models.  They made the final WRP harder to
reason about and, more importantly, could disagree with what CWA actually showed
at a turn.  Keep the measured/curve fitting work, but make the final fallback
strictly stock-only: sil/asf/kos six-metre pieces are shifted slightly toward
the outside miter and kept below the visible carriageway.

Dirt/gravel is deliberately untouched.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import playability as _p
from . import stock_road_emitted_seam_policy as _emitted
from . import stock_road_junction_policy as _junction
from . import stock_road_model_geometry as _geometry
from . import stock_road_turning_t_fallback_policy as _turning_t


_PAVED_FAMILIES = frozenset({"sil", "asf", "kos"})
STOCK_PAVED_HELPER_BIAS_METRES = -0.006
STOCK_PAVED_OUTSIDE_OVERLAP_METRES = 0.080
MAXIMUM_STOCK_PAVED_HELPER_SHIFT_METRES = 0.75
MAXIMUM_STOCK_PAVED_HELPER_LIFT_METRES = 0.035
MINIMUM_STOCK_PAVED_TERRAIN_CLEARANCE_METRES = 0.005

_ORIGINAL_TURNING_T_CAP = None
_INSTALLED = False


def _stock_family(model_path: str) -> str | None:
    match = _geometry.stock_straight_match(str(model_path))
    if match is None:
        return None
    family = match.group("family").casefold()
    return family if family in _PAVED_FAMILIES else None


def _stock_short_model(family: str) -> str:
    return rf"o\road\{family}6.p3d"


def _shifted_helper_centre(plan, family: str) -> tuple[float, float]:
    centre = (float(plan.centre[0]), float(plan.centre[1]))
    apex = getattr(plan, "outer_miter_apex", None)
    if apex is None:
        return centre

    dx = float(apex[0]) - centre[0]
    dz = float(apex[1]) - centre[1]
    distance = math.hypot(dx, dz)
    if distance <= 1.0e-9:
        return centre

    half_width = float(_geometry.STOCK_HALF_WIDTHS_METRES[family])
    shift = max(
        0.0,
        distance - half_width + STOCK_PAVED_OUTSIDE_OVERLAP_METRES,
    )
    shift = min(shift, MAXIMUM_STOCK_PAVED_HELPER_SHIFT_METRES)
    if shift <= 1.0e-9:
        return centre
    return (
        centre[0] + dx / distance * shift,
        centre[1] + dz / distance * shift,
    )


def _stock_helper_for_plan(plan, object_id, elevations, spec):
    family = _stock_family(str(plan.model_path))
    if family is None:
        return None

    centre = _shifted_helper_centre(plan, family)
    heading = float(plan.tangent_axis_degrees) % 360.0
    direction = (
        math.sin(math.radians(heading)),
        math.cos(math.radians(heading)),
    )
    half = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6]) * 0.5
    start = (
        centre[0] - direction[0] * half,
        centre[1] - direction[1] * half,
    )
    end = (
        centre[0] + direction[0] * half,
        centre[1] + direction[1] * half,
    )
    helper = _p._road_object_on_slope(
        int(object_id),
        _stock_short_model(family),
        start,
        end,
        elevations,
        spec,
        vertical_offset=(
            _p._STOCK_ROAD_VERTICAL_OFFSET_METRES
            + STOCK_PAVED_HELPER_BIAS_METRES
        ),
    )

    # A full stock strip must not be raised aggressively because its painted
    # borders would then win over the real approaches.  Permit only a tiny lift
    # when the outside miter would otherwise remain under terrain; larger
    # cross-slope failures are intentionally left for Road Inspector to report.
    apex = getattr(plan, "outer_miter_apex", None)
    if apex is not None and elevations is not None and spec is not None:
        terrain = _p._sample_elevation(
            elevations,
            spec.cells,
            spec.cell_size,
            float(apex[0]),
            float(apex[1]),
        )
        surface = _emitted._surface_height_at(
            helper,
            (float(apex[0]), float(apex[1])),
        )
        lift = max(
            0.0,
            terrain + MINIMUM_STOCK_PAVED_TERRAIN_CLEARANCE_METRES - surface,
        )
        if lift > 0.0:
            helper = replace(
                helper,
                y=float(helper.y)
                + min(lift, MAXIMUM_STOCK_PAVED_HELPER_LIFT_METRES),
            )
    return helper


def _plan_key(plan):
    apex = getattr(plan, "outer_miter_apex", None)
    if apex is not None:
        return (
            "apex",
            round(float(apex[0]), 3),
            round(float(apex[1]), 3),
            str(plan.model_path).casefold(),
        )
    return (
        "centre",
        round(float(plan.centre[0]), 3),
        round(float(plan.centre[1]), 3),
        round(float(plan.tangent_axis_degrees) % 180.0, 3),
        str(plan.model_path).casefold(),
    )


def _apply_stock_emitted_seam_covers(report, elevations, spec):
    """Replace generated paved miter/wedge objects with stock short pieces."""

    plans = tuple(_emitted._emitted_seam_cover_plans(report))
    wedge_plans = tuple(_emitted._terrain_wedge_cover_plans(report, elevations, spec))
    if not plans and not wedge_plans:
        return report

    objects = list(report.objects)
    next_id = max((int(obj.object_id) for obj in objects), default=0) + 1
    seen = set()
    added = 0
    for plan in (*plans, *wedge_plans):
        key = _plan_key(plan)
        if key in seen:
            continue
        seen.add(key)
        helper = _stock_helper_for_plan(plan, next_id, elevations, spec)
        if helper is None:
            continue
        objects.append(helper)
        next_id += 1
        added += 1

    if added == 0:
        return report

    required = len(objects)
    if (
        required > int(spec.max_road_objects)
        and not bool(getattr(spec, "advisory_object_limits", False))
    ):
        raise ValueError(
            "road object budget is too small after stock-only paved seam "
            f"coverage: requires {required:,} objects, "
            f"limit is {int(spec.max_road_objects):,}"
        )

    return replace(
        report,
        objects=tuple(objects),
        short_piece_objects=(
            int(getattr(report, "short_piece_objects", 0)) + added
        ),
    )


def _legacy_stock_cap_for_turning_t(
    current,
    source_node,
    incidents,
    family,
    elevations,
    spec,
):
    """Demote a turning T with the stock family short piece, never paved_fill."""

    if family not in _PAVED_FAMILIES:
        return current
    pair = _junction._dominant_pair(incidents)
    if pair is None:
        return current
    heading = _junction._heading(incidents[pair[0]].direction)
    length = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6])
    half = length * 0.5
    direction = (
        math.sin(math.radians(heading)),
        math.cos(math.radians(heading)),
    )
    start = (
        float(source_node[0]) - direction[0] * half,
        float(source_node[1]) - direction[1] * half,
    )
    end = (
        float(source_node[0]) + direction[0] * half,
        float(source_node[1]) + direction[1] * half,
    )
    fixed = _p._road_object_on_slope(
        int(current.object_id),
        _stock_short_model(family),
        start,
        end,
        elevations,
        spec,
        vertical_offset=_p._STOCK_ROAD_VERTICAL_OFFSET_METRES,
    )
    return replace(
        fixed,
        x=float(source_node[0]),
        z=float(source_node[1]),
        heading_degrees=heading % 360.0,
    )


def install_stock_road_stock_paved_only_policy() -> None:
    """Make every late paved fallback use only stock sil/asf/kos P3Ds."""

    global _ORIGINAL_TURNING_T_CAP, _INSTALLED
    if _INSTALLED:
        return
    if not _emitted._INSTALLED:
        raise RuntimeError("stock road emitted-seam policy must install first")
    if not _turning_t._INSTALLED:
        raise RuntimeError("turning-T fallback policy must install first")

    _ORIGINAL_TURNING_T_CAP = _turning_t._legacy_cap_for_turning_t
    _emitted._apply_emitted_seam_covers = _apply_stock_emitted_seam_covers
    _turning_t._legacy_cap_for_turning_t = _legacy_stock_cap_for_turning_t
    _INSTALLED = True
