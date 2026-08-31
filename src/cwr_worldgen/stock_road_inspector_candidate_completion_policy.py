# SPDX-License-Identifier: GPL-3.0-or-later
"""Complete Road Inspector paved candidates with wedge-only surface coverage.

Road Inspector's grass-wedge candidate has two ordered repairs: first try a
connector-locked native stock curve/curve chain, then, if the physical outside
triangle is still exposed, cover only that triangle with the bounded borderless
paved wedge.  The earlier candidate layer correctly stopped using complete
six-metre road strips on turns, but it also disabled the triangle fallback.  A
real WRP can therefore remain geometrically reportable and visibly green even
after all exact stock-curve searches have failed.

Keep the existing candidate hook as first refusal.  After its non-turn stock
repairs finish, run the final pitch-aware paved wedge planner against the updated
report and serialize only ``paved_wedge_q*`` triangles.  The temporary miter
object below is a height/pose reference and is never appended to the report.
No stock ``ces`` or generated-gravel seam participates in this policy.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import playability as _p
from . import stock_road_emitted_seam_policy as _emitted
from . import stock_road_inspector_candidate_policy as _candidate
from . import stock_road_model_geometry as _geometry


_PAVED_FAMILIES = frozenset({"sil", "asf", "kos"})
_ORIGINAL_APPLY = None
_INSTALLED = False


def _plan_family(plan) -> str | None:
    match = _geometry.stock_straight_match(str(plan.model_path))
    if match is None:
        return None
    family = match.group("family").casefold()
    return family if family in _PAVED_FAMILIES else None


def _append_borderless_wedges(report, elevations, spec):
    """Append only the bounded outside triangles requested by Road Inspector."""

    plans = tuple(_emitted._terrain_wedge_cover_plans(report, elevations, spec))
    if not plans:
        return report

    objects = list(report.objects)
    next_id = max((int(obj.object_id) for obj in objects), default=0) + 1
    half = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6]) * 0.5
    world_name = str(getattr(spec, "name", "cwr_worldgen"))
    seen = set()
    added = 0

    for plan in plans:
        family = _plan_family(plan)
        apex = getattr(plan, "outer_miter_apex", None)
        if family is None or apex is None:
            continue
        key = (
            family,
            round(float(apex[0]), 3),
            round(float(apex[1]), 3),
            round(float(getattr(plan, "turn_degrees", 0.0)), 2),
        )
        if key in seen:
            continue
        seen.add(key)

        heading = float(plan.tangent_axis_degrees) % 360.0
        direction = (
            math.sin(math.radians(heading)),
            math.cos(math.radians(heading)),
        )
        start = (
            float(plan.centre[0]) - direction[0] * half,
            float(plan.centre[1]) - direction[1] * half,
        )
        end = (
            float(plan.centre[0]) + direction[0] * half,
            float(plan.centre[1]) + direction[1] * half,
        )

        # This object is never emitted.  _terrain_clear_wedge_overlay uses its
        # slope/height as the reference pose while constructing the tiny
        # borderless triangle at the actual outside miter.
        reference = _p._road_object_on_slope(
            next_id,
            _emitted.paved_miter_model_path(
                world_name,
                float(plan.turn_degrees),
            ),
            start,
            end,
            elevations,
            spec,
            vertical_offset=(
                _p._STOCK_ROAD_VERTICAL_OFFSET_METRES
                + _emitted.EMITTED_SEAM_UNDERLAY_BIAS_METRES
            ),
        )
        overlay = _emitted._terrain_clear_wedge_overlay(
            plan,
            reference,
            next_id,
            elevations,
            spec,
            force=True,
        )
        if overlay is None:
            continue
        # Defensive guard: a future planner must not silently turn this narrow
        # candidate into another complete road/miter/fill overlay.
        filename = (
            str(overlay.model_path)
            .replace("/", "\\")
            .rsplit("\\", 1)[-1]
            .casefold()
        )
        if not filename.startswith("paved_wedge_q"):
            continue
        objects.append(overlay)
        next_id += 1
        added += 1

    if added == 0:
        return report

    if (
        len(objects) > int(spec.max_road_objects)
        and not bool(getattr(spec, "advisory_object_limits", False))
    ):
        raise ValueError(
            "road object budget is too small after Inspector paved wedge "
            f"coverage: requires {len(objects):,} objects, "
            f"limit is {int(spec.max_road_objects):,}"
        )

    return replace(
        report,
        objects=tuple(objects),
        short_piece_objects=(
            int(getattr(report, "short_piece_objects", 0)) + added
        ),
    )


def _apply_candidate_completion(report, elevations, spec):
    if _ORIGINAL_APPLY is None:
        raise RuntimeError("Inspector candidate completion policy is not installed")

    # The candidate layer first tries exact stock geometry and keeps only its
    # non-turn seam fallbacks.  Re-audit that updated report so an already-fixed
    # seam is never given a redundant triangle.
    fixed = _ORIGINAL_APPLY(report, elevations, spec)
    return _append_borderless_wedges(fixed, elevations, spec)


def install_stock_road_inspector_candidate_completion_policy() -> None:
    """Enable the Inspector's bounded borderless paved-wedge fallback."""

    global _ORIGINAL_APPLY, _INSTALLED
    if _INSTALLED:
        return
    if not _candidate._INSTALLED:
        raise RuntimeError("Inspector candidate policy must install first")

    _ORIGINAL_APPLY = _emitted._apply_emitted_seam_covers
    _emitted._apply_emitted_seam_covers = _apply_candidate_completion
    _INSTALLED = True
