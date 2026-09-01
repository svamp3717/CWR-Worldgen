# SPDX-License-Identifier: GPL-3.0-or-later
"""Own stock-only paved/dirt road output while allowing procedural gravel.

Two installation moments are intentionally retained. The early stock-helper
stage replaces generated paved seam/wedge fallbacks with stock short road P3Ds.
The final stage applies the reference-WRP construction rules and rejects any
generated paved or dirt road model that survives to serialization.

Keeping both responsibilities here makes the stock-output contract visible in
one place without changing when either hook becomes active.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import generator as _generator
from . import playability as _p
from . import stock_road_curve_usage_policy as _curve_usage
from . import stock_road_emitted_seam_policy as _emitted
from . import stock_road_inspector_candidate_enforcement_policy as _enforcement
from . import stock_road_inspector_candidate_policy as _candidate
from . import stock_road_junction_policy as _junction
from . import stock_road_local_fit_policy as _local
from . import stock_road_model_geometry as _geometry
from . import stock_road_sharp_turn_policy as _sharp
from .procedural_infrastructure import (
    paved_miter_angle_degrees,
    paved_wedge_angle_degrees,
)


_PAVED_FAMILIES = frozenset({"sil", "asf", "kos"})
STOCK_PAVED_HELPER_BIAS_METRES = -0.006
STOCK_PAVED_OUTSIDE_OVERLAP_METRES = 0.080
MAXIMUM_STOCK_PAVED_HELPER_SHIFT_METRES = 0.75
MAXIMUM_STOCK_PAVED_HELPER_LIFT_METRES = 0.035
MINIMUM_STOCK_PAVED_TERRAIN_CLEARANCE_METRES = 0.005
NATIVE_JUNCTION_HELPER_EXCLUSION_METRES = 0.75

STOCK_CURVE_SOURCE_CORRIDOR_METRES = 1.25
ORDINARY_PAVED_OVERLAP_METRES = 0.45
ORDINARY_PAVED_OVERLAP_TANGENT_TOLERANCE_DEGREES = 0.75
ORDINARY_PAVED_OVERLAP_ENDPOINT_TOLERANCE_METRES = 0.15
ORDINARY_PAVED_OVERLAP_MAXIMUM_END_SHORTFALL_METRES = 0.55

_ORIGINAL_NATIVE_T = None
_ORIGINAL_CHAIN = None
_ORIGINAL_FIT = None
_PAVED_HELPERS_INSTALLED = False
_INSTALLED = False


def _normalised_path(model_path: str) -> str:
    return str(model_path).replace("/", "\\").casefold()


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


def _native_junction_centres(report) -> tuple[tuple[float, float], ...]:
    cap_count = min(
        int(getattr(report, "junction_cap_objects", 0)),
        len(report.objects),
    )
    centres = []
    for cap in report.objects[:cap_count]:
        local = _geometry.native_junction_intersection_offset(str(cap.model_path))
        if local is None:
            continue
        centres.append(
            _geometry.transform_local(
                local,
                (float(cap.x), float(cap.z)),
                float(cap.heading_degrees),
            )
        )
    return tuple(centres)


def _plan_hits_native_junction(
    plan,
    centres: tuple[tuple[float, float], ...],
) -> bool:
    if not centres:
        return False
    points = [(float(plan.centre[0]), float(plan.centre[1]))]
    apex = getattr(plan, "outer_miter_apex", None)
    if apex is not None:
        points.append((float(apex[0]), float(apex[1])))
    return any(
        math.dist(point, centre) <= NATIVE_JUNCTION_HELPER_EXCLUSION_METRES
        for point in points
        for centre in centres
    )


def _apply_stock_emitted_seam_covers(report, elevations, spec):
    """Replace generated paved seam helpers with stock short pieces."""

    plans = tuple(_emitted._emitted_seam_cover_plans(report))
    wedge_plans = tuple(
        _emitted._terrain_wedge_cover_plans(report, elevations, spec)
    )
    if not plans and not wedge_plans:
        return report

    native_centres = _native_junction_centres(report)
    objects = list(report.objects)
    next_id = max((int(obj.object_id) for obj in objects), default=0) + 1
    seen = set()
    added = 0
    for plan in (*plans, *wedge_plans):
        key = _plan_key(plan)
        if key in seen:
            continue
        seen.add(key)
        if _plan_hits_native_junction(plan, native_centres):
            continue
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


_apply_stock_emitted_seam_covers.__name__ = "_apply_emitted_seam_covers"


def install_stock_road_stock_paved_only_policy() -> None:
    """Make late paved seam fallbacks use only stock sil/asf/kos P3Ds."""

    global _PAVED_HELPERS_INSTALLED
    if _PAVED_HELPERS_INSTALLED:
        return
    if not _emitted._INSTALLED:
        raise RuntimeError("stock road emitted-seam policy must install first")

    _emitted._apply_emitted_seam_covers = _apply_stock_emitted_seam_covers
    _PAVED_HELPERS_INSTALLED = True


def _generated_dirt_model(model_path: str) -> bool:
    path = _normalised_path(model_path)
    filename = path.rsplit("\\", 1)[-1]
    return "\\i\\" in path and filename.startswith("dirt") and filename.endswith(".p3d")


def _family_first_native_t(incidents):
    if len(incidents) != 3:
        return None
    pair = _junction._dominant_pair(incidents)
    if pair is None:
        return None
    first, second = pair
    branch = next(index for index in range(3) if index not in pair)
    main_family = incidents[first].family
    if main_family is None or incidents[second].family != main_family:
        return None
    branch_family = incidents[branch].family
    if branch_family is None:
        return None
    model = _junction._T_JUNCTION_MODELS.get((main_family, branch_family))
    if model is None:
        return None

    main_a = _junction._heading(incidents[first].direction)
    main_b = _junction._heading(incidents[second].direction)
    branch_heading = _junction._heading(incidents[branch].direction)
    fits = []
    for actual_zero, actual_180 in ((main_a, main_b), (main_b, main_a)):
        rotation, maximum_error = _junction._best_rotation(
            (
                (0.0, actual_zero),
                (180.0, actual_180),
                (_candidate.MEASURED_T_BRANCH_LOCAL_HEADING_DEGREES, branch_heading),
            )
        )
        fits.append((maximum_error, rotation))
    maximum_error, rotation = min(fits)
    return _junction._NativeJunction(
        model,
        rotation,
        maximum_error,
        main_family,
    )


def _stock_native_t_dispatch(incidents):
    if _ORIGINAL_NATIVE_T is None:
        raise RuntimeError("stock paved/dirt policy is not installed")

    if _enforcement._contains_generated_gravel(incidents):
        return _ORIGINAL_NATIVE_T(incidents)

    if _local._PLANNING_RELAXED_JUNCTION.get():
        return _family_first_native_t(incidents)

    return _candidate._measured_native_t_junction(incidents)


def _stock_paved_straight(piece) -> bool:
    match = _geometry.stock_straight_match(str(piece.model_path))
    return match is not None and match.group("family").casefold() in _PAVED_FAMILIES


def _heading(start, end) -> float:
    return math.degrees(
        math.atan2(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
    ) % 360.0


def _tangent_compatible_straight_chain(fitted) -> bool:
    if len(fitted) < 2 or any(not _stock_paved_straight(item[0]) for item in fitted):
        return False
    for previous, current in zip(fitted, fitted[1:]):
        previous_heading = _heading(previous[1], previous[2])
        current_heading = _heading(current[1], current[2])
        if _p._heading_difference(previous_heading, current_heading) > (
            ORDINARY_PAVED_OVERLAP_TANGENT_TOLERANCE_DEGREES + 1.0e-9
        ):
            return False
        if math.dist(previous[2], current[1]) > (
            ORDINARY_PAVED_OVERLAP_ENDPOINT_TOLERANCE_METRES + 1.0e-9
        ):
            return False
    return True


def _valid_overlapped_straight_chain(fitted) -> bool:
    if len(fitted) < 2 or any(not _stock_paved_straight(item[0]) for item in fitted):
        return False
    seam_limit = (
        ORDINARY_PAVED_OVERLAP_METRES
        + ORDINARY_PAVED_OVERLAP_ENDPOINT_TOLERANCE_METRES
    )
    for previous, current in zip(fitted, fitted[1:]):
        if _p._heading_difference(
            _heading(previous[1], previous[2]),
            _heading(current[1], current[2]),
        ) > ORDINARY_PAVED_OVERLAP_TANGENT_TOLERANCE_DEGREES + 1.0e-9:
            return False
        if math.dist(previous[2], current[1]) > seam_limit + 1.0e-9:
            return False
    return True


def _overlapped_stock_chain(
    measure,
    pieces,
    fitted,
    *,
    start_distance,
    preferred_end_distance,
    minimum_end_distance,
    maximum_end_distance,
):
    if not _tangent_compatible_straight_chain(fitted):
        return fitted
    sequence = tuple(item[0] for item in fitted)
    if any(not _stock_paved_straight(piece) for piece in sequence):
        return fitted

    current = float(start_distance)
    rebuilt = []
    final_progress = current
    for index, piece in enumerate(sequence):
        endpoint = measure.chord_endpoint(
            current,
            float(piece.length_metres),
            float(maximum_end_distance),
        )
        if endpoint is None:
            return fitted
        end_distance, end_x, end_z, _chord_heading = endpoint
        start_x, start_z, _start_heading = measure.point(current)
        rebuilt.append((piece, (start_x, start_z), (end_x, end_z)))
        final_progress = float(end_distance)
        if index + 1 < len(sequence):
            next_progress = final_progress - ORDINARY_PAVED_OVERLAP_METRES
            if next_progress <= current + 0.05:
                return fitted
            current = next_progress

    if final_progress < (
        float(minimum_end_distance)
        - ORDINARY_PAVED_OVERLAP_MAXIMUM_END_SHORTFALL_METRES
    ):
        return fitted
    if final_progress > float(maximum_end_distance) + 1.0e-6:
        return fitted
    if not _valid_overlapped_straight_chain(tuple(rebuilt)):
        return fitted
    return tuple(rebuilt)


def _stock_overlap_chain(
    measure,
    pieces,
    *,
    start_distance,
    preferred_end_distance,
    minimum_end_distance,
    maximum_end_distance,
):
    if _ORIGINAL_CHAIN is None:
        raise RuntimeError("stock paved overlap policy is not installed")
    fitted = _ORIGINAL_CHAIN(
        measure,
        pieces,
        start_distance=start_distance,
        preferred_end_distance=preferred_end_distance,
        minimum_end_distance=minimum_end_distance,
        maximum_end_distance=maximum_end_distance,
    )
    return _overlapped_stock_chain(
        measure,
        pieces,
        fitted,
        start_distance=start_distance,
        preferred_end_distance=preferred_end_distance,
        minimum_end_distance=minimum_end_distance,
        maximum_end_distance=maximum_end_distance,
    )


def _generated_road_model(model_path: str) -> bool:
    path = _normalised_path(model_path)
    filename = path.rsplit("\\", 1)[-1]
    return (
        filename == "paved_fill.p3d"
        or paved_miter_angle_degrees(filename) is not None
        or paved_wedge_angle_degrees(filename) is not None
        or _generated_dirt_model(path)
    )


def _fit(
    dataset,
    projection,
    elevations,
    spec,
    *,
    starting_id: int = 1,
    progress_callback=None,
):
    if _ORIGINAL_FIT is None:
        raise RuntimeError("stock paved/dirt final guard is not installed")
    report = _ORIGINAL_FIT(
        dataset,
        projection,
        elevations,
        spec,
        starting_id=starting_id,
        progress_callback=progress_callback,
    )
    forbidden = tuple(
        obj for obj in report.objects if _generated_road_model(str(obj.model_path))
    )
    if forbidden:
        sample = ", ".join(sorted({str(obj.model_path) for obj in forbidden})[:6])
        raise ValueError(
            "stock paved/dirt policy violation: generated paved or dirt road P3Ds "
            f"survived final fitting ({len(forbidden)} objects; {sample})"
        )
    return report


def install_stock_road_stock_assets_only_policy() -> None:
    """Install final stock paved/dirt rules while preserving generated gravel."""

    global _ORIGINAL_NATIVE_T, _ORIGINAL_CHAIN, _ORIGINAL_FIT, _INSTALLED
    if _INSTALLED:
        return
    if not _PAVED_HELPERS_INSTALLED:
        raise RuntimeError("stock paved helper policy must install first")
    if not _enforcement._FINAL_INSTALLED or not _curve_usage._INSTALLED:
        raise RuntimeError("candidate enforcement and curve usage must install first")

    _ORIGINAL_NATIVE_T = _enforcement._junction._native_t_junction
    _ORIGINAL_CHAIN = _curve_usage._ORIGINAL_CHAIN
    _curve_usage._ORIGINAL_CHAIN = _stock_overlap_chain

    _ORIGINAL_FIT = _p.fit_road_objects
    _generator.fit_road_objects = _fit

    _enforcement._junction._native_t_junction = _stock_native_t_dispatch
    _sharp._MAXIMUM_LOCKED_CORRIDOR_METRES = STOCK_CURVE_SOURCE_CORRIDOR_METRES

    _INSTALLED = True
