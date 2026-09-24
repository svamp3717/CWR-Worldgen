# SPDX-License-Identifier: GPL-3.0-or-later
"""Repair final paved road seams using the post-fit road inspector.

The normal road fitter, junction policies, bridge cleanup and final dedupe remain
authoritative.  This late pass inspects the WorldObject transforms that survived
those stages, asks road_inspector for stock-first repair plans, and applies only
ordinary paved seam repairs.  Dirt, gravel and protected junction-cap objects are
never replaced here.

Repairs are bounded and re-inspected.  This is important because generated paved
model names quantize length/curve parameters for asset reuse; if a first
replacement still leaves a visible boundary seam, the next pass may absorb the
adjacent paved piece instead of serializing the defect.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from statistics import median
from typing import Callable, Sequence

from . import final_road_dedup_policy as _dedup
from . import generator as _generator
from . import playability as _p
from . import procedural_infrastructure as _pi
from . import road_inspector as _inspector
from .model import WorldObject

_MAXIMUM_REPAIR_PASSES = 3
_RAW_PROGRESS_PERCENT = 99
_INSTALLED = False
_ORIGINAL_FIT = None


@dataclass(frozen=True, slots=True)
class FinalRoadInspectorRepairReport:
    passes: int
    stock_regions: int
    generated_regions: int
    removed_objects: int
    added_objects: int
    unresolved_stock_regions: int
    unresolved_generated_regions: int


_LAST_REPAIR_REPORT: FinalRoadInspectorRepairReport | None = None


def _rotate(
    local: tuple[float, float],
    yaw_degrees: float,
) -> tuple[float, float]:
    angle = math.radians(yaw_degrees)
    return (
        local[0] * math.cos(angle) + local[1] * math.sin(angle),
        -local[0] * math.sin(angle) + local[1] * math.cos(angle),
    )


def _direction(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(heading_degrees)
    return math.sin(angle), math.cos(angle)


def _component_vertical_offset(
    object_ids: Sequence[int],
    objects_by_id: dict[int, object],
    elevations: Sequence[float],
    spec,
) -> float:
    """Estimate the road-surface offset already used by the replaced component."""
    offsets: list[float] = []
    for object_id in object_ids:
        obj = objects_by_id.get(int(object_id))
        if obj is None:
            continue
        terrain = _p._sample_elevation(
            elevations,
            spec.cells,
            spec.cell_size,
            float(obj.x),
            float(obj.z),
        )
        offset = float(obj.y) - terrain
        if _pi.is_generated_paved_road_model(str(obj.model_path)):
            offset += _pi.GENERATED_GRAVEL_VISUAL_TOP_METRES * math.cos(
                math.radians(float(getattr(obj, "pitch_degrees", 0.0)))
            )
        if math.isfinite(offset):
            offsets.append(offset)
    if not offsets:
        return float(getattr(_p, "_STOCK_ROAD_VERTICAL_OFFSET_METRES", 0.035))
    # Keep pathological terrain sampling from turning a seam repair into a
    # levitating road. Existing paved chains/junction approaches live near
    # 0.035/0.060 m, but allow a little headroom for legacy worlds.
    return max(-0.05, min(0.15, float(median(offsets))))


def _stock_curve_object(
    object_id: int,
    family: str,
    radius: int,
    start: tuple[float, float],
    heading: float,
    turn_sign: int,
    elevations: Sequence[float],
    spec,
    *,
    vertical_offset: float,
) -> tuple[WorldObject, tuple[float, float], float]:
    begin, end = _inspector._curve_points(family, float(radius))
    if turn_sign > 0:
        local_start, local_end = begin, end
        yaw = heading
        next_heading = heading + 10.0
    else:
        local_start, local_end = end, begin
        yaw = heading - 190.0
        next_heading = heading - 10.0

    sx, sz = _rotate(local_start, yaw)
    origin = start[0] - sx, start[1] - sz
    ex, ez = _rotate(local_end, yaw)
    finish = origin[0] + ex, origin[1] + ez
    height = _p._sample_elevation(
        elevations,
        spec.cells,
        spec.cell_size,
        origin[0],
        origin[1],
    ) + vertical_offset
    return (
        WorldObject(
            object_id,
            rf"o\road\{family}10 {radius}.p3d",
            origin[0],
            height,
            origin[1],
            yaw % 360.0,
            0.0,
        ),
        finish,
        next_heading % 360.0,
    )


def _stock_straight_object(
    object_id: int,
    family: str,
    nominal: int,
    start: tuple[float, float],
    end: tuple[float, float],
    elevations: Sequence[float],
    spec,
    *,
    vertical_offset: float,
) -> WorldObject:
    return _p._road_object_on_slope(
        object_id,
        rf"o\road\{family}{nominal}.p3d",
        start,
        end,
        elevations,
        spec,
        vertical_offset=vertical_offset,
    )


def _stock_repair_objects(
    plan: _inspector.PavedStockRepairPlan,
    next_id: int,
    elevations: Sequence[float],
    spec,
    *,
    vertical_offset: float,
) -> tuple[tuple[WorldObject, ...], int]:
    point = tuple(plan.start)
    heading = float(plan.start_heading_degrees)
    objects: list[WorldObject] = []

    for _index in range(plan.first_turns):
        obj, point, heading = _stock_curve_object(
            next_id,
            plan.family,
            plan.first_radius,
            point,
            heading,
            plan.turn_sign,
            elevations,
            spec,
            vertical_offset=vertical_offset,
        )
        objects.append(obj)
        next_id += 1

    direction = _direction(heading)
    for _index in range(plan.middle_units):
        end = (
            point[0] + direction[0] * 6.25,
            point[1] + direction[1] * 6.25,
        )
        objects.append(
            _stock_straight_object(
                next_id,
                plan.family,
                6,
                point,
                end,
                elevations,
                spec,
                vertical_offset=vertical_offset,
            )
        )
        next_id += 1
        point = end

    for _index in range(plan.counter_turns):
        obj, point, heading = _stock_curve_object(
            next_id,
            plan.family,
            plan.counter_radius,
            point,
            heading,
            -plan.turn_sign,
            elevations,
            spec,
            vertical_offset=vertical_offset,
        )
        objects.append(obj)
        next_id += 1

    objects.append(
        _stock_straight_object(
            next_id,
            plan.family,
            plan.merge_nominal,
            point,
            tuple(plan.end),
            elevations,
            spec,
            vertical_offset=vertical_offset,
        )
    )
    return tuple(objects), next_id + 1


def _generated_repair_object(
    plan: _inspector.PavedReplacementPlan,
    object_id: int,
    elevations: Sequence[float],
    spec,
    *,
    vertical_offset: float,
) -> WorldObject:
    return _p._road_object_on_slope(
        object_id,
        plan.model_path,
        tuple(plan.start),
        tuple(plan.end),
        elevations,
        spec,
        vertical_offset=vertical_offset,
    )


def _apply_inspection_plans(
    report,
    inspection: _inspector.InspectionResult,
    elevations: Sequence[float],
    spec,
):
    protected_count = max(
        0,
        min(int(getattr(report, "junction_cap_objects", 0)), len(report.objects)),
    )
    protected_ids = {
        int(obj.object_id) for obj in report.objects[:protected_count]
    }
    objects_by_id = {int(obj.object_id): obj for obj in report.objects}

    selected: list[tuple[int, str, object]] = []
    used_ids: set[int] = set()
    for kind, plans in (
        ("stock", inspection.paved_stock_repairs),
        ("generated", inspection.paved_replacements),
    ):
        for plan in plans:
            ids = {int(value) for value in plan.replace_object_ids}
            if not ids or ids & protected_ids or ids & used_ids:
                continue
            if not ids.issubset(objects_by_id):
                continue
            used_ids.update(ids)
            selected.append((min(ids), kind, plan))

    if not selected:
        return report, 0, 0, 0, 0

    remove_ids = set(used_ids)
    objects = [
        obj for obj in report.objects
        if int(obj.object_id) not in remove_ids
    ]
    next_id = max(
        (int(obj.object_id) for obj in report.objects),
        default=0,
    ) + 1
    stock_regions = 0
    generated_regions = 0
    added_objects = 0

    for _minimum_id, kind, raw_plan in sorted(
        selected, key=lambda value: (value[0], value[1])
    ):
        vertical_offset = _component_vertical_offset(
            raw_plan.replace_object_ids,
            objects_by_id,
            elevations,
            spec,
        )
        if kind == "stock":
            plan = raw_plan
            additions, next_id = _stock_repair_objects(
                plan,
                next_id,
                elevations,
                spec,
                vertical_offset=vertical_offset,
            )
            objects.extend(additions)
            added_objects += len(additions)
            stock_regions += 1
        else:
            plan = raw_plan
            objects.append(
                _generated_repair_object(
                    plan,
                    next_id,
                    elevations,
                    spec,
                    vertical_offset=vertical_offset,
                )
            )
            next_id += 1
            added_objects += 1
            generated_regions += 1

    return (
        replace(report, objects=tuple(objects)),
        stock_regions,
        generated_regions,
        len(remove_ids),
        added_objects,
    )


def repair_final_road_geometry(
    report,
    elevations: Sequence[float],
    spec,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
):
    """Return the final road report after bounded inspector-driven paved repairs."""
    global _LAST_REPAIR_REPORT

    if (
        not report.objects
        or not bool(getattr(spec, "procedural_paved_road_fallback", False))
    ):
        _LAST_REPAIR_REPORT = FinalRoadInspectorRepairReport(0, 0, 0, 0, 0, 0, 0)
        return report

    current = report
    total_stock = total_generated = 0
    total_removed = total_added = 0
    completed_passes = 0

    for pass_index in range(1, _MAXIMUM_REPAIR_PASSES + 1):
        inspection = _inspector.inspect_road_objects(
            current.objects,
            world_name=str(getattr(spec, "name", "world")),
            topology_checks=False,
        )
        if not inspection.paved_stock_repairs and not inspection.paved_replacements:
            break

        repaired, stock_regions, generated_regions, removed, added = (
            _apply_inspection_plans(current, inspection, elevations, spec)
        )
        if repaired is current:
            break

        completed_passes = pass_index
        total_stock += stock_regions
        total_generated += generated_regions
        total_removed += removed
        total_added += added
        current = _dedup.deduplicate_final_road_objects(repaired, spec)

        if progress_callback is not None:
            progress_callback(
                _RAW_PROGRESS_PERCENT,
                "Inspector road repair "
                f"pass {pass_index}/{_MAXIMUM_REPAIR_PASSES}: "
                f"{stock_regions:,} stock region(s), "
                f"{generated_regions:,} generated region(s), "
                f"{removed:,} old object(s) replaced by {added:,}",
            )

    unresolved = _inspector.inspect_road_objects(
        current.objects,
        world_name=str(getattr(spec, "name", "world")),
        topology_checks=False,
    )
    _LAST_REPAIR_REPORT = FinalRoadInspectorRepairReport(
        completed_passes,
        total_stock,
        total_generated,
        total_removed,
        total_added,
        len(unresolved.paved_stock_repairs),
        len(unresolved.paved_replacements),
    )
    if progress_callback is not None and (
        total_stock
        or total_generated
        or unresolved.paved_stock_repairs
        or unresolved.paved_replacements
    ):
        progress_callback(
            _RAW_PROGRESS_PERCENT,
            "Inspector road repair complete: "
            f"{total_stock:,} stock region(s), "
            f"{total_generated:,} generated region(s); "
            f"{len(unresolved.paved_stock_repairs):,} stock and "
            f"{len(unresolved.paved_replacements):,} generated region(s) unresolved",
        )
    return current


def _fit(
    dataset,
    projection,
    elevations,
    spec,
    *,
    starting_id: int = 1,
    progress_callback=None,
):
    report = _ORIGINAL_FIT(
        dataset,
        projection,
        elevations,
        spec,
        starting_id=starting_id,
        progress_callback=progress_callback,
    )
    return repair_final_road_geometry(
        report,
        elevations,
        spec,
        progress_callback=progress_callback,
    )


def install_final_road_inspector_repair_policy() -> None:
    """Install after dedupe/bridge cleanup and before final building clearance."""
    global _INSTALLED, _ORIGINAL_FIT
    if _INSTALLED:
        return
    _ORIGINAL_FIT = _p.fit_road_objects
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
