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

from . import bridge_underlay_cleanup_policy as _bridge_underlay
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
    middle_length = float(_inspector._STOCK_REPAIR_STRAIGHT_LENGTHS[6])
    for _index in range(plan.middle_units):
        end = (
            point[0] + direction[0] * middle_length,
            point[1] + direction[1] * middle_length,
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
    *,
    protected_object_ids: Sequence[int] = (),
):
    protected_count = max(
        0,
        min(int(getattr(report, "junction_cap_objects", 0)), len(report.objects)),
    )
    protected_ids = {
        int(obj.object_id) for obj in report.objects[:protected_count]
    }
    protected_ids.update(int(value) for value in protected_object_ids)
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


def _bridge_terminal_underlay_ids(
    report,
    dataset,
    projection,
    elevations: Sequence[float],
    spec,
) -> frozenset[int]:
    """Return paved approach-mask IDs deliberately retained by bridge cleanup."""
    spans = _bridge_underlay._bridge_spans(
        dataset,
        projection,
        elevations,
        spec,
    )
    if not spans:
        return frozenset()

    protected: set[int] = set()
    for obj in report.objects:
        for span in spans:
            if _bridge_underlay._road_matches_terminal_underlay(
                obj,
                span.points,
                span.road_width,
            ):
                protected.add(int(obj.object_id))
                break
            if (
                span.source_points
                and span.source_end_measure > span.source_start_measure
                and _bridge_underlay._road_matches_terminal_underlay(
                    obj,
                    span.source_points,
                    span.road_width,
                    span.source_start_measure,
                    span.source_end_measure,
                )
            ):
                protected.add(int(obj.object_id))
                break
    return frozenset(protected)


def repair_final_road_geometry(
    report,
    elevations: Sequence[float],
    spec,
    *,
    protected_object_ids: Sequence[int] = (),
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
            _apply_inspection_plans(
                current,
                inspection,
                elevations,
                spec,
                protected_object_ids=protected_object_ids,
            )
        )
        if repaired is current:
            break

        completed_passes = pass_index
        total_stock += stock_regions
        total_generated += generated_regions
        total_removed += removed
        total_added += added
        current = repaired

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



def _at_grade_paved(tags) -> bool:
    if _p.road_is_dirt(tags):
        return False
    if _p._road_is_explicit_bridge(tags):
        return False
    tunnel = str(tags.get("tunnel", "")).strip().casefold()
    if tunnel not in {"", "no", "false", "0", "none"}:
        return False
    try:
        layer = float(str(tags.get("layer", "0")).replace(",", "."))
    except ValueError:
        layer = 0.0
    return abs(layer) <= 1.0e-9


def _expected_paved_junctions(dataset, projection, spec):
    """Return all-paved three/four-way nodes from normalized road topology."""

    incidents: dict[
        tuple[int, int],
        list[tuple[tuple[float, float], bool, str, str, str]],
    ] = {}
    positions: dict[tuple[int, int], tuple[float, float]] = {}
    projected = _p.projected_road_polylines(dataset, projection)
    for feature, raw_points in zip(dataset.roads, projected):
        if not _p.road_is_supported(
            feature.tags,
            include_minor=spec.include_minor_roads,
        ):
            continue
        points = tuple(_p._clean_road_points(raw_points))
        if len(points) < 2:
            continue
        dirt = _p.road_is_dirt(feature.tags)
        model = _p.road_model_for_tags(spec, feature.tags)
        at_grade = _at_grade_paved(feature.tags) if not dirt else True
        for index, (start, end) in enumerate(zip(points, points[1:])):
            if math.dist(start, end) <= 0.05:
                continue
            forward = _p._normalised_direction(start, end)
            reverse = (-forward[0], -forward[1])
            segment = f"{feature.osm_key}/{index:06d}"
            for point, direction in ((start, forward), (end, reverse)):
                key = _p._road_node_key(point)
                # Mark bridge/tunnel paved incidents as dirt-like for the sole
                # purpose of excluding the node from at-grade hub synthesis.
                synthetic_dirt = dirt or not at_grade
                incidents.setdefault(key, []).append(
                    (
                        direction,
                        synthetic_dirt,
                        model,
                        segment,
                        feature.osm_key,
                    )
                )
                positions.setdefault(key, point)

    result = []
    for key, raw_values in incidents.items():
        values = _p._unique_incidents(raw_values)
        if not (3 <= len(values) <= 4):
            continue
        if any(value[1] for value in values):
            continue
        directions = tuple(value[0] for value in values)
        models = tuple(value[2] for value in values)
        result.append((positions[key], directions, models))
    return tuple(result)



def _mixed_dirt_paved_points(dataset, projection, spec) -> tuple[tuple[float, float], ...]:
    """Return OSM nodes where plain dirt meets an at-grade paved road.

    These are deliberate underlay joins, never paved junction candidates. The
    late road-inspector guard must not reinterpret the paved slabs around them as
    a missing T/X hub after the dirt terminal piece has been placed underneath.
    """

    kinds: dict[tuple[int, int], set[str]] = {}
    positions: dict[tuple[int, int], tuple[float, float]] = {}
    projected = _p.projected_road_polylines(dataset, projection)
    for feature, raw_points in zip(dataset.roads, projected):
        if not _p.road_is_supported(
            feature.tags,
            include_minor=spec.include_minor_roads,
        ):
            continue
        if _p._is_plain_dirt_tags(feature.tags):
            kind = "dirt"
        elif _at_grade_paved(feature.tags):
            kind = "paved"
        else:
            continue
        points = tuple(_p._clean_road_points(raw_points))
        if len(points) < 2:
            continue
        for start, end in zip(points, points[1:]):
            if math.dist(start, end) <= 0.05:
                continue
            for point in (start, end):
                key = _p._road_node_key(point)
                kinds.setdefault(key, set()).add(kind)
                positions.setdefault(key, point)

    return tuple(
        positions[key]
        for key, values in kinds.items()
        if {"dirt", "paved"}.issubset(values)
    )


def _unit_from(
    centre: tuple[float, float],
    point: tuple[float, float],
) -> tuple[float, float] | None:
    dx = float(point[0]) - float(centre[0])
    dz = float(point[1]) - float(centre[1])
    length = math.hypot(dx, dz)
    if length <= 0.20:
        return None
    return dx / length, dz / length


def _unique_directions(
    directions: Sequence[tuple[float, float]],
    *,
    tolerance_degrees: float = 15.0,
) -> tuple[tuple[float, float], ...]:
    cosine = math.cos(math.radians(tolerance_degrees))
    result: list[tuple[float, float]] = []
    for direction in directions:
        length = math.hypot(*direction)
        if length <= 1.0e-9:
            continue
        value = direction[0] / length, direction[1] / length
        if any(
            value[0] * existing[0] + value[1] * existing[1] >= cosine
            for existing in result
        ):
            continue
        result.append(value)
    return tuple(result)


def _issue_paved_junction_candidate(issue, road_by_id):
    if issue.category not in {
        "paved_crossing_without_junction",
        "paved_t_without_junction",
        "intersection_without_junction",
    }:
        return None

    roads = tuple(
        road_by_id.get(int(object_id))
        for object_id in issue.object_ids
    )
    if not roads or any(road is None or road.road_type != "paved" for road in roads):
        return None

    centre = float(issue.x), float(issue.z)
    directions: list[tuple[float, float]] = []
    for road in roads:
        endpoints = tuple(road.endpoints)
        if len(endpoints) < 2:
            continue
        first = _unit_from(centre, endpoints[0].point)
        second = _unit_from(centre, endpoints[-1].point)
        first_distance = math.dist(centre, endpoints[0].point)
        second_distance = math.dist(centre, endpoints[-1].point)
        if first is not None and second is not None:
            dot = first[0] * second[0] + first[1] * second[1]
            # The crossing lies inside this road piece: both directions are
            # real arms. If both endpoints sit on the same side, it is a T stem
            # and only the farther outward direction belongs to the junction.
            if dot <= -0.35:
                directions.extend((first, second))
            elif first_distance >= second_distance:
                directions.append(first)
            else:
                directions.append(second)
        elif first is not None:
            directions.append(first)
        elif second is not None:
            directions.append(second)

    unique = _unique_directions(directions)
    if len(unique) not in {3, 4}:
        return None
    models = tuple(road.model_path for road in roads if road is not None)
    return centre, unique, models


def ensure_final_paved_junction_hubs(
    report,
    dataset,
    projection,
    elevations: Sequence[float],
    spec,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
):
    """Guarantee one visible generated hub at every surviving paved T/X node.

    This is deliberately a final visual guard. Earlier stock fitting may reserve
    a junction and later lose its cap during fallback/cleanup, while separately
    normalized paved slabs can also intersect without a shared OSM node. In both
    cases a raised generated hub is safer than leaving grass or raw crossing
    rectangles in the final WRP.
    """

    if (
        not report.objects
        or not bool(getattr(spec, "procedural_paved_road_fallback", False))
    ):
        return report

    inspection = _inspector.inspect_road_objects(
        report.objects,
        world_name=str(getattr(spec, "name", "world")),
        topology_checks=True,
    )
    existing_centres = [
        (road.x, road.z)
        for road in inspection.road_objects
        if road.kind.startswith("junction_")
    ]

    candidates = list(_expected_paved_junctions(dataset, projection, spec))
    mixed_dirt_paved_points = _mixed_dirt_paved_points(
        dataset,
        projection,
        spec,
    )
    road_by_id = {road.object_id: road for road in inspection.road_objects}
    for issue in inspection.issues:
        candidate = _issue_paved_junction_candidate(issue, road_by_id)
        if candidate is None:
            continue
        point = candidate[0]
        if any(
            math.dist(point, mixed_point) <= 2.0
            for mixed_point in mixed_dirt_paved_points
        ):
            continue
        if any(math.dist(point, current[0]) <= 2.0 for current in candidates):
            continue
        candidates.append(candidate)

    if not candidates:
        return report

    objects = list(report.objects)
    next_id = max((int(obj.object_id) for obj in objects), default=0) + 1
    added = 0
    for point, directions, models in candidates:
        if any(
            math.dist(point, mixed_point) <= 2.0
            for mixed_point in mixed_dirt_paved_points
        ):
            continue
        if any(
            math.dist(point, centre)
            <= _pi.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES + 1.0
            for centre in existing_centres
        ):
            continue
        try:
            signature, axis = _pi.paved_junction_signature_for_directions(
                directions
            )
        except ValueError:
            continue
        width = _pi.paved_junction_width_for_models(models)
        model_path = _pi.paved_junction_model_path(
            str(getattr(spec, "name", "world")),
            width,
            signature,
        )
        extent = _pi.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES
        start = (
            point[0] - axis[0] * extent,
            point[1] - axis[1] * extent,
        )
        end = (
            point[0] + axis[0] * extent,
            point[1] + axis[1] * extent,
        )
        objects.append(
            _p._road_object_on_slope(
                next_id,
                model_path,
                start,
                end,
                elevations,
                spec,
                vertical_offset=_p._junction_cap_vertical_offset(model_path),
            )
        )
        next_id += 1
        added += 1
        existing_centres.append(point)

    if not added:
        return report
    if progress_callback is not None:
        progress_callback(
            _RAW_PROGRESS_PERCENT,
            f"Final paved-junction guard added {added:,} missing generated hub(s)",
        )
    return replace(
        report,
        objects=tuple(objects),
        chain_count=report.chain_count + added,
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
    report = _ORIGINAL_FIT(
        dataset,
        projection,
        elevations,
        spec,
        starting_id=starting_id,
        progress_callback=progress_callback,
    )
    protected = _bridge_terminal_underlay_ids(
        report,
        dataset,
        projection,
        elevations,
        spec,
    )
    repaired = repair_final_road_geometry(
        report,
        elevations,
        spec,
        protected_object_ids=protected,
        progress_callback=progress_callback,
    )
    return ensure_final_paved_junction_hubs(
        repaired,
        dataset,
        projection,
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
