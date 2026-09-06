# SPDX-License-Identifier: GPL-3.0-or-later
"""Resolve final road/building conflicts by conservative right-of-way priority.

Step 2 moves buildings away from every final road surface when a bounded,
terrain-safe relocation exists.  This layer handles only the unresolved remainder.
Major paved roads and every junction remain authoritative.  Ordinary dirt/ces and
generated gravel pieces may instead be suppressed when preserving one mapped
building costs only a small, bounded amount of minor-road continuity.

Suppression is applied at final world-object assembly.  The fitted road report stays
immutable, which avoids renumbering or changing the road chain while non-road
objects are still being generated.  RVW4 serialization already renumbers final
objects, so the filtered assembly remains deterministic.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, replace
import math
import re
from typing import Callable, Sequence

from . import final_building_road_clearance_policy as _clearance
from . import generator as _generator
from . import osm as _osm

PointXZ = tuple[float, float]

_RAW_PROGRESS_PERCENT = 52
_BUILDING_BUCKET_METRES = 24.0
_MAXIMUM_SUPPRESSED_OBJECTS_PER_BUILDING = 2
_MAXIMUM_SUPPRESSED_ROAD_LENGTH_METRES = 25.75
_CACHE_REVISION = "final-road-building-clearance-v2-priority"

_MINOR_CES = re.compile(r"^ces(?P<nominal>25|12|6)\.p3d$", re.I)
_MINOR_GRAVEL = re.compile(
    r"^gravel(?P<nominal>25|12|6|3)(?:_[lr](?:05|10|15|20|30|45))?\.p3d$",
    re.I,
)

_INSTALLED = False
_ORIGINAL_RESOLVE = None
_ORIGINAL_ASSEMBLE = None
_ORIGINAL_LOAD_NONROAD_OBJECTS = None


@dataclass(frozen=True, slots=True)
class RoadBuildingPriorityState:
    road_objects_identity: int
    suppressed_object_ids: frozenset[int]
    preserved_buildings: int
    protected_rejections: int
    additional_road_checks: int


_PRIORITY_STATE: ContextVar[RoadBuildingPriorityState | None] = ContextVar(
    "cwr_road_building_priority_state", default=None
)


def _filename(path: str) -> str:
    return str(path).replace("/", "\\").rsplit("\\", 1)[-1].casefold()


def _plan_key(plan) -> tuple[object, ...]:
    return (
        getattr(plan, "osm_key", ""),
        int(getattr(plan, "geometry_index", 0)),
        getattr(plan, "geometry_kind", ""),
    )


def _bucket_range(minimum: float, maximum: float) -> range:
    return range(
        math.floor(minimum / _BUILDING_BUCKET_METRES),
        math.floor(maximum / _BUILDING_BUCKET_METRES) + 1,
    )


def _polygon_bounds(polygon: Sequence[PointXZ]) -> tuple[float, float, float, float]:
    return (
        min(point[0] for point in polygon),
        min(point[1] for point in polygon),
        max(point[0] for point in polygon),
        max(point[1] for point in polygon),
    )


class _AcceptedBuildingIndex:
    """Small append-only index used only for Step 3 restorations."""

    def __init__(self, polygons: Sequence[Sequence[PointXZ]]) -> None:
        self.polygons: list[tuple[PointXZ, ...]] = []
        self.buckets: dict[tuple[int, int], list[int]] = {}
        for polygon in polygons:
            if len(polygon) >= 3:
                self.add(polygon)

    def _keys(self, polygon: Sequence[PointXZ]) -> tuple[tuple[int, int], ...]:
        min_x, min_z, max_x, max_z = _polygon_bounds(polygon)
        return tuple(
            (bx, bz)
            for bz in _bucket_range(min_z, max_z)
            for bx in _bucket_range(min_x, max_x)
        )

    def add(self, polygon: Sequence[PointXZ]) -> None:
        stored = tuple(polygon)
        index = len(self.polygons)
        self.polygons.append(stored)
        for key in self._keys(stored):
            self.buckets.setdefault(key, []).append(index)

    def overlaps(self, polygon: Sequence[PointXZ]) -> bool:
        candidates: set[int] = set()
        for key in self._keys(polygon):
            candidates.update(self.buckets.get(key, ()))
        return any(
            _osm._polygons_intersect(polygon, self.polygons[index])
            for index in sorted(candidates)
        )


def _minor_object_length(obj, spec, protected_ids: frozenset[int]) -> float | None:
    object_id = int(getattr(obj, "object_id", -1))
    if object_id in protected_ids:
        return None
    filename = _filename(getattr(obj, "model_path", ""))
    match = _MINOR_CES.fullmatch(filename)
    if match is None:
        match = _MINOR_GRAVEL.fullmatch(filename)
    if match is None:
        return None
    nominal = int(match.group("nominal"))
    return float(spec.road_segment_length) * nominal / 25.0


def _suppression_budget_allows(
    object_ids: Sequence[int],
    object_map: dict[int, object],
    protected_ids: frozenset[int],
    spec,
) -> bool:
    unique = tuple(sorted(set(int(value) for value in object_ids)))
    if not unique or len(unique) > _MAXIMUM_SUPPRESSED_OBJECTS_PER_BUILDING:
        return False
    lengths: list[float] = []
    for object_id in unique:
        obj = object_map.get(object_id)
        if obj is None:
            return False
        length = _minor_object_length(obj, spec, protected_ids)
        if length is None:
            return False
        lengths.append(length)
    maximum = max(
        _MAXIMUM_SUPPRESSED_ROAD_LENGTH_METRES,
        float(spec.road_segment_length) + 0.75,
    )
    return sum(lengths) <= maximum + 1.0e-9


def _rebuild_plan_order(original_plans, base_plans, restored_by_key):
    base_by_key = {_plan_key(plan): plan for plan in base_plans}
    result = []
    for original in original_plans:
        key = _plan_key(original)
        if key in base_by_key:
            result.append(base_by_key[key])
        elif key in restored_by_key:
            result.append(restored_by_key[key])
    return tuple(result)


def resolve_road_building_priorities(
    plans,
    road_report,
    elevations,
    raster,
    spec,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
):
    """Run Step 2, then preserve buildings blocked only by minor road pieces."""
    plans = tuple(plans or ())
    road_objects = tuple(getattr(road_report, "objects", ()) or ())
    road_identity = id(getattr(road_report, "objects", ()))
    _PRIORITY_STATE.set(
        RoadBuildingPriorityState(road_identity, frozenset(), 0, 0, 0)
    )

    base_plans, base_report = _ORIGINAL_RESOLVE(
        plans,
        road_report,
        elevations,
        raster,
        spec,
        progress_callback=progress_callback,
    )
    rejected_total = max(0, int(getattr(base_report, "rejected", 0)))
    if not rejected_total or not plans or not road_objects:
        return base_plans, base_report

    base_keys = {_plan_key(plan) for plan in base_plans}
    rejected_plans = tuple(plan for plan in plans if _plan_key(plan) not in base_keys)
    if not rejected_plans:
        return base_plans, base_report

    primitives = _clearance._road_primitives(road_report, elevations, spec)
    road_index = _clearance._RoadPrimitiveIndex(primitives)
    object_map = {int(obj.object_id): obj for obj in road_objects}
    cap_count = max(
        0,
        min(int(getattr(road_report, "junction_cap_objects", 0)), len(road_objects)),
    )
    protected_ids = frozenset(
        int(obj.object_id) for obj in road_objects[:cap_count]
    )
    accepted_index = _AcceptedBuildingIndex(
        tuple(tuple(plan.support_polygon) for plan in base_plans)
    )

    restored_by_key: dict[tuple[object, ...], object] = {}
    suppressed_ids: set[int] = set()
    extra_checks = 0

    for plan in rejected_plans:
        polygon = tuple(getattr(plan, "support_polygon", ()))
        if len(polygon) < 3:
            continue
        conflicts, checked = _clearance._conflicts(polygon, road_index)
        extra_checks += checked
        conflict_ids = tuple(sorted({int(item.object_id) for item in conflicts}))
        if not _suppression_budget_allows(
            conflict_ids, object_map, protected_ids, spec
        ):
            continue
        # Step 2 removes a rejected building from its mutable footprint index.
        # A later building may therefore have moved into that vacancy. Never
        # restore the original footprint if doing so would create a new building
        # overlap just to solve a road overlap.
        if accepted_index.overlaps(polygon):
            continue
        restored_by_key[_plan_key(plan)] = plan
        accepted_index.add(polygon)
        suppressed_ids.update(conflict_ids)

    preserved = len(restored_by_key)
    protected_rejections = max(0, rejected_total - preserved)
    state = RoadBuildingPriorityState(
        road_identity,
        frozenset(suppressed_ids),
        preserved,
        protected_rejections,
        extra_checks,
    )
    _PRIORITY_STATE.set(state)

    if progress_callback is not None:
        progress_callback(
            _RAW_PROGRESS_PERCENT,
            "Resolving road/building priorities "
            f"({rejected_total:,} unresolved after relocation; {preserved:,} buildings kept; "
            f"{len(suppressed_ids):,} minor road pieces suppressed; "
            f"{protected_rejections:,} buildings rejected; {extra_checks:,} nearby road surfaces)",
        )

    if not restored_by_key:
        if extra_checks and hasattr(base_report, "nearby_road_checks"):
            base_report = replace(
                base_report,
                nearby_road_checks=int(base_report.nearby_road_checks) + extra_checks,
            )
        return base_plans, base_report

    final_plans = _rebuild_plan_order(plans, base_plans, restored_by_key)
    updates = {}
    if hasattr(base_report, "rejected"):
        updates["rejected"] = protected_rejections
    if hasattr(base_report, "nearby_road_checks"):
        updates["nearby_road_checks"] = int(base_report.nearby_road_checks) + extra_checks
    if updates:
        base_report = replace(base_report, **updates)
    return final_plans, base_report


def _filter_suppressed_roads(road_objects: Sequence[object]) -> tuple[object, ...]:
    state = _PRIORITY_STATE.get()
    if (
        state is None
        or not state.suppressed_object_ids
        or id(road_objects) != state.road_objects_identity
    ):
        return tuple(road_objects)
    return tuple(
        obj
        for obj in road_objects
        if int(getattr(obj, "object_id", -1)) not in state.suppressed_object_ids
    )


def install_road_building_priority_policy() -> None:
    """Install Step 3 after the final-road clearance and geometry policies."""
    global _INSTALLED, _ORIGINAL_RESOLVE, _ORIGINAL_ASSEMBLE
    global _ORIGINAL_LOAD_NONROAD_OBJECTS
    if _INSTALLED:
        return

    _ORIGINAL_RESOLVE = _clearance.resolve_final_building_road_conflicts
    _ORIGINAL_ASSEMBLE = _generator._assemble_world_objects
    _ORIGINAL_LOAD_NONROAD_OBJECTS = _generator._load_nonroad_objects

    # Cache entries written by Step 2 alone do not encode which minor road pieces
    # Step 3 suppresses. Force one new placement-cache generation, then recompute
    # the tiny spatial priority decision on later cache hits so final assembly has
    # the suppression set even when generate_world_objects itself was skipped.
    _clearance._CACHE_REVISION = _CACHE_REVISION

    def priority_assemble(
        road_objects,
        nonroads,
        semantic_objects,
        *,
        renumber: bool = True,
    ):
        filtered = _filter_suppressed_roads(road_objects)
        return _ORIGINAL_ASSEMBLE(
            filtered,
            nonroads,
            semantic_objects,
            renumber=renumber,
        )

    def priority_load_nonroad_objects(
        dataset,
        projection,
        raster,
        elevations,
        spec,
        **kwargs,
    ):
        result = _ORIGINAL_LOAD_NONROAD_OBJECTS(
            dataset,
            projection,
            raster,
            elevations,
            spec,
            **kwargs,
        )
        cache_hit = bool(result[2]) if len(result) > 2 else False
        if cache_hit:
            road_report = _clearance._road_context_matches(
                dataset, projection, elevations, spec
            )
            plans = kwargs.get("building_placement_plans")
            if road_report is not None and plans is not None:
                # Discard the plans here. The cached non-road payload was produced
                # with this same versioned policy; this call only restores the
                # ephemeral suppression state needed by final assembly.
                resolve_road_building_priorities(
                    plans,
                    road_report,
                    elevations,
                    raster,
                    spec,
                    progress_callback=kwargs.get("progress_callback"),
                )
        return result

    _clearance.resolve_final_building_road_conflicts = resolve_road_building_priorities
    _generator._assemble_world_objects = priority_assemble
    _generator._load_nonroad_objects = priority_load_nonroad_objects
    _INSTALLED = True
