# SPDX-License-Identifier: GPL-3.0-or-later
"""Final invariant audit for post-processed roads and procedural buildings.

Steps 1-3 deliberately mutate different layers of the road/building decision:
road slabs are deduplicated, buildings are relocated, then a very small set of
minor road pieces may be suppressed to preserve otherwise-unmovable buildings.
This final pass sees the exact surviving road set and exact surviving building
footprints and enforces the invariant that no ground-level road surface still
intersects a procedural building.

The audit is intentionally cheap. It reuses the same road primitive builder and
spatial index as the relocation pass. Any unexpected survivor is rejected rather
than silently serialized through a road. Suppressions that are no longer needed
after such a rejection are released again before final WRP assembly.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Callable

from . import final_building_road_clearance_policy as _clearance
from . import road_building_priority_policy as _priority

_RAW_PROGRESS_PERCENT = 52
_PROGRESS_BUCKET_PERCENT = 5
_INSTALLED = False
_ORIGINAL_RESOLVE = None


@dataclass(frozen=True, slots=True)
class FinalRoadBuildingAuditReport:
    buildings_checked: int
    final_road_objects: int
    final_road_primitives: int
    violations: int
    rejected: int
    released_suppressions: int
    nearby_road_checks: int


_FINAL_AUDIT: ContextVar[FinalRoadBuildingAuditReport | None] = ContextVar(
    "cwr_final_road_building_audit", default=None
)


def _suppression_state_for_report(road_report):
    state = _priority._PRIORITY_STATE.get()
    if state is None:
        return None
    objects = getattr(road_report, "objects", ())
    if id(objects) != state.road_objects_identity:
        return None
    return state


def _filtered_road_objects(road_report, suppressed_ids: frozenset[int]):
    return tuple(
        obj
        for obj in getattr(road_report, "objects", ())
        if int(getattr(obj, "object_id", -1)) not in suppressed_ids
    )


def _road_report_with_objects(objects):
    # _road_primitives only requires ``objects``. Keeping this tiny avoids
    # rebuilding or mutating the fitted RoadFitReport merely for the audit.
    return SimpleNamespace(objects=tuple(objects))


def _audit_plans_against_roads(
    plans,
    road_objects,
    elevations,
    spec,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
):
    plans = tuple(plans or ())
    primitives = _clearance._road_primitives(
        _road_report_with_objects(road_objects), elevations, spec
    )
    road_index = _clearance._RoadPrimitiveIndex(primitives) if primitives else None
    kept = []
    rejected = []
    checks = 0
    last_bucket = -1
    total = len(plans)

    def report(completed: int, *, force: bool = False) -> None:
        nonlocal last_bucket
        if progress_callback is None:
            return
        percent = 100 if total <= 0 else min(100, int(completed * 100 / total))
        bucket = percent // _PROGRESS_BUCKET_PERCENT
        if (
            not force
            and completed not in {1, total}
            and bucket <= last_bucket
        ):
            return
        last_bucket = bucket
        progress_callback(
            _RAW_PROGRESS_PERCENT,
            "Final road/building conflict audit "
            f"({completed:,}/{total:,}, {percent}%; {len(rejected):,} violations; "
            f"{checks:,} nearby road surfaces)",
        )

    report(0, force=True)
    for completed, plan in enumerate(plans, start=1):
        polygon = tuple(getattr(plan, "support_polygon", ()))
        if len(polygon) < 3 or road_index is None:
            kept.append(plan)
            report(completed)
            continue
        conflicts, tested = _clearance._conflicts(polygon, road_index)
        checks += tested
        if conflicts:
            rejected.append(plan)
        else:
            kept.append(plan)
        report(completed)
    report(total, force=True)
    return tuple(kept), tuple(rejected), len(primitives), checks


def _suppressed_ids_still_needed(
    plans,
    road_report,
    suppressed_ids: frozenset[int],
    elevations,
    spec,
):
    """Return suppressions that still intersect at least one surviving building."""
    if not suppressed_ids or not plans:
        return frozenset(), 0
    primitives = _clearance._road_primitives(road_report, elevations, spec)
    relevant = tuple(
        primitive
        for primitive in primitives
        if int(primitive.object_id) in suppressed_ids
    )
    if not relevant:
        return frozenset(), 0
    index = _clearance._RoadPrimitiveIndex(relevant)
    needed: set[int] = set()
    checks = 0
    for plan in plans:
        polygon = tuple(getattr(plan, "support_polygon", ()))
        if len(polygon) < 3:
            continue
        conflicts, tested = _clearance._conflicts(polygon, index)
        checks += tested
        needed.update(int(primitive.object_id) for primitive in conflicts)
    return frozenset(needed), checks


def audit_final_road_building_conflicts(
    plans,
    road_report,
    elevations,
    raster,
    spec,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
):
    """Run Steps 2-3, then enforce zero conflicts against final surviving roads."""
    plans, conflict_report = _ORIGINAL_RESOLVE(
        plans,
        road_report,
        elevations,
        raster,
        spec,
        progress_callback=progress_callback,
    )

    state = _suppression_state_for_report(road_report)
    suppressed = state.suppressed_object_ids if state is not None else frozenset()
    final_roads = _filtered_road_objects(road_report, suppressed)
    kept, violations, primitive_count, checks = _audit_plans_against_roads(
        plans,
        final_roads,
        elevations,
        spec,
        progress_callback=progress_callback,
    )

    released = 0
    suppression_checks = 0
    if state is not None and suppressed:
        needed, suppression_checks = _suppressed_ids_still_needed(
            kept,
            road_report,
            suppressed,
            elevations,
            spec,
        )
        released = len(suppressed - needed)
        if needed != suppressed or violations:
            # If an unexpected audit rejection removed the only building that
            # justified a gap, restore that minor road instead of leaving a hole.
            rejected_restored = 0
            if violations:
                full_primitives = _clearance._road_primitives(
                    road_report, elevations, spec
                )
                suppressed_primitives = tuple(
                    primitive
                    for primitive in full_primitives
                    if int(primitive.object_id) in suppressed
                )
                if suppressed_primitives:
                    suppressed_index = _clearance._RoadPrimitiveIndex(
                        suppressed_primitives
                    )
                    for plan in violations:
                        polygon = tuple(getattr(plan, "support_polygon", ()))
                        if len(polygon) < 3:
                            continue
                        conflicts, tested = _clearance._conflicts(
                            polygon, suppressed_index
                        )
                        suppression_checks += tested
                        rejected_restored += int(bool(conflicts))
            _priority._PRIORITY_STATE.set(replace(
                state,
                suppressed_object_ids=needed,
                preserved_buildings=max(
                    0, int(state.preserved_buildings) - rejected_restored
                ),
                protected_rejections=int(state.protected_rejections) + len(violations),
                additional_road_checks=(
                    int(state.additional_road_checks) + checks + suppression_checks
                ),
            ))

    total_checks = checks + suppression_checks
    rejected_count = len(violations)
    if rejected_count:
        updates = {}
        if hasattr(conflict_report, "rejected"):
            updates["rejected"] = int(conflict_report.rejected) + rejected_count
        if hasattr(conflict_report, "nearby_road_checks"):
            updates["nearby_road_checks"] = (
                int(conflict_report.nearby_road_checks) + total_checks
            )
        if updates:
            conflict_report = replace(conflict_report, **updates)
    elif total_checks and hasattr(conflict_report, "nearby_road_checks"):
        conflict_report = replace(
            conflict_report,
            nearby_road_checks=int(conflict_report.nearby_road_checks) + total_checks,
        )

    report = FinalRoadBuildingAuditReport(
        buildings_checked=len(plans),
        final_road_objects=len(final_roads),
        final_road_primitives=primitive_count,
        violations=rejected_count,
        rejected=rejected_count,
        released_suppressions=released,
        nearby_road_checks=total_checks,
    )
    _FINAL_AUDIT.set(report)

    if progress_callback is not None:
        progress_callback(
            _RAW_PROGRESS_PERCENT,
            "Final road/building conflict audit complete "
            f"({len(kept):,} buildings kept; {rejected_count:,} final violations rejected; "
            f"{released:,} unused road suppressions released; {total_checks:,} nearby road surfaces)",
        )
    return kept, conflict_report


def install_final_road_building_audit_policy() -> None:
    """Install the final invariant after Step 3 priority resolution is live."""
    global _INSTALLED, _ORIGINAL_RESOLVE
    if _INSTALLED:
        return
    _ORIGINAL_RESOLVE = _priority.resolve_road_building_priorities
    _priority.resolve_road_building_priorities = audit_final_road_building_conflicts
    # Step 2's generate-world wrapper resolves this symbol dynamically.
    _clearance.resolve_final_building_road_conflicts = audit_final_road_building_conflicts
    _INSTALLED = True
