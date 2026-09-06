# SPDX-License-Identifier: GPL-3.0-or-later
"""Separate preferred road clearance from destructive physical-overlap decisions.

The final road/building pass prefers 0.75 m of breathing room around rendered road
surfaces. That margin is useful while searching for a nicer building placement, but
it is not part of the road mesh. A source-backed building must therefore never be
deleted, nor a minor road suppressed, solely because that optional margin cannot be
satisfied.

This policy keeps Step 2's preferred-clearance relocation search intact. Before the
search it identifies original building footprints with no physical road overlap and
keeps those footprints reserved if the relocation search later gives up. Step 3 and
Step 4 then run destructive right-of-way/safety decisions using the actual rendered
road half-width only.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import replace
from typing import Callable, Sequence

from . import final_building_road_clearance_policy as _clearance
from . import final_road_building_audit_policy as _audit
from . import road_building_priority_policy as _priority

PointXZ = tuple[float, float]

_PHYSICAL_OVERLAP_CLEARANCE_METRES = 0.0
_PREFERRED_CLEARANCE_METRES = float(_clearance._ROAD_CLEARANCE_METRES)
_RAW_PROGRESS_PERCENT = 52
_CACHE_REVISION = "final-road-building-clearance-v3-physical-overlap"

_CONFLICT_CLEARANCE: ContextVar[float] = ContextVar(
    "cwr_final_road_conflict_clearance_metres",
    default=_PREFERRED_CLEARANCE_METRES,
)
_PRESERVE_ORIGINAL_INDICES: ContextVar[frozenset[int]] = ContextVar(
    "cwr_preserve_clearance_only_building_indices",
    default=frozenset(),
)

_INSTALLED = False
_STEP2_RESOLVE = None
_STEP3_RESOLVE = None
_STEP4_RESOLVE = None
_ORIGINAL_BUILDING_INDEX_UPDATE = None


def _primitive_intersects_polygon_at_clearance(
    primitive,
    polygon: Sequence[PointXZ],
    clearance_metres: float,
) -> bool:
    """Test one road capsule against a building using an explicit extra margin."""
    if len(polygon) < 3:
        return False

    clearance = max(0.0, float(clearance_metres))
    limit = max(0.0, float(primitive.half_width)) + clearance
    min_x, min_z, max_x, max_z = _clearance._polygon_bounds(polygon)
    p_min_x = min(primitive.start[0], primitive.end[0]) - limit
    p_min_z = min(primitive.start[1], primitive.end[1]) - limit
    p_max_x = max(primitive.start[0], primitive.end[0]) + limit
    p_max_z = max(primitive.start[1], primitive.end[1]) + limit
    if (
        max_x < p_min_x
        or p_max_x < min_x
        or max_z < p_min_z
        or p_max_z < min_z
    ):
        return False

    if _clearance._point_in_polygon(primitive.start, polygon) or _clearance._point_in_polygon(
        primitive.end, polygon
    ):
        return True

    limit_sq = limit * limit
    previous = polygon[-1]
    for current in polygon:
        if _clearance._segment_distance_sq(
            primitive.start,
            primitive.end,
            previous,
            current,
        ) <= limit_sq:
            return True
        previous = current
    return False


def conflicts_at_clearance(
    polygon: Sequence[PointXZ],
    road_index,
    clearance_metres: float,
):
    """Return nearby road primitives intersecting ``polygon`` at one margin."""
    conflicts = []
    checked = 0
    for index in road_index.candidates(polygon):
        checked += 1
        primitive = road_index.primitives[index]
        if _primitive_intersects_polygon_at_clearance(
            primitive, polygon, clearance_metres
        ):
            conflicts.append(primitive)
    conflicts.sort(
        key=lambda primitive: (
            primitive.object_id,
            primitive.start,
            primitive.end,
        )
    )
    return tuple(conflicts), checked


def _contextual_conflicts(polygon: Sequence[PointXZ], road_index):
    return conflicts_at_clearance(
        polygon,
        road_index,
        _CONFLICT_CLEARANCE.get(),
    )


def _plan_key(plan) -> tuple[object, ...]:
    return (
        getattr(plan, "osm_key", ""),
        int(getattr(plan, "geometry_index", 0)),
        getattr(plan, "geometry_kind", ""),
    )


def _physically_clear_original_indices(plans, road_report, elevations, spec):
    """Return original plan indices whose actual footprint does not touch a road."""
    primitives = _clearance._road_primitives(road_report, elevations, spec)
    if not primitives:
        return frozenset(range(len(plans))), 0
    road_index = _clearance._RoadPrimitiveIndex(primitives)
    clear: set[int] = set()
    checks = 0
    for index, plan in enumerate(plans):
        polygon = tuple(getattr(plan, "support_polygon", ()))
        if len(polygon) < 3:
            clear.add(index)
            continue
        conflicts, tested = conflicts_at_clearance(
            polygon,
            road_index,
            _PHYSICAL_OVERLAP_CLEARANCE_METRES,
        )
        checks += tested
        if not conflicts:
            clear.add(index)
    return frozenset(clear), checks


def _guarded_building_index_update(self, index: int, polygon) -> None:
    """Keep clearance-only originals reserved if Step 2 gives up relocating them."""
    if polygon is None and int(index) in _PRESERVE_ORIGINAL_INDICES.get():
        return
    _ORIGINAL_BUILDING_INDEX_UPDATE(self, index, polygon)


def _preferred_step2_resolve(
    plans,
    road_report,
    elevations,
    raster,
    spec,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
):
    """Run Step 2 with preferred clearance but forbid clearance-only deletion."""
    plans = tuple(plans or ())
    physically_clear, physical_checks = _physically_clear_original_indices(
        plans, road_report, elevations, spec
    )

    mode_token = _CONFLICT_CLEARANCE.set(_PREFERRED_CLEARANCE_METRES)
    preserve_token = _PRESERVE_ORIGINAL_INDICES.set(physically_clear)
    try:
        resolved, report = _STEP2_RESOLVE(
            plans,
            road_report,
            elevations,
            raster,
            spec,
            progress_callback=progress_callback,
        )
    finally:
        _PRESERVE_ORIGINAL_INDICES.reset(preserve_token)
        _CONFLICT_CLEARANCE.reset(mode_token)

    resolved_by_key = {_plan_key(plan): plan for plan in resolved}
    restored = 0
    ordered = []
    for index, original in enumerate(plans):
        key = _plan_key(original)
        selected = resolved_by_key.get(key)
        if selected is not None:
            ordered.append(selected)
            continue
        if index in physically_clear:
            # Step 2 tried the preferred 0.75 m margin but could not find a safe
            # relocation. The source footprint does not touch the rendered road,
            # so keeping it is strictly better than deleting it.
            ordered.append(original)
            restored += 1

    updates = {}
    if hasattr(report, "nearby_road_checks"):
        updates["nearby_road_checks"] = int(report.nearby_road_checks) + physical_checks
    if restored and hasattr(report, "rejected"):
        updates["rejected"] = max(0, int(report.rejected) - restored)
    if updates:
        report = replace(report, **updates)

    if restored and progress_callback is not None:
        progress_callback(
            _RAW_PROGRESS_PERCENT,
            "Preserving buildings that miss only the preferred road-clearance margin "
            f"({restored:,} kept at mapped footprints; physical road overlap required for rejection)",
        )
    return tuple(ordered), report


def _physical_step3_resolve(*args, **kwargs):
    """Run Step 3 suppression decisions against actual road surfaces only."""
    token = _CONFLICT_CLEARANCE.set(_PHYSICAL_OVERLAP_CLEARANCE_METRES)
    try:
        return _STEP3_RESOLVE(*args, **kwargs)
    finally:
        _CONFLICT_CLEARANCE.reset(token)


def _physical_step4_resolve(*args, **kwargs):
    """Run the final invariant audit against actual road surfaces only."""
    token = _CONFLICT_CLEARANCE.set(_PHYSICAL_OVERLAP_CLEARANCE_METRES)
    try:
        return _STEP4_RESOLVE(*args, **kwargs)
    finally:
        _CONFLICT_CLEARANCE.reset(token)


def install_physical_road_overlap_policy() -> None:
    """Make physical overlap, rather than preference clearance, destructive."""
    global _INSTALLED, _STEP2_RESOLVE, _STEP3_RESOLVE, _STEP4_RESOLVE
    global _ORIGINAL_BUILDING_INDEX_UPDATE
    if _INSTALLED:
        return

    # Step 4 has already installed by the time this policy runs. Capture each
    # layer separately so Step 2 can keep its relocation margin while Steps 3/4
    # inherit physical-only conflict semantics.
    _STEP2_RESOLVE = _priority._ORIGINAL_RESOLVE
    _STEP3_RESOLVE = _audit._ORIGINAL_RESOLVE
    _STEP4_RESOLVE = _clearance.resolve_final_building_road_conflicts
    _ORIGINAL_BUILDING_INDEX_UPDATE = _clearance._BuildingFootprintIndex.update

    _clearance._conflicts = _contextual_conflicts
    _clearance._BuildingFootprintIndex.update = _guarded_building_index_update

    # Step 3 calls this pointer dynamically for its relocation phase. The wrapper
    # temporarily switches back to preferred-clearance mode, then returns to the
    # physical mode used by Step 3 itself.
    _priority._ORIGINAL_RESOLVE = _preferred_step2_resolve
    _audit._ORIGINAL_RESOLVE = _physical_step3_resolve

    # Step 4 is the public live resolver after installation. Keep the same target
    # in the priority module because its cache-hit restoration closure resolves
    # that module-global symbol dynamically.
    _clearance.resolve_final_building_road_conflicts = _physical_step4_resolve
    _priority.resolve_road_building_priorities = _physical_step4_resolve
    _audit.audit_final_road_building_conflicts = _physical_step4_resolve

    # Cached non-road placement from the earlier 0.75 m destructive rule may be
    # missing source buildings. Force one fresh placement cache generation.
    _clearance._CACHE_REVISION = _CACHE_REVISION
    _INSTALLED = True
