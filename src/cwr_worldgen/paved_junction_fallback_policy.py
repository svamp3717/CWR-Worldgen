# SPDX-License-Identifier: GPL-3.0-or-later
"""Restore ordinary road geometry when a stock paved junction cannot be fitted."""
from __future__ import annotations

from dataclasses import replace
import math

from . import generator as _generator
from . import paved_junction_policy as _paved
from . import playability as _p
from . import road_quality_policy as _rq

_SUCCESS_DISTANCE_METRES = 0.75
_MAX_STABILIZATION_REFITS = 2
# A failed stock junction changes trimming around its 32 m reserve.  Include the
# target-search reach, clear zone and one full stock slab when proactively marking
# neighbouring plans dirty.  The parallel planner also compares complete target
# fingerprints, so long road-chain changes outside this radius are still caught.
_FALLBACK_DEPENDENCY_RADIUS_METRES = (
    float(_paved._APPROACH_RESERVE)
    + 55.0
    + float(_paved._CLEAR_RADIUS)
    + max(float(value) for value in _paved._STRAIGHTS.values())
)
_ORIGINAL_FIT = None
_INSTALLED = False


def _junction_geometry(dataset, projection, spec):
    """Reserve stock-junction space only for plans still eligible for stock fitting.

    ``paved_junction_policy`` normally exposes only stock-paved junction geometry.
    During a fallback refit, ``_PLANS`` contains the subset that actually fitted in
    the previous pass. Failed stock-paved plans are restored to the ordinary road
    quality geometry so their arms are no longer trimmed back by the 32 m stock
    approach reserve. Non-stock junctions remain excluded exactly as before.
    """

    base = dict(_paved._ORIGINAL_GEOMETRY(dataset, projection, spec))
    all_plans = _paved._plans(dataset, projection, spec)
    active_plans = _paved._PLANS.get()
    if active_plans is None:
        active_plans = all_plans
        fallback_keys: set[tuple[int, int]] = set()
    else:
        fallback_keys = set(all_plans).difference(active_plans)

    result = {
        key: base[key]
        for key in fallback_keys
        if key in base
    }
    for key, plan in active_plans.items():
        if key not in base:
            continue
        result[key] = replace(
            base[key],
            axis=plan.axis,
            half_length=_paved._APPROACH_RESERVE,
            half_width=_paved._APPROACH_RESERVE,
            directions=tuple(connector.direction for connector in plan.connectors),
        )
    return result


def _successful_plan_keys(report, plans) -> frozenset[tuple[int, int]]:
    """Identify plans that actually emitted their stock T/X junction model."""

    wanted_models = {
        plan.model_path.replace("/", "\\").casefold()
        for plan in plans.values()
    }
    emitted_by_model: dict[str, list[tuple[float, float]]] = {}
    for obj in getattr(report, "objects", ()):
        model = obj.model_path.replace("/", "\\").casefold()
        if model not in wanted_models:
            continue
        emitted_by_model.setdefault(model, []).append((float(obj.x), float(obj.z)))

    successful = []
    for key, plan in plans.items():
        model = plan.model_path.replace("/", "\\").casefold()
        if any(
            math.dist(position, plan.point) <= _SUCCESS_DISTANCE_METRES
            for position in emitted_by_model.get(model, ())
        ):
            successful.append(key)
    return frozenset(successful)


def _base_refit(
    dataset,
    projection,
    elevations,
    spec,
    active_plans,
    *,
    starting_id: int,
    progress_callback,
):
    token = _paved._PLANS.set(dict(active_plans))
    try:
        return _paved._ORIGINAL_FIT(
            dataset,
            projection,
            elevations,
            spec,
            starting_id=starting_id,
            progress_callback=progress_callback,
        )
    finally:
        _paved._PLANS.reset(token)


def _planning_runtime():
    """Return the late parallel planner only when its indexed base is live.

    Keeping this conditional matters for low-level tests and custom callers that
    deliberately replace ``_apply_plans``.  Production package startup installs
    the indexed performance policy before road fitting begins.
    """
    from . import paved_junction_performance_policy as performance
    from . import paved_junction_parallel_planning_policy as parallel

    current = _paved._apply_plans
    if current is parallel.apply_paved_junctions_parallel:
        return parallel
    if current is performance.apply_paved_junctions_fast:
        parallel.install_paved_junction_parallel_planning_policy()
        return parallel
    return None


def _affected_plan_keys(
    plans,
    active_keys,
    changed_keys,
    *,
    radius: float = _FALLBACK_DEPENDENCY_RADIUS_METRES,
) -> frozenset[tuple[int, int]]:
    """Return active stock plans near junctions that just became ordinary."""
    changed_points = tuple(
        plans[key].point
        for key in changed_keys
        if key in plans
    )
    if not changed_points:
        return frozenset()
    result = []
    for key in active_keys:
        plan = plans.get(key)
        if plan is None:
            continue
        if any(math.dist(plan.point, point) <= radius for point in changed_points):
            result.append(key)
    return frozenset(result)


def _fit(
    dataset,
    projection,
    elevations,
    spec,
    *,
    starting_id: int = 1,
    progress_callback=None,
):
    if not bool(getattr(spec, "stock_road_piece_fitting", False)):
        return _ORIGINAL_FIT(
            dataset,
            projection,
            elevations,
            spec,
            starting_id=starting_id,
            progress_callback=progress_callback,
        )

    plans = _paved._plans(dataset, projection, spec)
    if not plans:
        return _ORIGINAL_FIT(
            dataset,
            projection,
            elevations,
            spec,
            starting_id=starting_id,
            progress_callback=progress_callback,
        )

    planning = _planning_runtime()
    session = None
    session_token = None
    if planning is not None:
        session, session_token = planning.begin_planning_session(
            "Planning paved-junction approaches, initial pass"
        )

    try:
        # First preserve the normal paved-junction path. Most maps never need the
        # fallback and therefore pay only a cheap scan for emitted junction models.
        report = _ORIGINAL_FIT(
            dataset,
            projection,
            elevations,
            spec,
            starting_id=starting_id,
            progress_callback=progress_callback,
        )
        successful_keys = _successful_plan_keys(report, plans)
        if len(successful_keys) == len(plans):
            return report

        active = {
            key: plans[key]
            for key in successful_keys
        }
        changed_keys = frozenset(set(plans).difference(successful_keys))
        snapshot = session.latest if session is not None else None
        affected = _affected_plan_keys(plans, active, changed_keys)
        if progress_callback is not None:
            suffix = (
                f"; {len(affected):,} nearby paved plan(s) marked for fresh search"
                if planning is not None and active
                else ""
            )
            progress_callback(
                99,
                "Refitting paved junction fallbacks: "
                f"{len(plans) - len(active):,} stock junction(s) use ordinary connected roads"
                + suffix,
            )

        # Refit from the road-quality layer with failed plans restored to ordinary
        # junction geometry.  The expensive paved approach search is not repeated
        # wholesale: unchanged target fingerprints reuse their previous solution,
        # while nearby/changed candidate sets are solved again.  A neighbouring
        # fallback can still invalidate another stock approach, so retain the same
        # bounded two-pass stabilization safety net.
        for attempt in range(_MAX_STABILIZATION_REFITS):
            base_report = _base_refit(
                dataset,
                projection,
                elevations,
                spec,
                active,
                starting_id=starting_id,
                progress_callback=progress_callback,
            )
            if not active:
                return base_report

            if planning is not None and session is not None:
                planning.configure_planning_session(
                    session,
                    reuse=snapshot,
                    replan_keys=affected,
                    label=(
                        "Replanning paved junctions affected by fallbacks, "
                        f"stabilization {attempt + 1}/{_MAX_STABILIZATION_REFITS}"
                    ),
                )

            fitted = _paved._apply_plans(base_report, active, elevations, spec)
            next_snapshot = session.latest if session is not None else None
            stable_keys = _successful_plan_keys(fitted, active)
            if len(stable_keys) == len(active):
                return fitted

            newly_failed = frozenset(set(active).difference(stable_keys))
            active = {
                key: active[key]
                for key in stable_keys
            }
            snapshot = next_snapshot
            affected = _affected_plan_keys(plans, active, newly_failed)

        # Geometry remained coupled after the bounded stabilization attempts. An
        # ordinary junction is visually less fancy but, unlike a 30 m grass gap,
        # remains a road.
        return _base_refit(
            dataset,
            projection,
            elevations,
            spec,
            {},
            starting_id=starting_id,
            progress_callback=progress_callback,
        )
    finally:
        if planning is not None and session_token is not None:
            planning.end_planning_session(session_token)


def install_paved_junction_fallback_policy() -> None:
    global _ORIGINAL_FIT, _INSTALLED
    if _INSTALLED:
        return

    _ORIGINAL_FIT = _p.fit_road_objects

    # Replace the geometry hook before gravel-junction policy captures it as its
    # underlying implementation. This preserves the existing wrapper order.
    _paved._junction_geometry = _junction_geometry
    _rq._junction_geometry = _junction_geometry
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
