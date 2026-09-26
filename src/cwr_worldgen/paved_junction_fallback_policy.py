# SPDX-License-Identifier: GPL-3.0-or-later
"""Use generated paved hubs when a stock paved junction cannot be fitted."""
from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

from . import generator as _generator
from . import paved_junction_policy as _paved
from . import playability as _p
from . import procedural_infrastructure as _pi
from . import road_quality_policy as _rq

_SUCCESS_DISTANCE_METRES = 0.75
_SUCCESS_BUCKET_METRES = _SUCCESS_DISTANCE_METRES
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
    the previous pass. Failed stock-paved plans return to the road-quality geometry,
    which now reserves the generated paved hub and its actual arm extent instead of
    the 32 m stock-approach envelope. Non-stock junctions remain unchanged.
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


def _success_bucket(point: tuple[float, float]) -> tuple[int, int]:
    return (
        math.floor(float(point[0]) / _SUCCESS_BUCKET_METRES),
        math.floor(float(point[1]) / _SUCCESS_BUCKET_METRES),
    )


def _successful_plan_keys(report, plans, progress_callback=None) -> frozenset[tuple[int, int]]:
    """Identify plans that actually emitted their stock T/X junction model.

    Junction models are heavily reused.  Grouping only by model path and then
    linearly scanning every emitted position for every plan therefore approaches
    O(plans * emitted junctions) on large worlds.  Keep the exact 0.75 m success
    predicate, but spatially bucket emitted positions by model so each plan checks
    only its own bucket and the eight neighbours.
    """

    wanted_models = {
        plan.model_path.replace("/", "\\").casefold()
        for plan in plans.values()
    }
    emitted_by_model: dict[
        str, dict[tuple[int, int], list[tuple[float, float]]]
    ] = {}
    objects = tuple(getattr(report, "objects", ()))
    total_objects = len(objects)
    progress_interval = max(1, total_objects // 50) if total_objects else 1

    if progress_callback is not None and total_objects:
        progress_callback(
            99,
            f"Indexing fitted paved junctions for fallback validation (0/{total_objects:,})",
        )

    for object_index, obj in enumerate(objects, start=1):
        model = obj.model_path.replace("/", "\\").casefold()
        if model in wanted_models:
            position = (float(obj.x), float(obj.z))
            buckets = emitted_by_model.setdefault(model, {})
            buckets.setdefault(_success_bucket(position), []).append(position)
        if (
            progress_callback is not None
            and (
                object_index == total_objects
                or object_index % progress_interval == 0
            )
        ):
            progress_callback(
                99,
                "Indexing fitted paved junctions for fallback validation "
                f"({object_index:,}/{total_objects:,})",
            )

    successful = []
    total_plans = len(plans)
    plan_interval = max(1, total_plans // 50) if total_plans else 1
    for plan_index, (key, plan) in enumerate(plans.items(), start=1):
        model = plan.model_path.replace("/", "\\").casefold()
        buckets = emitted_by_model.get(model, {})
        bx, bz = _success_bucket(plan.point)
        matched = False
        for nx in range(bx - 1, bx + 2):
            if matched:
                break
            for nz in range(bz - 1, bz + 2):
                if any(
                    math.dist(position, plan.point) <= _SUCCESS_DISTANCE_METRES
                    for position in buckets.get((nx, nz), ())
                ):
                    matched = True
                    break
        if matched:
            successful.append(key)
        if (
            progress_callback is not None
            and total_plans
            and (
                plan_index == total_plans
                or plan_index % plan_interval == 0
            )
        ):
            progress_callback(
                99,
                "Validating fitted paved junctions "
                f"({plan_index:,}/{total_plans:,}; {len(successful):,} successful)",
            )
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



def _generated_local_headings(
    length: float,
    signed_curve_degrees: float,
) -> tuple[float, float]:
    if abs(signed_curve_degrees) <= 1.0e-9:
        return 0.0, 0.0
    theta = math.radians(abs(signed_curve_degrees))
    radius = length / max(1.0e-9, 2.0 * math.sin(theta * 0.5))
    sagitta = math.copysign(
        radius * (1.0 - math.cos(theta * 0.5)),
        signed_curve_degrees,
    )
    control_x = sagitta * 2.0
    return (
        math.degrees(math.atan2(2.0 * control_x, length)),
        math.degrees(math.atan2(-2.0 * control_x, length)),
    )


def _generated_curve_choice(
    start: tuple[float, float],
    end: tuple[float, float],
    start_direction: tuple[float, float],
    end_direction: tuple[float, float],
) -> float:
    """Choose the generated curve bucket that best matches both approach tangents."""

    length = max(0.01, math.dist(start, end))
    chord = _paved._heading(_paved._unit((
        end[0] - start[0],
        end[1] - start[1],
    )))
    start_heading = _paved._heading(start_direction)
    end_heading = _paved._heading(end_direction)
    choices = (0.0,) + tuple(
        value
        for amount in _pi.GENERATED_PAVED_CURVE_BUCKETS
        for value in (float(amount), -float(amount))
    )
    best = None
    for value in choices:
        first_local, last_local = _generated_local_headings(length, value)
        first_error = _paved._angle(
            _paved._direction(chord + first_local),
            _paved._direction(start_heading),
        )
        last_error = _paved._angle(
            _paved._direction(chord + last_local),
            _paved._direction(end_heading),
        )
        score = (
            max(first_error, last_error),
            first_error + last_error,
            abs(value),
        )
        if best is None or score < best[0]:
            best = score, value
    return 0.0 if best is None else float(best[1])


def _stitch_generated_fallback_approaches(
    report,
    fallback_plans,
    elevations,
    spec,
):
    """Make failed-stock fallback hubs own their approach seams in the base fit.

    This runs inside paved-junction generation, before WRP serialization and
    before the road inspector.  The old fallback path inserted a generated hub
    but could leave the stock chain that had been fitted for the failed prefab.
    That produced both a short slab inside the hub and 0.3-1.3 m connector gaps.
    Rebuild only the terminal paved piece of each fallback arm from the exact
    generated connector to its existing outward endpoint.
    """

    if (
        not fallback_plans
        or not report.objects
        or not hasattr(report, "junction_cap_objects")
    ):
        return report

    objects_by_id = {int(obj.object_id): obj for obj in report.objects}
    remove_ids: set[int] = set()
    replacements: dict[int, object] = {}
    used_ids: set[int] = set()
    radius = float(_pi.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES)
    tolerance = 0.12

    for plan in fallback_plans.values():
        arms = tuple(getattr(plan, "arms", ()))
        if not arms:
            # Compatibility for low-level callers/tests that only carry
            # connector directions. Production paved plans always have arms.
            arms = tuple(
                SimpleNamespace(source_direction=connector.direction)
                for connector in getattr(plan, "connectors", ())
            )
        if not arms:
            continue
        candidates = []
        for obj in report.objects[report.junction_cap_objects:]:
            family = _paved._family(obj.model_path)
            if family is None or _paved._kind(family) != "paved":
                continue
            filename = obj.model_path.replace("/", "\\").rsplit("\\", 1)[-1].casefold()
            if filename.startswith("paved_j"):
                continue
            axis = _paved._object_axis(obj, spec)
            if axis is None:
                continue
            distances = tuple(math.dist(plan.point, point) for point in axis)
            radial_directions = tuple(
                _paved._unit((
                    point[0] - plan.point[0],
                    point[1] - plan.point[1],
                ))
                for point in axis
            )
            nearest_arm_error = min(
                _paved._angle(direction, arm.source_direction)
                for direction in radial_directions
                for arm in arms
            )
            if nearest_arm_error > 35.0:
                continue

            # A short slab with one endpoint deep inside the hub is stale stock
            # approach geometry. Remove it before selecting the real terminal
            # piece, otherwise it becomes the inspector's "extra approach".
            if (
                min(distances) < radius - 1.0
                and max(distances) <= radius + 3.0
            ):
                remove_ids.add(int(obj.object_id))
                continue
            candidates.append((obj, axis, distances))

        for arm in arms:
            direction = _paved._unit(arm.source_direction)
            connector = (
                plan.point[0] + direction[0] * radius,
                plan.point[1] + direction[1] * radius,
            )
            choices = []
            for obj, axis, distances in candidates:
                object_id = int(obj.object_id)
                if object_id in remove_ids or object_id in used_ids:
                    continue
                for near_index in (0, 1):
                    near = axis[near_index]
                    far = axis[1 - near_index]
                    near_radius = distances[near_index]
                    far_radius = distances[1 - near_index]
                    if far_radius <= near_radius + 0.10:
                        continue
                    radial = _paved._unit((
                        near[0] - plan.point[0],
                        near[1] - plan.point[1],
                    ))
                    radial_error = _paved._angle(radial, direction)
                    if radial_error > 30.0:
                        continue
                    gap = math.dist(connector, near)
                    if gap > 3.0:
                        continue
                    choices.append((
                        gap,
                        radial_error,
                        object_id,
                        obj,
                        near,
                        far,
                    ))

            if not choices:
                continue
            gap, _radial_error, object_id, old, near, far = min(
                choices,
                key=lambda value: (value[0], value[1], value[2]),
            )
            used_ids.add(object_id)
            if gap <= tolerance:
                continue

            family = _paved._family(old.model_path) or "sil"
            width = 2.0 * float(_paved._WIDTH.get(family, 4.55))
            length = math.dist(connector, far)
            if length <= 0.05:
                remove_ids.add(object_id)
                continue
            continuation = _paved._unit((
                far[0] - near[0],
                far[1] - near[1],
            ))
            curve = _generated_curve_choice(
                connector,
                far,
                direction,
                continuation,
            )
            model_path = _pi.paved_fallback_model_path(
                str(getattr(spec, "name", "world")),
                width,
                length,
                curve,
            )
            replacements[object_id] = _p._road_object_on_slope(
                object_id,
                model_path,
                connector,
                far,
                elevations,
                spec,
                vertical_offset=_p._STOCK_ROAD_VERTICAL_OFFSET_METRES,
            )

    if not remove_ids and not replacements:
        return report
    return replace(
        report,
        objects=tuple(
            replacements.get(int(obj.object_id), obj)
            for obj in report.objects
            if int(obj.object_id) not in remove_ids
        ),
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
        successful_keys = _successful_plan_keys(
            report, plans, progress_callback=progress_callback
        )
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
                f"{len(plans) - len(active):,} stock junction(s) use generated paved hubs"
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
            fallback_plans = {
                key: plans[key]
                for key in plans
                if key not in active
            }
            base_report = _stitch_generated_fallback_approaches(
                base_report,
                fallback_plans,
                elevations,
                spec,
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
            stable_keys = _successful_plan_keys(
                fitted, active, progress_callback=progress_callback
            )
            if len(stable_keys) == len(active):
                return fitted

            newly_failed = frozenset(set(active).difference(stable_keys))
            active = {
                key: active[key]
                for key in stable_keys
            }
            snapshot = next_snapshot
            affected = _affected_plan_keys(plans, active, newly_failed)

        # Geometry remained coupled after the bounded stabilization attempts.
        # Refit without stock plans so every paved junction uses its generated
        # angle-matched hub instead of leaving a grass gap or overlapping slabs.
        final_report = _base_refit(
            dataset,
            projection,
            elevations,
            spec,
            {},
            starting_id=starting_id,
            progress_callback=progress_callback,
        )
        return _stitch_generated_fallback_approaches(
            final_report,
            plans,
            elevations,
            spec,
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
