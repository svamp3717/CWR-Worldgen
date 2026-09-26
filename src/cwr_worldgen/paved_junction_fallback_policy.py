# SPDX-License-Identifier: GPL-3.0-or-later
"""Restore ordinary road geometry when a stock paved junction cannot be fitted."""
from __future__ import annotations

from dataclasses import replace
import math
import re

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
    the previous pass. Failed stock-paved plans are restored to ordinary road
    quality geometry. Generated paved T hubs never need the 32 m prefab approach
    reserve at all, so they use a compact seam envelope matching their real arm
    extent. Non-stock junctions remain excluded exactly as before.
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
        directions = tuple(
            connector.direction for connector in plan.connectors
        )
        if _pi.is_generated_paved_junction_model(plan.model_path):
            # Generated hubs are already present in the base fitter and own
            # their seam at the real 6.25 m arm radius. Do not apply the old
            # ~32 m stock-junction approach reserve here: doing so can strand
            # the nearest road slab tens of metres from a perfectly valid hub,
            # after which hub-presence validation incorrectly calls the plan
            # successful. A compact square quality envelope keeps every arm
            # within the 3 m seam-stitch search even at a 45-degree heading.
            seam_extent = (
                float(_pi.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES)
                + float(_pi.GENERATED_PAVED_JUNCTION_APPROACH_CLEARANCE_METRES)
                + float(_rq._JUNCTION_OVERLAP)
            )
            result[key] = replace(
                base[key],
                axis=plan.axis,
                half_length=seam_extent,
                half_width=seam_extent,
                directions=directions,
            )
        else:
            result[key] = replace(
                base[key],
                axis=plan.axis,
                half_length=_paved._APPROACH_RESERVE,
                half_width=_paved._APPROACH_RESERVE,
                directions=directions,
            )
    return result


def _success_bucket(point: tuple[float, float]) -> tuple[int, int]:
    return (
        math.floor(float(point[0]) / _SUCCESS_BUCKET_METRES),
        math.floor(float(point[1]) / _SUCCESS_BUCKET_METRES),
    )


def _successful_plan_keys(
    report,
    plans,
    spec=None,
    progress_callback=None,
) -> frozenset[tuple[int, int]]:
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
    generated_connector_buckets: dict[
        tuple[int, int],
        list[tuple[tuple[float, float], tuple[float, float]]],
    ] = {}
    generated_connector_plans = (
        spec is not None
        and any(
            _pi.is_generated_paved_junction_model(plan.model_path)
            for plan in plans.values()
        )
    )
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
            generated_connector_plans
            and object_index > int(getattr(report, "junction_cap_objects", 0))
        ):
            axis = _generated_paved_axis(obj, spec)
            if axis is not None:
                for endpoint_index in (0, 1):
                    endpoint = axis[endpoint_index]
                    other = axis[1 - endpoint_index]
                    continuation = _paved._unit((
                        other[0] - endpoint[0],
                        other[1] - endpoint[1],
                    ))
                    generated_connector_buckets.setdefault(
                        _success_bucket(endpoint), []
                    ).append((endpoint, continuation))
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
        if (
            matched
            and generated_connector_plans
            and _pi.is_generated_paved_junction_model(plan.model_path)
        ):
            # A generated hub sitting at the node is not sufficient. Every arm
            # must actually meet an emitted paved approach near the connector;
            # otherwise the old 32 m reserve can masquerade as a successful
            # junction while leaving a large visible gap.
            for connector in plan.connectors:
                cbx, cbz = _success_bucket(connector.point)
                connected = False
                for nx in range(cbx - 1, cbx + 2):
                    if connected:
                        break
                    for nz in range(cbz - 1, cbz + 2):
                        for endpoint, continuation in generated_connector_buckets.get(
                            (nx, nz), ()
                        ):
                            if (
                                math.dist(endpoint, connector.point)
                                <= _SUCCESS_DISTANCE_METRES
                                and _paved._angle(
                                    continuation, connector.direction
                                ) <= 30.0
                            ):
                                connected = True
                                break
                        if connected:
                            break
                if not connected:
                    matched = False
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
    """Choose a generated paved curve matching both seam tangents."""

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


def _generated_paved_axis(obj, spec):
    """Return endpoint geometry for stock or generated paved approach pieces."""

    if _pi.is_generated_paved_road_model(obj.model_path):
        filename = (
            obj.model_path.replace("/", "\\").rsplit("\\", 1)[-1].casefold()
        )
        match = re.fullmatch(
            r"paved_w\d{3}_l(?P<length>\d{4})(?:_[lr]\d{2})?\.p3d",
            filename,
        )
        if match is not None:
            return _p._model_axis(obj, int(match.group("length")) / 10.0)
    return _paved._object_axis(obj, spec)


def _stitch_generated_hub_approaches(
    report,
    plans,
    elevations,
    spec,
):
    """Make generated paved hubs own the terminal approach seam.

    The uploaded terrtest40 PBO showed the actual failure: an angle-matched
    generated plan existed, but the final WRP still contained a sil6 cap and
    stock approaches because the stock approach-template solver rejected the
    plan. Base fitting now guarantees the hub itself. This pass independently
    rebuilds only the terminal approach piece to land on the generated hub's
    explicit connector, so template-search failure cannot reintroduce a gap.
    """

    generated_plans = tuple(
        plan
        for plan in plans.values()
        if _pi.is_generated_paved_junction_model(plan.model_path)
    )
    if not generated_plans or not report.objects:
        return report

    remove_ids: set[int] = set()
    replacements: dict[int, object] = {}
    used_ids: set[int] = set()
    radius = float(_pi.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES)

    for plan in generated_plans:
        candidates = []
        for obj in report.objects[report.junction_cap_objects:]:
            if int(obj.object_id) in used_ids:
                continue
            family = _paved._family(obj.model_path)
            generated_road = _pi.is_generated_paved_road_model(obj.model_path)
            if (
                not generated_road
                and (family is None or _paved._kind(family) != "paved")
            ):
                continue
            axis = _generated_paved_axis(obj, spec)
            if axis is None:
                continue
            distances = tuple(math.dist(plan.point, point) for point in axis)

            # Remove a short stale approach slab that sits mostly inside the
            # generated hub. It would otherwise poke through the hub visual.
            if (
                min(distances) < radius - 1.0
                and max(distances) <= radius + 3.0
            ):
                remove_ids.add(int(obj.object_id))
                continue
            candidates.append((obj, axis, distances))

        for arm in plan.arms:
            connector = tuple(arm.connector.point)
            source_direction = _paved._unit(arm.source_direction)
            connector_direction = _paved._unit(arm.connector.direction)
            connector_radius = math.dist(plan.point, connector)
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
                    radial_error = _paved._angle(radial, source_direction)
                    if radial_error > 30.0:
                        continue

                    endpoint_gap = math.dist(connector, near)
                    segment_gap = _p._point_segment_distance(
                        connector,
                        near,
                        far,
                    )
                    straddles_connector = (
                        near_radius < connector_radius < far_radius
                        and segment_gap <= 1.50
                    )
                    if endpoint_gap > 3.0 and not straddles_connector:
                        continue
                    gap = segment_gap if straddles_connector else endpoint_gap
                    choices.append((
                        gap,
                        0 if straddles_connector else 1,
                        radial_error,
                        object_id,
                        obj,
                        near,
                        far,
                    ))

            if not choices:
                continue
            (
                gap,
                _straddle_priority,
                _radial_error,
                object_id,
                old,
                near,
                far,
            ) = min(
                choices,
                key=lambda value: (value[0], value[1], value[2], value[3]),
            )
            used_ids.add(object_id)

            continuation = _paved._unit((
                far[0] - near[0],
                far[1] - near[1],
            ))
            tangent_error = _paved._angle(
                continuation,
                connector_direction,
            )
            # Endpoint coincidence alone is not a clean seam. terrtest41 had
            # two generated approaches within millimetres of their connectors
            # but one arrived about 11 degrees off-axis, exposing a triangular
            # corner. Rebuild whenever the inner tangent is visibly different.
            if gap <= 0.12 and tangent_error <= 2.0:
                continue

            width = _p._generated_paved_half_width(old.model_path) * 2.0
            length = math.dist(connector, far)
            if length <= 0.05:
                remove_ids.add(object_id)
                continue
            curve = _generated_curve_choice(
                connector,
                far,
                connector_direction,
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
        report = _stitch_generated_hub_approaches(
            report,
            plans,
            elevations,
            spec,
        )
        successful_keys = _successful_plan_keys(
            report, plans, spec=spec, progress_callback=progress_callback
        )
        if len(successful_keys) == len(plans):
            return report

        active = {
            key: plans[key]
            for key in successful_keys
        }
        failed_stock_keys = frozenset(
            set(plans).difference(successful_keys)
        )

        # Give failed vanilla T/X plans one recovery pass against ordinary road
        # geometry before considering a generated hub. The initial stock search
        # reserves roughly 32 m around every candidate junction. On some dense or
        # curved layouts that reservation can hide the very target slab the stock
        # approach solver needs. Refit failed nodes without that reserve, then let
        # the same stock planner clear/rebuild its approaches from the connected
        # road chain. terrtest46's map-centre T is the concrete regression.
        if failed_stock_keys:
            recovery_base = _base_refit(
                dataset,
                projection,
                elevations,
                spec,
                active,
                starting_id=starting_id,
                progress_callback=progress_callback,
            )
            recovery_fit = _paved._apply_plans(
                recovery_base,
                plans,
                elevations,
                spec,
            )
            recovery_success = _successful_plan_keys(
                recovery_fit,
                plans,
                spec=spec,
                progress_callback=progress_callback,
            )
            if len(recovery_success) == len(plans):
                return recovery_fit
            if recovery_success:
                successful_keys = frozenset(
                    set(successful_keys) | set(recovery_success)
                )
                active = {
                    key: plans[key]
                    for key in successful_keys
                }
                failed_stock_keys = frozenset(
                    set(plans).difference(successful_keys)
                )

        # Stock is authoritative. Only junctions that actually failed the stock
        # model/approach validation may be promoted to a generated exact-heading
        # T. This keeps ordinary vanilla-compatible intersections vanilla while
        # retaining the custom hub for genuinely skewed/problematic nodes.
        generated_fallbacks = {}
        if bool(
            getattr(spec, "procedural_paved_road_fallback", False)
        ):
            for key in failed_stock_keys:
                stock_plan = plans[key]
                generated = _paved._generated_plan(
                    stock_plan.point,
                    tuple(
                        (arm.source_direction, arm.family)
                        for arm in stock_plan.arms
                    ),
                    world_name=str(getattr(spec, "name", "world")),
                )
                if generated is not None:
                    generated_fallbacks[key] = generated

        if generated_fallbacks:
            generated_active = dict(active)
            generated_active.update(generated_fallbacks)
            generated_base = _base_refit(
                dataset,
                projection,
                elevations,
                spec,
                generated_active,
                starting_id=starting_id,
                progress_callback=progress_callback,
            )
            generated_fit = _paved._apply_plans(
                generated_base,
                generated_active,
                elevations,
                spec,
            )
            generated_fit = _stitch_generated_hub_approaches(
                generated_fit,
                generated_fallbacks,
                elevations,
                spec,
            )
            generated_success = _successful_plan_keys(
                generated_fit,
                generated_active,
                spec=spec,
                progress_callback=progress_callback,
            )
            if len(generated_success) == len(generated_active):
                return generated_fit
            active = {
                key: generated_active[key]
                for key in generated_success
            }

        changed_keys = frozenset(
            set(plans).difference(active)
        )
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
            base_report = _stitch_generated_hub_approaches(
                base_report,
                plans,
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
                fitted, active, spec=spec, progress_callback=progress_callback
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

        # Geometry remained coupled after the bounded stabilization attempts. An
        # ordinary junction is visually less fancy but, unlike a 30 m grass gap,
        # remains a road.
        final_report = _base_refit(
            dataset,
            projection,
            elevations,
            spec,
            {},
            starting_id=starting_id,
            progress_callback=progress_callback,
        )
        return _stitch_generated_hub_approaches(
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
