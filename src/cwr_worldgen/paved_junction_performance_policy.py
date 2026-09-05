# SPDX-License-Identifier: GPL-3.0-or-later
"""Speed up paved-junction post-processing and expose live progress.

The stock paved-junction pass used to rescan every fitted road object for every
junction arm, recompute the same approach solution in a cleanup wrapper, then
perform broad object-by-junction cleanup scans. Large maps therefore appeared
to freeze immediately after the road-quality audit had already reached 100%.

This policy keeps the same geometric acceptance rules but indexes reusable road
axes/endpoints once, computes each junction approach once, and limits cleanup to
spatially relevant candidates.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import product
import math
from typing import Callable, Iterable

from . import paved_junction_policy as _paved
from . import road_audit_performance_policy as _audit_progress

_RAW_PERCENT = 99
_BUCKET_METRES = 32.0
_PATH_BUCKET_METRES = 8.0
_PROGRESS_BUCKET_PERCENT = 2
_INSTALLED = False
_ORIGINAL_APPLY = None


@dataclass(frozen=True, slots=True)
class _AxisEntry:
    object_id: int
    axis: tuple[tuple[float, float], tuple[float, float]]
    midpoint: tuple[float, float]
    half_length: float


@dataclass(frozen=True, slots=True)
class _EndpointEntry:
    object_id: int
    point: tuple[float, float]
    other: tuple[float, float]
    continuation: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _PathTemplate:
    order: int
    turn_sign: int
    first_turns: int
    first_radius: int
    middle_units: int
    counter_turns: int
    counter_radius: int
    point: tuple[float, float]
    heading: float


@dataclass(frozen=True, slots=True)
class _PathTemplateIndex:
    templates: tuple[_PathTemplate, ...]
    buckets: dict[tuple[int, int], tuple[_PathTemplate, ...]]


@dataclass(slots=True)
class _SpatialState:
    axis_entries: tuple[_AxisEntry, ...]
    noncap_axis_entries: tuple[_AxisEntry, ...]
    axis_buckets: dict[tuple[int, int], tuple[_AxisEntry, ...]]
    endpoint_buckets: dict[tuple[int, int], tuple[_EndpointEntry, ...]]
    maximum_half_length: float


class _Reporter:
    def __init__(self, callback: Callable[[int, str], None] | None) -> None:
        self.callback = callback
        self._last_bucket: dict[str, int] = {}

    @staticmethod
    def _percent(completed: int, total: int) -> int:
        if total <= 0:
            return 100
        return min(100, max(0, int(completed * 100 / total)))

    def report(
        self,
        key: str,
        label: str,
        completed: int,
        total: int,
        *,
        detail: str = "",
        force: bool = False,
    ) -> None:
        if self.callback is None:
            return
        percent = self._percent(completed, total)
        bucket = percent // _PROGRESS_BUCKET_PERCENT
        previous = self._last_bucket.get(key, -1)
        if (
            not force
            and completed not in {0, 1, total}
            and bucket <= previous
        ):
            return
        self._last_bucket[key] = bucket
        suffix = f"; {detail}" if detail else ""
        self.callback(
            _RAW_PERCENT,
            f"{label} ({completed:,}/{total:,}, {percent}%{suffix})",
        )


def _bucket(value: float, size: float = _BUCKET_METRES) -> int:
    return math.floor(float(value) / float(size))


def _bucket_keys_for_bbox(
    minimum_x: float,
    maximum_x: float,
    minimum_z: float,
    maximum_z: float,
    *,
    size: float = _BUCKET_METRES,
) -> Iterable[tuple[int, int]]:
    bx0 = _bucket(minimum_x, size)
    bx1 = _bucket(maximum_x, size)
    bz0 = _bucket(minimum_z, size)
    bz1 = _bucket(maximum_z, size)
    for bx in range(bx0, bx1 + 1):
        for bz in range(bz0, bz1 + 1):
            yield bx, bz


def _freeze_buckets(values):
    return {key: tuple(entries) for key, entries in values.items()}


def _build_spatial_state(report, spec, reporter: _Reporter) -> _SpatialState:
    axis_entries: list[_AxisEntry] = []
    noncap_entries: list[_AxisEntry] = []
    axis_buckets: dict[tuple[int, int], list[_AxisEntry]] = {}
    endpoint_buckets: dict[tuple[int, int], list[_EndpointEntry]] = {}
    maximum_half_length = 0.0
    total = len(report.objects)
    cap_count = int(report.junction_cap_objects)

    reporter.report(
        "index",
        "Indexing paved-junction road geometry",
        0,
        total,
        force=True,
    )
    for object_index, obj in enumerate(report.objects):
        axis = _paved._object_axis(obj, spec)
        if axis is not None:
            dx = float(axis[1][0]) - float(axis[0][0])
            dz = float(axis[1][1]) - float(axis[0][1])
            length = math.hypot(dx, dz)
            if length > 1.0e-9:
                midpoint = (
                    (float(axis[0][0]) + float(axis[1][0])) * 0.5,
                    (float(axis[0][1]) + float(axis[1][1])) * 0.5,
                )
                entry = _AxisEntry(
                    int(obj.object_id),
                    axis,
                    midpoint,
                    length * 0.5,
                )
                axis_entries.append(entry)
                axis_buckets.setdefault(
                    (_bucket(midpoint[0]), _bucket(midpoint[1])), []
                ).append(entry)
                maximum_half_length = max(maximum_half_length, entry.half_length)

                if object_index >= cap_count:
                    noncap_entries.append(entry)
                    for endpoint_index in (0, 1):
                        point = axis[endpoint_index]
                        other = axis[1 - endpoint_index]
                        continuation = _paved._unit(
                            (
                                float(other[0]) - float(point[0]),
                                float(other[1]) - float(point[1]),
                            )
                        )
                        endpoint = _EndpointEntry(
                            int(obj.object_id),
                            point,
                            other,
                            continuation,
                        )
                        endpoint_buckets.setdefault(
                            (_bucket(point[0]), _bucket(point[1])), []
                        ).append(endpoint)
        reporter.report(
            "index",
            "Indexing paved-junction road geometry",
            object_index + 1,
            total,
            detail=(
                f"{len(axis_entries):,} reusable axes"
                if object_index + 1 == total
                else ""
            ),
        )

    return _SpatialState(
        tuple(axis_entries),
        tuple(noncap_entries),
        _freeze_buckets(axis_buckets),
        _freeze_buckets(endpoint_buckets),
        maximum_half_length,
    )


def _target_candidates(
    state: _SpatialState,
    plan,
    arm,
):
    radius = float(_paved._APPROACH_RESERVE) + 55.0
    px, pz = plan.point
    result = []
    for key in _bucket_keys_for_bbox(
        px - radius, px + radius, pz - radius, pz + radius
    ):
        for endpoint in state.endpoint_buckets.get(key, ()):
            distance = math.dist(plan.point, endpoint.point)
            other_distance = math.dist(plan.point, endpoint.other)
            if not (
                _paved._APPROACH_RESERVE - 2.0
                <= distance
                <= _paved._APPROACH_RESERVE + 55.0
                and other_distance > distance + 0.20
            ):
                continue
            radial = _paved._unit(
                (
                    endpoint.point[0] - plan.point[0],
                    endpoint.point[1] - plan.point[1],
                )
            )
            if (
                _paved._angle(radial, arm.source_direction) > 28.0
                or _paved._angle(
                    endpoint.continuation, arm.source_direction
                )
                > 28.0
            ):
                continue
            result.append(
                _paved._Target(
                    endpoint.object_id,
                    endpoint.point,
                    endpoint.continuation,
                )
            )

    result.sort(
        key=lambda target: (
            _paved._angle(
                _paved._unit(
                    (
                        target.point[0] - plan.point[0],
                        target.point[1] - plan.point[1],
                    )
                ),
                arm.source_direction,
            ),
            abs(
                math.dist(plan.point, target.point)
                - _paved._APPROACH_RESERVE
            ),
            target.object_id,
        )
    )
    return tuple(result[:10])


@lru_cache(maxsize=2)
def _path_template_index(turn_sign: int) -> _PathTemplateIndex:
    """Precompute every turn/straight shape once in connector-local space."""
    sign = 1 if int(turn_sign) >= 0 else -1
    radii = tuple(
        int(value) for value in _paved._catalogue()["stock_curve_radii"]
    )
    templates: list[_PathTemplate] = []
    buckets: dict[tuple[int, int], list[_PathTemplate]] = {}
    order = 0

    for first_turns in range(0, 7):
        for counter_turns in range(0, 5):
            if first_turns == 0 and counter_turns > 0:
                continue
            first_radii = radii if first_turns else (radii[0],)
            counter_radii = radii if counter_turns else (radii[0],)
            for first_radius in first_radii:
                for counter_radius in counter_radii:
                    for middle_units in range(0, 8):
                        point = (0.0, 0.0)
                        heading = 0.0
                        for _index in range(first_turns):
                            point, heading = _paved._arc_step(
                                point,
                                heading,
                                sign,
                                first_radius,
                            )
                        if middle_units:
                            direction = _paved._direction(heading)
                            distance = (
                                middle_units * _paved._STRAIGHTS[6]
                            )
                            point = (
                                point[0] + direction[0] * distance,
                                point[1] + direction[1] * distance,
                            )
                        for _index in range(counter_turns):
                            point, heading = _paved._arc_step(
                                point,
                                heading,
                                -sign,
                                counter_radius,
                            )
                        template = _PathTemplate(
                            order,
                            sign,
                            first_turns,
                            first_radius,
                            middle_units,
                            counter_turns,
                            counter_radius,
                            point,
                            heading,
                        )
                        order += 1
                        templates.append(template)
                        buckets.setdefault(
                            (
                                _bucket(point[0], _PATH_BUCKET_METRES),
                                _bucket(point[1], _PATH_BUCKET_METRES),
                            ),
                            [],
                        ).append(template)

    return _PathTemplateIndex(
        tuple(templates),
        _freeze_buckets(buckets),
    )


def _target_local_point(arm, target) -> tuple[float, float]:
    direction = arm.connector.direction
    right = direction[1], -direction[0]
    dx = target.point[0] - arm.connector.point[0]
    dz = target.point[1] - arm.connector.point[1]
    return (
        dx * right[0] + dz * right[1],
        dx * direction[0] + dz * direction[1],
    )


def _candidate_templates(
    arm,
    target,
    tolerance: float,
    turn_sign: int,
):
    local = _target_local_point(arm, target)
    radius = max(float(value) for value in _paved._STRAIGHTS.values()) + tolerance
    index = _path_template_index(turn_sign)
    candidates = []
    for key in _bucket_keys_for_bbox(
        local[0] - radius,
        local[0] + radius,
        local[1] - radius,
        local[1] + radius,
        size=_PATH_BUCKET_METRES,
    ):
        candidates.extend(index.buckets.get(key, ()))
    candidates.sort(key=lambda template: template.order)
    return candidates


def _approach_choice_to_target(
    plan,
    arm,
    target,
    tolerance: float,
):
    connector = arm.connector
    delta = _paved._signed_angle(
        connector.direction, target.continuation
    )
    preferred_sign = 1 if delta >= 0.0 else -1
    initial_heading = _paved._heading(connector.direction)
    best = None

    for turn_sign in (preferred_sign, -preferred_sign):
        for path in _candidate_templates(
            arm, target, tolerance, turn_sign
        ):
            world_point = _paved._world(
                path.point,
                connector.point,
                connector.direction,
            )
            merge_vector = (
                target.point[0] - world_point[0],
                target.point[1] - world_point[1],
            )
            merge_distance = math.hypot(*merge_vector)
            if merge_distance <= 0.05:
                continue

            # Reject by stock straight length before doing expensive heading
            # comparisons. This is mathematically independent of the headings
            # and discards almost every path candidate on real junctions.
            nominal_errors = []
            for nominal in (6, 12, 25):
                length_error = abs(
                    merge_distance - _paved._STRAIGHTS[nominal]
                )
                if length_error <= tolerance:
                    nominal_errors.append((nominal, length_error))
            if not nominal_errors:
                continue

            merge_direction = (
                merge_vector[0] / merge_distance,
                merge_vector[1] / merge_distance,
            )
            path_direction = _paved._direction(
                (initial_heading + path.heading) % 360.0
            )
            in_error = _paved._angle(
                path_direction, merge_direction
            )
            out_error = _paved._angle(
                merge_direction, target.continuation
            )
            if max(in_error, out_error) > 12.0:
                continue

            for nominal, length_error in nominal_errors:
                piece_count = (
                    path.first_turns
                    + path.counter_turns
                    + path.middle_units
                    + 1
                )
                score = (
                    length_error * 20.0
                    + in_error
                    + out_error
                    + piece_count * 0.25
                    + path.first_radius * 0.001
                    + path.counter_radius * 0.001
                )
                choice = _paved._ApproachChoice(
                    path.turn_sign,
                    path.first_turns,
                    path.first_radius,
                    path.middle_units,
                    path.counter_turns,
                    path.counter_radius,
                    nominal,
                    target.point,
                )
                if best is None or score < best[0]:
                    best = score, choice
    return best


def _arm_options(state: _SpatialState, plan, arm, spec):
    tolerance = max(
        0.20,
        min(
            0.40,
            float(getattr(spec, "road_connection_tolerance", 0.35)),
        ),
    )
    result = []
    for target in _target_candidates(state, plan, arm):
        match = _approach_choice_to_target(
            plan, arm, target, tolerance
        )
        if match is not None:
            score, choice = match
            result.append((score, target, choice))
    result.sort(key=lambda item: (item[0], item[1].object_id))
    return tuple(result[:8])


def _plan_application(state: _SpatialState, plan, spec):
    options = tuple(
        _arm_options(state, plan, arm, spec)
        for arm in plan.arms
    )
    if any(not values for values in options):
        return None
    best = None
    for combination in product(*options):
        object_ids = tuple(
            value[1].object_id for value in combination
        )
        if len(set(object_ids)) != len(object_ids):
            continue
        score = sum(value[0] for value in combination)
        if best is None or score < best[0]:
            best = score, combination
    return None if best is None else best[1]


def _application_point_buckets(applications):
    buckets: dict[tuple[int, int], list[tuple]] = {}
    for application in applications:
        plan = application[0]
        buckets.setdefault(
            (_bucket(plan.point[0]), _bucket(plan.point[1])), []
        ).append(application)
    return _freeze_buckets(buckets)


def _applications_near_axis(
    plan_buckets,
    entry: _AxisEntry,
    radius: float,
):
    start, end = entry.axis
    for key in _bucket_keys_for_bbox(
        min(start[0], end[0]) - radius,
        max(start[0], end[0]) + radius,
        min(start[1], end[1]) - radius,
        max(start[1], end[1]) + radius,
    ):
        yield from plan_buckets.get(key, ())


def _clear_base_junction_zones(
    state: _SpatialState,
    applications,
    protected_ids: set[int],
    reporter: _Reporter,
) -> set[int]:
    plan_buckets = _application_point_buckets(applications)
    remove_ids: set[int] = set()
    total = len(state.noncap_axis_entries)
    reporter.report(
        "clear",
        "Clearing paved-junction approach zones",
        0,
        total,
        force=True,
    )
    for completed, entry in enumerate(
        state.noncap_axis_entries, start=1
    ):
        if entry.object_id not in protected_ids:
            for plan, _cap_index, _choices in _applications_near_axis(
                plan_buckets,
                entry,
                _paved._CLEAR_RADIUS,
            ):
                if (
                    _paved._segment_distance(plan.point, entry.axis)
                    < _paved._CLEAR_RADIUS
                ):
                    remove_ids.add(entry.object_id)
                    break
        reporter.report(
            "clear",
            "Clearing paved-junction approach zones",
            completed,
            total,
            detail=f"{len(remove_ids):,} road pieces removed",
        )
    return remove_ids


def _axis_entries_in_corridor(
    state: _SpatialState,
    start: tuple[float, float],
    end: tuple[float, float],
):
    margin = 6.0 + state.maximum_half_length + 0.25
    seen: set[int] = set()
    for key in _bucket_keys_for_bbox(
        min(start[0], end[0]) - margin,
        max(start[0], end[0]) + margin,
        min(start[1], end[1]) - margin,
        max(start[1], end[1]) + margin,
    ):
        for entry in state.axis_buckets.get(key, ()):
            if entry.object_id in seen:
                continue
            seen.add(entry.object_id)
            yield entry


def _premerge_cleanup_ids(
    state: _SpatialState,
    applications,
    protected_ids: set[int],
    reporter: _Reporter,
) -> set[int]:
    remove_ids: set[int] = set()
    total = sum(
        len(plan.arms)
        for plan, _cap, _choices in applications
    )
    completed = 0
    reporter.report(
        "corridor",
        "Cleaning paved-junction merge corridors",
        0,
        total,
        force=True,
    )

    for plan, _cap_index, choices in applications:
        for arm, (_score, target, _choice) in zip(
            plan.arms, choices
        ):
            sx, sz = arm.source_direction
            target_along = (
                (target.point[0] - plan.point[0]) * sx
                + (target.point[1] - plan.point[1]) * sz
            )
            if target_along > _paved._CLEAR_RADIUS + 0.20:
                corridor_end = (
                    plan.point[0] + sx * target_along,
                    plan.point[1] + sz * target_along,
                )
                for entry in _axis_entries_in_corridor(
                    state, plan.point, corridor_end
                ):
                    if (
                        entry.object_id in protected_ids
                        or entry.object_id in remove_ids
                    ):
                        continue
                    axis = entry.axis
                    midpoint = entry.midpoint
                    samples = (axis[0], midpoint, axis[1])
                    along = tuple(
                        (point[0] - plan.point[0]) * sx
                        + (point[1] - plan.point[1]) * sz
                        for point in samples
                    )
                    if (
                        max(along)
                        <= _paved._CLEAR_RADIUS - 0.20
                        or min(along) >= target_along - 0.20
                    ):
                        continue
                    lateral = min(
                        abs(
                            (point[0] - plan.point[0]) * sz
                            - (point[1] - plan.point[1]) * sx
                        )
                        for point in samples
                    )
                    if lateral <= 6.0:
                        remove_ids.add(entry.object_id)

            completed += 1
            reporter.report(
                "corridor",
                "Cleaning paved-junction merge corridors",
                completed,
                total,
                detail=f"{len(remove_ids):,} stale pieces removed",
            )
    return remove_ids


def _remove_stale_caps(
    objects,
    report,
    applications,
    spec,
    reporter: _Reporter,
):
    if report.junction_cap_objects <= 0:
        return objects

    active_buckets = _application_point_buckets(applications)
    original_cap_ids = {
        int(obj.object_id)
        for obj in report.objects[: report.junction_cap_objects]
    }
    current_by_id = {
        int(obj.object_id): obj
        for obj in objects
        if int(obj.object_id) in original_cap_ids
    }
    cap_ids = tuple(sorted(original_cap_ids))
    remove_ids: set[int] = set()
    total = len(cap_ids)
    reporter.report(
        "caps",
        "Removing stale paved-junction caps",
        0,
        total,
        force=True,
    )

    for completed, object_id in enumerate(cap_ids, start=1):
        obj = current_by_id.get(object_id)
        if obj is not None:
            axis = _paved._object_axis(obj, spec)
            if axis is not None:
                dx = axis[1][0] - axis[0][0]
                dz = axis[1][1] - axis[0][1]
                midpoint = (
                    (axis[0][0] + axis[1][0]) * 0.5,
                    (axis[0][1] + axis[1][1]) * 0.5,
                )
                entry = _AxisEntry(
                    object_id,
                    axis,
                    midpoint,
                    math.hypot(dx, dz) * 0.5,
                )
                for plan, _cap, _choices in _applications_near_axis(
                    active_buckets,
                    entry,
                    _paved._CLEAR_RADIUS,
                ):
                    if (
                        _paved._segment_distance(plan.point, axis)
                        < _paved._CLEAR_RADIUS
                    ):
                        remove_ids.add(object_id)
                        break
        reporter.report(
            "caps",
            "Removing stale paved-junction caps",
            completed,
            total,
            detail=f"{len(remove_ids):,} stale caps removed",
        )

    if not remove_ids:
        return objects
    return [
        obj for obj in objects
        if int(obj.object_id) not in remove_ids
    ]


def apply_paved_junctions_fast(
    report,
    plans,
    elevations,
    spec,
):
    """Apply paved-junction plans with spatial indexing and live progress."""
    if not plans or report.junction_cap_objects <= 0:
        return report

    callback = _audit_progress._PROGRESS_CALLBACK.get()
    reporter = _Reporter(callback)
    state = _build_spatial_state(report, spec, reporter)

    applications = []
    used_caps: set[int] = set()
    ordered = tuple(sorted(plans))
    total_plans = len(ordered)
    reporter.report(
        "plans",
        "Planning paved-junction approaches",
        0,
        total_plans,
        force=True,
    )
    for completed, key in enumerate(ordered, start=1):
        plan = plans[key]
        cap_index = _paved._cap_index(
            report, plan, used_caps
        )
        if cap_index is not None:
            choices = _plan_application(state, plan, spec)
            if choices is not None:
                used_caps.add(cap_index)
                applications.append((plan, cap_index, choices))
        reporter.report(
            "plans",
            "Planning paved-junction approaches",
            completed,
            total_plans,
            detail=f"{len(applications):,} applicable",
        )

    if not applications:
        reporter.report(
            "complete",
            "Paved-junction post-processing complete",
            1,
            1,
            detail="0 applicable junctions",
            force=True,
        )
        return report

    target_ids = {
        int(target.object_id)
        for _plan, _cap_index, choices in applications
        for _score, target, _choice in choices
    }
    base_remove_ids = _clear_base_junction_zones(
        state,
        applications,
        target_ids,
        reporter,
    )

    objects = [
        obj for obj in report.objects
        if int(obj.object_id) not in base_remove_ids
    ]
    index_by_id = {
        int(obj.object_id): index
        for index, obj in enumerate(objects)
    }
    original_caps = report.objects[: report.junction_cap_objects]
    protected_cap_ids: set[int] = set()

    for plan, cap_index, _choices in applications:
        old_id = int(original_caps[cap_index].object_id)
        protected_cap_ids.add(old_id)
        current_index = index_by_id[old_id]
        start = (
            plan.point[0] - plan.axis[0] * _paved._JUNCTION_RADIUS,
            plan.point[1] - plan.axis[1] * _paved._JUNCTION_RADIUS,
        )
        end = (
            plan.point[0] + plan.axis[0] * _paved._JUNCTION_RADIUS,
            plan.point[1] + plan.axis[1] * _paved._JUNCTION_RADIUS,
        )
        objects[current_index] = _paved._p._road_object_on_slope(
            old_id,
            plan.model_path,
            start,
            end,
            elevations,
            spec,
            vertical_offset=0.060,
        )

    next_id = max(
        (int(obj.object_id) for obj in objects),
        default=0,
    ) + 1
    total_arms = sum(
        len(plan.arms)
        for plan, _cap, _choices in applications
    )
    emitted_arms = 0
    reporter.report(
        "emit",
        "Generating paved-junction approaches",
        0,
        total_arms,
        force=True,
    )
    for plan, _cap_index, choices in applications:
        for arm, (_score, _target, choice) in zip(
            plan.arms, choices
        ):
            additions, next_id = _paved._approach_objects(
                plan,
                arm,
                choice,
                next_id,
                elevations,
                spec,
            )
            objects.extend(additions)
            emitted_arms += 1
            reporter.report(
                "emit",
                "Generating paved-junction approaches",
                emitted_arms,
                total_arms,
            )

    # Preserve the historical post-apply pre-merge cleanup, but only inspect
    # road axes whose midpoint can possibly intersect one approach corridor.
    corridor_protected = target_ids | protected_cap_ids
    corridor_remove_ids = _premerge_cleanup_ids(
        state,
        applications,
        corridor_protected,
        reporter,
    )
    if corridor_remove_ids:
        objects = [
            obj for obj in objects
            if int(obj.object_id) not in corridor_remove_ids
        ]

    # Preserve gravel_junction_policy's stale base-cap cleanup. Every application
    # above has just emitted its selected stock junction at plan.point, so its
    # active-plan discovery condition is satisfied by construction.
    objects = _remove_stale_caps(
        objects,
        report,
        applications,
        spec,
        reporter,
    )

    reporter.report(
        "complete",
        "Paved-junction post-processing complete",
        1,
        1,
        detail=(
            f"{len(applications):,} junctions; "
            f"{len(objects):,} road objects"
        ),
        force=True,
    )
    return replace(report, objects=tuple(objects))


def _install_audit_completion_deduplication() -> None:
    """Avoid the normal+forced duplicate 100% audit status line."""
    reporter_type = _audit_progress._AuditReporter
    if getattr(reporter_type, "_cwr_completion_dedupe", False):
        return

    original_index = reporter_type.index
    original_junction = reporter_type.junction

    def index(self, completed, total, *, force=False):
        marker = (int(completed), int(total))
        if (
            force
            and marker
            == getattr(self, "_cwr_last_index_completion", None)
        ):
            return
        original_index(self, completed, total, force=force)
        if completed == total:
            self._cwr_last_index_completion = marker

    def junction(
        self,
        completed,
        *,
        failed,
        indexed_axes,
        force=False,
    ):
        marker = int(completed)
        if (
            force
            and marker
            == getattr(self, "_cwr_last_junction_completion", None)
        ):
            return
        original_junction(
            self,
            completed,
            failed=failed,
            indexed_axes=indexed_axes,
            force=force,
        )
        if completed == self.junction_total:
            self._cwr_last_junction_completion = marker

    reporter_type.index = index
    reporter_type.junction = junction
    reporter_type._cwr_completion_dedupe = True


def install_paved_junction_performance_policy() -> None:
    """Replace only the paved post-fit apply phase after all road wrappers."""
    global _ORIGINAL_APPLY, _INSTALLED
    if _INSTALLED:
        return

    _install_audit_completion_deduplication()
    _ORIGINAL_APPLY = _paved._apply_plans
    _paved._apply_plans = apply_paved_junctions_fast
    _INSTALLED = True
