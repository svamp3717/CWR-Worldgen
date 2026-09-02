# SPDX-License-Identifier: GPL-3.0-or-later
"""Fit paved roads to stock CWA T/X junction and turn P3Ds."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import permutations, product
import json
import math
from pathlib import Path
import re

from . import generator as _generator
from . import playability as _p
from . import road_quality_policy as _rq

_JUNCTION_RADIUS = 6.25
_APPROACH_RESERVE = 32.0
_CLEAR_RADIUS = 30.0
_TURN_DEGREES = 10.0
_WIDTH = {"sil": 4.55, "kos": 4.55, "asf": 3.50}
_STRAIGHTS = {25: 25.0, 12: 12.5, 6: 6.25}
_T = re.compile(r"kr_new_(sil|asf|kos)_(sil|asf|kos)_t\.p3d$", re.I)
_CURVE = re.compile(r"(?:sil|asf|kos)10 (?:25|50|75|100)\.p3d$", re.I)
_CATALOGUE = Path(__file__).with_name("data") / "road_types.json"


@dataclass(frozen=True, slots=True)
class _Connector:
    family: str
    point: tuple[float, float]
    direction: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _Arm:
    family: str
    source_direction: tuple[float, float]
    connector: _Connector


@dataclass(frozen=True, slots=True)
class _Plan:
    model_path: str
    point: tuple[float, float]
    axis: tuple[float, float]
    arms: tuple[_Arm, ...]

    @property
    def connectors(self) -> tuple[_Connector, ...]:
        return tuple(arm.connector for arm in self.arms)


@dataclass(frozen=True, slots=True)
class _Target:
    object_id: int
    point: tuple[float, float]
    continuation: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _ApproachChoice:
    turn_sign: int
    first_turns: int
    first_radius: int
    middle_units: int
    counter_turns: int
    counter_radius: int
    merge_nominal: int
    merge_target: tuple[float, float]


_PLANS: ContextVar[dict[tuple[int, int], _Plan] | None] = ContextVar(
    "cwr_paved_junction_plans", default=None
)
_ORIGINAL_FIT = None
_ORIGINAL_GEOMETRY = None
_ORIGINAL_TRUSTED_ASSETS = None
_INSTALLED = False


def _unit(direction: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(*direction)
    return (0.0, 1.0) if length <= 1e-9 else (
        direction[0] / length,
        direction[1] / length,
    )


def _heading(direction: tuple[float, float]) -> float:
    return math.degrees(math.atan2(direction[0], direction[1])) % 360.0


def _direction(heading: float) -> tuple[float, float]:
    radians = math.radians(heading)
    return math.sin(radians), math.cos(radians)


def _signed_angle(first: tuple[float, float], second: tuple[float, float]) -> float:
    return (_heading(second) - _heading(first) + 180.0) % 360.0 - 180.0


def _angle(first: tuple[float, float], second: tuple[float, float]) -> float:
    return abs(_signed_angle(first, second))


@lru_cache(maxsize=1)
def _catalogue() -> dict:
    return json.loads(_CATALOGUE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _family_info() -> dict[str, dict]:
    return {
        str(value["name"]).casefold(): value
        for value in _catalogue()["families"]
    }


def _family(path: str) -> str | None:
    if _p.is_generated_gravel_road_model(path):
        return "gravel"
    value = path.replace("/", "\\").casefold()
    for entry in _catalogue()["families"]:
        root = str(entry.get("root", "")).casefold()
        if root and value.startswith(root):
            return str(entry["name"]).casefold()
    return None


def _kind(family: str) -> str:
    return str(_family_info().get(family, {}).get("kind", "unknown")).casefold()


def _junction_family(family: str) -> str:
    return str(_family_info().get(family, {}).get("junction_family", family)).casefold()


def _cost(actual: str, model: str) -> int:
    if actual == model or _junction_family(actual) == model:
        return 0
    return 1 if _kind(actual) == "paved" and model in _WIDTH else 10


@lru_cache(maxsize=1)
def _t_models() -> tuple[tuple[str, str, str], ...]:
    result = []
    for path in _catalogue()["stock_t_junctions"]:
        match = _T.search(str(path).replace("/", "\\"))
        if match:
            result.append((
                str(path),
                match.group(1).casefold(),
                match.group(2).casefold(),
            ))
    return tuple(sorted(result, key=lambda value: value[0].casefold()))


def stock_junction_model_paths() -> tuple[str, ...]:
    return tuple(path for path, _main, _side in _t_models()) + tuple(
        str(value) for value in _catalogue()["stock_x_junctions"]
    )


def stock_turn_model_paths() -> tuple[str, ...]:
    return tuple(
        rf"o\road\{family}10 {radius}.p3d"
        for family in ("sil", "asf", "kos")
        for radius in _catalogue()["stock_curve_radii"]
    )


def stock_approach_model_paths() -> tuple[str, ...]:
    return tuple(
        rf"o\road\{family}{nominal}.p3d"
        for family in ("sil", "asf", "kos")
        for nominal in _catalogue()["stock_straight_lengths"]
    ) + stock_turn_model_paths() + stock_junction_model_paths()


def _world(local, origin, axis) -> tuple[float, float]:
    right = axis[1], -axis[0]
    return (
        origin[0] + right[0] * local[0] + axis[0] * local[1],
        origin[1] + right[1] * local[0] + axis[1] * local[1],
    )


def _main_pair(directions) -> tuple[int, int]:
    return min(
        (
            (a, b)
            for a in range(len(directions))
            for b in range(a + 1, len(directions))
        ),
        key=lambda pair: (
            directions[pair[0]][0] * directions[pair[1]][0]
            + directions[pair[0]][1] * directions[pair[1]][1]
        ),
    )


def _connector(local, local_heading, family, point, axis) -> _Connector:
    right = axis[1], -axis[0]
    radians = math.radians(local_heading)
    return _Connector(
        family,
        _world(local, point, axis),
        _unit((
            right[0] * math.sin(radians) + axis[0] * math.cos(radians),
            right[1] * math.sin(radians) + axis[1] * math.cos(radians),
        )),
    )


def _assign_arms(incidents, connectors) -> tuple[_Arm, ...]:
    best = None
    for order in permutations(range(len(connectors))):
        errors = tuple(
            _angle(incidents[i][0], connectors[j].direction)
            for i, j in enumerate(order)
        )
        family_cost = sum(
            _cost(incidents[i][1], connectors[j].family)
            for i, j in enumerate(order)
        )
        score = (max(errors, default=0.0), sum(errors), family_cost, order)
        if best is None or score < best[0]:
            best = score, order
    assert best is not None
    return tuple(
        _Arm(incidents[i][1], incidents[i][0], connectors[j])
        for i, j in enumerate(best[1])
    )


def _plan(point, incidents) -> _Plan | None:
    if len(incidents) not in {3, 4} or not all(
        _kind(family) == "paved" for _direction_value, family in incidents
    ):
        return None

    directions = tuple(value[0] for value in incidents)
    first, second = _main_pair(directions)
    axis = _unit((
        directions[first][0] - directions[second][0],
        directions[first][1] - directions[second][1],
    ))

    if len(incidents) == 4:
        models = tuple(str(value) for value in _catalogue()["stock_x_junctions"])
        if not models:
            return None
        model_path = models[0]
        definitions = (
            ((0.0, _JUNCTION_RADIUS), 0.0, "sil"),
            ((0.0, -_JUNCTION_RADIUS), 180.0, "sil"),
            ((_JUNCTION_RADIUS, 0.0), 90.0, "sil"),
            ((-_JUNCTION_RADIUS, 0.0), 270.0, "sil"),
        )
    else:
        branch = next(i for i in range(3) if i not in {first, second})
        right = axis[1], -axis[0]
        if (
            directions[branch][0] * right[0]
            + directions[branch][1] * right[1]
            > 0.0
        ):
            axis = -axis[0], -axis[1]
        main_families = incidents[first][1], incidents[second][1]
        branch_family = incidents[branch][1]
        candidates = [
            (
                _cost(main_families[0], main)
                + _cost(main_families[1], main)
                + _cost(branch_family, side),
                path.casefold(),
                path,
                main,
                side,
            )
            for path, main, side in _t_models()
        ]
        if not candidates:
            return None
        _score, _name, model_path, main, side = min(candidates)
        cx = (_JUNCTION_RADIUS - _WIDTH[main]) * 0.5
        definitions = (
            ((cx, _JUNCTION_RADIUS), 0.0, main),
            ((cx, -_JUNCTION_RADIUS), 180.0, main),
            ((cx - _JUNCTION_RADIUS, 0.0), 270.0, side),
        )

    connectors = tuple(
        _connector(local, heading, family, point, axis)
        for local, heading, family in definitions
    )
    return _Plan(model_path, point, axis, _assign_arms(incidents, connectors))


def _plans(dataset, projection, spec) -> dict[tuple[int, int], _Plan]:
    incidents = {}
    positions = {}
    for feature, projected in zip(
        dataset.roads, _p.projected_road_polylines(dataset, projection)
    ):
        if not _p.road_is_supported(
            feature.tags, include_minor=spec.include_minor_roads
        ):
            continue
        points = tuple(_p._clean_road_points(projected))
        model = _p.road_model_for_tags(spec, feature.tags)
        family = _family(model)
        if family is None:
            continue
        dirt = _p.road_is_dirt(feature.tags)
        for index, (start, end) in enumerate(zip(points, points[1:])):
            if math.dist(start, end) <= 0.05:
                continue
            forward = _rq._unit(start, end)
            reverse = -forward[0], -forward[1]
            segment = f"{feature.osm_key}/{index:06d}"
            for node, direction in ((start, forward), (end, reverse)):
                key = _p._road_node_key(node)
                incidents.setdefault(key, []).append(
                    (direction, dirt, model, segment, feature.osm_key)
                )
                positions.setdefault(key, node)

    result = {}
    for key, raw in incidents.items():
        values = _p._unique_incidents(raw)
        if len(values) not in {3, 4}:
            continue
        typed = tuple((value[0], _family(value[2])) for value in values)
        if any(family is None for _direction_value, family in typed):
            continue
        plan = _plan(
            positions[key],
            tuple(
                (direction, family)
                for direction, family in typed
                if family is not None
            ),
        )
        if plan is not None:
            result[key] = plan
    return result


def _junction_geometry(dataset, projection, spec):
    base = dict(_ORIGINAL_GEOMETRY(dataset, projection, spec))
    plans = _PLANS.get() or _plans(dataset, projection, spec)
    return {
        key: replace(
            base[key],
            axis=plan.axis,
            half_length=_APPROACH_RESERVE,
            half_width=_APPROACH_RESERVE,
            directions=tuple(connector.direction for connector in plan.connectors),
        )
        for key, plan in plans.items()
        if key in base
    }


def _arc_step(point, heading, turn_sign, radius):
    direction = _direction(heading)
    right = direction[1], -direction[0]
    angle = math.radians(_TURN_DEGREES)
    side = turn_sign * radius * (1.0 - math.cos(angle))
    forward = radius * math.sin(angle)
    return (
        point[0] + right[0] * side + direction[0] * forward,
        point[1] + right[1] * side + direction[1] * forward,
    ), (heading + turn_sign * _TURN_DEGREES) % 360.0


def _curve_points(family: str, radius: float):
    angle = math.radians(_TURN_DEGREES)
    half = angle * 0.5
    chord = 2.0 * radius * math.sin(half)
    width = _WIDTH[family]
    midpoint = (
        width * (1.0 - math.cos(angle)) * 0.5,
        -width * math.sin(angle) * 0.5,
    )
    unit = math.sin(half), math.cos(half)
    return (
        (
            midpoint[0] - unit[0] * chord * 0.5,
            midpoint[1] - unit[1] * chord * 0.5,
        ),
        (
            midpoint[0] + unit[0] * chord * 0.5,
            midpoint[1] + unit[1] * chord * 0.5,
        ),
    )


def _rotate(local, yaw):
    angle = math.radians(yaw)
    return (
        local[0] * math.cos(angle) + local[1] * math.sin(angle),
        -local[0] * math.sin(angle) + local[1] * math.cos(angle),
    )


def _curve_object(
    object_id, family, radius, start, heading, turn_sign, elevations, spec
):
    begin, end = _curve_points(family, float(radius))
    if turn_sign > 0:
        local_start, local_end = begin, end
        yaw = heading
        next_heading = heading + _TURN_DEGREES
    else:
        local_start, local_end = end, begin
        yaw = heading - (180.0 + _TURN_DEGREES)
        next_heading = heading - _TURN_DEGREES

    sx, sz = _rotate(local_start, yaw)
    origin = start[0] - sx, start[1] - sz
    ex, ez = _rotate(local_end, yaw)
    finish = origin[0] + ex, origin[1] + ez
    height = _p._sample_elevation(
        elevations, spec.cells, spec.cell_size, origin[0], origin[1]
    ) + 0.060
    return (
        _p.WorldObject(
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


def _straight_object(
    object_id, family, nominal, start, end, elevations, spec
):
    return _p._road_object_on_slope(
        object_id,
        rf"o\road\{family}{nominal}.p3d",
        start,
        end,
        elevations,
        spec,
        vertical_offset=0.060,
    )


def _object_axis(obj, spec):
    family = _family(obj.model_path)
    if family is None or _kind(family) != "paved":
        return None
    filename = obj.model_path.replace("/", "\\").rsplit("\\", 1)[-1].casefold()
    if _CURVE.fullmatch(filename) or filename.startswith("kr_"):
        return None
    length = _rq._piece_length(obj.model_path, spec.road_segment_length)
    return _p._model_axis(obj, length)


def _target_candidates(report, plan: _Plan, arm: _Arm, spec):
    result = []
    for obj in report.objects[report.junction_cap_objects:]:
        axis = _object_axis(obj, spec)
        if axis is None:
            continue
        for index in (0, 1):
            point, other = axis[index], axis[1 - index]
            distance = math.dist(plan.point, point)
            other_distance = math.dist(plan.point, other)
            if not (
                _APPROACH_RESERVE - 2.0 <= distance <= _APPROACH_RESERVE + 55.0
                and other_distance > distance + 0.20
            ):
                continue
            radial = _unit((
                point[0] - plan.point[0],
                point[1] - plan.point[1],
            ))
            continuation = _unit((
                other[0] - point[0],
                other[1] - point[1],
            ))
            if (
                _angle(radial, arm.source_direction) > 28.0
                or _angle(continuation, arm.source_direction) > 28.0
            ):
                continue
            result.append(_Target(obj.object_id, point, continuation))
    result.sort(key=lambda target: (
        _angle(
            _unit((
                target.point[0] - plan.point[0],
                target.point[1] - plan.point[1],
            )),
            arm.source_direction,
        ),
        abs(math.dist(plan.point, target.point) - _APPROACH_RESERVE),
        target.object_id,
    ))
    return tuple(result[:10])


def _approach_choice_to_target(
    plan: _Plan, arm: _Arm, target: _Target, tolerance: float
):
    connector = arm.connector
    delta = _signed_angle(connector.direction, target.continuation)
    preferred_sign = 1 if delta >= 0.0 else -1
    radii = tuple(int(value) for value in _catalogue()["stock_curve_radii"])
    best = None

    for turn_sign in (preferred_sign, -preferred_sign):
        for first_turns in range(0, 7):
            for counter_turns in range(0, 5):
                if first_turns == 0 and counter_turns > 0:
                    continue
                first_radii = radii if first_turns else (radii[0],)
                counter_radii = radii if counter_turns else (radii[0],)
                for first_radius in first_radii:
                    for counter_radius in counter_radii:
                        for middle_units in range(0, 8):
                            point = connector.point
                            heading = _heading(connector.direction)
                            for _index in range(first_turns):
                                point, heading = _arc_step(
                                    point, heading, turn_sign, first_radius
                                )
                            if middle_units:
                                direction = _direction(heading)
                                distance = middle_units * _STRAIGHTS[6]
                                point = (
                                    point[0] + direction[0] * distance,
                                    point[1] + direction[1] * distance,
                                )
                            for _index in range(counter_turns):
                                point, heading = _arc_step(
                                    point, heading, -turn_sign, counter_radius
                                )

                            merge_vector = (
                                target.point[0] - point[0],
                                target.point[1] - point[1],
                            )
                            merge_distance = math.hypot(*merge_vector)
                            if merge_distance <= 0.05:
                                continue
                            merge_direction = _unit(merge_vector)
                            in_error = _angle(_direction(heading), merge_direction)
                            out_error = _angle(merge_direction, target.continuation)
                            if max(in_error, out_error) > 12.0:
                                continue

                            for nominal in (6, 12, 25):
                                length_error = abs(merge_distance - _STRAIGHTS[nominal])
                                if length_error > tolerance:
                                    continue
                                piece_count = (
                                    first_turns + counter_turns + middle_units + 1
                                )
                                score = (
                                    length_error * 20.0
                                    + in_error
                                    + out_error
                                    + piece_count * 0.25
                                    + first_radius * 0.001
                                    + counter_radius * 0.001
                                )
                                choice = _ApproachChoice(
                                    turn_sign,
                                    first_turns,
                                    first_radius,
                                    middle_units,
                                    counter_turns,
                                    counter_radius,
                                    nominal,
                                    target.point,
                                )
                                if best is None or score < best[0]:
                                    best = score, choice
    return best


def _arm_options(report, plan, arm, spec):
    tolerance = max(
        0.20, min(0.40, float(getattr(spec, "road_connection_tolerance", 0.35)))
    )
    result = []
    for target in _target_candidates(report, plan, arm, spec):
        match = _approach_choice_to_target(plan, arm, target, tolerance)
        if match is not None:
            score, choice = match
            result.append((score, target, choice))
    result.sort(key=lambda item: (item[0], item[1].object_id))
    return tuple(result[:8])


def _plan_application(report, plan, spec):
    options = tuple(_arm_options(report, plan, arm, spec) for arm in plan.arms)
    if any(not values for values in options):
        return None
    best = None
    for combination in product(*options):
        object_ids = tuple(value[1].object_id for value in combination)
        if len(set(object_ids)) != len(object_ids):
            continue
        score = sum(value[0] for value in combination)
        if best is None or score < best[0]:
            best = score, combination
    return None if best is None else best[1]


def _cap_index(report, plan, used):
    best = None
    for index, obj in enumerate(report.objects[:report.junction_cap_objects]):
        if index in used:
            continue
        distance = math.dist((obj.x, obj.z), plan.point)
        if best is None or distance < best[0]:
            best = distance, index
    if best is None or best[0] > 0.50:
        return None
    return best[1]


def _segment_distance(point, axis):
    return _p._point_segment_distance(point, axis[0], axis[1])


def _approach_objects(plan, arm, choice, next_id, elevations, spec):
    family = arm.connector.family
    point = arm.connector.point
    heading = _heading(arm.connector.direction)
    objects = []

    for _index in range(choice.first_turns):
        obj, point, heading = _curve_object(
            next_id, family, choice.first_radius, point, heading,
            choice.turn_sign, elevations, spec
        )
        objects.append(obj)
        next_id += 1

    direction = _direction(heading)
    for _index in range(choice.middle_units):
        end = (
            point[0] + direction[0] * _STRAIGHTS[6],
            point[1] + direction[1] * _STRAIGHTS[6],
        )
        objects.append(
            _straight_object(next_id, family, 6, point, end, elevations, spec)
        )
        next_id += 1
        point = end

    for _index in range(choice.counter_turns):
        obj, point, heading = _curve_object(
            next_id, family, choice.counter_radius, point, heading,
            -choice.turn_sign, elevations, spec
        )
        objects.append(obj)
        next_id += 1

    objects.append(
        _straight_object(
            next_id, family, choice.merge_nominal, point,
            choice.merge_target, elevations, spec
        )
    )
    return tuple(objects), next_id + 1


def _apply_plans(report, plans, elevations, spec):
    if not plans or report.junction_cap_objects <= 0:
        return report

    applications = []
    used_caps = set()
    for key in sorted(plans):
        plan = plans[key]
        cap_index = _cap_index(report, plan, used_caps)
        if cap_index is None:
            continue
        choices = _plan_application(report, plan, spec)
        if choices is None:
            continue
        used_caps.add(cap_index)
        applications.append((plan, cap_index, choices))

    if not applications:
        return report

    protected_ids = {
        target.object_id
        for _plan, _cap_index_value, choices in applications
        for _score, target, _choice in choices
    }
    remove_ids = set()
    for obj in report.objects[report.junction_cap_objects:]:
        if obj.object_id in protected_ids:
            continue
        axis = _object_axis(obj, spec)
        if axis is None:
            continue
        if any(
            _segment_distance(plan.point, axis) < _CLEAR_RADIUS
            for plan, _cap_index_value, _choices in applications
        ):
            remove_ids.add(obj.object_id)

    objects = [
        obj for obj in report.objects
        if obj.object_id not in remove_ids
    ]
    index_by_id = {obj.object_id: index for index, obj in enumerate(objects)}
    original_caps = report.objects[:report.junction_cap_objects]

    for plan, cap_index, _choices in applications:
        old_id = original_caps[cap_index].object_id
        current_index = index_by_id[old_id]
        start = (
            plan.point[0] - plan.axis[0] * _JUNCTION_RADIUS,
            plan.point[1] - plan.axis[1] * _JUNCTION_RADIUS,
        )
        end = (
            plan.point[0] + plan.axis[0] * _JUNCTION_RADIUS,
            plan.point[1] + plan.axis[1] * _JUNCTION_RADIUS,
        )
        objects[current_index] = _p._road_object_on_slope(
            old_id, plan.model_path, start, end, elevations, spec,
            vertical_offset=0.060,
        )

    next_id = max((obj.object_id for obj in objects), default=0) + 1
    for plan, _cap_index_value, choices in applications:
        for arm, (_score, _target, choice) in zip(plan.arms, choices):
            additions, next_id = _approach_objects(
                plan, arm, choice, next_id, elevations, spec
            )
            objects.extend(additions)

    return replace(report, objects=tuple(objects))


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
            dataset, projection, elevations, spec,
            starting_id=starting_id, progress_callback=progress_callback,
        )

    plans = _plans(dataset, projection, spec)
    token = _PLANS.set(plans)
    try:
        report = _ORIGINAL_FIT(
            dataset, projection, elevations, spec,
            starting_id=starting_id, progress_callback=progress_callback,
        )
    finally:
        _PLANS.reset(token)
    return _apply_plans(report, plans, elevations, spec)


def _trusted_legacy_asset_paths(spec, milestone_number: int):
    base = tuple(_ORIGINAL_TRUSTED_ASSETS(spec, milestone_number))
    if (
        milestone_number < 8
        or not bool(getattr(spec, "stock_road_piece_fitting", False))
    ):
        return base
    trusted = set(base)
    trusted.update(
        _generator.canonical_asset_path(path)
        for path in stock_approach_model_paths()
    )
    return tuple(sorted(path for path in trusted if path))


def install_paved_junction_policy() -> None:
    global _ORIGINAL_FIT, _ORIGINAL_GEOMETRY
    global _ORIGINAL_TRUSTED_ASSETS, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_FIT = _p.fit_road_objects
    _ORIGINAL_GEOMETRY = _rq._junction_geometry
    _ORIGINAL_TRUSTED_ASSETS = _generator._trusted_legacy_asset_paths
    _rq._junction_geometry = _junction_geometry
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _generator._trusted_legacy_asset_paths = _trusted_legacy_asset_paths
    _INSTALLED = True
