# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen.model import WorldObject
from cwr_worldgen.stock_road_junction_policy import _Incident
from cwr_worldgen import stock_road_local_fit_policy as _local
from cwr_worldgen.stock_road_relaxation_policy import (
    _Obstacle,
    _ObstacleIndex,
    _simplify_open_run,
)


def _direction(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(heading_degrees)
    return math.sin(angle), math.cos(angle)


def _empty_obstacles() -> _ObstacleIndex:
    return _ObstacleIndex((), {})


def test_shallow_dogleg_inside_corridor_is_simplified():
    points = ((0.0, 0.0), (0.0, 6.25), (1.25, 12.40))
    assert _simplify_open_run(points, _empty_obstacles()) == (points[0], points[-1])


def test_large_visual_bend_is_not_flattened_even_below_heading_gate():
    points = (
        (0.0, 0.0),
        (0.0, 10.0),
        (
            math.sin(math.radians(10.0)) * 20.0,
            10.0 + math.cos(math.radians(10.0)) * 20.0,
        ),
    )
    assert _simplify_open_run(points, _empty_obstacles()) == points


def test_wider_paved_t_matcher_exists_only_during_planning():
    incidents = (
        _Incident(_direction(0.0), "sil", r"o\road\sil25.p3d"),
        _Incident(_direction(180.0), "sil", r"o\road\sil25.p3d"),
        _Incident(_direction(288.0), "sil", r"o\road\sil25.p3d"),
    )

    assert _local._strict_native_junction_for_incidents(incidents) is None
    assert _local._transaction_native_junction_for_incidents(incidents) is None

    token = _local._PLANNING_RELAXED_JUNCTION.set(True)
    try:
        planned = _local._transaction_native_junction_for_incidents(incidents)
    finally:
        _local._PLANNING_RELAXED_JUNCTION.reset(token)

    assert planned is not None
    assert planned.model_path.casefold().endswith(r"kr_new_sil_sil_t.p3d")
    assert _local._transaction_native_junction_for_incidents(incidents) is None


def test_relaxation_group_is_keyed_by_whole_junction_node():
    projected = (
        ((0.0, 0.0), (10.0, 0.0)),
        ((0.0, 0.0), (0.0, 10.0)),
    )
    plans = {
        (0, 0, 1): (1.0, 0.5),
        (1, 0, 1): (0.5, 1.0),
    }

    grouped = _local._group_relaxations(projected, plans)
    assert len(grouped) == 1
    assert next(iter(grouped.values())) == plans


def test_one_blocked_arm_rejects_the_complete_junction_edit():
    projected = (
        ((0.0, 0.0), (12.0, 0.0)),
        ((0.0, 0.0), (0.0, 12.0)),
    )
    plans = {
        (0, 0, 1): (2.0, 0.0),
        (1, 0, 1): (0.0, 2.0),
    }
    obstacle = _Obstacle(0.5, -0.5, 1.5, 0.5)
    index = _ObstacleIndex(
        (obstacle,),
        {(-1, -1): (0,), (-1, 0): (0,), (0, -1): (0,), (0, 0): (0,)},
    )

    assert not _local._group_is_obstacle_safe(projected, plans, index)


def test_clear_junction_edit_survives_obstacle_transaction_check():
    projected = (
        ((0.0, 0.0), (12.0, 0.0)),
        ((0.0, 0.0), (0.0, 12.0)),
    )
    plans = {
        (0, 0, 1): (2.0, 0.0),
        (1, 0, 1): (0.0, 2.0),
    }

    assert _local._group_is_obstacle_safe(projected, plans, _empty_obstacles())


def test_normal_straight_junction_cap_does_not_receive_repair_slab():
    cap = WorldObject(1, r"o\road\sil6.p3d", 0.0, 0.041, 0.0, 90.0, 0.0)
    approach = WorldObject(2, r"o\road\sil6.p3d", -6.0, 0.035, 0.0, 90.0, 0.0)
    report = SimpleNamespace(objects=(cap, approach), junction_cap_objects=1)

    assert _local._connector_cover_plans(report) == ()
