# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

from cwr_worldgen.stock_road_junction_policy import _Incident
from cwr_worldgen.stock_road_local_fit_policy import (
    _PLANNING_RELAXED_JUNCTION,
    _group_is_obstacle_safe,
    _group_relaxations,
    _strict_native_junction_for_incidents,
    _transaction_native_junction_for_incidents,
)
from cwr_worldgen.stock_road_relaxation_policy import _Obstacle, _ObstacleIndex


def _direction(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(heading_degrees)
    return math.sin(angle), math.cos(angle)


def test_wider_paved_t_matcher_exists_only_during_planning():
    incidents = (
        _Incident(_direction(0.0), "sil", r"o\road\sil25.p3d"),
        _Incident(_direction(180.0), "sil", r"o\road\sil25.p3d"),
        _Incident(_direction(288.0), "sil", r"o\road\sil25.p3d"),
    )

    assert _strict_native_junction_for_incidents(incidents) is None

    token = _PLANNING_RELAXED_JUNCTION.set(True)
    try:
        planned = _transaction_native_junction_for_incidents(incidents)
    finally:
        _PLANNING_RELAXED_JUNCTION.reset(token)

    assert planned is not None
    assert planned.model_path.casefold().endswith(r"kr_new_sil_sil_t.p3d")
    assert _transaction_native_junction_for_incidents(incidents) is None


def test_relaxation_group_is_keyed_by_whole_junction_node():
    projected = (
        ((0.0, 0.0), (10.0, 0.0)),
        ((0.0, 0.0), (0.0, 10.0)),
    )
    plans = {
        (0, 0, 1): (1.0, 0.5),
        (1, 0, 1): (0.5, 1.0),
    }

    grouped = _group_relaxations(projected, plans)

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

    assert not _group_is_obstacle_safe(projected, plans, index)


def test_clear_junction_edit_survives_obstacle_transaction_check():
    projected = (
        ((0.0, 0.0), (12.0, 0.0)),
        ((0.0, 0.0), (0.0, 12.0)),
    )
    plans = {
        (0, 0, 1): (2.0, 0.0),
        (1, 0, 1): (0.0, 2.0),
    }

    assert _group_is_obstacle_safe(projected, plans, _ObstacleIndex((), {}))
