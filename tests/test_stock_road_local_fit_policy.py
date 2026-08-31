# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen.model import WorldObject
from cwr_worldgen.stock_road_junction_policy import _Incident
from cwr_worldgen.stock_road_local_fit_policy import (
    _connector_cover_plans,
    _native_junction_for_incidents,
    _same_family_paved_t,
)
from cwr_worldgen.stock_road_relaxation_policy import (
    _ObstacleIndex,
    _simplify_open_run,
)
from cwr_worldgen.stock_road_relaxation_transaction_policy import (
    _PLANNING_RELAXED_JUNCTION,
)


def _direction(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(heading_degrees)
    return math.sin(angle), math.cos(angle)


def _empty_obstacles() -> _ObstacleIndex:
    return _ObstacleIndex((), {})


def test_shallow_dogleg_inside_corridor_is_simplified():
    # A synthetic shallow dog-leg: enough angular noise to create a visible
    # miter with short rigid slabs, but still well inside the allowed corridor.
    points = ((0.0, 0.0), (0.0, 6.25), (1.25, 12.40))

    simplified = _simplify_open_run(points, _empty_obstacles())

    assert simplified == (points[0], points[-1])


def test_large_visual_bend_is_not_flattened_even_below_heading_gate():
    points = (
        (0.0, 0.0),
        (0.0, 10.0),
        (math.sin(math.radians(10.0)) * 20.0, 10.0 + math.cos(math.radians(10.0)) * 20.0),
    )

    simplified = _simplify_open_run(points, _empty_obstacles())

    assert simplified == points


def test_same_family_paved_skew_t_is_provisional_only_during_transaction():
    incidents = (
        _Incident(_direction(0.0), "sil", r"o\road\sil25.p3d"),
        _Incident(_direction(180.0), "sil", r"o\road\sil25.p3d"),
        _Incident(_direction(282.0), "sil", r"o\road\sil25.p3d"),
    )

    assert _same_family_paved_t(incidents)
    assert _native_junction_for_incidents(incidents) is None

    token = _PLANNING_RELAXED_JUNCTION.set(True)
    try:
        native = _native_junction_for_incidents(incidents)
    finally:
        _PLANNING_RELAXED_JUNCTION.reset(token)

    assert native is not None
    assert native.model_path.casefold().endswith(r"kr_new_sil_sil_t.p3d")


def test_normal_straight_junction_cap_does_not_receive_repair_slab():
    cap = WorldObject(1, r"o\road\sil6.p3d", 0.0, 0.041, 0.0, 90.0, 0.0)
    approach = WorldObject(2, r"o\road\sil6.p3d", -6.0, 0.035, 0.0, 90.0, 0.0)
    report = SimpleNamespace(objects=(cap, approach), junction_cap_objects=1)

    assert _connector_cover_plans(report) == ()
