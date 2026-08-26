# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

from cwr_worldgen.model import WorldObject
from cwr_worldgen.stock_road_relaxation_policy import (
    MAXIMUM_LOCAL_RELAXATION_METRES,
    _Obstacle,
    _ObstacleIndex,
    _axis_index,
    _connector_already_covered,
    _simplify_open_run,
)
from cwr_worldgen import stock_road_surface_overlap_policy as _surface


def _empty_index() -> _ObstacleIndex:
    return _ObstacleIndex((), {})


def test_open_micro_bend_is_replaced_by_one_longer_chord():
    points = ((0.0, 0.0), (0.30, 12.0), (0.0, 25.0))

    simplified = _simplify_open_run(points, _empty_index())

    assert simplified == (points[0], points[-1])
    assert 0.30 < MAXIMUM_LOCAL_RELAXATION_METRES


def test_nearby_building_vetoes_the_same_local_shortcut():
    points = ((0.0, 0.0), (0.30, 12.0), (0.0, 25.0))
    obstacle = _Obstacle(-0.5, 10.0, 0.5, 14.0)
    # One obstacle in the only bucket touched by this synthetic road.
    index = _ObstacleIndex((obstacle,), {(0, 0): (0,), (-1, 0): (0,)})

    simplified = _simplify_open_run(points, index)

    assert simplified == points


def test_real_ten_degree_bend_is_not_straightened_as_source_noise():
    points = (
        (0.0, 0.0),
        (0.0, 10.0),
        (math.sin(math.radians(10.0)) * 20.0, 10.0 + math.cos(math.radians(10.0)) * 20.0),
    )

    simplified = _simplify_open_run(points, _empty_index())

    assert simplified == points


def test_long_approach_crossing_connector_suppresses_short_repair_piece():
    # A 25 m paved approach already spans the connector point in its interior.
    approach = WorldObject(
        1,
        r"O\Road\sil25.p3d",
        0.0,
        0.035,
        12.5,
        0.0,
        0.0,
    )
    connector = _surface._Connector((0.0, 6.25), (0.0, 1.0), "sil")

    assert _connector_already_covered(_axis_index((approach,)), connector)


def test_connector_beyond_approach_end_still_needs_gap_handling():
    approach = WorldObject(
        1,
        r"O\Road\sil6.p3d",
        0.0,
        0.035,
        3.125,
        0.0,
        0.0,
    )
    connector = _surface._Connector((0.0, 7.0), (0.0, 1.0), "sil")

    assert not _connector_already_covered(_axis_index((approach,)), connector)
