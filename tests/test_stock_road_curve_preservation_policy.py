# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

from cwr_worldgen import stock_road_geometry_policy as _geometry
from cwr_worldgen import stock_road_path_conditioning_policy as _path
from cwr_worldgen import stock_road_relaxation_policy as _relax
from cwr_worldgen.stock_road_path_conditioning_policy import (
    _candidate_is_sustained_curve,
)


def _empty_obstacles() -> _relax._ObstacleIndex:
    return _relax._ObstacleIndex((), {})


def _stock_like_arc(radius: float = 100.0):
    # Five samples over ten degrees. The complete arc is only about 0.38 m away
    # from its endpoint chord, so a plain sub-metre simplifier would erase it.
    result = []
    for degrees in (0.0, 2.5, 5.0, 7.5, 10.0):
        angle = math.radians(degrees)
        result.append(
            (
                radius * (1.0 - math.cos(angle)),
                radius * math.sin(angle),
            )
        )
    return tuple(result)


def test_stock_radius_ten_degree_arc_is_recognised_as_sustained_curvature():
    points = _stock_like_arc()

    assert _candidate_is_sustained_curve(points, 0, len(points) - 1)


def test_pre_fit_conditioner_does_not_flatten_stock_like_arc():
    points = _stock_like_arc()

    simplified = _path._simplify_path(points, set(), _empty_obstacles())

    assert simplified == points


def test_post_rounding_relaxation_does_not_flatten_stock_like_arc():
    points = _stock_like_arc()

    simplified = _relax._simplify_open_run(points, _empty_obstacles())

    assert simplified == points


def test_micro_bend_cleanup_keeps_consecutive_same_direction_curve_samples():
    points = _stock_like_arc()

    assert _geometry._simplify_micro_bends(points) == points


def test_nearly_straight_dogleg_is_still_allowed_to_simplify():
    points = ((0.0, 0.0), (0.30, 12.0), (0.0, 25.0))

    assert not _candidate_is_sustained_curve(points, 0, len(points) - 1)
    assert _path._simplify_path(points, set(), _empty_obstacles()) == (
        points[0],
        points[-1],
    )
