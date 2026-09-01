# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

from cwr_worldgen import stock_road_geometry_policy as _geometry
from cwr_worldgen import stock_road_path_conditioning_policy as _path
from cwr_worldgen import stock_road_relaxation_policy as _relax
from cwr_worldgen.playability import _unique_incidents
from cwr_worldgen.stock_road_path_conditioning_policy import (
    _candidate_is_sustained_curve,
    _condition_paths_with_count,
    _merge_compatible_paths,
    _protected_node_keys,
    _simplify_path,
)
from cwr_worldgen.stock_road_relaxation_policy import _Obstacle, _ObstacleIndex


def _empty_obstacles() -> _ObstacleIndex:
    return _ObstacleIndex((), {})


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


def test_unambiguous_same_type_fragments_merge_into_one_run():
    paths = (
        ((0.0, 0.0), (10.0, 0.0)),
        ((10.0, 0.0), (20.0, 0.0)),
    )
    keys = (("sil",), ("sil",))

    merged = _merge_compatible_paths(paths, keys)

    assert merged[0] == ((0.0, 0.0), (10.0, 0.0), (20.0, 0.0))
    assert merged[1] == ()


def test_three_same_type_arms_remain_separate_at_t_junction():
    paths = (
        ((0.0, 0.0), (10.0, 0.0)),
        ((10.0, 0.0), (20.0, 0.0)),
        ((10.0, 0.0), (10.0, 10.0)),
    )
    keys = (("sil",), ("sil",), ("sil",))

    assert _merge_compatible_paths(paths, keys) == paths


def test_main_road_can_merge_through_different_surface_branch():
    paths = (
        ((0.0, 0.0), (10.0, 0.0)),
        ((10.0, 0.0), (20.0, 0.0)),
        ((10.0, 0.0), (10.0, 8.0)),
    )
    keys = (("sil",), ("sil",), ("gravel",))

    merged = _merge_compatible_paths(paths, keys)

    assert merged[0] == ((0.0, 0.0), (10.0, 0.0), (20.0, 0.0))
    assert merged[1] == ()
    assert merged[2] == paths[2]
    # The shared node stays as a real vertex, so later junction discovery still
    # sees the branch even though the main road is fitted as one continuous run.
    assert (10.0, 0.0) in merged[0]


def test_merged_through_road_keeps_both_incident_directions_at_t_node():
    values = (
        ((1.0, 0.0), False, r"o\road\sil25.p3d", "owner/000000", "owner"),
        ((-1.0, 0.0), False, r"o\road\sil25.p3d", "owner/000001", "owner"),
        ((0.0, 1.0), False, r"o\road\asf25.p3d", "branch/000000", "branch"),
    )

    unique = _unique_incidents(values)

    assert len(unique) == 3
    assert {value[0] for value in unique} == {(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0)}


def test_half_metre_source_noise_is_removed_before_piece_fitting():
    points = ((0.0, 0.0), (0.25, 5.0), (0.0, 10.0))

    simplified = _simplify_path(points, set(), _empty_obstacles())

    assert simplified == (points[0], points[-1])


def test_conditioned_degree_two_vertices_remain_reportable_as_suppressed_caps():
    points = tuple((float(index) * 10.0, 0.0) for index in range(6))

    conditioned, suppressed = _condition_paths_with_count(
        (points,), (("sil",),), _empty_obstacles()
    )

    assert conditioned == ((points[0], points[-1]),)
    assert suppressed == 4


def test_surface_transition_node_is_a_simplification_anchor():
    paths = (
        ((0.0, 0.0), (0.25, 5.0)),
        ((0.25, 5.0), (0.0, 10.0)),
    )
    keys = (("sil",), ("asf",))
    protected = _protected_node_keys(paths, keys)

    assert (2, 50) in protected


def test_sharp_corner_is_a_simplification_anchor():
    points = ((0.0, 0.0), (0.0, 10.0), (10.0, 10.0))
    paths = (points,)
    keys = (("sil",),)

    protected = _protected_node_keys(paths, keys)
    simplified = _simplify_path(points, protected, _empty_obstacles())

    assert (0, 100) in protected
    assert simplified == points


def test_obstacle_corridor_vetoes_source_shortcut():
    points = ((0.0, 0.0), (0.25, 5.0), (0.0, 10.0))
    obstacle = _Obstacle(-0.2, 4.8, 0.2, 5.2)
    obstacles = _ObstacleIndex((obstacle,), {(0, 0): (0,)})

    simplified = _simplify_path(points, set(), obstacles)

    assert simplified == points


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
