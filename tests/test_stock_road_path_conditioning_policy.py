# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from cwr_worldgen.stock_road_path_conditioning_policy import (
    _merge_compatible_paths,
    _protected_node_keys,
    _simplify_path,
)
from cwr_worldgen.stock_road_relaxation_policy import _Obstacle, _ObstacleIndex


def _empty_obstacles() -> _ObstacleIndex:
    return _ObstacleIndex((), {})


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


def test_half_metre_source_noise_is_removed_before_piece_fitting():
    points = ((0.0, 0.0), (0.25, 5.0), (0.0, 10.0))

    simplified = _simplify_path(points, set(), _empty_obstacles())

    assert simplified == (points[0], points[-1])


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
