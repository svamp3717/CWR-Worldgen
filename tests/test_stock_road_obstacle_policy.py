# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from types import SimpleNamespace

from cwr_worldgen.osm import OsmLineFeature
from cwr_worldgen.stock_road_relaxation_policy import (
    _build_obstacle_index,
    _line_obstacles,
    _shortcut_clear,
)


class _IdentityProjection:
    @staticmethod
    def to_world(point):
        return point


def _line(tags, points):
    return OsmLineFeature("way/test", tags, tuple(points))


def test_bent_fence_is_indexed_as_segments_not_one_large_rectangle():
    feature = _line(
        {"barrier": "fence"},
        ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0)),
    )

    obstacles = _line_obstacles((feature,), _IdentityProjection(), default_radius=0.50)

    assert len(obstacles) == 2
    horizontal, vertical = obstacles
    assert horizontal.max_z < 1.0
    assert vertical.min_x > 9.0
    # The empty inside of the L must not become an obstacle merely because the
    # complete feature has a large axis-aligned bounding box.
    assert not any(
        obstacle.min_x <= 5.0 <= obstacle.max_x
        and obstacle.min_z <= 5.0 <= obstacle.max_z
        for obstacle in obstacles
    )


def test_hedge_gets_wider_source_footprint_than_wire_fence():
    fence = _line({"barrier": "fence"}, ((0.0, 0.0), (10.0, 0.0)))
    hedge = _line({"barrier": "hedge"}, ((0.0, 0.0), (10.0, 0.0)))

    fence_box = _line_obstacles((fence,), _IdentityProjection(), default_radius=0.50)[0]
    hedge_box = _line_obstacles((hedge,), _IdentityProjection(), default_radius=0.50)[0]

    assert hedge_box.max_z - hedge_box.min_z > fence_box.max_z - fence_box.min_z


def test_mapped_barrier_vetoes_a_relaxed_road_chord():
    dataset = SimpleNamespace(
        barriers=(
            _line({"barrier": "wall"}, ((5.0, -2.0), (5.0, 2.0))),
        ),
        tree_rows=(),
    )

    index = _build_obstacle_index(dataset, _IdentityProjection())

    assert not _shortcut_clear(index, (0.0, 0.0), (10.0, 0.0))


def test_tree_row_is_also_a_physical_relaxation_obstacle():
    dataset = SimpleNamespace(
        barriers=(),
        tree_rows=(
            _line({"natural": "tree_row"}, ((5.0, -3.0), (5.0, 3.0))),
        ),
    )

    index = _build_obstacle_index(dataset, _IdentityProjection())

    assert not _shortcut_clear(index, (0.0, 0.0), (10.0, 0.0))
