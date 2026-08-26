# SPDX-License-Identifier: GPL-3.0-or-later
"""Extend stock-road relaxation obstacles to mapped linear roadside objects.

The base relaxation policy protects source buildings, utility points and mapped
individual trees. OSM also carries explicit fence, wall, hedge and retaining-wall
ways plus tree rows that later become physical world objects. A locally straighter
road must not be allowed to move through those merely because they are linear
features rather than points/polygons.

Add each source segment as its own small obstacle box. Segment-level boxes avoid
turning one long rural fence into a map-wide rectangular exclusion zone while the
existing road-width margin still checks the complete visible carriageway.
"""
from __future__ import annotations

import math

from . import stock_road_relaxation_policy as _relax

_BARRIER_HALF_WIDTH_METRES = {
    "hedge": 1.25,
    "wall": 0.60,
    "retaining_wall": 0.75,
    "fence": 0.45,
}
_TREE_ROW_HALF_WIDTH_METRES = 1.75
_ORIGINAL_BUILD_OBSTACLE_INDEX = None
_INSTALLED = False


def _line_radius(feature, default: float) -> float:
    kind = str(getattr(feature, "tags", {}).get("barrier", "")).strip().casefold()
    return _BARRIER_HALF_WIDTH_METRES.get(kind, default)


def _line_obstacles(features, projection, *, default_radius: float):
    result = []
    for feature in features:
        points = tuple(projection.to_world(point) for point in feature.points)
        if len(points) < 2:
            continue
        radius = max(0.0, _line_radius(feature, default_radius))
        for start, end in zip(points, points[1:]):
            if math.dist(start, end) <= 1.0e-6:
                continue
            result.append(
                _relax._Obstacle(
                    min(start[0], end[0]) - radius,
                    min(start[1], end[1]) - radius,
                    max(start[0], end[0]) + radius,
                    max(start[1], end[1]) + radius,
                )
            )
    return result


def _reindex(obstacles):
    bucket_lists: dict[tuple[int, int], list[int]] = {}
    for index, obstacle in enumerate(obstacles):
        for bx in _relax._bucket_range(
            obstacle.min_x, obstacle.max_x, _relax._OBSTACLE_BUCKET_METRES
        ):
            for bz in _relax._bucket_range(
                obstacle.min_z, obstacle.max_z, _relax._OBSTACLE_BUCKET_METRES
            ):
                bucket_lists.setdefault((bx, bz), []).append(index)
    return _relax._ObstacleIndex(
        tuple(obstacles),
        {key: tuple(values) for key, values in bucket_lists.items()},
    )


def _build_obstacle_index(dataset, projection):
    if _ORIGINAL_BUILD_OBSTACLE_INDEX is None:
        raise RuntimeError("stock road obstacle policy is not installed")
    base = _ORIGINAL_BUILD_OBSTACLE_INDEX(dataset, projection)
    obstacles = list(base.obstacles)
    obstacles.extend(
        _line_obstacles(
            getattr(dataset, "barriers", ()),
            projection,
            default_radius=0.50,
        )
    )
    # Tree rows are represented by repeated stock tree objects. Give the source
    # centreline a modest crown/trunk envelope before the ordinary road-width
    # safety margin is added by _shortcut_clear.
    obstacles.extend(
        _line_obstacles(
            getattr(dataset, "tree_rows", ()),
            projection,
            default_radius=_TREE_ROW_HALF_WIDTH_METRES,
        )
    )
    if len(obstacles) == len(base.obstacles):
        return base
    return _reindex(obstacles)


def install_stock_road_obstacle_policy() -> None:
    global _ORIGINAL_BUILD_OBSTACLE_INDEX, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_BUILD_OBSTACLE_INDEX = _relax._build_obstacle_index
    _relax._build_obstacle_index = _build_obstacle_index
    _INSTALLED = True
