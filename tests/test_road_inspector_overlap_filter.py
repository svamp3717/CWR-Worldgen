# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

from cwr_worldgen import road_inspector as _core
from cwr_worldgen import road_inspector_overlap_filter as _overlap
from cwr_worldgen import stock_road_model_geometry as _geometry


def _straight(
    object_id: int,
    center: tuple[float, float],
    heading: float,
    *,
    family: str = "sil",
    y: float = 0.0,
):
    length = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6])
    direction = (
        math.sin(math.radians(float(heading))),
        math.cos(math.radians(float(heading))),
    )
    half = length * 0.5
    start = (
        float(center[0]) - direction[0] * half,
        float(center[1]) - direction[1] * half,
    )
    end = (
        float(center[0]) + direction[0] * half,
        float(center[1]) + direction[1] * half,
    )
    model = rf"o\road\{family}6.p3d"
    endpoints = (
        _core._endpoint(
            object_id=object_id,
            model_path=model,
            family=family,
            kind="straight",
            endpoint_index=0,
            point=start,
            tangent_axis_degrees=heading,
            outward_heading_degrees=heading + 180.0,
        ),
        _core._endpoint(
            object_id=object_id,
            model_path=model,
            family=family,
            kind="straight",
            endpoint_index=1,
            point=end,
            tangent_axis_degrees=heading,
            outward_heading_degrees=heading,
        ),
    )
    return _core.RoadObject(
        object_id,
        model,
        float(center[0]),
        float(y),
        float(center[1]),
        float(heading),
        0.0,
        family,
        "straight",
        length,
        (float(center[0]), float(center[1])),
        endpoints,
    )


def _issue(
    object_ids=(1, 2),
    *,
    x: float = 0.0,
    z: float = 0.0,
    family: str = "sil",
    category: str = "connector_gap",
):
    model = rf"o\road\{family}6.p3d"
    return _core.RoadIssue(
        "RI-test",
        "high",
        70.0,
        category,
        float(x),
        float(z),
        tuple(object_ids),
        (model, model),
        "test seam",
        "test fix",
        {},
    )


def test_adjacent_dual_underlay_fans_are_not_visible_seams():
    roads = (
        _straight(1, (-3.0, 0.0), 90.0, y=-0.01),
        _straight(2, (-3.0, 0.0), 100.0, y=-0.01),
        _straight(3, (3.0, 0.0), 100.0, y=-0.01),
        _straight(4, (3.0, 0.0), 110.0, y=-0.01),
    )

    assert _overlap._dual_underlay_seam(_issue((2, 3)), roads)


def test_one_dual_underlay_member_does_not_hide_normal_seam():
    roads = (
        _straight(1, (-3.0, 0.0), 90.0, y=-0.01),
        _straight(2, (-3.0, 0.0), 100.0, y=-0.01),
        _straight(3, (3.0, 0.0), 100.0, y=0.0),
    )

    assert not _overlap._dual_underlay_seam(_issue((2, 3)), roads)


def test_aligned_third_paved_straight_bridge_suppresses_gap():
    roads = (
        _straight(1, (-3.525, 0.0), 90.0),
        _straight(2, (3.525, 0.0), 90.0),
        _straight(3, (0.0, 0.0), 90.0, y=-0.01),
    )

    assert _overlap._bridged_by_aligned_paved_straight(_issue(), roads)


def test_wrong_axis_third_road_does_not_hide_gap():
    roads = (
        _straight(1, (-3.525, 0.0), 90.0),
        _straight(2, (3.525, 0.0), 90.0),
        _straight(3, (0.0, 0.0), 0.0, y=-0.01),
    )

    assert not _overlap._bridged_by_aligned_paved_straight(_issue(), roads)


def test_dirt_repair_geometry_remains_visible_to_inspector():
    roads = (
        _straight(1, (-3.0, 0.0), 90.0, family="ces", y=-0.01),
        _straight(2, (-3.0, 0.0), 100.0, family="ces", y=-0.01),
        _straight(3, (3.0, 0.0), 100.0, family="ces", y=-0.01),
        _straight(4, (3.0, 0.0), 110.0, family="ces", y=-0.01),
    )

    issue = _issue((2, 3), family="ces")
    assert not _overlap._dual_underlay_seam(issue, roads)
    assert not _overlap._bridged_by_aligned_paved_straight(issue, roads)
