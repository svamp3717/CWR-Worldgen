from __future__ import annotations

import math

# Install the public read-only inspector stack in production order.
import cwr_worldgen.road_inspector_entry  # noqa: F401
from cwr_worldgen import road_inspector as _core


def _straight(
    object_id: int,
    *,
    begin: tuple[float, float],
    heading: float,
    length: float = 6.25,
    center_y: float = 0.035,
) -> _core.RoadObject:
    angle = math.radians(heading)
    direction = (math.sin(angle), math.cos(angle))
    end = (
        begin[0] + direction[0] * length,
        begin[1] + direction[1] * length,
    )
    center = ((begin[0] + end[0]) * 0.5, (begin[1] + end[1]) * 0.5)
    first = _core.RoadEndpoint(
        object_id,
        r"o\road\sil6.p3d",
        "sil",
        "straight",
        0,
        begin,
        heading % 180.0,
        (heading + 180.0) % 360.0,
        4.55,
    )
    second = _core.RoadEndpoint(
        object_id,
        r"o\road\sil6.p3d",
        "sil",
        "straight",
        1,
        end,
        heading % 180.0,
        heading % 360.0,
        4.55,
    )
    return _core.RoadObject(
        object_id,
        r"o\road\sil6.p3d",
        center[0],
        center_y,
        center[1],
        heading,
        0.0,
        "sil",
        "straight",
        length,
        center,
        (first, second),
    )


def _cap(object_id: int, node: tuple[float, float]) -> _core.RoadObject:
    half = 6.25 * 0.5
    first = _core.RoadEndpoint(
        object_id,
        r"o\road\sil6.p3d",
        "sil",
        "straight",
        0,
        (node[0], node[1] - half),
        0.0,
        180.0,
        4.55,
    )
    second = _core.RoadEndpoint(
        object_id,
        r"o\road\sil6.p3d",
        "sil",
        "straight",
        1,
        (node[0], node[1] + half),
        0.0,
        0.0,
        4.55,
    )
    return _core.RoadObject(
        object_id,
        r"o\road\sil6.p3d",
        node[0],
        0.031,
        node[1],
        0.0,
        0.0,
        "sil",
        "straight",
        6.25,
        node,
        (first, second),
    )


def test_inset_approaches_are_matched_to_logical_source_node() -> None:
    node = (100.0, 100.0)
    roads = (
        _cap(1, node),
        _straight(2, begin=(100.0, 103.0), heading=0.0),
        _straight(3, begin=(100.0, 97.0), heading=180.0),
        _straight(4, begin=(103.0, 100.0), heading=90.0),
    )
    source = (_core.SourceJunction(node, (0.0, 90.0, 180.0)),)

    issues = _core._source_intersection_issues(
        roads,
        source,
        match_tolerance=0.75,
    )

    # All three real approaches stop three metres before the OSM node because
    # the central cap owns that space. They are still directionally exact and
    # must not be reported as missing/misaligned arms.
    assert not [
        issue
        for issue in issues
        if issue.category
        in {"intersection_missing_cap", "intersection_approach_mismatch"}
    ]


def test_inset_matching_does_not_snap_a_piece_that_points_across_the_node() -> None:
    node = (100.0, 100.0)
    roads = (
        _cap(1, node),
        _straight(2, begin=(100.0, 103.0), heading=0.0),
        _straight(3, begin=(100.0, 97.0), heading=180.0),
        # The endpoint happens to be east of the node but the piece points
        # north. Proximity alone must not invent a correct eastern approach.
        _straight(4, begin=(103.0, 100.0), heading=0.0),
    )
    source = (_core.SourceJunction(node, (0.0, 90.0, 180.0)),)

    issues = _core._source_intersection_issues(
        roads,
        source,
        match_tolerance=0.75,
    )

    mismatch = next(
        issue for issue in issues if issue.category == "intersection_approach_mismatch"
    )
    assert mismatch.metrics["maximum_approach_heading_error_degrees"] > 80.0
