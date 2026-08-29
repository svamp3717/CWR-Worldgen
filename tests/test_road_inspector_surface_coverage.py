# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import road_inspector as _core
from cwr_worldgen import road_inspector_overlap_filter as _overlap
from cwr_worldgen import road_inspector_surface_coverage as _coverage
from cwr_worldgen import stock_road_model_geometry as _geometry


def _straight(
    object_id: int,
    *,
    seam: tuple[float, float],
    heading: float,
    seam_endpoint: int,
    family: str = "sil",
    y: float = 0.0,
):
    length = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6])
    direction = (
        math.sin(math.radians(heading)),
        math.cos(math.radians(heading)),
    )
    sign = -1.0 if seam_endpoint == 1 else 1.0
    center = (
        seam[0] + direction[0] * length * 0.5 * sign,
        seam[1] + direction[1] * length * 0.5 * sign,
    )
    first = (
        center[0] - direction[0] * length * 0.5,
        center[1] - direction[1] * length * 0.5,
    )
    second = (
        center[0] + direction[0] * length * 0.5,
        center[1] + direction[1] * length * 0.5,
    )
    model = rf"o\road\{family}6.p3d"
    endpoints = (
        _core._endpoint(
            object_id=object_id,
            model_path=model,
            family=family,
            kind="straight",
            endpoint_index=0,
            point=first,
            tangent_axis_degrees=heading,
            outward_heading_degrees=heading + 180.0,
        ),
        _core._endpoint(
            object_id=object_id,
            model_path=model,
            family=family,
            kind="straight",
            endpoint_index=1,
            point=second,
            tangent_axis_degrees=heading,
            outward_heading_degrees=heading,
        ),
    )
    return _core.RoadObject(
        object_id,
        model,
        center[0],
        y,
        center[1],
        heading,
        0.0,
        family,
        "straight",
        length,
        center,
        endpoints,
    )


def _issue(*, family: str = "sil"):
    model = rf"o\road\{family}6.p3d"
    return _core.RoadIssue(
        "RI-test",
        "high",
        70.0,
        "straight_miter",
        0.0,
        0.0,
        (1, 2),
        (model, model),
        "test seam",
        "test fix",
        {},
    )


def test_same_family_underlay_cover_suppresses_raw_paved_miter():
    seam = (0.0, 0.0)
    first = _straight(1, seam=seam, heading=340.0, seam_endpoint=1)
    second = _straight(2, seam=seam, heading=308.0, seam_endpoint=0)
    # This is the average undirected axis of the two exposed road edges, the
    # same placement used by the production straight-seam underlay policy.
    cover = _straight(3, seam=seam, heading=144.0, seam_endpoint=0)
    # Recenter the helper on the seam instead of putting one endpoint there.
    direction = (
        math.sin(math.radians(144.0)),
        math.cos(math.radians(144.0)),
    )
    half = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6]) * 0.5
    center_cover = _core.RoadObject(
        cover.object_id,
        cover.model_path,
        0.0,
        0.0,
        0.0,
        144.0,
        0.0,
        "sil",
        "straight",
        half * 2.0,
        seam,
        (
            _core._endpoint(
                object_id=3,
                model_path=cover.model_path,
                family="sil",
                kind="straight",
                endpoint_index=0,
                point=(-direction[0] * half, -direction[1] * half),
                tangent_axis_degrees=144.0,
                outward_heading_degrees=324.0,
            ),
            _core._endpoint(
                object_id=3,
                model_path=cover.model_path,
                family="sil",
                kind="straight",
                endpoint_index=1,
                point=(direction[0] * half, direction[1] * half),
                tangent_axis_degrees=144.0,
                outward_heading_degrees=144.0,
            ),
        ),
    )

    assert _coverage._covered_by_other_paved_surface(
        _issue(), (first, second, center_cover)
    )


def test_nearby_but_wrongly_oriented_road_does_not_hide_open_edges():
    seam = (0.0, 0.0)
    first = _straight(1, seam=seam, heading=340.0, seam_endpoint=1)
    second = _straight(2, seam=seam, heading=308.0, seam_endpoint=0)
    unrelated = _straight(3, seam=seam, heading=90.0, seam_endpoint=0)

    assert not _coverage._covered_by_other_paved_surface(
        _issue(), (first, second, unrelated)
    )


def test_dirt_seams_are_not_filtered_by_paved_surface_coverage():
    seam = (0.0, 0.0)
    first = _straight(1, seam=seam, heading=340.0, seam_endpoint=1, family="ces")
    second = _straight(2, seam=seam, heading=308.0, seam_endpoint=0, family="ces")
    cover = _straight(3, seam=seam, heading=144.0, seam_endpoint=0, family="ces")

    assert not _coverage._covered_by_other_paved_surface(
        _issue(family="ces"), (first, second, cover)
    )


def _overlap_road(object_id, x, z, heading, *, family="sil", y=0.0):
    return SimpleNamespace(
        object_id=int(object_id),
        family=family,
        kind="straight",
        x=float(x),
        y=float(y),
        z=float(z),
        heading_degrees=float(heading),
    )


def test_near_coincident_paved_straights_are_not_reported_as_seam():
    roads = (
        _overlap_road(1, 100.0, 200.0, 15.0),
        _overlap_road(2, 100.18, 200.03, 15.4),
    )

    assert _overlap._overlapping_paved_pair(_issue(), roads)


def test_normal_end_to_end_paved_straights_are_not_overlap_filtered():
    roads = (
        _overlap_road(1, 100.0, 200.0, 15.0),
        _overlap_road(2, 101.6, 206.0, 15.2),
    )

    assert not _overlap._overlapping_paved_pair(_issue(), roads)


def test_near_coincident_dirt_straights_remain_visible_to_inspector():
    roads = (
        _overlap_road(1, 100.0, 200.0, 15.0, family="ces"),
        _overlap_road(2, 100.18, 200.03, 15.4, family="ces"),
    )

    assert not _overlap._overlapping_paved_pair(_issue(family="ces"), roads)
