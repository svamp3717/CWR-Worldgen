# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

from cwr_worldgen import road_inspector as _core
from cwr_worldgen import road_inspector_overlap_filter as _overlap
from cwr_worldgen import road_inspector_surface_coverage as _coverage
from cwr_worldgen import road_inspector_grass_wedge as _wedge
from cwr_worldgen import stock_road_model_geometry as _geometry
from cwr_worldgen.procedural_infrastructure import (
    GENERATED_PAVED_FILL_RADIUS_METRES,
    GENERATED_PAVED_WEDGE_BASE_OVERLAP_METRES,
    paved_wedge_local_points,
)
from cwr_worldgen.pbo import pack_directory
from cwr_worldgen.wrp import write_rvw4


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


def test_bisecting_straight_does_not_reach_outer_miter_apex():
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

    assert not _coverage._covered_by_other_paved_surface(
        _issue(), (first, second, center_cover)
    )


def test_circular_paved_fill_does_not_hide_outside_miter_wedge():
    seam = (0.0, 0.0)
    first = _straight(1, seam=seam, heading=340.0, seam_endpoint=1)
    second = _straight(2, seam=seam, heading=308.0, seam_endpoint=0)
    fill = _core.RoadObject(
        3,
        r"wg_test\i\paved_fill.p3d",
        seam[0],
        -0.01,
        seam[1],
        144.0,
        0.0,
        "sil",
        "paved_fill",
        9.1,
        seam,
        (),
    )

    assert not _coverage._covered_by_other_paved_surface(
        _issue(), (first, second, fill)
    )


def test_angle_matched_paved_miter_suppresses_raw_paved_miter():
    seam = (0.0, 0.0)
    first = _straight(1, seam=seam, heading=340.0, seam_endpoint=1)
    second = _straight(2, seam=seam, heading=308.0, seam_endpoint=0)
    miter = _core.RoadObject(
        3,
        r"wg_test\i\paved_miter_q128.p3d",
        seam[0],
        -0.01,
        seam[1],
        144.0,
        0.0,
        "sil",
        "paved_miter",
        9.5,
        seam,
        (),
    )

    assert _coverage._covered_by_other_paved_surface(
        _issue(), (first, second, miter)
    )


def test_buried_miter_needs_terrain_clear_outer_wedge():
    seam = (0.0, 0.0)
    first = _straight(1, seam=seam, heading=340.0, seam_endpoint=1)
    second = _straight(2, seam=seam, heading=308.0, seam_endpoint=0)
    miter = _core.RoadObject(
        3,
        r"wg_test\i\paved_miter_q128.p3d",
        0.0,
        0.025,
        0.0,
        144.0,
        0.0,
        "sil",
        "paved_miter",
        9.5,
        seam,
        (),
    )
    terrain = ((0.0, 0.0, 0.0, 0.0), 2, 25.0)
    issue = _issue()
    assert not _coverage._covered_by_other_paved_surface(
        issue, (first, second, miter), terrain
    )

    geometry = _wedge._grass_wedge_geometry(first.endpoints[1], second.endpoints[0])
    assert geometry is not None
    apex = geometry[3]
    radial = math.hypot(apex[0], apex[1])
    ux, uz = apex[0] / radial, apex[1] / radial
    turn = 32.0
    base_distance = (
        GENERATED_PAVED_FILL_RADIUS_METRES * math.cos(math.radians(turn * 0.5))
        - GENERATED_PAVED_WEDGE_BASE_OVERLAP_METRES
    )
    origin = (ux * base_distance, uz * base_distance)
    wedge_points = paved_wedge_local_points(turn)
    wedge = _core.RoadObject(
        4,
        r"wg_test\i\paved_wedge_q128.p3d",
        origin[0],
        0.031,
        origin[1],
        math.degrees(math.atan2(ux, uz)) % 360.0,
        0.0,
        "sil",
        "paved_wedge",
        wedge_points[0][2],
        origin,
        (),
    )
    assert _coverage._covered_by_other_paved_surface(
        issue, (first, second, miter, wedge), terrain
    )


def test_pbo_terrain_context_reads_landgrid_and_quantized_heights(
    tmp_path: Path,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.cpp").write_text(
        "class CfgWorlds { class wg_test { landGrid = 10; }; };\n",
        encoding="ascii",
    )
    write_rvw4(
        source / "wg_test.wrp",
        2,
        2,
        (0.0, 1.0, 2.0, 3.0),
        (0, 0, 0, 0),
        (r"wg_test\data\ground.pac",),
        (),
        height_scale=0.05,
    )
    pbo = tmp_path / "wg_test.pbo"
    pack_directory(source, pbo)

    terrain = _coverage._terrain_context(pbo)

    assert terrain is not None
    elevations, cells, cell_size = terrain
    assert elevations == (0.0, 1.0, 2.0, 3.0)
    assert cells == 2
    assert cell_size == 10.0
    assert math.isclose(
        _coverage._terrain_height(terrain, (5.0, 5.0)),
        1.5,
        abs_tol=1.0e-9,
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
