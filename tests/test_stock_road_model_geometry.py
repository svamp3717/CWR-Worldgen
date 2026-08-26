# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

from cwr_worldgen.stock_road_model_geometry import (
    STOCK_JUNCTION_CONNECTOR_RADIUS_METRES,
    STOCK_STRAIGHT_LENGTHS_METRES,
    native_junction_intersection_offset,
    solve_planar_connector_transform,
    stock_curve_connectors,
    stock_straight_length,
    transform_local,
)


def test_memory_lod_straight_connector_lengths_are_used_verbatim():
    assert STOCK_STRAIGHT_LENGTHS_METRES == {25: 25.0, 12: 12.5, 6: 6.25}
    assert stock_straight_length(r"O\Road\sil25.p3d") == 25.0
    assert stock_straight_length(r"O\Road\sil12.p3d") == 12.5
    assert stock_straight_length(r"O\Road\sil6.p3d") == 6.25
    assert stock_straight_length(r"custom\road6.p3d") is None


def test_curve_connectors_include_real_chord_heading_and_origin_offset():
    geometry = stock_curve_connectors(r"O\Road\sil10 75.p3d")

    assert geometry is not None
    expected_chord = 2.0 * 75.0 * math.sin(math.radians(5.0))
    assert math.isclose(geometry.chord_length_metres, expected_chord, abs_tol=1.0e-12)
    assert math.isclose(math.dist(geometry.begin, geometry.end), expected_chord, abs_tol=1.0e-12)
    assert math.isclose(geometry.local_chord_heading_degrees, 5.0, abs_tol=1.0e-12)

    midpoint = (
        (geometry.begin[0] + geometry.end[0]) * 0.5,
        (geometry.begin[1] + geometry.end[1]) * 0.5,
    )
    assert not math.isclose(midpoint[0], 0.0, abs_tol=1.0e-6)
    assert not math.isclose(midpoint[1], 0.0, abs_tol=1.0e-6)


def test_connector_transform_maps_both_curve_endpoints_exactly():
    geometry = stock_curve_connectors(r"O\Road\ces10 50.p3d")
    assert geometry is not None

    world_begin = (100.0, 200.0)
    chord_heading = 37.0
    angle = math.radians(chord_heading)
    world_end = (
        world_begin[0] + math.sin(angle) * geometry.chord_length_metres,
        world_begin[1] + math.cos(angle) * geometry.chord_length_metres,
    )
    origin, heading = solve_planar_connector_transform(
        world_begin, world_end, geometry.begin, geometry.end
    )

    assert math.dist(transform_local(geometry.begin, origin, heading), world_begin) < 1.0e-9
    assert math.dist(transform_local(geometry.end, origin, heading), world_end) < 1.0e-9
    assert math.isclose(heading, chord_heading - 5.0, abs_tol=1.0e-9)


def test_native_t_logical_center_is_not_the_model_origin():
    assert STOCK_JUNCTION_CONNECTOR_RADIUS_METRES == 6.25
    sil_offset = native_junction_intersection_offset(r"o\road\kr_new_sil_ces_t.p3d")
    asf_offset = native_junction_intersection_offset(r"o\road\kr_new_asf_ces_t.p3d")
    cross_offset = native_junction_intersection_offset(r"o\road\kr_new_silxsil.p3d")

    assert sil_offset is not None and math.isclose(sil_offset[0], 0.85, abs_tol=1.0e-12)
    assert asf_offset is not None and math.isclose(asf_offset[0], 1.375, abs_tol=1.0e-12)
    assert cross_offset == (0.0, 0.0)
