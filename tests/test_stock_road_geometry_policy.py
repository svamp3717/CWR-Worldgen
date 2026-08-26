# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

from cwr_worldgen import playability as _p
from cwr_worldgen.stock_road_geometry_policy import (
    _MAXIMUM_STOCK_FILLET_DEVIATION_METRES,
    _MAXIMUM_TANGENT_TURN_ERROR_DEGREES,
    _curve_turn_error_degrees,
)


def _arc(radius: float, turn_degrees: float, samples: int = 101):
    angle = math.radians(turn_degrees)
    return tuple(
        (
            radius * (1.0 - math.cos(angle * index / (samples - 1))),
            radius * math.sin(angle * index / (samples - 1)),
        )
        for index in range(samples)
    )


def test_ten_degree_source_arc_is_safe_for_native_curve_connectors():
    run = _arc(50.0, 10.0)
    error = _curve_turn_error_degrees(run, run[0], run[-1])

    assert error < _MAXIMUM_TANGENT_TURN_ERROR_DEGREES


def test_irregular_source_turn_is_rejected_before_it_can_leave_a_large_wedge():
    run = _arc(50.0, 15.0)
    error = _curve_turn_error_degrees(run, run[0], run[-1])

    assert error > _MAXIMUM_TANGENT_TURN_ERROR_DEGREES


def test_micro_bend_is_simplified_instead_of_creating_rotated_straight_slabs():
    # The middle point is only 20 cm off the direct centreline. Keeping it would
    # rotate neighbouring stock rectangles for no useful geographic fidelity.
    rounded = _p._rounded_road_run(((0.0, 0.0), (0.20, 20.0), (0.0, 40.0)))

    assert rounded == ((0.0, 0.0), (0.0, 40.0))


def test_gentle_corner_uses_constant_radius_fillet_inside_road_corridor():
    # A ten-degree, long-segment bend can use the native 100 m radius. Measure
    # the generated samples against the hard-corner source and ensure the local
    # relaxation stays within the conservative sub-metre corridor.
    length = 100.0
    turn = math.radians(10.0)
    points = (
        (0.0, -length),
        (0.0, 0.0),
        (math.sin(turn) * length, math.cos(turn) * length),
    )
    rounded = _p._rounded_road_run(points)

    assert len(rounded) > 4
    assert rounded[0] == points[0]
    assert rounded[-1] == points[-1]
    maximum_cut = 0.0
    for point in rounded[1:-1]:
        distance = min(
            _p._point_segment_distance(point, points[0], points[1]),
            _p._point_segment_distance(point, points[1], points[2]),
        )
        maximum_cut = max(maximum_cut, distance)
    assert maximum_cut <= _MAXIMUM_STOCK_FILLET_DEVIATION_METRES + 1.0e-6
