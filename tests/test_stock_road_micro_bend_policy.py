# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import stock_road_micro_bend_policy as _micro
from cwr_worldgen import stock_road_sharp_turn_policy as _sharp


def _pieces():
    return (
        SimpleNamespace(
            model_path=r"o\road\sil25.p3d",
            length_metres=25.0,
            nominal_length=25,
        ),
        SimpleNamespace(
            model_path=r"o\road\sil12.p3d",
            length_metres=12.5,
            nominal_length=12,
        ),
        SimpleNamespace(
            model_path=r"o\road\sil6.p3d",
            length_metres=6.25,
            nominal_length=6,
        ),
    )


def _ten_degree_radius_100_arc():
    radius = 100.0
    centre = (radius, 0.0)
    start_vector = (-radius, 0.0)
    points = []
    for degrees in (0.0, 2.5, 5.0, 7.5, 10.0):
        angle = -math.radians(degrees)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotated = (
            cosine * start_vector[0] - sine * start_vector[1],
            sine * start_vector[0] + cosine * start_vector[1],
        )
        points.append((centre[0] + rotated[0], centre[1] + rotated[1]))
    return tuple(points)


def test_micro_bend_beam_accepts_one_native_curve():
    source = _ten_degree_radius_100_arc()
    pieces = _pieces()

    # The original sharp-turn beam deliberately requires two curves and cannot
    # finish this one-section bend. The late micro-bend wrapper should.
    assert _micro._ORIGINAL_BEAM is not None
    assert _micro._ORIGINAL_BEAM(source, 1, 0.0, 10.0, pieces) is None

    locked = _sharp._beam_stock_path(source, 1, 0.0, 10.0, pieces)
    assert locked is not None
    exact = _sharp._recover_exact_actions(locked, pieces, 1)
    assert exact is not None
    assert _sharp._curve_count(exact) == 1
    assert exact[0][0].model_path.casefold() == r"o\road\sil10 100.p3d"


def test_micro_bend_policy_lowers_the_sustained_turn_gate():
    assert math.isclose(
        _sharp._MINIMUM_SUSTAINED_TOTAL_TURN_DEGREES,
        _micro.MINIMUM_MICRO_BEND_TOTAL_TURN_DEGREES,
        abs_tol=1.0e-12,
    )
    assert _sharp._MINIMUM_SUSTAINED_TOTAL_TURN_DEGREES < 10.0
