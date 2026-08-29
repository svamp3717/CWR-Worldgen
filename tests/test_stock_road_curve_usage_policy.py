# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_curve_usage_policy as _usage
from cwr_worldgen import stock_road_junction_endpoint_policy as _junction_endpoint
from cwr_worldgen import stock_road_sharp_turn_policy as _sharp


def _coherent_twenty_degree_bend(radius: float = 75.0):
    points = [(0.0, -20.0)]
    for degrees in range(0, 21, 5):
        angle = math.radians(degrees)
        points.append((
            radius * (1.0 - math.cos(angle)),
            radius * math.sin(angle),
        ))
    end = points[-1]
    tangent = (math.sin(math.radians(20.0)), math.cos(math.radians(20.0)))
    points.append((end[0] + tangent[0] * 20.0, end[1] + tangent[1] * 20.0))
    return tuple(points)


def test_curve_usage_gate_accepts_coherent_moderate_paved_bend():
    bend = _usage._dominant_bend(_coherent_twenty_degree_bend())

    assert bend is not None
    _sign, total = bend
    assert 15.0 <= total <= 25.0


def test_curve_usage_gate_rejects_s_bend_direction_reversal():
    points = (
        (0.0, 0.0),
        (0.0, 20.0),
        (4.0, 40.0),
        (0.0, 60.0),
        (0.0, 80.0),
    )

    assert _usage._dominant_bend(points) is None


def test_exact_native_curve_sequence_has_zero_internal_tangent_error():
    pieces = _p.road_model_variants(r"o\road\sil25.p3d", 25.0)
    family = _sharp._paved_family(pieces)
    assert family is not None
    actions = [
        action
        for action in _sharp._actions(pieces, *family, 1)
        if action.turn_sign == 1 and action.radius_metres == 100.0
    ]
    assert len(actions) == 1
    action = actions[0]

    first_state = _sharp._State(0.0, 0.0, 0.0, 0.0, 0.0, (), 0)
    first_end, first_heading, _samples = _sharp._advance(first_state, action)
    second_state = _sharp._State(
        0.0,
        first_end[0],
        first_end[1],
        first_heading,
        0.0,
        (),
        1,
    )
    second_end, _second_heading, _samples = _sharp._advance(second_state, action)
    fitted = (
        (action.piece, (0.0, 0.0), first_end),
        (action.piece, first_end, second_end),
    )

    assert _usage._maximum_internal_tangent_error(fitted, 1) <= 1.0e-9


def test_curve_usage_policy_remains_inside_final_endpoint_wrapper():
    assert _junction_endpoint._ORIGINAL_CHAIN is _usage._curve_promotion_chain
    assert _p._stock_piece_chain is _junction_endpoint._junction_endpoint_chain
