# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_curve_usage_policy as _usage
from cwr_worldgen import stock_road_inspector_candidate_policy as _candidate
from cwr_worldgen import stock_road_junction_policy as _junction
from cwr_worldgen import stock_road_model_geometry as _geometry
from cwr_worldgen import stock_road_sharp_exact_policy as _exact
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


def test_curve_first_success_does_not_call_straight_baseline(monkeypatch):
    measure = _p._PolylineMeasure.create(
        ((0.0, 0.0), (0.0, 10.0), (1.75, 20.0), (5.1, 29.4), (9.8, 38.2))
    )
    pieces = _p.road_model_variants(r"o\road\sil25.p3d", 25.0)
    geometry = _geometry.stock_curve_connectors(r"o\road\sil10 100.p3d")
    assert geometry is not None
    curve_piece = _p._RoadPiece(
        r"o\road\sil10 100.p3d",
        geometry.chord_length_metres,
        10,
    )
    exact = ((curve_piece, measure.points[0], measure.points[-1]),)
    preferred = measure.total - 0.5

    def baseline_should_not_run(*args, **kwargs):
        raise AssertionError("faceted baseline ran before successful native curve fit")

    monkeypatch.setattr(_usage, "_ORIGINAL_CHAIN", baseline_should_not_run)
    monkeypatch.setattr(_usage, "_dominant_bend", lambda points: (1, 20.0))
    monkeypatch.setattr(
        _exact,
        "_measure_slice",
        lambda current, start, end: (current.points, 0.0, 20.0),
    )
    monkeypatch.setattr(
        _exact,
        "_quantised_stock_exit_heading",
        lambda entry, source_exit, sign: 20.0,
    )
    monkeypatch.setattr(
        _sharp,
        "_beam_stock_path",
        lambda source, sign, entry, exit, available: (
            measure.points[0],
            measure.points[-1],
        ),
    )
    monkeypatch.setattr(_usage, "_path_is_obstacle_safe", lambda path: True)
    monkeypatch.setattr(_exact, "_recover_exact_actions", lambda path, available, sign: exact)
    monkeypatch.setattr(_exact, "_curve_count", lambda fitted: 1)
    monkeypatch.setattr(_usage, "_maximum_internal_tangent_error", lambda fitted, sign: 0.0)
    monkeypatch.setattr(
        _sharp,
        "_nearest_forward",
        lambda current, point, minimum, maximum: (0.0, preferred),
    )

    fitted = _usage._curve_promotion_chain(
        measure,
        pieces,
        start_distance=0.0,
        preferred_end_distance=preferred,
        minimum_end_distance=preferred - 1.0,
        maximum_end_distance=measure.total,
    )

    assert fitted == exact


def test_curve_usage_and_inspector_search_remain_inside_endpoint_wrapper():
    assert _candidate._ORIGINAL_PIECE_CHAIN is _usage._curve_promotion_chain
    assert _junction._ORIGINAL_ENDPOINT_CHAIN is _candidate._candidate_exact_curve_chain
    assert _p._stock_piece_chain is _junction._junction_endpoint_chain
