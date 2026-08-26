# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

from cwr_worldgen import playability as _p
from cwr_worldgen import road_quality_policy as _quality
from cwr_worldgen.stock_road_curve_policy import (
    _curve_reverse_for_run,
    _piece_length,
    _stock_curve_model_paths,
)
from cwr_worldgen.stock_road_geometry_policy import stock_curve_geometry


def _arc(radius: float, arc_length: float, samples: int = 33):
    angle = arc_length / radius
    return tuple(
        (
            radius * (1.0 - math.cos(angle * index / (samples - 1))),
            radius * math.sin(angle * index / (samples - 1)),
        )
        for index in range(samples)
    )


def test_stock_curve_models_are_exposed_as_trusted_road_variants():
    paths = _p.road_model_variant_paths(r"O\Road\sil25.p3d", 25.0)

    assert paths[:3] == (
        r"O\Road\sil25.p3d",
        r"O\Road\sil12.p3d",
        r"O\Road\sil6.p3d",
    )
    assert paths[3:] == (
        r"O\Road\sil10 25.p3d",
        r"O\Road\sil10 50.p3d",
        r"O\Road\sil10 75.p3d",
        r"O\Road\sil10 100.p3d",
    )
    assert _stock_curve_model_paths(r"O\Road\ces25.p3d")[0] == r"O\Road\ces10 25.p3d"
    assert _stock_curve_model_paths(r"custom\road25.p3d") == ()


def test_stock_curve_geometry_is_ten_degrees_not_ten_metres():
    chord, turn, sagitta = stock_curve_geometry(50, 1.0)

    assert math.isclose(turn, 10.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(
        chord,
        2.0 * 50.0 * math.sin(math.radians(5.0)),
        rel_tol=1e-12,
    )
    assert math.isclose(
        sagitta,
        50.0 * (1.0 - math.cos(math.radians(5.0))),
        rel_tol=1e-12,
    )
    assert math.isclose(
        _piece_length(r"O\Road\sil10 50.p3d", 24.5),
        chord,
        rel_tol=1e-12,
    )


def test_stock_models_use_real_connectors_even_when_spacing_setting_is_24_5():
    pieces = _p.road_model_variants(r"O\Road\sil25.p3d", 24.5)

    assert [(piece.nominal_length, piece.length_metres) for piece in pieces] == [
        (25, 25.0),
        (12, 12.0),
        (6, 6.0),
    ]

    # The P3D itself is never scaled by WRP placement, so curve radii/chords are
    # likewise physical model dimensions rather than 24.5/25-shrunk spacing.
    expected = {
        25: 2.0 * 25.0 * math.sin(math.radians(5.0)),
        50: 2.0 * 50.0 * math.sin(math.radians(5.0)),
        75: 2.0 * 75.0 * math.sin(math.radians(5.0)),
        100: 2.0 * 100.0 * math.sin(math.radians(5.0)),
    }
    for radius, expected_chord in expected.items():
        chord, turn, _sagitta = stock_curve_geometry(radius, 24.5 / 25.0)
        assert math.isclose(chord, expected_chord, rel_tol=1e-12)
        assert turn == 10.0


def test_stock_curve_chain_prefers_native_radius_on_matching_bend():
    measure = _p._PolylineMeasure.create(_arc(50.0, 30.0))
    pieces = _p.road_model_variants(r"O\Road\sil25.p3d", 24.5)

    token = _quality._CONTEXT.set(None)
    try:
        fitted = _p._stock_piece_chain(
            measure,
            pieces,
            start_distance=0.0,
            preferred_end_distance=measure.total,
            minimum_end_distance=measure.total - 3.0,
            maximum_end_distance=measure.total + 3.0,
        )
    finally:
        _quality._CONTEXT.reset(token)

    models = [piece.model_path for piece, _start, _end in fitted]
    assert r"O\Road\sil10 50.p3d" in models
    assert models.count(r"O\Road\sil6.p3d") < len(models)

    for piece, start, end in fitted:
        if "10 " not in piece.model_path:
            continue
        actual_chord = math.dist(start, end)
        expected_chord = _piece_length(piece.model_path, 24.5)
        assert math.isclose(actual_chord, expected_chord, rel_tol=0.0, abs_tol=1e-6)


def test_stock_curve_chain_keeps_straights_on_straight_road():
    measure = _p._PolylineMeasure.create(((0.0, 0.0), (0.0, 60.0)))
    pieces = _p.road_model_variants(r"O\Road\ces25.p3d", 24.5)

    token = _quality._CONTEXT.set(None)
    try:
        fitted = _p._stock_piece_chain(
            measure,
            pieces,
            start_distance=0.0,
            preferred_end_distance=60.0,
            minimum_end_distance=57.0,
            maximum_end_distance=63.0,
        )
    finally:
        _quality._CONTEXT.reset(token)

    assert fitted
    assert all("10 " not in piece.model_path for piece, _start, _end in fitted)


def test_curve_orientation_flips_for_opposite_turn_handedness():
    right = _arc(50.0, math.radians(20.0) * 50.0)
    left = tuple((-x, z) for x, z in right)

    assert not _curve_reverse_for_run(right, right[0], right[-1])
    assert _curve_reverse_for_run(left, left[0], left[-1])
