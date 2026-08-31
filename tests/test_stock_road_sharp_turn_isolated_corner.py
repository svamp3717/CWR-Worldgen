# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

from cwr_worldgen import playability as _p
from cwr_worldgen.road_pipeline import ROAD_PIPELINE_STAGES
from cwr_worldgen import stock_road_model_geometry as _model_geometry
from cwr_worldgen import stock_road_sharp_turn_policy as _sharp


def _isolated_corner(turn_degrees: float):
    """Five points with one turn and quiet boundary segments on both sides."""

    incoming = (
        (0.0, -25.0),
        (0.0, -12.5),
        (0.0, 0.0),
    )
    angle = math.radians(float(turn_degrees))
    direction = (math.sin(angle), math.cos(angle))
    return (
        *incoming,
        (direction[0] * 12.5, direction[1] * 12.5),
        (direction[0] * 25.0, direction[1] * 25.0),
    )


def _fit_with_production_cover(chain, measure, pieces):
    trim = 3.125 - 0.70
    cover = 3.125 + 0.15
    return chain(
        measure,
        pieces,
        start_distance=trim,
        preferred_end_distance=measure.total - trim,
        minimum_end_distance=measure.total - cover,
        maximum_end_distance=measure.total + 0.70,
    )


def _curve_count(fitted) -> int:
    return sum(
        _model_geometry.stock_curve_match(str(piece.model_path)) is not None
        for piece, _start, _end in fitted
    )


def test_isolated_twelve_degree_corner_is_handed_to_curve_beam() -> None:
    points = _isolated_corner(12.0)

    spans = _sharp._sharp_turn_spans(points)

    assert (1, 3, 1) in spans


def test_isolated_thirty_two_degree_corner_is_handed_to_curve_beam() -> None:
    points = _isolated_corner(32.0)

    spans = _sharp._sharp_turn_spans(points)

    assert (1, 3, 1) in spans


def test_isolated_twelve_degree_corner_changes_production_fit_to_native_curve() -> None:
    points = _isolated_corner(12.0)
    measure = _p._PolylineMeasure.create(_p._rounded_road_run(points))
    pieces = _p.road_model_variants(r"o\road\sil25.p3d", 25.0)

    baseline = _fit_with_production_cover(_sharp._ORIGINAL_CHAIN, measure, pieces)
    fitted = _fit_with_production_cover(_p._stock_piece_chain, measure, pieces)

    assert _curve_count(fitted) > _curve_count(baseline), [
        piece.model_path for piece, _start, _end in fitted
    ]
    for previous, current in zip(fitted, fitted[1:]):
        assert math.dist(previous[2], current[1]) <= 1.0e-4


def test_isolated_thirty_two_degree_corner_changes_production_fit_to_native_curves() -> None:
    points = _isolated_corner(32.0)
    measure = _p._PolylineMeasure.create(_p._rounded_road_run(points))
    pieces = _p.road_model_variants(r"o\road\sil25.p3d", 25.0)

    baseline = _fit_with_production_cover(_sharp._ORIGINAL_CHAIN, measure, pieces)
    fitted = _fit_with_production_cover(_p._stock_piece_chain, measure, pieces)

    assert _curve_count(fitted) >= _curve_count(baseline) + 2, [
        piece.model_path for piece, _start, _end in fitted
    ]
    for previous, current in zip(fitted, fitted[1:]):
        assert math.dist(previous[2], current[1]) <= 1.0e-4


def test_small_heading_noise_is_not_promoted_to_stock_curve_span() -> None:
    points = _isolated_corner(5.0)

    assert _sharp._isolated_single_vertex_spans(points) == ()


def test_boundary_corner_is_left_to_endpoint_or_junction_fitting() -> None:
    points = _isolated_corner(12.0)[1:]

    assert _sharp._isolated_single_vertex_spans(points) == ()


def test_existing_sustained_span_wins_over_single_vertex_augmentation() -> None:
    points = _isolated_corner(12.0)
    existing = ((0, 3, 1),)

    assert _sharp._isolated_single_vertex_spans(points, existing) == ()


def test_single_vertex_behavior_is_owned_by_sharp_turn_stage() -> None:
    assert _sharp._INSTALLED
    assert "single_vertex_bend" not in ROAD_PIPELINE_STAGES
