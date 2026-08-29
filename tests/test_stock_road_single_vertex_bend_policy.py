# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

from cwr_worldgen import stock_road_sharp_turn_policy as _sharp
from cwr_worldgen import stock_road_single_vertex_bend_policy as _single


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


def test_isolated_twelve_degree_corner_is_handed_to_curve_beam() -> None:
    points = _isolated_corner(12.0)

    spans = _sharp._sharp_turn_spans(points)

    assert (1, 3, 1) in spans


def test_isolated_thirty_two_degree_corner_is_handed_to_curve_beam() -> None:
    points = _isolated_corner(32.0)

    spans = _sharp._sharp_turn_spans(points)

    assert (1, 3, 1) in spans


def test_small_heading_noise_is_not_promoted_to_stock_curve_span() -> None:
    points = _isolated_corner(5.0)

    assert _single._isolated_single_vertex_spans(points) == ()


def test_boundary_corner_is_left_to_endpoint_or_junction_fitting() -> None:
    points = _isolated_corner(12.0)[1:]

    assert _single._isolated_single_vertex_spans(points) == ()


def test_existing_sustained_span_wins_over_single_vertex_augmentation() -> None:
    points = _isolated_corner(12.0)
    existing = ((0, 3, 1),)

    assert _single._isolated_single_vertex_spans(points, existing) == ()


def test_single_vertex_policy_is_active_on_package_import() -> None:
    assert _single._INSTALLED
    assert _sharp._sharp_turn_spans is _single._single_vertex_sharp_turn_spans
