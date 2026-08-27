# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_model_geometry as _model_geometry
from cwr_worldgen import stock_road_sharp_turn_policy as _sharp


def _lundby_sharp_turn_points():
    # Real normalized D957 geometry from the Lundby source bundle around the
    # two reported visible seam locations near (3211,3176) and (3143,3187).
    return (
        (3223.5000, 3181.5000),
        (3206.2500, 3176.2500),
        (3183.0000, 3174.0000),
        (3161.2500, 3176.2500),
        (3141.0000, 3181.5000),
        (3126.7500, 3188.9990),
        (3085.5000, 3213.7500),
    )


def _fit(chain, measure, pieces):
    return chain(
        measure,
        pieces,
        start_distance=0.0,
        preferred_end_distance=measure.total,
        minimum_end_distance=max(0.0, measure.total - 0.35),
        maximum_end_distance=measure.total + 3.125,
    )


def _curve_count(fitted) -> int:
    return sum(
        _model_geometry.stock_curve_match(str(piece.model_path)) is not None
        for piece, _start, _end in fitted
    )


def test_lundby_sharp_asphalt_turn_uses_native_curves_instead_of_sil6_facets():
    points = _lundby_sharp_turn_points()
    rounded = _p._rounded_road_run(points)
    measure = _p._PolylineMeasure.create(rounded)
    pieces = _p.road_model_variants(r"o\road\sil25.p3d", 25.0)

    baseline = _fit(_sharp._ORIGINAL_CHAIN, measure, pieces)
    fitted = _fit(_p._stock_piece_chain, measure, pieces)

    assert _curve_count(fitted) >= 3, [piece.model_path for piece, _a, _b in fitted]
    assert _curve_count(fitted) > _curve_count(baseline)
    for previous, current in zip(fitted, fitted[1:]):
        assert math.dist(previous[2], current[1]) <= 1.0e-4


def test_lundby_locked_path_stays_inside_narrow_source_corridor():
    points = _lundby_sharp_turn_points()
    pieces = _p.road_model_variants(r"o\road\sil25.p3d", 25.0)
    entry_heading = _sharp._heading(points[0], points[1])
    exit_heading = _sharp._heading(points[-2], points[-1])

    locked = _sharp._beam_stock_path(
        points[:-1],
        1,
        entry_heading,
        exit_heading,
        pieces,
    )

    assert locked is not None
    source = _p._PolylineMeasure.create(points[:-1])
    maximum = 0.0
    for point in locked:
        nearest = _sharp._nearest_forward(source, point, 0.0, source.total)
        assert nearest is not None
        maximum = max(maximum, nearest[0])
    assert maximum <= _sharp._MAXIMUM_LOCKED_CORRIDOR_METRES + 1.0e-9


def test_sharp_turn_policy_does_not_apply_to_ces_family():
    points = _lundby_sharp_turn_points()
    measure = _p._PolylineMeasure.create(_p._rounded_road_run(points))
    pieces = _p.road_model_variants(r"o\road\ces25.p3d", 25.0)

    baseline = _fit(_sharp._ORIGINAL_CHAIN, measure, pieces)
    fitted = _fit(_p._stock_piece_chain, measure, pieces)

    assert fitted == baseline
