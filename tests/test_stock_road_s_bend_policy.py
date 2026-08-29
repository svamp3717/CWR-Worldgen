# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_curve_policy as _curve
from cwr_worldgen import stock_road_model_geometry as _geometry
from cwr_worldgen import stock_road_s_bend_exact_policy as _s_exact
from cwr_worldgen import stock_road_s_bend_policy as _s_bend
from cwr_worldgen import stock_road_sharp_exact_policy as _exact
from cwr_worldgen import stock_road_sharp_turn_policy as _sharp


def _lundby20_bad_turn_points():
    # Real normalized D957 geometry from the bend around the historical
    # screenshot location near (879.27, 3535.87).
    return (
        (1053.750, 3564.750),
        (1012.500, 3559.500),
        (999.000, 3556.500),
        (984.000, 3552.000),
        (944.250, 3540.000),
        (921.000, 3534.000),
        (906.000, 3531.751),
        (894.000, 3531.000),
        (873.000, 3534.749),
        (859.500, 3534.749),
        (842.250, 3531.000),
        (825.750, 3525.750),
        (805.500, 3515.250),
        (751.500, 3471.000),
        (729.750, 3453.000),
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


def _production_window(measure):
    trim = 3.125 - 0.70
    cover = 3.125 + 0.15
    return trim, measure.total - trim, measure.total - cover, measure.total + 0.70


def _fit_with_production_junction_cover(chain, measure, pieces):
    start, preferred, minimum, maximum = _production_window(measure)
    return chain(
        measure,
        pieces,
        start_distance=start,
        preferred_end_distance=preferred,
        minimum_end_distance=minimum,
        maximum_end_distance=maximum,
    )


def _curve_count(fitted):
    return sum(
        _geometry.stock_curve_match(str(piece.model_path)) is not None
        for piece, _start, _end in fitted
    )


def _piece_tangents(item, source_points):
    piece, start, end = item
    chord = _sharp._heading(start, end)
    if _geometry.stock_curve_match(str(piece.model_path)) is None:
        return chord, chord
    reverse = _curve._curve_reverse_for_run(source_points, start, end)
    if reverse:
        return (chord + 5.0) % 360.0, (chord - 5.0) % 360.0
    return (chord - 5.0) % 360.0, (chord + 5.0) % 360.0


def _maximum_seam_tangent_error(fitted, source_points):
    maximum = 0.0
    for previous, current in zip(fitted, fitted[1:]):
        previous_end = _piece_tangents(previous, source_points)[1]
        current_start = _piece_tangents(current, source_points)[0]
        maximum = max(maximum, _p._heading_difference(previous_end, current_start))
    return maximum


def test_lundby20_local_s_bend_gets_native_curve_geometry():
    points = _lundby20_bad_turn_points()
    measure = _p._PolylineMeasure.create(_p._rounded_road_run(points))
    pieces = _p.road_model_variants(r"o\road\sil25.p3d", 25.0)

    locked = _s_bend._s_bend_replacement(measure, pieces)
    assert locked is not None

    baseline = _fit(_sharp._ORIGINAL_CHAIN, measure, pieces)
    fitted = _fit(_p._stock_piece_chain, measure, pieces)

    assert _curve_count(fitted) >= 3, [piece.model_path for piece, _a, _b in fitted]
    assert _curve_count(fitted) > _curve_count(baseline)

    seams = tuple(zip(fitted, fitted[1:]))
    reported = (879.27, 3535.87)
    previous, current = min(seams, key=lambda pair: math.dist(pair[0][2], reported))
    previous_curve = _geometry.stock_curve_match(str(previous[0].model_path)) is not None
    current_curve = _geometry.stock_curve_match(str(current[0].model_path)) is not None
    heading_change = _p._heading_difference(
        _sharp._heading(previous[1], previous[2]),
        _sharp._heading(current[1], current[2]),
    )
    assert previous_curve or current_curve or heading_change <= 1.0, (
        previous[0].model_path,
        current[0].model_path,
        heading_change,
        previous[2],
    )


def test_lundby20_production_s_bend_retains_exact_stock_actions():
    points = _lundby20_bad_turn_points()
    measure = _p._PolylineMeasure.create(_p._rounded_road_run(points))
    pieces = _p.road_model_variants(r"o\road\sil25.p3d", 25.0)
    start, preferred, _minimum, _maximum = _production_window(measure)

    source_points, entry_heading, source_exit_heading = _exact._measure_slice(
        measure, start, preferred
    )
    stock_exit_heading = _s_exact._quantised_exit_heading(
        entry_heading, source_exit_heading
    )
    locked_path = _s_exact._long_exact_s_bend_path(
        source_points, entry_heading, stock_exit_heading, pieces
    )
    assert locked_path is not None
    exact_actions = _s_exact._recover_exact_actions(locked_path, pieces)
    assert exact_actions is not None
    assert _curve_count(exact_actions) >= _s_exact.MINIMUM_EXACT_S_BEND_CURVES
    assert (
        _s_exact._maximum_internal_tangent_error(exact_actions, source_points)
        <= _s_exact.MAXIMUM_EXACT_INTERNAL_TANGENT_ERROR_DEGREES
    )

    baseline = _fit_with_production_junction_cover(_s_exact._ORIGINAL_CHAIN, measure, pieces)
    fitted = _fit_with_production_junction_cover(_p._stock_piece_chain, measure, pieces)
    baseline_error = _maximum_seam_tangent_error(baseline, measure.points)
    fitted_error = _maximum_seam_tangent_error(fitted, measure.points)

    assert _s_exact._INSTALLED
    assert _curve_count(fitted) >= _s_exact.MINIMUM_EXACT_S_BEND_CURVES
    assert _curve_count(fitted) >= _curve_count(baseline)
    assert fitted != baseline
    assert baseline_error - fitted_error >= _s_exact.MINIMUM_TANGENT_IMPROVEMENT_DEGREES
    assert fitted_error <= _s_exact.MAXIMUM_EXACT_INTERNAL_TANGENT_ERROR_DEGREES
    for previous, current in zip(fitted, fitted[1:]):
        assert math.dist(previous[2], current[1]) <= 1.0e-4


def test_s_bend_policy_does_not_apply_to_ces_roads():
    points = _lundby20_bad_turn_points()
    measure = _p._PolylineMeasure.create(_p._rounded_road_run(points))
    pieces = _p.road_model_variants(r"o\road\ces25.p3d", 25.0)

    assert _s_bend._s_bend_replacement(measure, pieces) is None
