# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_model_geometry as _geometry
from cwr_worldgen import stock_road_s_bend_policy as _s_bend
from cwr_worldgen import stock_road_sharp_turn_policy as _sharp


def _lundby20_bad_turn_points():
    # Real normalized D957 source geometry around the screenshot at roughly
    # (879.27, 3535.87). The road first bends right, then reverses immediately.
    # Lundby20 rendered this section almost entirely from independently rotated
    # sil6/sil12 rectangles, making the underlay helpers visibly look like
    # mismatched road pieces along the inside edge.
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


def _curve_count(fitted):
    return sum(
        _geometry.stock_curve_match(str(piece.model_path)) is not None
        for piece, _start, _end in fitted
    )


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

    # The screenshot location itself must not remain a multi-degree
    # straight-to-straight miter. A native curve on either side is acceptable;
    # otherwise the two visible straight slabs must be essentially collinear.
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


def test_s_bend_policy_does_not_apply_to_ces_roads():
    points = _lundby20_bad_turn_points()
    measure = _p._PolylineMeasure.create(_p._rounded_road_run(points))
    pieces = _p.road_model_variants(r"o\road\ces25.p3d", 25.0)

    assert _s_bend._s_bend_replacement(measure, pieces) is None
