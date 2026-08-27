# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_model_geometry as _model_geometry
from cwr_worldgen import stock_road_sharp_exact_policy as _exact
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


def _lundby_junction_to_junction_turn_points():
    # The production network really splits D957 at both the track junction at
    # 3206.25/3176.25 and the junction at 3126.75/3188.999. This is therefore
    # the exact curved run emitted by the normal world build, not a synthetic
    # enlargement around the reported coordinates.
    return _lundby_sharp_turn_points()[1:-1]


def _fit(chain, measure, pieces):
    return chain(
        measure,
        pieces,
        start_distance=0.0,
        preferred_end_distance=measure.total,
        minimum_end_distance=max(0.0, measure.total - 0.35),
        maximum_end_distance=measure.total + 3.125,
    )


def _fit_with_production_junction_cover(chain, measure, pieces):
    # A stock sil6 junction cap is 6.25 m long. Production trims each branch to
    # 3.125 - 0.70 m and accepts coverage out to 3.125 + 0.15 m.
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


def _is_curve(piece) -> bool:
    return _model_geometry.stock_curve_match(str(piece.model_path)) is not None


def _curve_count(fitted) -> int:
    return sum(_is_curve(piece) for piece, _start, _end in fitted)


def _right_turn_tangents(item):
    piece, start, end = item
    chord_heading = _sharp._heading(start, end)
    if _is_curve(piece):
        # Stock curve connectors traverse a ten-degree right turn. Their chord
        # heading is exactly halfway between the two endpoint tangents.
        return (chord_heading - 5.0) % 360.0, (chord_heading + 5.0) % 360.0
    return chord_heading, chord_heading


def _seam_tangent_errors(fitted):
    errors = []
    for previous, current in zip(fitted, fitted[1:]):
        previous_end = _right_turn_tangents(previous)[1]
        current_start = _right_turn_tangents(current)[0]
        errors.append(_p._heading_difference(previous_end, current_start))
    return tuple(errors)


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


def test_reported_lundby_seams_are_not_straight_to_straight_miters():
    """The two in-game bug coordinates must no longer land on a faceted miter."""

    points = _lundby_sharp_turn_points()
    measure = _p._PolylineMeasure.create(_p._rounded_road_run(points))
    pieces = _p.road_model_variants(r"o\road\sil25.p3d", 25.0)
    fitted = _fit(_p._stock_piece_chain, measure, pieces)
    seams = tuple(zip(fitted, fitted[1:]))

    for reported in ((3211.0, 3176.0), (3143.0, 3187.0)):
        previous, current = min(
            seams,
            key=lambda pair: math.dist(pair[0][2], reported),
        )
        assert math.dist(previous[2], current[1]) <= 1.0e-4
        previous_heading = _sharp._heading(previous[1], previous[2])
        current_heading = _sharp._heading(current[1], current[2])
        heading_change = _p._heading_difference(previous_heading, current_heading)
        assert (
            _is_curve(previous[0])
            or _is_curve(current[0])
            or heading_change <= 1.0
        ), (
            reported,
            previous[0].model_path,
            current[0].model_path,
            heading_change,
            previous[2],
        )


def test_lundby_production_split_has_no_exposed_tangent_miters():
    """The actual split D957 run must have tangent-continuous interior seams.

    Matching only centreline endpoints is insufficient: two road rectangles can
    share the same centre point while their outer edges diverge into a triangular
    grass wedge. The production replacement therefore has to match the tangent
    on both sides of every exposed internal connector.
    """

    points = _lundby_junction_to_junction_turn_points()
    measure = _p._PolylineMeasure.create(_p._rounded_road_run(points))
    pieces = _p.road_model_variants(r"o\road\sil25.p3d", 25.0)

    baseline = _fit_with_production_junction_cover(_exact._ORIGINAL_CHAIN, measure, pieces)
    fitted = _fit_with_production_junction_cover(_p._stock_piece_chain, measure, pieces)

    assert _curve_count(fitted) >= 3, [piece.model_path for piece, _a, _b in fitted]
    assert _curve_count(fitted) > _curve_count(baseline)
    assert len(fitted) <= len(baseline)

    # The old greedy result contains several 2-5 degree tangent discontinuities,
    # which are the visible half-road grass wedges reported in CWA.
    assert max(_seam_tangent_errors(baseline)) > 2.0

    for previous, current in zip(fitted, fitted[1:]):
        assert math.dist(previous[2], current[1]) <= 1.0e-4
    assert max(_seam_tangent_errors(fitted), default=0.0) <= 1.0e-4

    # Do not cure the visual seam by moving the road somewhere else. Every
    # connector stays in the same narrow source-centreline corridor used by the
    # sharp-turn beam search.
    for _piece, start, end in fitted:
        for point in (start, end):
            nearest = _sharp._nearest_forward(measure, point, 0.0, measure.total)
            assert nearest is not None
            assert nearest[0] <= _sharp._MAXIMUM_LOCKED_CORRIDOR_METRES + 1.0e-9


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
