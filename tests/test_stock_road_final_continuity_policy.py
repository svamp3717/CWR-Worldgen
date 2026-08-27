# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import playability as _p
from cwr_worldgen import road_quality_policy as _quality
from cwr_worldgen import stock_road_curve_regularization_policy as _regular
from cwr_worldgen import stock_road_final_continuity_policy as _final
from cwr_worldgen import stock_road_geometry_policy as _geometry
from cwr_worldgen import stock_road_junction_policy as _junction


def _direction(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(heading_degrees)
    return math.sin(angle), math.cos(angle)


def _incident(heading_degrees: float, family: str = "sil"):
    return _junction._Incident(
        _direction(heading_degrees),
        family,
        rf"o\road\{family}25.p3d",
    )


def test_sampled_100m_arc_has_stable_ten_degree_tangent_change():
    points = [(500.0, 500.0)]
    for degrees in range(5, 21, 5):
        angle = math.radians(degrees)
        points.append(
            (
                500.0 + 100.0 * (1.0 - math.cos(angle)),
                500.0 + 100.0 * math.sin(angle),
            )
        )
    measure = _p._PolylineMeasure.create(points)
    ten_degree_point = points[2]
    start_distance = _final._distance_on_measure(measure, points[0])
    end_distance = _final._distance_on_measure(measure, ten_degree_point)
    start_heading = _final._smoothed_measure_heading(measure, start_distance)
    end_heading = _final._smoothed_measure_heading(measure, end_distance)

    assert abs(_p._signed_heading_delta(start_heading, end_heading) - 10.0) < 0.8
    assert _final._smoothed_curve_turn_error_degrees(
        points, points[0], ten_degree_point
    ) < 0.8


def _roadlab_curve_points():
    return (
        (500.2496, 300.0),
        (500.2496, 399.7504),
        (500.2496, 500.2496),
        (500.2496, 508.4992),
        (501.7504, 517.5008),
        (503.2512, 525.7504),
        (506.2496, 534.0),
        (509.2512, 542.2496),
        (513.7504, 549.7504),
        (518.2496, 557.2512),
        (523.5008, 564.0),
        (588.0, 641.2512),
        (651.7504, 717.7504),
    )


def _roadlab_curve_fixture(*, rounded: bool = False):
    points = _roadlab_curve_points()
    if rounded:
        points = _p._rounded_road_run(points)
    measure = _p._PolylineMeasure.create(points)
    pieces = _p.road_model_variants(r"o\road\sil25.p3d", 25.0)
    return measure, pieces


def _assert_coherent_100m_curve(fitted, *, minimum_count: int = 3):
    models = [piece.model_path.casefold() for piece, _start, _end in fitted]
    assert models.count(r"o\road\sil10 100.p3d") >= minimum_count, models
    assert r"o\road\sil10 75.p3d" not in models, models
    assert r"o\road\sil10 50.p3d" not in models, models


def _fit_fixture(measure, pieces):
    return _p._stock_piece_chain(
        measure,
        pieces,
        start_distance=0.0,
        preferred_end_distance=measure.total,
        minimum_end_distance=max(0.0, measure.total - 0.35),
        maximum_end_distance=measure.total + 3.125,
    )


def test_roadlab_curve_chain_prefers_100m_native_radius():
    # Use the post-normalization RoadLab coordinates, including the first 5-degree
    # sample whose sub-metre lateral offset was rounded onto the entry straight.
    measure, pieces = _roadlab_curve_fixture()
    _assert_coherent_100m_curve(_fit_fixture(measure, pieces))


def test_roadlab_curve_chain_stays_coherent_after_production_rounding():
    # This is the regression the generated WRP exposed.  The old per-corner
    # fillet changed the quantized 100 m arc into 75/50/straight/75 pieces even
    # though the selector handled the unrounded source correctly.  Production
    # rounding now snaps the full forty-degree bend to four native sections.
    measure, pieces = _roadlab_curve_fixture(rounded=True)
    _assert_coherent_100m_curve(_fit_fixture(measure, pieces), minimum_count=4)


def test_roadlab_regularized_arc_stays_inside_existing_fillet_corridor():
    points = _roadlab_curve_points()
    spans = _regular._sustained_curve_spans(points)
    assert (2, 10, 1) in spans
    arc = _regular._regularized_stock_arc(points, 2, 10, 1)
    assert arc is not None
    # Four exact ten-degree stock sections sampled every 2.5 degrees.
    assert len(arc) == 17
    maximum_allowed = _geometry._MAXIMUM_STOCK_FILLET_DEVIATION_METRES
    assert math.dist(arc[0], points[2]) <= maximum_allowed + 1.0e-9
    assert math.dist(arc[-1], points[10]) <= maximum_allowed + 1.0e-9

    maximum = 0.0
    for point in points[2:11]:
        nearest = min(
            _geometry._point_segment_distance(point, start, end)
            for start, end in zip(arc, arc[1:])
        )
        maximum = max(maximum, nearest)
    assert maximum <= maximum_allowed + 1.0e-9


def test_curve_regularizer_leaves_small_source_wiggle_to_existing_rounder():
    points = (
        (450.0, 1600.0),
        (451.0, 1650.0),
        (450.0, 1700.0),
        (449.0, 1750.0),
        (450.0, 1800.0),
    )
    assert _regular._sustained_curve_spans(points) == ()
    assert _regular._curve_regularized_rounded_run(points) == _regular._ORIGINAL_ROUNDED(points)


def test_curve_regularizer_leaves_one_hard_corner_to_existing_rounder():
    points = ((0.0, 0.0), (0.0, 50.0), (25.0, 93.3012701892))
    assert _regular._sustained_curve_spans(points) == ()
    assert _regular._curve_regularized_rounded_run(points) == _regular._ORIGINAL_ROUNDED(points)


def test_roadlab_curve_chain_stays_coherent_with_flat_terrain_context():
    measure, pieces = _roadlab_curve_fixture(rounded=True)
    spec = SimpleNamespace(cells=64, cell_size=50.0, road_connection_tolerance=0.35)
    context = _quality._Context(
        elevations=(24.0,) * (64 * 64),
        spec=spec,
        junctions={},
    )
    token = _quality._CONTEXT.set(context)
    try:
        fitted = _fit_fixture(measure, pieces)
    finally:
        _quality._CONTEXT.reset(token)
    _assert_coherent_100m_curve(fitted, minimum_count=4)


def test_45_degree_sil_t_uses_native_mesh_when_connector_stays_inside_branch():
    incidents = (
        _incident(90.0),
        _incident(270.0),
        _incident(45.0),
    )

    native = _final._same_family_paved_skew_t(incidents, "sil")

    assert native is not None
    assert native.model_path == r"o\road\kr_new_sil_sil_t.p3d"
    assert math.isclose(native.heading_degrees, 270.0, abs_tol=1.0e-9)
    assert math.isclose(native.maximum_heading_error_degrees, 45.0, abs_tol=1.0e-9)


def test_too_skewed_branch_keeps_legacy_fallback():
    incidents = (
        _incident(90.0),
        _incident(270.0),
        _incident(60.0),
    )

    assert _final._same_family_paved_skew_t(incidents, "sil") is None


def test_curve_seam_repair_underlays_are_disabled():
    marker = object()
    assert _final._disable_curve_seam_underlays(marker, (), None) is marker
