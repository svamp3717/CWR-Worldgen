# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_final_continuity_policy as _final
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
