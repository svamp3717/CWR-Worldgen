# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_junction_policy as _junction
from cwr_worldgen import stock_road_model_geometry as _geometry
from cwr_worldgen import stock_road_visual_finish_policy as _finish


def test_skew_t_legacy_cap_uses_continuous_main_axis():
    incidents = (
        _junction._Incident((1.0, 0.0), "sil", r"o\road\sil25.p3d"),
        _junction._Incident((-1.0, 0.0), "sil", r"o\road\sil25.p3d"),
        _junction._Incident((1.0, 1.0), "sil", r"o\road\sil25.p3d"),
    )

    heading = _finish._dominant_cap_heading(incidents, "sil")

    assert heading is not None
    assert _finish._axis_heading_difference(heading, 90.0) < 1.0e-9
    assert _finish._axis_heading_difference(heading, 45.0) > 40.0


def test_native_curve_tangents_meet_at_the_rendered_edge(monkeypatch):
    first_start = (0.0, 0.0)
    first_end = (math.sin(math.radians(5.0)) * 10.0, math.cos(math.radians(5.0)) * 10.0)
    second_start = first_end
    second_end = (
        second_start[0] + math.sin(math.radians(15.0)) * 10.0,
        second_start[1] + math.cos(math.radians(15.0)) * 10.0,
    )
    piece = SimpleNamespace(model_path=r"o\road\sil10 100.p3d")
    measure = SimpleNamespace(points=(first_start, first_end, second_end))

    headings = {
        first_start: 0.0,
        first_end: 10.0,
        second_start: 10.0,
        second_end: 20.0,
    }
    monkeypatch.setattr(
        _p,
        "_nearest_polyline_heading",
        lambda _points, point: headings[tuple(point)],
    )

    first_tangents = _finish._piece_tangents(measure, piece, first_start, first_end)
    second_tangents = _finish._piece_tangents(measure, piece, second_start, second_end)

    assert math.isclose(first_tangents[0], 0.0, abs_tol=1.0e-9)
    assert math.isclose(first_tangents[1], 10.0, abs_tol=1.0e-9)
    assert math.isclose(second_tangents[0], 10.0, abs_tol=1.0e-9)
    assert math.isclose(second_tangents[1], 20.0, abs_tol=1.0e-9)
    assert _p._heading_difference(first_tangents[1], second_tangents[0]) == 0.0


def _object(object_id, model_path, x, z, heading):
    return SimpleNamespace(
        object_id=object_id,
        model_path=model_path,
        x=float(x),
        z=float(z),
        heading_degrees=float(heading),
        pitch_degrees=0.0,
    )


def test_curve_straight_tangent_mismatch_gets_low_seam_cover():
    model = r"o\road\sil10 100.p3d"
    curve_geometry = _geometry.stock_curve_connectors(model)
    assert curve_geometry is not None
    curve = _object(1, model, 0.0, 0.0, 0.0)
    seam = _geometry.transform_local(curve_geometry.end, (0.0, 0.0), 0.0)

    straight_heading = 11.5
    half = _geometry.STOCK_STRAIGHT_LENGTHS_METRES[6] * 0.5
    angle = math.radians(straight_heading)
    direction = (math.sin(angle), math.cos(angle))
    straight = _object(
        2,
        r"o\road\sil6.p3d",
        seam[0] + direction[0] * half,
        seam[1] + direction[1] * half,
        straight_heading,
    )
    report = SimpleNamespace(objects=(curve, straight), junction_cap_objects=0)

    plans = _finish._curve_seam_cover_plans(report)

    assert len(plans) == 1
    assert plans[0].model_path.casefold() == r"o\road\sil6.p3d"
    assert math.dist(plans[0].centre, seam) < 1.0e-9
    assert math.isclose(plans[0].tangent_axis_degrees, 10.75, abs_tol=1.0e-9)


def test_continuous_native_curve_run_does_not_get_seam_cover():
    model = r"o\road\sil10 100.p3d"
    geometry = _geometry.stock_curve_connectors(model)
    assert geometry is not None
    first = _object(1, model, 0.0, 0.0, 0.0)
    seam = _geometry.transform_local(geometry.end, (0.0, 0.0), 0.0)

    second_heading = 10.0
    begin_offset = _geometry.rotate_local(geometry.begin, second_heading)
    second_origin = (seam[0] - begin_offset[0], seam[1] - begin_offset[1])
    second = _object(2, model, second_origin[0], second_origin[1], second_heading)
    report = SimpleNamespace(objects=(first, second), junction_cap_objects=0)

    plans = _finish._curve_seam_cover_plans(report)

    assert plans == ()
