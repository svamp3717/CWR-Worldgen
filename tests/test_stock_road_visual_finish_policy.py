# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_final_continuity_policy as _continuity
from cwr_worldgen import stock_road_junction_policy as _junction
from cwr_worldgen import stock_road_model_geometry as _geometry
from cwr_worldgen import stock_road_visual_finish_policy as _finish


def _direction(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(float(heading_degrees))
    return math.sin(angle), math.cos(angle)


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


def _legacy_cap_report(heading: float):
    cap = SimpleNamespace(
        object_id=1,
        model_path=r"o\road\sil6.p3d",
        x=0.0,
        y=0.041,
        z=0.0,
        heading_degrees=float(heading),
        pitch_degrees=0.0,
    )
    return SimpleNamespace(objects=(cap,), junction_cap_objects=1)


def _install_cap_realign_test_doubles(monkeypatch, incidents, captured):
    node = (0.0, 0.0)
    monkeypatch.setattr(
        _finish,
        "_junction_incident_map",
        lambda _dataset, _projection, _spec: {
            _p._road_node_key(node): (node, incidents)
        },
    )

    def fake_slope(
        object_id,
        model_path,
        start,
        end,
        _elevations,
        _spec,
        *,
        vertical_offset,
    ):
        captured.append(float(vertical_offset))
        heading = math.degrees(
            math.atan2(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
        ) % 360.0
        return SimpleNamespace(
            object_id=int(object_id),
            model_path=str(model_path),
            x=(float(start[0]) + float(end[0])) * 0.5,
            y=float(vertical_offset),
            z=(float(start[1]) + float(end[1])) * 0.5,
            heading_degrees=heading,
            pitch_degrees=0.0,
        )

    monkeypatch.setattr(_p, "_road_object_on_slope", fake_slope)

    def namespace_replace(obj, **changes):
        values = vars(obj).copy()
        values.update(changes)
        return SimpleNamespace(**values)

    monkeypatch.setattr(_finish, "replace", namespace_replace)


def test_turning_legacy_cap_is_lowered_even_when_axis_already_matches(monkeypatch):
    # The dominant pair is 160 degrees apart, so the source through road turns
    # twenty degrees at the node. The old policy saw the cap already aligned to
    # the first arm and skipped it, leaving the +6 mm legacy cap visibly above
    # the turning approaches.
    incidents = (
        _junction._Incident(_direction(90.0), "sil", r"o\road\sil25.p3d"),
        _junction._Incident(_direction(250.0), "sil", r"o\road\sil25.p3d"),
        _junction._Incident(_direction(0.0), "sil", r"o\road\sil25.p3d"),
    )
    captured = []
    _install_cap_realign_test_doubles(monkeypatch, incidents, captured)

    result = _finish._realign_legacy_caps(
        _legacy_cap_report(90.0), None, None, (), SimpleNamespace()
    )

    assert len(captured) == 1
    assert math.isclose(
        captured[0],
        _p._STOCK_ROAD_VERTICAL_OFFSET_METRES
        + _finish.TURNING_LEGACY_CAP_VERTICAL_BIAS_METRES,
        abs_tol=1.0e-9,
    )
    assert _finish.TURNING_LEGACY_CAP_VERTICAL_BIAS_METRES < 0.0
    assert math.isclose(result.objects[0].y, captured[0], abs_tol=1.0e-9)
    assert _finish._axis_heading_difference(result.objects[0].heading_degrees, 90.0) < 1.0e-9


def test_aligned_straight_through_cap_is_not_needlessly_rebuilt(monkeypatch):
    incidents = (
        _junction._Incident(_direction(90.0), "sil", r"o\road\sil25.p3d"),
        _junction._Incident(_direction(270.0), "sil", r"o\road\sil25.p3d"),
        _junction._Incident(_direction(0.0), "sil", r"o\road\sil25.p3d"),
    )
    captured = []
    _install_cap_realign_test_doubles(monkeypatch, incidents, captured)
    report = _legacy_cap_report(90.0)

    result = _finish._realign_legacy_caps(
        report, None, None, (), SimpleNamespace()
    )

    assert captured == []
    assert result is report


def test_native_curve_tangents_meet_at_the_rendered_edge(monkeypatch):
    first_start = (0.0, 0.0)
    first_end = (
        math.sin(math.radians(5.0)) * 10.0,
        math.cos(math.radians(5.0)) * 10.0,
    )
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

    first_tangents = _finish._piece_tangents(
        measure, piece, first_start, first_end
    )
    second_tangents = _finish._piece_tangents(
        measure, piece, second_start, second_end
    )

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
        y=0.0,
        z=float(z),
        heading_degrees=float(heading),
        pitch_degrees=0.0,
    )


def _curve_straight_report(straight_heading: float):
    model = r"o\road\sil10 100.p3d"
    curve_geometry = _geometry.stock_curve_connectors(model)
    assert curve_geometry is not None
    curve = _object(1, model, 0.0, 0.0, 0.0)
    seam = _geometry.transform_local(
        curve_geometry.end, (0.0, 0.0), 0.0
    )

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
    return SimpleNamespace(
        objects=(curve, straight),
        junction_cap_objects=0,
    ), seam


def test_curve_straight_tangent_mismatch_is_detected():
    report, seam = _curve_straight_report(11.5)

    plans = _finish._curve_seam_cover_plans(report)

    assert len(plans) == 1
    assert plans[0].model_path.casefold() == r"o\road\sil6.p3d"
    assert math.dist(plans[0].centre, seam) < 1.0e-9
    assert math.isclose(
        plans[0].tangent_axis_degrees, 10.75, abs_tol=1.0e-9
    )


def test_final_physical_seam_budget_keeps_lundby_sized_turns():
    # The retired curve-seam wrapper existed partly to widen this bound to 8
    # degrees. Keep that effective production value without keeping the wrapper.
    assert _finish.MAXIMUM_CURVE_SEAM_TANGENT_ERROR_DEGREES == 8.0

    report, _seam = _curve_straight_report(16.0)
    assert len(_finish._curve_seam_cover_plans(report)) == 1


def test_continuous_native_curve_run_does_not_get_seam_plan():
    model = r"o\road\sil10 100.p3d"
    geometry = _geometry.stock_curve_connectors(model)
    assert geometry is not None
    first = _object(1, model, 0.0, 0.0, 0.0)
    seam = _geometry.transform_local(geometry.end, (0.0, 0.0), 0.0)

    second_heading = 10.0
    begin_offset = _geometry.rotate_local(geometry.begin, second_heading)
    second_origin = (
        seam[0] - begin_offset[0],
        seam[1] - begin_offset[1],
    )
    second = _object(
        2, model, second_origin[0], second_origin[1], second_heading
    )
    report = SimpleNamespace(
        objects=(first, second),
        junction_cap_objects=0,
    )

    assert _finish._curve_seam_cover_plans(report) == ()


def test_final_continuity_owns_intermediate_visual_seam_policy():
    # Intermediate road-underlay wrappers are gone. Curve selection/continuity
    # owns the visible geometry; the later emitted-seam audit owns final WRP
    # connector gaps.
    assert (
        _finish._apply_curve_seam_covers
        is _continuity._disable_curve_seam_underlays
    )
