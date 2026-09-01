# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_final_continuity_policy as _continuity
from cwr_worldgen import stock_road_junction_policy as _junction
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
    # the first arm and skipped it, leaving the legacy cap visibly above the
    # turning approaches.
    incidents = (
        _junction._Incident(_direction(90.0), "sil", r"o\road\sil25.p3d"),
        _junction._Incident(_direction(250.0), "sil", r"o\road\sil25.p3d"),
        _junction._Incident(_direction(0.0), "sil", r"o\road\sil25.p3d"),
    )
    captured = []
    _install_cap_realign_test_doubles(monkeypatch, incidents, captured)

    realign = _continuity._ORIGINAL_REALIGN_LEGACY_CAPS
    assert realign is not None
    result = realign(
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

    realign = _continuity._ORIGINAL_REALIGN_LEGACY_CAPS
    assert realign is not None
    result = realign(report, None, None, (), SimpleNamespace())

    assert captured == []
    assert result is report


def test_retired_intermediate_visual_seam_hook_is_a_noop():
    report = SimpleNamespace(objects=(), junction_cap_objects=0)

    assert _finish._apply_curve_seam_covers(report, (), SimpleNamespace()) is report
