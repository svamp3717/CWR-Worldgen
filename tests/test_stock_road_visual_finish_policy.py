# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_junction_policy as _junction
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
