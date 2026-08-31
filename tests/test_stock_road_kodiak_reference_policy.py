# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import math

from cwr_worldgen import generator as _generator
from cwr_worldgen import playability as _p
from cwr_worldgen import road_quality_policy as _quality
from cwr_worldgen import stock_road_curve_usage_policy as _curve_usage
from cwr_worldgen import stock_road_reference_wrp_policy as _reference
from cwr_worldgen import stock_road_sharp_turn_policy as _sharp
from cwr_worldgen import stock_road_stock_assets_only_policy as _stock_only


@dataclass(frozen=True)
class _Report:
    objects: tuple
    junction_cap_objects: int


def _piece(model_path: str, length: float, nominal: int):
    return _p._RoadPiece(model_path, length, nominal)


def test_kodiak_stage_remains_in_final_curve_first_chain() -> None:
    assert _reference._INSTALLED
    assert _reference._KODIAK_INSTALLED
    assert _stock_only._INSTALLED

    assert _p.fit_road_objects is _generator.fit_road_objects
    assert _generator.fit_road_objects is _stock_only._fit
    assert _stock_only._ORIGINAL_FIT is _reference._fit

    assert _quality._quality_window is _reference._quality_window
    assert _curve_usage._MINIMUM_BASELINE_SHORT_STRAIGHTS == 0
    assert math.isclose(
        _curve_usage._MINIMUM_TOTAL_TURN_DEGREES,
        _reference.KODIAK_MINIMUM_CURVE_PROMOTION_TURN_DEGREES,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        _curve_usage._MAXIMUM_TOTAL_TURN_DEGREES,
        _reference.KODIAK_MAXIMUM_CURVE_PROMOTION_TURN_DEGREES,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        _curve_usage._MAXIMUM_PROMOTION_RUN_METRES,
        _reference.KODIAK_MAXIMUM_CURVE_PROMOTION_RUN_METRES,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        _sharp._MAXIMUM_SPAN_METRES,
        _reference.KODIAK_MAXIMUM_CURVE_PROMOTION_RUN_METRES,
        abs_tol=1.0e-12,
    )


def test_paved_junction_window_overlaps_measured_footprint(monkeypatch) -> None:
    measure = _p._PolylineMeasure.create(((0.0, 0.0), (0.0, 100.0)))
    pieces = (
        _piece(r"o\road\sil25.p3d", 25.0, 25),
        _piece(r"o\road\sil12.p3d", 12.5, 12),
        _piece(r"o\road\sil6.p3d", 6.25, 6),
    )
    start_key = _p._road_node_key(measure.points[0])
    end_key = _p._road_node_key(measure.points[-1])
    junction = _quality._Junction(
        point=(0.0, 0.0),
        axis=(0.0, 1.0),
        half_length=6.25,
        half_width=6.25,
        directions=((0.0, 1.0), (0.0, -1.0), (1.0, 0.0)),
    )
    context = _quality._Context(
        (), SimpleNamespace(), {start_key: junction, end_key: junction}
    )
    monkeypatch.setattr(
        _reference,
        "_ORIGINAL_QUALITY_WINDOW",
        lambda _measure, _pieces, start, preferred, minimum, maximum, _context: (
            start,
            preferred,
            minimum,
            maximum,
        ),
    )

    start, preferred, minimum, maximum = _reference._quality_window(
        measure,
        pieces,
        6.25,
        93.75,
        90.0,
        100.0,
        context,
    )

    expected = 6.25 - _reference.KODIAK_PAVED_JUNCTION_OVERLAP_METRES
    assert math.isclose(start, expected, abs_tol=1.0e-9)
    assert math.isclose(preferred, 100.0 - expected, abs_tol=1.0e-9)
    assert minimum == 90.0
    assert maximum >= preferred


def test_kodiak_junction_overlap_does_not_change_ces(monkeypatch) -> None:
    measure = _p._PolylineMeasure.create(((0.0, 0.0), (0.0, 100.0)))
    pieces = (_piece(r"o\road\ces25.p3d", 25.0, 25),)
    key = _p._road_node_key(measure.points[0])
    junction = _quality._Junction(
        point=(0.0, 0.0),
        axis=(0.0, 1.0),
        half_length=6.25,
        half_width=6.25,
        directions=((0.0, 1.0), (0.0, -1.0), (1.0, 0.0)),
    )
    context = _quality._Context((), SimpleNamespace(), {key: junction})
    baseline = (6.25, 93.75, 90.0, 100.0)
    monkeypatch.setattr(
        _reference,
        "_ORIGINAL_QUALITY_WINDOW",
        lambda *_args, **_kwargs: baseline,
    )

    assert _reference._quality_window(
        measure,
        pieces,
        *baseline,
        context,
    ) == baseline


def test_native_node_to_connector_sil6_is_removed(monkeypatch) -> None:
    node = (500.0, 500.0)
    cap = _p.WorldObject(
        1,
        r"o\road\kr_new_sil_sil_t.p3d",
        500.0,
        0.0,
        500.0,
        0.0,
        0.0,
    )
    stale = _p.WorldObject(
        2,
        r"o\road\sil6.p3d",
        500.0,
        0.0,
        503.125,
        0.0,
        0.0,
    )
    outside = _p.WorldObject(
        3,
        r"o\road\sil25.p3d",
        500.0,
        0.0,
        518.75,
        0.0,
        0.0,
    )
    report = _Report(
        objects=(cap, stale, outside),
        junction_cap_objects=1,
    )
    monkeypatch.setattr(
        _reference._finish,
        "_junction_incident_map",
        lambda *_args, **_kwargs: {
            _p._road_node_key(node): (node, ()),
        },
    )

    fixed = _reference._drop_native_node_stubs(
        report,
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
    )

    ids = {int(obj.object_id) for obj in fixed.objects}
    assert ids == {1, 3}
