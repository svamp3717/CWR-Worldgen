# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import playability as _p
from cwr_worldgen import road_quality_policy as _quality
from cwr_worldgen.model import WorldObject
from cwr_worldgen.road_quality_policy import _Context, _Junction
from cwr_worldgen.stock_road_model_geometry import (
    STOCK_JUNCTION_CONNECTOR_RADIUS_METRES,
)
from cwr_worldgen.stock_road_surface_overlap_policy import (
    CONNECTOR_COVER_SPAN_METRES,
    _apply_connector_covers,
    _connector_cover_plans,
)


def _report(*objects, caps: int):
    return _p.RoadFitReport(
        objects=tuple(objects),
        chain_count=1,
        connection_count=caps,
        failed_connections=0,
        maximum_connection_gap=0.0,
        maximum_chain_gap=0.0,
        truncated=False,
        junction_cap_objects=caps,
    )


def test_native_t_connector_gap_gets_same_family_underlay():
    # The sil/sil T logical center is +0.85 m in local X. Shift the object
    # origin so the synthetic logical intersection is world (0, 0).
    cap = WorldObject(
        1,
        r"o\road\kr_new_sil_sil_t.p3d",
        -0.85,
        0.04,
        0.0,
        0.0,
        0.0,
    )
    # The north connector is z=6.25. Leave a one-metre physical gap before a
    # normal sil6 approach begins.
    branch = WorldObject(
        2,
        r"o\road\sil6.p3d",
        0.0,
        0.035,
        10.375,
        0.0,
        0.0,
    )

    plans = _connector_cover_plans(_report(cap, branch, caps=1))

    assert len(plans) == 1
    plan = plans[0]
    assert plan.model_path.casefold() == r"o\road\sil6.p3d"
    assert math.isclose(plan.centre[0], 0.0, abs_tol=1.0e-6)
    assert math.isclose(plan.centre[1], 6.75, abs_tol=1.0e-6)
    assert plan.direction[1] > 0.99


def test_exact_native_connector_does_not_add_redundant_underlay():
    cap = WorldObject(
        1,
        r"o\road\kr_new_sil_sil_t.p3d",
        -0.85,
        0.04,
        0.0,
        0.0,
        0.0,
    )
    branch = WorldObject(
        2,
        r"o\road\sil6.p3d",
        0.0,
        0.035,
        9.375,
        0.0,
        0.0,
    )

    assert _connector_cover_plans(_report(cap, branch, caps=1)) == ()


def test_plain_paved_cap_connector_gap_is_covered_without_dirt_model():
    cap = WorldObject(
        1,
        r"o\road\sil6.p3d",
        0.0,
        0.041,
        0.0,
        0.0,
        0.0,
    )
    branch = WorldObject(
        2,
        r"o\road\sil6.p3d",
        0.0,
        0.035,
        7.125,
        0.0,
        0.0,
    )

    plans = _connector_cover_plans(_report(cap, branch, caps=1))

    assert len(plans) == 1
    assert plans[0].model_path.casefold() == r"o\road\sil6.p3d"
    assert math.isclose(plans[0].centre[1], 3.5625, abs_tol=1.0e-6)


def test_connector_underlay_is_appended_after_existing_road_objects():
    cap = WorldObject(
        1,
        r"o\road\kr_new_sil_sil_t.p3d",
        -0.85,
        0.04,
        0.0,
        0.0,
        0.0,
    )
    branch = WorldObject(
        2,
        r"o\road\sil6.p3d",
        0.0,
        0.035,
        10.375,
        0.0,
        0.0,
    )
    report = _report(cap, branch, caps=1)
    cells = 64
    spec = SimpleNamespace(
        cells=cells,
        cell_size=1.0,
        max_road_objects=100,
        advisory_object_limits=False,
    )

    fixed = _apply_connector_covers(
        report,
        (0.0,) * (cells * cells),
        spec,
    )

    assert len(fixed.objects) == 3
    cover = fixed.objects[-1]
    assert cover.object_id == 3
    assert cover.model_path.casefold() == r"o\road\sil6.p3d"
    assert fixed.short_piece_objects == 1
    assert math.isclose(CONNECTOR_COVER_SPAN_METRES, 6.0, abs_tol=1.0e-12)


def test_measured_stock_junction_keeps_fitter_free_to_continue_under_cap():
    measure = _p._PolylineMeasure.create(((0.0, 0.0), (0.0, 20.0)))
    end_key = _p._road_node_key(measure.points[-1])
    extent = STOCK_JUNCTION_CONNECTOR_RADIUS_METRES
    junction = _Junction(
        point=measure.points[-1],
        axis=(0.0, 1.0),
        half_length=extent,
        half_width=extent,
        directions=((0.0, -1.0), (0.0, 1.0), (1.0, 0.0)),
    )
    context = _Context(
        elevations=(),
        spec=SimpleNamespace(cells=1, cell_size=1.0),
        junctions={end_key: junction},
    )
    pieces = (_p._RoadPiece(r"o\road\sil6.p3d", 6.25, 6),)

    start, preferred_end, minimum_end, maximum_end = _quality._quality_window(
        measure,
        pieces,
        0.0,
        20.0,
        20.0,
        20.0,
        context,
    )

    assert start == 0.0
    assert math.isclose(preferred_end, measure.total, abs_tol=1.0e-9)
    assert minimum_end >= measure.total - 0.10 - 1.0e-9
    assert maximum_end >= measure.total + 3.125 - 1.0e-9
