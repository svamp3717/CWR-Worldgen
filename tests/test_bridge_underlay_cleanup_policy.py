from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cwr_worldgen import bridge_render_policy as bridge_render
from cwr_worldgen import bridge_underlay_cleanup_policy as cleanup
from cwr_worldgen.model import WorldObject
from cwr_worldgen.playability import RoadFitReport


def _report(*objects: WorldObject) -> RoadFitReport:
    return RoadFitReport(
        objects=tuple(objects),
        chain_count=1,
        connection_count=0,
        failed_connections=0,
        maximum_connection_gap=0.0,
        maximum_chain_gap=0.0,
        truncated=False,
    )


def test_tinybjorsund_submerged_road_chain_is_removed_beneath_bridge() -> None:
    span = cleanup._BridgeSpan(
        points=((605.08, 675.70), (871.81, 854.84)),
        road_width=7.0,
    )
    underlay = (
        WorldObject(1070, r"o\road\sil25.p3d", 689.4, -3.02, 732.5, 56.11),
        WorldObject(1071, r"o\road\sil25.p3d", 709.8, -4.93, 746.2, 56.11),
        WorldObject(1072, r"o\road\sil25.p3d", 730.1, -4.97, 759.8, 56.11),
        WorldObject(1073, r"o\road\sil25.p3d", 750.4, -4.67, 773.5, 56.11),
        WorldObject(1074, r"o\road\sil12.p3d", 765.5, -3.97, 783.6, 56.11),
        WorldObject(1075, r"o\road\sil12.p3d", 775.3, -3.05, 790.2, 56.11),
        WorldObject(1076, r"o\road\sil12.p3d", 785.0, -1.89, 796.7, 56.11),
        WorldObject(1077, r"o\road\sil12.p3d", 794.8, -0.74, 803.3, 56.11),
    )
    crossing = WorldObject(2000, r"o\road\sil25.p3d", 738.4, 0.0, 765.3, 146.11)

    cleaned, removed = cleanup._remove_bridge_underlays(_report(*underlay, crossing), (span,))

    assert removed == len(underlay)
    assert {obj.object_id for obj in cleaned.objects} == {2000}


def test_cleanup_keeps_parallel_road_outside_stock_bridge_footprint() -> None:
    span = cleanup._BridgeSpan(points=((0.0, 0.0), (200.0, 0.0)), road_width=6.0)
    underlay = WorldObject(1, r"o\road\sil25.p3d", 100.0, 0.0, 0.0, 90.0)
    parallel = WorldObject(2, r"o\road\sil25.p3d", 100.0, 0.0, 7.5, 90.0)

    cleaned, removed = cleanup._remove_bridge_underlays(_report(underlay, parallel), (span,))

    assert removed == 1
    assert tuple(obj.object_id for obj in cleaned.objects) == (2,)


def test_cleanup_keeps_road_under_first_bridge_module_at_both_ends() -> None:
    span = cleanup._BridgeSpan(points=((0.0, 0.0), (250.0, 0.0)), road_width=7.0)
    first_end = WorldObject(1, r"o\road\sil25.p3d", 25.0, 0.0, 0.0, 90.0)
    interior = WorldObject(2, r"o\road\sil25.p3d", 125.0, 0.0, 0.0, 90.0)
    second_end = WorldObject(3, r"o\road\sil25.p3d", 225.0, 0.0, 0.0, 90.0)

    cleaned, removed = cleanup._remove_bridge_underlays(_report(first_end, interior, second_end), (span,))

    assert removed == 1
    assert tuple(obj.object_id for obj in cleaned.objects) == (1, 3)


def test_missing_terminal_underlays_are_explicitly_generated() -> None:
    span = cleanup._BridgeSpan(
        points=((0.0, 100.0), (250.0, 100.0)),
        road_width=7.0,
        road_model_path=r"o\road\sil25.p3d",
    )
    spec = SimpleNamespace(cells=16, cell_size=50.0)
    elevations = (6.30,) * (spec.cells * spec.cells)

    filled, added = cleanup._add_terminal_underlays(_report(), (span,), elevations, spec)

    assert added == 4
    assert len(filled.objects) == 4
    assert all(obj.model_path.casefold() == r"o\road\sil25.p3d" for obj in filled.objects)
    assert [obj.x for obj in filled.objects] == pytest.approx([12.5, 37.5, 212.5, 237.5])
    assert all(obj.z == pytest.approx(100.0) for obj in filled.objects)
    assert all(obj.y == pytest.approx(6.335) for obj in filled.objects)
    assert all(obj.heading_degrees == pytest.approx(90.0) for obj in filled.objects)


def test_cleanup_removes_curved_source_road_that_bows_away_from_straight_bridge() -> None:
    source = ((0.0, 0.0), (100.0, 8.0), (200.0, 0.0))
    source_length = sum(math.dist(start, end) for start, end in zip(source, source[1:]))
    span = cleanup._BridgeSpan(
        points=((0.0, 0.0), (200.0, 0.0)),
        road_width=6.0,
        source_points=source,
        source_start_measure=0.0,
        source_end_measure=source_length,
    )
    curved_underlay = WorldObject(1, r"o\road\sil25.p3d", 100.0, -4.9, 8.0, 90.0)
    parallel = WorldObject(2, r"o\road\sil25.p3d", 100.0, 0.0, 12.0, 90.0)
    crossing = WorldObject(3, r"o\road\sil25.p3d", 100.0, 0.0, 8.0, 0.0)

    cleaned, removed = cleanup._remove_bridge_underlays(
        _report(curved_underlay, parallel, crossing), (span,)
    )

    assert removed == 1
    assert tuple(obj.object_id for obj in cleaned.objects) == (2, 3)


def test_cleanup_preserves_dry_approach_roads_outside_emitted_bridge_span() -> None:
    span = cleanup._BridgeSpan(points=((400.0, 100.0), (600.0, 100.0)), road_width=7.0)
    dry_before = WorldObject(1, r"o\road\sil25.p3d", 250.0, 0.0, 100.0, 90.0)
    under_bridge = WorldObject(2, r"o\road\sil25.p3d", 500.0, 0.0, 100.0, 90.0)
    dry_after = WorldObject(3, r"o\road\sil25.p3d", 750.0, 0.0, 100.0, 90.0)

    cleaned, removed = cleanup._remove_bridge_underlays(
        _report(dry_before, under_bridge, dry_after), (span,)
    )

    assert removed == 1
    assert tuple(obj.object_id for obj in cleaned.objects) == (1, 3)


def test_bridge_spans_use_same_wet_only_stock_plan_as_bridge_renderer() -> None:
    feature = SimpleNamespace(tags={"highway": "primary", "bridge": "yes"})
    dataset = SimpleNamespace(roads=(feature,))
    spec = SimpleNamespace(
        bridges_enabled=True,
        maximum_bridge_objects=1000,
        advisory_object_limits=True,
        procedural_bridges=True,
        bridge_module_length=30.0,
        cells=64,
        cell_size=10.0,
        sea_level=0.0,
        world_size=640.0,
        paved_road_model=r"o\road\sil25.p3d",
        dirt_road_model=r"o\road\ces25.p3d",
    )
    points = ((20.0, 100.0), (620.0, 100.0))
    plan = bridge_render.StockBridgeSpanPlan(
        points=((219.6, 100.0), (420.4, 100.0)),
        module_count=4,
        wet_start=(220.0, 100.0),
        wet_end=(420.0, 100.0),
        wet_length=200.0,
    )

    with (
        patch.object(cleanup._osm, "projected_road_polylines", return_value=(points,)),
        patch.object(cleanup._osm, "road_bridge_crosses_ditch_only", return_value=False),
        patch.object(cleanup, "_feature_needs_bridge", return_value=True),
        patch.object(bridge_render, "stock_bridge_span_plan", return_value=plan),
        patch.object(cleanup._osm, "_bridge_module_chunks", return_value=((0.0,) * 8,)),
    ):
        spans = cleanup._bridge_spans(dataset, None, [0.0] * (64 * 64), spec)

    assert len(spans) == 1
    assert spans[0].points == plan.points
    assert spans[0].points != points
    assert spans[0].source_points == points
    assert spans[0].source_start_measure < spans[0].source_end_measure
    assert spans[0].road_model_path.casefold() == r"o\road\sil25.p3d"


def test_bridge_object_budget_uses_clipped_stock_module_count() -> None:
    feature = SimpleNamespace(tags={"highway": "primary", "bridge": "yes"})
    dataset = SimpleNamespace(roads=(feature,))
    spec = SimpleNamespace(
        bridges_enabled=True,
        maximum_bridge_objects=3,
        advisory_object_limits=False,
        procedural_bridges=True,
        bridge_module_length=30.0,
        cells=64,
        cell_size=10.0,
        sea_level=0.0,
        world_size=640.0,
        paved_road_model=r"o\road\sil25.p3d",
        dirt_road_model=r"o\road\ces25.p3d",
    )
    points = ((20.0, 100.0), (620.0, 100.0))
    plan = bridge_render.StockBridgeSpanPlan(
        points=((200.0, 100.0), (450.0, 100.0)),
        module_count=4,
        wet_start=(210.0, 100.0),
        wet_end=(440.0, 100.0),
        wet_length=230.0,
    )

    with (
        patch.object(cleanup._osm, "projected_road_polylines", return_value=(points,)),
        patch.object(cleanup._osm, "road_bridge_crosses_ditch_only", return_value=False),
        patch.object(cleanup, "_feature_needs_bridge", return_value=True),
        patch.object(bridge_render, "stock_bridge_span_plan", return_value=plan),
        patch.object(cleanup._osm, "_bridge_module_chunks", return_value=((0.0,) * 8,)),
    ):
        spans = cleanup._bridge_spans(dataset, None, [0.0] * (64 * 64), spec)

    assert spans == ()
