from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from cwr_worldgen import bridge_abutment_terrain_policy as abutment
from cwr_worldgen import bridge_render_policy as bridge
from cwr_worldgen import bridge_runtime_policy as runtime
from cwr_worldgen import bridge_source_water_policy as source_water


def _spec():
    return SimpleNamespace(cells=64, cell_size=50.0, world_size=3200.0)


def _plan():
    step = float(bridge._STOCK_MODULE_SPACING_METRES)
    return bridge.StockBridgeSpanPlan(
        points=((175.0, 200.0), (175.0 + 3.0 * step, 200.0)),
        module_count=3,
        wet_start=(185.0, 200.0),
        wet_end=(315.0, 200.0),
        wet_length=130.0,
    )


def test_final_component_corridor_reuses_original_osm_pre_fill_plan() -> None:
    spec = _spec()
    plan = _plan()
    original_osm_points = (
        (20.0, 200.0),
        (150.0, 200.0),
        (350.0, 200.0),
        (620.0, 200.0),
    )
    # Model the old ten-module physical component. Final reconciliation asks the
    # planner with this synthetic corridor, not with the OSM polyline above.
    corridor = (
        (0.0, 200.0),
        (10.0 * float(bridge._STOCK_MODULE_SPACING_METRES), 200.0),
    )

    abutment._PLAN_CACHE.clear()
    abutment._PLAN_CACHE[abutment._plan_key(original_osm_points, spec)] = plan
    try:
        with patch.object(
            source_water,
            "_mapped_water_stock_plan",
            side_effect=AssertionError("post-fill terrain must not re-plan the bridge"),
        ):
            resolved = runtime._runtime_stock_bridge_span_plan(corridor, (), spec)
    finally:
        abutment._PLAN_CACHE.clear()

    assert resolved is plan
    assert resolved.points == plan.points
    assert resolved.module_count == 3


def test_corridor_cache_match_rejects_unrelated_nearby_bridge() -> None:
    spec = _spec()
    plan = _plan()
    original_osm_points = ((20.0, 200.0), (620.0, 200.0))
    unrelated_corridor = ((0.0, 280.0), (500.0, 280.0))

    abutment._PLAN_CACHE.clear()
    abutment._PLAN_CACHE[abutment._plan_key(original_osm_points, spec)] = plan
    try:
        resolved = abutment._cached_bridge_plan_for_corridor(
            unrelated_corridor,
            spec,
        )
    finally:
        abutment._PLAN_CACHE.clear()

    assert resolved is None
