from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from cwr_worldgen import bridge_abutment_terrain_policy as abutment
from cwr_worldgen import bridge_final_alignment_policy as final_alignment
from cwr_worldgen import bridge_final_count_policy as final_count
from cwr_worldgen import bridge_or_causeway_terrain_policy as terrain_policy
from cwr_worldgen import bridge_render_policy as render_policy
from cwr_worldgen import bridge_road_connection_policy as road_connection
from cwr_worldgen import bridge_runtime_policy as runtime_policy
from cwr_worldgen import bridge_source_water_policy as source_water
from cwr_worldgen import bridge_underlay_cleanup_policy as underlay_cleanup
from cwr_worldgen import bridge_water_deck_clamp_policy as water_clamp
from cwr_worldgen import build_cache_policy as build_cache


def test_package_runtime_installs_complete_bridge_policy_chain() -> None:
    assert runtime_policy._INSTALLED
    assert render_policy._INSTALLED
    assert water_clamp._INSTALLED
    assert underlay_cleanup._INSTALLED
    assert source_water._INSTALLED
    assert road_connection._INSTALLED
    assert final_alignment._INSTALLED
    assert terrain_policy._INSTALLED
    assert abutment._INSTALLED
    assert final_count._INSTALLED

    assert source_water._ORIGINAL_WATER_TEST is not None
    assert source_water._ORIGINAL_STOCK_PLAN is not None
    assert terrain_policy._ORIGINAL_SOLVE is not None
    assert abutment._ORIGINAL_SOLVE is not None
    assert final_count._ORIGINAL_ANCHOR is not None

    assert render_policy.stock_bridge_span_plan is runtime_policy._runtime_stock_bridge_span_plan
    assert source_water._ORIGINAL_STOCK_PLAN is water_clamp._ORIGINAL_STOCK_BRIDGE_SPAN_PLAN
    assert build_cache.BUILD_CACHE_REVISION == "v11-authoritative-pre-fill-bridge-plan"


def test_runtime_planner_reuses_pre_fill_wet_plan() -> None:
    plan = render_policy.StockBridgeSpanPlan(
        points=((100.0, 100.0), (551.7, 100.0)),
        module_count=9,
        wet_start=(110.0, 100.0),
        wet_end=(541.0, 100.0),
        wet_length=431.0,
    )
    spec = SimpleNamespace(cells=32, cell_size=50.0, world_size=1600.0)

    with (
        patch.object(abutment, "_cached_bridge_plan", return_value=plan),
        patch.object(
            source_water,
            "_mapped_water_stock_plan",
            side_effect=AssertionError("cached wet plan should win"),
        ),
    ):
        resolved = runtime_policy._runtime_stock_bridge_span_plan(plan.points, (), spec)

    assert resolved == plan
    assert resolved.module_count == 9


def test_runtime_planner_matches_final_component_corridor_to_pre_fill_plan() -> None:
    plan = render_policy.StockBridgeSpanPlan(
        points=((180.0, 200.0), (330.5708541870117, 200.0)),
        module_count=3,
        wet_start=(190.0, 200.0),
        wet_end=(320.0, 200.0),
        wet_length=130.0,
    )
    spec = SimpleNamespace(cells=64, cell_size=50.0, world_size=3200.0)
    corridor = ((0.0, 200.0), (501.90284729003906, 200.0))

    with (
        patch.object(abutment, "_cached_bridge_plan", return_value=None),
        patch.object(abutment, "_cached_bridge_plan_for_corridor", return_value=plan),
        patch.object(
            source_water,
            "_mapped_water_stock_plan",
            side_effect=AssertionError("post-fill replanning must not move the bridge"),
        ),
    ):
        resolved = runtime_policy._runtime_stock_bridge_span_plan(corridor, (), spec)

    assert resolved is plan
    assert resolved.module_count == 3


def test_installed_bridge_runtime_accepts_donor_module_spacing() -> None:
    """The real build replaces the planner; donor fitting must survive that wrapper."""
    spec = SimpleNamespace(
        cells=8, cell_size=25.0, world_size=200.0, sea_level=0.0,
        bridge_donor_model=r"haus\bridge03.p3d",
    )
    source = ((20.0, 100.0), (180.0, 100.0))
    underwater = (-3.0,) * 64

    with (
        patch.object(
            abutment, "_cached_bridge_plan",
            side_effect=AssertionError("stock bridge cache must not affect donor"),
        ),
        patch.object(
            abutment, "_cached_bridge_plan_for_corridor",
            side_effect=AssertionError("stock corridor cache must not affect donor"),
        ),
    ):
        # This is the precise call path that raised TypeError in the user's
        # build: _extend_stock_bridge_to_wet_span -> runtime planner.
        corridor = render_policy._extend_stock_bridge_to_wet_span(
            source, underwater, spec, 40.0,
        )
        plan = runtime_policy._runtime_stock_bridge_span_plan(
            source, underwater, spec, module_spacing=40.0,
        )

    assert plan is not None
    assert plan.module_count == 4
    assert plan.length == 160.0
    assert corridor == plan.points


def test_source_water_fallback_respects_custom_donor_spacing() -> None:
    """Mapped-water fallback must not silently revert to Nogova's 50.19 m."""
    spec = SimpleNamespace(
        cells=8, cell_size=25.0, world_size=200.0, sea_level=0.0,
        road_width=5.0,
    )
    corridor = ((20.0, 100.0), (180.0, 100.0))
    with (
        patch.object(
            source_water, "_ORIGINAL_STOCK_PLAN",
            side_effect=lambda points, elevations, spec, **kw: None,
        ) as original_plan,
        patch.object(
            source_water, "_source_mapped_water_interval",
            return_value=(50.0, 110.0),
        ),
        patch.object(
            source_water, "_mapped_water_is_representable",
            return_value=True,
        ),
    ):
        plan = source_water._mapped_water_stock_plan(
            corridor, (), spec, module_spacing=40.0,
        )

    assert original_plan.call_args.kwargs["module_spacing"] == 40.0
    assert plan is not None
    assert plan.module_count == 2
    assert plan.length == 80.0
