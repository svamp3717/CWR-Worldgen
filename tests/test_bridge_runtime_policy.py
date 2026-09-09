from __future__ import annotations

from cwr_worldgen import bridge_final_alignment_policy as final_alignment
from cwr_worldgen import bridge_or_causeway_terrain_policy as terrain_policy
from cwr_worldgen import bridge_render_policy as render_policy
from cwr_worldgen import bridge_road_connection_policy as road_connection
from cwr_worldgen import bridge_runtime_policy as runtime_policy
from cwr_worldgen import bridge_source_water_policy as source_water
from cwr_worldgen import bridge_underlay_cleanup_policy as underlay_cleanup
from cwr_worldgen import bridge_water_deck_clamp_policy as water_clamp
from cwr_worldgen import build_cache_policy as build_cache


def test_package_runtime_installs_complete_bridge_policy_chain() -> None:
    # Importing cwr_worldgen runs the package installers before this test module
    # is imported. These assertions prevent bridge policies from becoming a
    # test-only collection of helpers that real GUI/CLI builds never execute.
    assert runtime_policy._INSTALLED
    assert render_policy._INSTALLED
    assert water_clamp._INSTALLED
    assert underlay_cleanup._INSTALLED
    assert source_water._INSTALLED
    assert road_connection._INSTALLED
    assert final_alignment._INSTALLED
    assert terrain_policy._INSTALLED

    assert source_water._ORIGINAL_WATER_TEST is not None
    assert source_water._ORIGINAL_STOCK_PLAN is not None
    assert terrain_policy._ORIGINAL_SOLVE is not None

    # Horizontal length is finalized by the wet-only/source-water planner, not
    # by the tide-safe or connected-road wrappers that can walk far onto land.
    assert render_policy.stock_bridge_span_plan is source_water._mapped_water_stock_plan
    assert (
        source_water._ORIGINAL_STOCK_PLAN
        is water_clamp._ORIGINAL_STOCK_BRIDGE_SPAN_PLAN
    )
    assert build_cache.BUILD_CACHE_REVISION == "v3-water-authoritative-bridge-length"
