from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest

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

    # Horizontal length starts from nominal/source water and gets only the small
    # maximum-tide shoreline extension performed by the final runtime planner.
    assert render_policy.stock_bridge_span_plan is runtime_policy._runtime_stock_bridge_span_plan
    assert (
        source_water._ORIGINAL_STOCK_PLAN
        is water_clamp._ORIGINAL_STOCK_BRIDGE_SPAN_PLAN
    )
    assert build_cache.BUILD_CACHE_REVISION == "v4-maximum-tide-bridge-envelope"


def test_maximum_tide_envelope_adds_one_module_not_hundreds_of_metres() -> None:
    module = render_policy._STOCK_MODULE_SPACING_METRES
    current_length = 9.0 * module
    plan = render_policy.StockBridgeSpanPlan(
        points=((100.0, 100.0), (100.0 + current_length, 100.0)),
        module_count=9,
        wet_start=(110.0, 100.0),
        wet_end=(90.0 + current_length, 100.0),
        wet_length=current_length - 20.0,
    )
    spec = SimpleNamespace(
        cells=64,
        cell_size=50.0,
        sea_level=0.0,
        world_size=3200.0,
    )
    feature = SimpleNamespace()
    projected = plan.points
    start_path = ((100.0, 100.0), (0.0, 100.0))
    end_path = (
        (100.0 + current_length, 100.0),
        (700.0, 100.0),
    )

    def connected(_feature, _dataset, _projection, endpoint, *_args):
        return start_path if endpoint[0] < 200.0 else end_path

    def terrain(_elevations, _cells, _cell_size, x, _z):
        # Max tide is +5 m. The first stable dry terrain is 10 m outside the
        # start and 27 m outside the end, matching the test19 failure scale.
        if x <= 90.0 or x >= 100.0 + current_length + 27.0:
            return 6.0
        return 2.0

    token = source_water._CONTEXT.set(
        SimpleNamespace(dataset=object(), projection=object())
    )
    try:
        with (
            patch.object(
                road_connection,
                "_matching_bridge_feature",
                return_value=(feature, projected),
            ),
            patch.object(
                road_connection._osm,
                "_connected_bridge_approach_path",
                side_effect=connected,
            ),
            patch.object(
                road_connection._osm,
                "_sample_elevation",
                side_effect=terrain,
            ),
        ):
            expanded = runtime_policy._maximum_tide_stock_plan(
                plan,
                projected,
                (),
                spec,
            )
    finally:
        source_water._CONTEXT.reset(token)

    assert expanded is not None
    assert expanded.module_count == 10
    assert math.dist(*expanded.points) == pytest.approx(10.0 * module)
    assert expanded.wet_start[0] == pytest.approx(90.0, abs=0.51)
    assert expanded.wet_end[0] == pytest.approx(
        100.0 + current_length + 27.0,
        abs=0.51,
    )

    # Whole-module rounding may overlap a little stable bank, but never another
    # hundred metres of dry approach road.
    assert 0.0 < expanded.wet_start[0] - expanded.points[0][0] < module
    assert 0.0 < expanded.points[1][0] - expanded.wet_end[0] < module
