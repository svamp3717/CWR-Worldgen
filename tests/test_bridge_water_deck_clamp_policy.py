from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from cwr_worldgen import bridge_render_policy as bridge_render
from cwr_worldgen import bridge_water_deck_clamp_policy as clamp
from cwr_worldgen import osm


def _spec(*, sea_level: float = 0.0):
    return SimpleNamespace(
        sea_level=sea_level,
        cells=64,
        cell_size=10.0,
        world_size=640.0,
    )


def test_low_shoreline_reserves_maximum_tide_headroom_before_tuning() -> None:
    spec = _spec(sea_level=0.0)

    with patch.object(clamp, "_ORIGINAL_DRY_APPROACH_HEIGHT", return_value=0.05):
        pre_tuning = clamp._safe_dry_approach_height(
            (0.0, 0.0),
            (1.0, 0.0),
            None,
            (),
            spec,
        )

    final_deck = pre_tuning - bridge_render._BRIDGE_WORLD_DOWNWARD_OFFSET_METRES
    assert abs(final_deck - 5.5) < 1e-12


def test_high_bank_keeps_existing_point_seven_metre_tuning() -> None:
    spec = _spec(sea_level=0.0)
    untuned_road_surface = 7.035

    with patch.object(
        clamp,
        "_ORIGINAL_DRY_APPROACH_HEIGHT",
        return_value=untuned_road_surface,
    ):
        pre_tuning = clamp._safe_dry_approach_height(
            (0.0, 0.0),
            (1.0, 0.0),
            None,
            (),
            spec,
        )

    assert pre_tuning == untuned_road_surface
    final_deck = pre_tuning - bridge_render._BRIDGE_WORLD_DOWNWARD_OFFSET_METRES
    assert abs(final_deck - 6.335) < 1e-12


def test_tide_safe_span_extends_only_to_nearest_safe_bank_road() -> None:
    spec = _spec()
    points = ((20.0, 100.0), (220.0, 100.0))
    base = bridge_render.StockBridgeSpanPlan(
        points=((90.0, 100.0), (150.0, 100.0)),
        module_count=2,
        wet_start=(90.0, 100.0),
        wet_end=(150.0, 100.0),
        wet_length=60.0,
    )

    # Terrain is tide-safe on dry land up to x=70 and again from x=170 onward.
    # The source bridge way is 200 m long, so a three-module stock chain can sit
    # around the 100 m safe interval while leaving substantial normal-road
    # prefix/suffix on both sides.
    def ground(_points, _cumulative, distance, _elevations, _spec):
        x = 20.0 + float(distance)
        if x <= 70.0 or x >= 170.0:
            return 7.0
        if 90.0 <= x <= 150.0:
            return -5.0
        return 3.0

    with (
        patch.object(clamp, "_ORIGINAL_STOCK_BRIDGE_SPAN_PLAN", return_value=base),
        patch.object(bridge_render, "_ground_at_measure", side_effect=ground),
    ):
        plan = clamp._tide_safe_stock_bridge_span_plan(points, (), spec)

    assert plan is not None
    assert plan.wet_start == base.wet_start
    assert plan.wet_end == base.wet_end
    assert plan.module_count == 3
    assert plan.points[0][0] < 70.0
    assert plan.points[1][0] > 170.0
    assert plan.points[0][0] > points[0][0]
    assert plan.points[1][0] < points[-1][0]


def test_none_from_underlying_sampler_is_preserved() -> None:
    with patch.object(clamp, "_ORIGINAL_DRY_APPROACH_HEIGHT", return_value=None):
        assert clamp._safe_dry_approach_height(
            (0.0, 0.0),
            (1.0, 0.0),
            None,
            (),
            _spec(),
        ) is None
