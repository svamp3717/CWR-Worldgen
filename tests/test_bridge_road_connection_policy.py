from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
import math
import pytest

from cwr_worldgen import bridge_render_policy as bridge
from cwr_worldgen import bridge_road_connection_policy as policy
from cwr_worldgen import bridge_source_water_policy as source


def _spec():
    return SimpleNamespace(
        cells=64,
        cell_size=50.0,
        sea_level=0.0,
        world_size=3200.0,
    )


def test_short_bridge_expands_to_two_modules_on_connected_safe_roads() -> None:
    spec = _spec()
    points = ((100.0, 100.0), (120.0, 100.0))
    plan = bridge.StockBridgeSpanPlan(
        points=((84.9, 100.0), (135.1, 100.0)),
        module_count=1,
        wet_start=(105.0, 100.0),
        wet_end=(115.0, 100.0),
        wet_length=10.0,
    )
    feature = SimpleNamespace()
    start_path = ((100.0, 100.0), (0.0, 100.0))
    end_path = ((120.0, 100.0), (240.0, 100.0))

    def connected(_feature, _dataset, _projection, endpoint, *_args):
        return start_path if endpoint[0] < 110.0 else end_path

    token = source._CONTEXT.set(SimpleNamespace(dataset=object(), projection=object()))
    try:
        with (
            patch.object(
                policy,
                "_matching_bridge_feature",
                return_value=(feature, points),
            ),
            patch.object(
                policy._osm,
                "_connected_bridge_approach_path",
                side_effect=connected,
            ),
            patch.object(
                policy,
                "_road_surface_at",
                return_value=5.6,
            ),
        ):
            fitted = policy._connected_stock_plan(
                plan,
                points,
                (),
                spec,
            )
    finally:
        source._CONTEXT.reset(token)

    assert fitted.module_count == 2
    assert math.dist(*fitted.points) == pytest.approx(
        2.0 * bridge._STOCK_MODULE_SPACING_METRES,
        abs=policy._MAXIMUM_ENDPOINT_LENGTH_ERROR_METRES,
    )
    # A short crossing may use one extra stock module to meet both roads.
    assert fitted.points[0][0] < 72.5
    assert fitted.points[1][0] > 147.5


def test_long_dry_source_bridge_does_not_redefine_water_crossing() -> None:
    """A Tostero-style dry OSM bridge must not turn ~3 wet modules into ~10."""
    spec = _spec()
    points = ((100.0, 100.0), (500.0, 100.0))
    plan = bridge.StockBridgeSpanPlan(
        points=((224.7, 100.0), (375.3, 100.0)),
        module_count=3,
        wet_start=(225.0, 100.0),
        wet_end=(375.0, 100.0),
        wet_length=150.0,
    )
    feature = SimpleNamespace()
    start_path = ((100.0, 100.0), (0.0, 100.0))
    end_path = ((500.0, 100.0), (700.0, 100.0))

    def connected(_feature, _dataset, _projection, endpoint, *_args):
        return start_path if endpoint[0] < 300.0 else end_path

    token = source._CONTEXT.set(SimpleNamespace(dataset=object(), projection=object()))
    try:
        with (
            patch.object(
                policy,
                "_matching_bridge_feature",
                return_value=(feature, points),
            ),
            patch.object(
                policy._osm,
                "_connected_bridge_approach_path",
                side_effect=connected,
            ),
            patch.object(
                policy,
                "_road_surface_at",
                return_value=5.6,
            ),
        ):
            fitted = policy._connected_stock_plan(
                plan,
                points,
                (),
                spec,
            )
    finally:
        source._CONTEXT.reset(token)

    # Connected dry roads would demand roughly ten stock modules here.  Keep
    # the wet-authoritative three-module plan and let terrain/road grading solve
    # the approaches instead.
    assert fitted == plan


def test_connected_adjustment_rejects_large_endpoint_drift() -> None:
    plan = bridge.StockBridgeSpanPlan(
        points=((100.0, 100.0), (250.0, 100.0)),
        module_count=3,
        wet_start=(100.0, 100.0),
        wet_end=(250.0, 100.0),
        wet_length=150.0,
    )

    # Preserving bridge length is not enough: a fitted bridge must not slide an
    # entire crossing away from its water-derived position.
    assert not policy._connected_adjustment_is_bounded(
        plan,
        3,
        (40.0, 100.0),
        (190.0, 100.0),
    )

    assert policy._connected_adjustment_is_bounded(
        plan,
        4,
        (75.0, 100.0),
        (275.0, 100.0),
    )


def test_missing_safe_approach_keeps_existing_bridge_plan() -> None:
    spec = _spec()
    points = ((100.0, 100.0), (120.0, 100.0))
    plan = bridge.StockBridgeSpanPlan(
        points=((84.9, 100.0), (135.1, 100.0)),
        module_count=1,
        wet_start=(105.0, 100.0),
        wet_end=(115.0, 100.0),
        wet_length=10.0,
    )
    feature = SimpleNamespace()
    start_path = ((100.0, 100.0), (0.0, 100.0))
    end_path = ((120.0, 100.0), (240.0, 100.0))

    def connected(_feature, _dataset, _projection, endpoint, *_args):
        return start_path if endpoint[0] < 110.0 else end_path

    token = source._CONTEXT.set(SimpleNamespace(dataset=object(), projection=object()))
    try:
        with (
            patch.object(
                policy,
                "_matching_bridge_feature",
                return_value=(feature, points),
            ),
            patch.object(
                policy._osm,
                "_connected_bridge_approach_path",
                side_effect=connected,
            ),
            patch.object(
                policy,
                "_road_surface_at",
                return_value=2.0,
            ),
        ):
            fitted = policy._connected_stock_plan(
                plan,
                points,
                (),
                spec,
            )
    finally:
        source._CONTEXT.reset(token)

    assert fitted == plan
