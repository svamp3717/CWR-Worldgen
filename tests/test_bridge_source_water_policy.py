from __future__ import annotations

from types import SimpleNamespace

from cwr_worldgen import bridge_render_policy as bridge_render
from cwr_worldgen import bridge_source_water_policy as source_water
from cwr_worldgen import terrain_solver
from cwr_worldgen.osm import (
    BboxProjection,
    GeoPolygon,
    OsmDataset,
    OsmLineFeature,
    OsmPolygonFeature,
)


def _fixture():
    projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 100.0)

    def ll(x: float, z: float):
        return projection.to_latlon((x, z))

    # A 20 m explicit bridge crosses only a 10 m mapped-water strip. Every
    # terrain sample used below is +2 m, so the historical terrain-only water
    # test says this is dry on a coarse grid.
    road = OsmLineFeature(
        "way/bridge",
        {"highway": "primary", "bridge": "yes", "layer": "1"},
        (ll(40.0, 50.0), ll(60.0, 50.0)),
    )
    water = OsmPolygonFeature(
        "way/water",
        {"natural": "water"},
        (
            GeoPolygon(
                outer=(
                    ll(45.0, 40.0),
                    ll(55.0, 40.0),
                    ll(55.0, 60.0),
                    ll(45.0, 60.0),
                    ll(45.0, 40.0),
                )
            ),
        ),
    )
    dataset = OsmDataset(
        source_generator="test",
        element_count=2,
        coastlines=(),
        water=(water,),
        forests=(),
        farmland=(),
        urban=(),
        roads=(road,),
    )
    points = tuple(projection.to_world(point) for point in road.points)
    elevations = (2.0,) * 16
    spec = SimpleNamespace(
        cells=4,
        cell_size=25.0,
        sea_level=0.0,
        world_size=100.0,
    )
    return dataset, projection, points, elevations, spec


def test_mapped_water_fills_coarse_terrain_bridge_blind_spot() -> None:
    dataset, projection, points, elevations, spec = _fixture()
    context = source_water._make_context(dataset, projection)
    interval = source_water._source_mapped_water_interval(points, context)

    assert interval is not None
    assert abs(interval[0] - 5.0) < 0.05
    assert abs(interval[1] - 15.0) < 0.05

    token = source_water._CONTEXT.set(context)
    try:
        assert not source_water._ORIGINAL_WATER_TEST(
            points,
            elevations,
            cells=spec.cells,
            cell_size=spec.cell_size,
            sea_level=spec.sea_level,
            width=7.0,
        )
        assert source_water._source_aware_water_test(
            points,
            elevations,
            cells=spec.cells,
            cell_size=spec.cell_size,
            sea_level=spec.sea_level,
            width=7.0,
        )
    finally:
        source_water._CONTEXT.reset(token)


def test_mapped_water_fallback_emits_one_stock_module_not_full_dry_way() -> None:
    dataset, projection, points, elevations, spec = _fixture()
    context = source_water._make_context(dataset, projection)
    token = source_water._CONTEXT.set(context)
    try:
        plan = source_water._mapped_water_stock_plan(points, elevations, spec)
    finally:
        source_water._CONTEXT.reset(token)

    assert plan is not None
    assert plan.module_count == 1
    assert abs(plan.wet_length - 10.0) < 0.1
    assert abs(plan.length - bridge_render._STOCK_MODULE_SPACING_METRES) < 1e-9
    assert plan.length < 60.0


def test_ordinary_water_road_fallback_is_tide_safe() -> None:
    assert source_water._TIDE_SAFE_ROAD_CLEARANCE_METRES == 5.5
    assert terrain_solver.ROAD_WATER_MINIMUM_CLEARANCE_METRES >= 5.5


def test_no_mapped_water_does_not_invent_bridge_water() -> None:
    dataset, projection, points, _elevations, _spec = _fixture()
    dry_dataset = OsmDataset(
        source_generator=dataset.source_generator,
        element_count=1,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=dataset.roads,
    )
    context = source_water._make_context(dry_dataset, projection)
    assert source_water._source_mapped_water_interval(points, context) is None
